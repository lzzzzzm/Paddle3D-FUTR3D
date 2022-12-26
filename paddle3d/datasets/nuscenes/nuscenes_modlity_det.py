# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path as osp
import pickle
from functools import reduce
from typing import List, Optional, Union
import numbers

import numpy as np
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from collections.abc import Mapping, Sequence
from typing import List

import paddle
from paddle3d.apis import manager
from paddle3d.datasets import BaseDataset
from paddle3d.sample import Sample
from paddle3d.geometries import BBoxes3D
from paddle3d.geometries import CoordMode
import paddle3d.transforms as T
from paddle3d.transforms.reader import LoadMultiViewImageFromFiles, Collect3D
from paddle3d.transforms.normalize import NormalizeMultiviewImage


@manager.DATASETS.add_component
class NuscenesModlityDataset(BaseDataset):
    """
    """

    def __init__(self,
                 anno_file: str,
                 dataset_root: str,
                 mode='train',
                 with_velocity=True,
                 use_valid_flag=True,
                 transforms=None,
                 use_modality=None):
        super(NuscenesModlityDataset, self).__init__()
        self.dataset_root = dataset_root
        self.mode = mode
        self.with_velocity = with_velocity
        self.use_valid_flag = use_valid_flag
        self.use_modality = use_modality
        if self.use_modality is None:
            self.use_modality = dict(
                use_camera=True,
                use_lidar=False,
                use_radar=True,
                use_map=False,
                use_external=False,
            )
        self.CLASSES = ('car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                        'traffic_cone')
        self.transforms = transforms
        if self.transforms is None:
            self.transforms = [
                LoadMultiViewImageFromFiles(),
                NormalizeMultiviewImage(
                    mean=[103.530, 116.280, 123.675],
                    std=[1.0, 1.0, 1.0]
                ),
                Collect3D(key=['img'])
            ]

        if isinstance(self.transforms, list):
            self.transforms = T.Compose(self.transforms)

        self.data_infos = self.load_annotations(anno_file)

    def load_annotations(self, anno_file):
        """Load annotations from ann_file.
        Args:
            ann_file (str): Path of the annotation file.
        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        # print('ann_file:', ann_file)
        # ann_file = 'data/nuscenes/radar_nuscenes_5sweeps_infos_val.pkl'
        file = open(anno_file, "rb")
        data = pickle.load(file)
        file.close()
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::1]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def load_anno_info(self, index, sample):
        info = self.data_infos[index]
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)
        sample.labels = gt_labels_3d
        origin = np.array([0.5, 0.5, 0.5])
        dst = np.array([0.5, 0.5, 0])
        src = np.array(origin)
        gt_bboxes_3d[:, :3] += gt_bboxes_3d[:, 3:6] * (dst - src)
        sample.bboxes_3d = paddle.to_tensor(gt_bboxes_3d, dtype='float32')
        bottom_center = gt_bboxes_3d[:, :3]
        # calc gravity_center
        gravity_center = paddle.zeros_like(paddle.to_tensor(bottom_center), dtype='float32')
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + gt_bboxes_3d[:, 5] * 0.5
        sample['gravity_center'] = gravity_center
        return sample

    def get_data_info(self, index):
        info = self.data_infos[index]
        path = {
            'image_paths': None,
            'lidar_path': None,
            'radar_paths': None
        }
        if self.use_modality['use_camera']:
            image_paths = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
            path['image_paths'] = image_paths
        if self.use_modality['use_radar']:
            path['radar_paths'] = info['radars']
        if self.use_modality['use_lidar']:
            path['lidar_path'] = info['lidar_path']

        sample = Sample(path=path, modality='multimodal')
        sample.sample_idx = info['token']
        sample.timestamp = info['timestamp'] / 1e6
        if self.use_modality['use_camera']:
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            for cam_type, cam_info in info['cams'].items():
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                                  'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt.T)
            init_dict = {
                'lidar2img': lidar2img_rts,
                'intrinsics': intrinsics,
                'extrinsics': extrinsics
            }
            sample['img_meta'] = []
            sample['img_meta'].append(init_dict)

        if self.is_train_mode:
            sample = self.load_anno_info(index, sample)

        return sample

    def __getitem__(self, index):
        sample = self.get_data_info(index)
        sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.data_infos)

    def collate_fn(self, batch: List):
        """
        """
        sample = batch[0]
        if isinstance(sample, np.ndarray):
            batch = np.stack(batch, axis=0)
            return batch
        # TODO
        elif isinstance(sample, List):
            return batch
        elif isinstance(sample, paddle.Tensor):
            return paddle.stack(batch, axis=0)
        elif isinstance(sample, numbers.Number):
            batch = np.array(batch)
            return batch
        elif isinstance(sample, (str, bytes)):
            return batch
        elif isinstance(sample, Sample):
            valid_keys = [
                key for key, value in sample.items() if value is not None
            ]
            self.padding_sample(batch)

            return {
                key: self.collate_fn([d[key] for d in batch])
                for key in valid_keys
            }
        elif isinstance(sample, Mapping):
            return {
                key: self.collate_fn([d[key] for d in batch])
                for key in sample
            }
        elif isinstance(sample, Sequence):
            sample_fields_num = len(sample)
            if not all(
                    len(sample) == sample_fields_num for sample in iter(batch)):
                raise RuntimeError(
                    "fileds number not same among samples in a batch")
            return [self.collate_fn(fields) for fields in zip(*batch)]


        elif isinstance(sample, type(None)):
            return batch

        raise TypeError(
            "batch data con only contains: tensor, numpy.ndarray, "
            "dict, list, number, paddle3d.Sample, but got {}".format(
                type(sample)))
