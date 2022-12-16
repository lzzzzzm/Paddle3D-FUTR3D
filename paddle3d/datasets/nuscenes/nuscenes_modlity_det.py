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

import paddle
from paddle3d.apis import manager
from paddle3d.datasets import BaseDataset
from paddle3d.sample import ModlitySample, Sample
import paddle3d.transforms as T
from paddle3d.transforms.reader import LoadMultiViewImageFromFiles, Collect3D
from paddle3d.transforms.normalize import NormalizeMultiviewImage

@manager.DATASETS.add_component
class NuscenesModlityDataset(BaseDataset):
    """
    """
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')


    def __init__(self,
                 anno_file: str,
                 dataset_root: str,
                 mode='train',
                 use_valid_flag=True,
                 transforms=None,
                 use_modality=None):
        super(NuscenesModlityDataset, self).__init__()
        self.dataset_root = dataset_root
        self.mode = mode
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
        self.transforms = transforms
        if self.transforms is None:
            self.transforms=[
                LoadMultiViewImageFromFiles(),
                NormalizeMultiviewImage(
                    mean=[103.530, 116.280, 123.675],
                    std=[1.0, 1.0, 1.0]
                ),
                Collect3D(key=['img'])
            ]

        if isinstance(self.transforms, list):
            self.transforms= T.Compose(self.transforms)

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
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)
        sample.labels = gt_labels_3d
        sample.bboxes_3d = gt_bboxes_3d

        return sample

    def get_data_info(self, index):
        info = self.data_infos[index]
        modlity_sample = ModlitySample(use_modality=self.use_modality)
        modlity_sample.sample_idx = info['token']
        modlity_sample.lidar_path = info['lidar_path']
        # modlity_sample.sweeps = info['sweeps']
        modlity_sample.timestamp = info['timestamp'] / 1e6
        modlity_sample.radar = info['radars']
        if self.use_modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
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
                'img_filename' : image_paths,
                'lidar2img' : lidar2img_rts,
                'intrinsics' : intrinsics,
                'extrinsics' : extrinsics
            }
            modlity_sample.img_meta.append(init_dict)
            # modlity_sample.img_meta['img_filename'] = image_paths
            # modlity_sample.img_meta['lidar2img'] = lidar2img_rts
            # modlity_sample.img_meta['img_filename'] = image_paths
            # modlity_sample.img_meta['img_filename'] = image_paths
            # modlity_sample.img_meta.img_filename = image_paths
            # modlity_sample.img_meta.lidar2img = lidar2img_rts
            # modlity_sample.img_meta.intrinsics = intrinsics
            # modlity_sample.img_meta.extrinsics = extrinsics
        if not self.is_test_mode:
            modlity_sample = self.load_anno_info(index, modlity_sample)

        return modlity_sample


    def __getitem__(self, index):
        modlity_sample = self.get_data_info(index)
        modlity_sample = self.transforms(modlity_sample)
        return modlity_sample

    def __len__(self):
        return len(self.data_infos)

    def collate_fn(self, batch):

        sample = ModlitySample(use_modality=self.use_modality)
        # img batch
        if self.use_modality['use_camera']:
            sample.img = paddle.stack([batch[i].img for i in range(len(batch))], axis=0)
            for i in range(len(batch)):
                sample.img_meta.append(*batch[i].img_meta)
        # radar batch
        if self.use_modality['use_radar']:
            sample.radar = paddle.concat([batch[i].radar for i in range(len(batch))], axis=0)
        # label
        return sample





# if __name__ == '__main__':
#     dataset = NuscenesModlityDataset(
#         anno_file='F:\Pycharm_Project\论文复现\Paddle3D\data\\nuscenes\\radar_nuscenes_5sweeps_infos_val.pkl',
#         dataset_root='F:\Pycharm_Project\论文复现\Paddle3D\data\\nuscenes',
#         use_modality=None
#     )
    # modlity_sample = dataset.get_data_info(0)
    # print(modlity_sample)

    # for data in dataset:
    #     break
