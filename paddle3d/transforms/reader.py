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

from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import paddle
from PIL import Image

import paddle.vision.transforms as T

from paddle3d.apis import manager
from paddle3d.datasets.kitti import kitti_utils
from paddle3d.datasets.semantic_kitti.semantic_kitti import \
    SemanticKITTIDataset
from paddle3d.geometries import PointCloud, RadarPointCloud
from paddle3d.geometries.bbox import points_in_convex_polygon_3d_jit
from paddle3d.sample import Sample
from paddle3d.transforms import functional as F
from paddle3d.transforms.base import TransformABC
from paddle3d.utils.logger import logger

__all__ = [
    "LoadImage", "LoadMultiViewImageFromFiles",
    "LoadPointCloud", "LoadRadarPointsMultiSweeps", "RemoveCameraInvisiblePointsKITTI",
    "LoadSemanticKITTIRange", 'ObjectRangeFilter',
    "Collect3D"
]


@manager.TRANSFORMS.add_component
class Collect3D(TransformABC):
    """
        only support multimodel sample
        'img': (1) transpose (2) to tensor (3) collect img_meta --- use paddle transform to_tensor
        'radar' (1) to_tensor
    """

    def __init__(self, key):
        self.key = key
        use_camera, use_lidar, use_radar, use_map, use_external = False, False, False, False, False
        if 'img' in self.key:
            use_camera = True
        if 'lidar' in self.key:
            use_lidar = True
        if 'radar' in self.key:
            use_radar = True
        if 'map' in self.key:
            use_map = True
        if 'use_external' in self.key:
            use_external = True
        self.use_modality = dict(
            use_camera=use_camera,
            use_lidar=use_lidar,
            use_radar=use_radar,
            use_map=use_map,
            use_external=use_external,
        )
        self.to_tensor = T.ToTensor()


    def __call__(self, sample):
        collect_sample = Sample(path=sample.path, modality=sample.modality)
        if 'img' in self.key:
            collect_sample['img'] = paddle.stack([self.to_tensor(img) for img in sample.img])
            collect_sample['img_meta'] = sample.img_meta
            # collect_sample.img = paddle.stack([self.to_tensor(img) for img in sample.img])
            # collect_sample.img_meta = sample.img_meta
        if 'radar' in self.key:
            collect_sample['radar'] = self.to_tensor(sample.radar)
            # collect_sample.radar = self.to_tensor(sample.radar)
        # collect label
        if sample.labels is not None:
            collect_sample.labels = sample.labels
            collect_sample.bboxes_3d = sample.bboxes_3d
            collect_sample['gravity_center'] = sample.gravity_center
            # collect_sample.update({
            #     'gravity_center': sample.gravity_center
            # })

        return collect_sample


@manager.TRANSFORMS.add_component
class LoadImage(TransformABC):
    """
    """
    _READER_MAPPER = {"cv2": cv2.imread, "pillow": Image.open}

    def __init__(self,
                 to_chw: bool = True,
                 to_rgb: bool = True,
                 reader: str = "cv2"):
        if reader not in self._READER_MAPPER.keys():
            raise ValueError('Unsupported reader {}'.format(reader))

        self.reader = reader
        self.to_rgb = to_rgb
        self.to_chw = to_chw

    def __call__(self, sample: Sample) -> Sample:
        """
        """
        sample.data = np.array(self._READER_MAPPER[self.reader](sample.path))

        sample.meta.image_reader = self.reader
        sample.meta.image_format = "bgr" if self.reader == "cv2" else "rgb"
        sample.meta.channel_order = "hwc"

        if sample.meta.image_format != "rgb" and self.to_rgb:
            if sample.meta.image_format == "bgr":
                sample.data = cv2.cvtColor(sample.data, cv2.COLOR_BGR2RGB)
                sample.meta.image_format = "rgb"
            else:
                raise RuntimeError('Unsupported image format {}'.format(
                    sample.meta.image_format))

        if self.to_chw:
            sample.data = sample.data.transpose((2, 0, 1))
            sample.meta.channel_order = "chw"

        return sample


@manager.TRANSFORMS.add_component
class LoadMultiViewImageFromFiles(TransformABC):
    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, sample):
        """
         only support multimodel sample
        """
        filename = sample.path['image_paths']
        # img is of shape (h, w, c, num_views)
        img = []
        for name in filename:
            im = cv2.imread(name)
            im = cv2.resize(im, (im.shape[0]//4, im.shape[1]//4))
            img.append(im)
        img = np.array(img)
        img = np.transpose(img, (1, 2, 3, 0))
        # img = np.stack(
        #     [cv2.imread(name) for name in filename], axis=-1)

        if self.to_float32:
            img = img.astype(np.float32)

        sample['img'] = np.array([img[..., i] for i in range(img.shape[-1])])
        # sample.img = np.array([img[..., i] for i in range(img.shape[-1])])
        update_dict = {
            'img_shape' : img.shape,
            'ori_shape' : img.shape,
            'pad_shape' : img.shape,
            'scale_factor' : 1.0
        }
        sample.img_meta[0].update(update_dict)
        return sample

@manager.TRANSFORMS.add_component
class LoadRadarPointsMultiSweeps(TransformABC):
    """
        only support ModlitySample
    """

    def __init__(self,
                 load_dim=18,
                 use_dim=[0, 1, 2, 3, 4],
                 sweeps_num=3,
                 max_num=300,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 test_mode=False):
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.sweeps_num = sweeps_num
        self.max_num = max_num
        self.test_mode = test_mode
        self.pc_range = pc_range

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.
        Args:
            pts_filename (str): Filename of point clouds data.
        Returns:
            np.ndarray: An array containing point clouds data.
            [N, 18]
        """
        radar_obj = RadarPointCloud.from_file(pts_filename)

        #[18, N]
        points = radar_obj.points

        return points.transpose().astype(np.float32)

    def _pad_or_drop(self, points):
        '''
        points: [N, 18]
        '''

        num_points = points.shape[0]

        if num_points == self.max_num:
            masks = np.ones((num_points, 1),
                            dtype=points.dtype)

            return points, masks

        if num_points > self.max_num:
            points = np.random.permutation(points)[:self.max_num, :]
            masks = np.ones((self.max_num, 1),
                            dtype=points.dtype)

            return points, masks

        if num_points < self.max_num:
            zeros = np.zeros((self.max_num - num_points, points.shape[1]),
                             dtype=points.dtype)
            masks = np.ones((num_points, 1),
                            dtype=points.dtype)

            points = np.concatenate((points, zeros), axis=0)
            masks = np.concatenate((masks, zeros.copy()[:, [0]]), axis=0)

            return points, masks

    def __call__(self, sample):
        radars_dict = sample.path['radar_paths']
        # radars_dict = sample.radar
        points_sweep_list = []
        for key, sweeps in radars_dict.items():
            if len(sweeps) < self.sweeps_num:
                idxes = list(range(len(sweeps)))
            else:
                idxes = list(range(self.sweeps_num))
            ts = sweeps[0]['timestamp'] * 1e-6
            for idx in idxes:
                sweep = sweeps[idx]

                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                timestamp = sweep['timestamp'] * 1e-6
                time_diff = ts - timestamp
                time_diff = np.ones((points_sweep.shape[0], 1)) * time_diff

                # velocity compensated by the ego motion in sensor frame
                velo_comp = points_sweep[:, 8:10]
                velo_comp = np.concatenate(
                    (velo_comp, np.zeros((velo_comp.shape[0], 1))), 1)
                velo_comp = velo_comp @ sweep['sensor2lidar_rotation'].T
                velo_comp = velo_comp[:, :2]

                # velocity in sensor frame
                velo = points_sweep[:, 6:8]
                velo = np.concatenate(
                    (velo, np.zeros((velo.shape[0], 1))), 1)
                velo = velo @ sweep['sensor2lidar_rotation'].T
                velo = velo[:, :2]

                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']

                points_sweep_ = np.concatenate(
                    [points_sweep[:, :6], velo,
                     velo_comp, points_sweep[:, 10:],
                     time_diff], axis=1)
                points_sweep_list.append(points_sweep_)

        points = np.concatenate(points_sweep_list, axis=0)

        points = points[:, self.use_dim]

        points[:, 0:1] = (points[:, 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        points[:, 1:2] = (points[:, 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        points[:, 2:3] = (points[:, 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

        if self.max_num > 0:
            points, mask = self._pad_or_drop(points)
        else:
            mask = np.ones((points.shape[0], points.shape[1]),
                           dtype=points.dtype)

        points = np.concatenate((points, mask), axis=-1)
        sample['radar'] = points.astype(np.float32)
        # sample.radar = points.astype(np.float32)
        return sample

@manager.TRANSFORMS.add_component
class ObjectRangeFilter(TransformABC):
    def __init__(self, point_cloud_range, bbox_instance, with_gravity_center=False):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)
        self.bbox_instance = bbox_instance
        self.with_gravity_center = with_gravity_center

    def __call__(self, sample):
        """Call function to filter objects by the range.

        Args:
            sample (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if self.bbox_instance == 'LiDARInstance3DBoxes' or self.bbox_instance == 'DepthInstance3DBoxes':
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif self.bbox_instance == 'CameraInstance3DBoxes':
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = sample['bboxes_3d']
        gt_labels_3d = sample['labels']
        if self.with_gravity_center:
            gravity_center = sample['gravity_center']
        bev_index = paddle.to_tensor([0, 1, 3, 4, 6])
        bbox_bev = paddle.index_select(gt_bboxes_3d, bev_index, axis=1)
        # bbox_bev = gt_bboxes_3d[:, [0, 1, 3, 4, 6]]
        mask = ((bbox_bev[:, 0] > bev_range[0])
                          & (bbox_bev[:, 1] > bev_range[1])
                          & (bbox_bev[:, 0] < bev_range[2])
                          & (bbox_bev[:, 1] < bev_range[3]))

        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]
        gravity_center = gravity_center[mask]
        # limit rad to [-pi, pi]
        gt_bboxes_3d[:, 6] = gt_bboxes_3d[:, 6] - paddle.floor(gt_bboxes_3d[:, 6] / (2*np.pi) + 0.5) * 2*np.pi
        sample['bboxes_3d'] = gt_bboxes_3d
        sample['labels'] = gt_labels_3d
        sample['gravity_center'] = gravity_center
        return sample


@manager.TRANSFORMS.add_component
class LoadPointCloud(TransformABC):
    """
    Load point cloud.

    Args:
        dim: The dimension of each point.
        use_dim: The dimension of each point to use.
        use_time_lag: Whether to use time lag.
        sweep_remove_radius: The radius within which points are removed in sweeps.
    """

    def __init__(self,
                 dim,
                 use_dim: Union[int, List[int]] = None,
                 use_time_lag: bool = False,
                 sweep_remove_radius: float = 1):
        self.dim = dim
        self.use_dim = range(use_dim) if isinstance(use_dim, int) else use_dim
        self.use_time_lag = use_time_lag
        self.sweep_remove_radius = sweep_remove_radius

    def __call__(self, sample: Sample):
        """
        """
        if sample.modality != "lidar":
            raise ValueError('{} Only Support samples in modality lidar'.format(
                self.__class__.__name__))

        if sample.data is not None:
            raise ValueError(
                'The data for this sample has been processed before.')

        data = np.fromfile(sample.path, np.float32).reshape(-1, self.dim)

        if self.use_dim is not None:
            data = data[:, self.use_dim]

        if self.use_time_lag:
            time_lag = np.zeros((data.shape[0], 1), dtype=data.dtype)
            data = np.hstack([data, time_lag])

        if len(sample.sweeps) > 0:
            data_sweep_list = [
                data,
            ]
            for i in np.random.choice(
                    len(sample.sweeps), len(sample.sweeps), replace=False):
                sweep = sample.sweeps[i]
                sweep_data = np.fromfile(sweep.path, np.float32).reshape(
                    -1, self.dim)
                if self.use_dim:
                    sweep_data = sweep_data[:, self.use_dim]
                sweep_data = sweep_data.T

                # Remove points that are in a certain radius from origin.
                x_filter_mask = np.abs(
                    sweep_data[0, :]) < self.sweep_remove_radius
                y_filter_mask = np.abs(
                    sweep_data[1, :]) < self.sweep_remove_radius
                not_close = np.logical_not(
                    np.logical_and(x_filter_mask, y_filter_mask))
                sweep_data = sweep_data[:, not_close]

                # Homogeneous transform of current sample to reference coordinate
                if sweep.meta.ref_from_curr is not None:
                    sweep_data[:3, :] = sweep.meta.ref_from_curr.dot(
                        np.vstack((sweep_data[:3, :],
                                   np.ones(sweep_data.shape[1]))))[:3, :]
                sweep_data = sweep_data.T
                if self.use_time_lag:
                    curr_time_lag = sweep.meta.time_lag * np.ones(
                        (sweep_data.shape[0], 1)).astype(sweep_data.dtype)
                    sweep_data = np.hstack([sweep_data, curr_time_lag])
                data_sweep_list.append(sweep_data)
            data = np.concatenate(data_sweep_list, axis=0)

        sample.data = PointCloud(data)
        return sample


@manager.TRANSFORMS.add_component
class RemoveCameraInvisiblePointsKITTI(TransformABC):
    """
    Remove camera invisible points for KITTI dataset.
    """

    def __call__(self, sample: Sample):
        calibs = sample.calibs
        C, Rinv, T = kitti_utils.projection_matrix_decomposition(calibs[2])

        im_path = (Path(sample.path).parents[1] / "image_2" / Path(
            sample.path).stem).with_suffix(".png")
        im_shape = np.array(cv2.imread(str(im_path)).shape[:2], dtype=np.int32)
        im_bbox = [0, 0, im_shape[1], im_shape[0]]

        frustum = F.get_frustum(im_bbox, C)
        frustum = (Rinv @ (frustum - T).T).T
        frustum = kitti_utils.coord_camera_to_velodyne(frustum, calibs)
        frustum_normals = F.corner_to_surface_normal(frustum[None, ...])

        indices = points_in_convex_polygon_3d_jit(sample.data[:, :3],
                                                  frustum_normals)
        sample.data = sample.data[indices.reshape([-1])]

        return sample


@manager.TRANSFORMS.add_component
class LoadSemanticKITTIRange(TransformABC):
    """
    Load SemanticKITTI range image.
    Please refer to <https://github.com/PRBonn/semantic-kitti-api/blob/master/auxiliary/laserscan.py>.

    Args:
        project_label (bool, optional): Whether project label to range view or not.
    """

    def __init__(self, project_label=True):
        self.project_label = project_label
        self.proj_H = 64
        self.proj_W = 1024
        self.upper_inclination = 3. / 180. * np.pi
        self.lower_inclination = -25. / 180. * np.pi
        self.fov = self.upper_inclination - self.lower_inclination

        self.remap_lut = SemanticKITTIDataset.build_remap_lut()

    def _remap_semantic_labels(self, sem_label):
        """
        Remap semantic labels to cross entropy format.
        Please refer to <https://github.com/PRBonn/semantic-kitti-api/blob/master/remap_semantic_labels.py>.
        """

        return self.remap_lut[sem_label]

    def __call__(self, sample: Sample) -> Sample:
        raw_scan = np.fromfile(sample.path, dtype=np.float32).reshape((-1, 4))
        points = raw_scan[:, 0:3]
        remissions = raw_scan[:, 3]

        # get depth of all points (L-2 norm of [x, y, z])
        depth = np.linalg.norm(points, ord=2, axis=1)

        # get angles of all points
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (
                pitch + abs(self.lower_inclination)) / self.fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        proj_x_copy = np.copy(
            proj_x
        )  # save a copy in original order, for each point, where it is in the range image

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        proj_y_copy = np.copy(
            proj_y
        )  # save a copy in original order, for each point, where it is in the range image

        # unproj_range_copy = np.copy(depth)   # copy of depth in original order

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = points[order]
        remission = remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # projected range image - [H,W] range (-1 is no data)
        proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        proj_remission = np.full((self.proj_H, self.proj_W),
                                 -1,
                                 dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)

        proj_range[proj_y, proj_x] = depth
        proj_xyz[proj_y, proj_x] = points
        proj_remission[proj_y, proj_x] = remission
        proj_idx[proj_y, proj_x] = indices
        proj_mask = proj_idx > 0  # mask containing for each pixel, if it contains a point or not

        sample.data = np.concatenate([
            proj_range[None, ...],
            proj_xyz.transpose([2, 0, 1]), proj_remission[None, ...]
        ])

        sample.meta["proj_mask"] = proj_mask.astype(np.float32)
        sample.meta["proj_x"] = proj_x_copy
        sample.meta["proj_y"] = proj_y_copy

        if sample.labels is not None:
            # load labels
            raw_label = np.fromfile(
                sample.labels, dtype=np.uint32).reshape((-1))
            # only fill in attribute if the right size
            if raw_label.shape[0] == points.shape[0]:
                sem_label = raw_label & 0xFFFF  # semantic label in lower half
                sem_label = self._remap_semantic_labels(sem_label)
                # inst_label = raw_label >> 16  # instance id in upper half
            else:
                logger.error("Point cloud shape: {}".format(points.shape))
                logger.error("Label shape: {}".format(raw_label.shape))
                raise ValueError(
                    "Scan and Label don't contain same number of points. {}".
                        format(sample.path))
            # # sanity check
            # assert ((sem_label + (inst_label << 16) == raw_label).all())

            if self.project_label:
                # project label to range view
                # semantics
                proj_sem_label = np.zeros((self.proj_H, self.proj_W),
                                          dtype=np.int32)  # [H,W]  label
                proj_sem_label[proj_mask] = sem_label[proj_idx[proj_mask]]

                # # instances
                # proj_inst_label = np.zeros((self.proj_H, self.proj_W),
                #                            dtype=np.int32)  # [H,W]  label
                # proj_inst_label[proj_mask] = self.inst_label[proj_idx[proj_mask]]

                sample.labels = proj_sem_label.astype(np.int64)[None, ...]
            else:
                sample.labels = sem_label.astype(np.int64)

        return sample


@manager.TRANSFORMS.add_component
class LoadSemanticKITTIPointCloud(TransformABC):
    """
    Load SemanticKITTI range image.
    Please refer to <https://github.com/PRBonn/semantic-kitti-api/blob/master/auxiliary/laserscan.py>.
    """

    def __init__(self, use_dim: List[int] = None):
        self.proj_H = 64
        self.proj_W = 1024
        self.upper_inclination = 3. / 180. * np.pi
        self.lower_inclination = -25. / 180. * np.pi
        self.fov = self.upper_inclination - self.lower_inclination

        self.remap_lut = SemanticKITTIDataset.build_remap_lut()

        self.use_dim = use_dim

    def _remap_semantic_labels(self, sem_label):
        """
        Remap semantic labels to cross entropy format.
        Please refer to <https://github.com/PRBonn/semantic-kitti-api/blob/master/remap_semantic_labels.py>.
        """

        return self.remap_lut[sem_label]

    def __call__(self, sample: Sample) -> Sample:
        raw_scan = np.fromfile(sample.path, dtype=np.float32).reshape(-1, 4)
        points = raw_scan[:, 0:3]

        sample.data = PointCloud(raw_scan[:, self.use_dim])

        if sample.labels is not None:
            # load labels
            raw_label = np.fromfile(sample.labels, dtype=np.int32).reshape(-1)
            # only fill in attribute if the right size
            if raw_label.shape[0] == points.shape[0]:
                sem_label = raw_label & 0xFFFF  # semantic label in lower half
                sem_label = self._remap_semantic_labels(sem_label)
                # self.inst_label = raw_label >> 16  # instance id in upper half
            else:
                logger.error("Point cloud shape: {}".format(points.shape))
                logger.error("Label shape: {}".format(raw_label.shape))
                raise ValueError(
                    "Scan and Label don't contain same number of points. {}".
                        format(sample.path))
            # # sanity check
            # assert ((sem_label + (inst_label << 16) == raw_label).all())

            sample.labels = sem_label

        return sample
