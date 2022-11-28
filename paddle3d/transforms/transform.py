# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

from typing import Any, List, Tuple, Union

import numpy as np
import cv2

from paddle3d.apis import manager
from paddle3d.geometries.bbox import points_in_convex_polygon_3d_jit
from paddle3d.sample import Sample
from paddle3d.transforms import functional as F
from paddle3d.transforms.base import TransformABC
from paddle3d.transforms.functional import points_to_voxel

__all__ = [
    "RandomHorizontalFlip", "RandomVerticalFlip", "GlobalRotate", "GlobalScale",
    "GlobalTranslate", "ShufflePoint", "FilterBBoxOutsideRange", "HardVoxelize",
    "RandomObjectPerturb"
]


@manager.TRANSFORMS.add_component
class RandomHorizontalFlip(TransformABC):
    """
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, sample: Sample):
        if np.random.random() < self.prob:
            if sample.modality == "image":
                sample.data = F.horizontal_flip(sample.data)
                h, w, c = sample.data.shape
            elif sample.modality == "lidar":
                sample.data.flip(axis=1)

            # Flip camera intrinsics
            if "camera_intrinsic" in sample.meta:
                sample.meta.camera_intrinsic[
                    0, 2] = w - sample.meta.camera_intrinsic[0, 2] - 1

            # Flip bbox
            if sample.bboxes_3d is not None:
                sample.bboxes_3d.horizontal_flip()
            if sample.bboxes_2d is not None and sample.modality == "image":
                sample.bboxes_2d.horizontal_flip(image_width=w)
        return sample


@manager.TRANSFORMS.add_component
class RandomVerticalFlip(TransformABC):
    """
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, sample: Sample):
        if np.random.random() < self.prob:
            if sample.modality == "image":
                sample.data = F.vertical_flip(sample.data)
                h, w, c = sample.data.shape
            elif sample.modality == "lidar":
                sample.data.flip(axis=0)

            # Flip camera intrinsics
            if "camera_intrinsic" in sample.meta:
                sample.meta.camera_intrinsic[
                    1, 2] = h - sample.meta.camera_intrinsic[1, 2] - 1

            # Flip bbox
            if sample.bboxes_3d is not None:
                sample.bboxes_3d.vertical_flip()
            if sample.bboxes_2d is not None and sample.modality == "image":
                sample.bboxes_2d.vertical_flip(image_height=h)

        return sample


@manager.TRANSFORMS.add_component
class GlobalRotate(TransformABC):
    """
    """

    def __init__(self, min_rot: float = -np.pi / 4, max_rot: float = np.pi / 4):
        self.min_rot = min_rot
        self.max_rot = max_rot

    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError("GlobalRotate only supports lidar data!")
        angle = np.random.uniform(self.min_rot, self.max_rot)
        # Rotate points
        sample.data.rotate_around_z(angle)
        # Rotate bboxes_3d
        if sample.bboxes_3d is not None:
            sample.bboxes_3d.rotate_around_z(angle)
        return sample


@manager.TRANSFORMS.add_component
class GlobalScale(TransformABC):
    """
    """

    def __init__(self,
                 min_scale: float = 0.95,
                 max_scale: float = 1.05,
                 size=None):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.size = size

    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError("GlobalScale only supports lidar data!")
        factor = np.random.uniform(
            self.min_scale, self.max_scale, size=self.size)
        # Scale points
        sample.data.scale(factor)
        # Scale bboxes_3d
        if sample.bboxes_3d is not None:
            sample.bboxes_3d.scale(factor)
        return sample


@manager.TRANSFORMS.add_component
class GlobalTranslate(TransformABC):
    """
    Translate sample by a random offset.

    Args:
        translation_std (Union[float, List[float], Tuple[float]], optional):
            The standard deviation of the translation offset. Defaults to (.2, .2, .2).
        distribution (str):
            The random distribution. Defaults to normal.
    """

    def __init__(
            self,
            translation_std: Union[float, List[float], Tuple[float]] = (.2, .2,
                                                                        .2),
            distribution="normal"):
        if not isinstance(translation_std, (list, tuple)):
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        self.translation_std = translation_std
        self.distribution = distribution

    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError("GlobalScale only supports lidar data!")
        if self.distribution not in ["normal", "uniform"]:
            raise ValueError(
                "GlobalScale only supports normal and uniform random distribution!"
            )

        if self.distribution == "normal":
            translation = np.random.normal(scale=self.translation_std, size=3)
        elif self.distribution == "uniform":
            translation = np.random.uniform(
                low=-self.translation_std[0],
                high=self.translation_std[0],
                size=3)
        else:
            raise ValueError(
                "GlobalScale only supports normal and uniform random distribution!"
            )

        sample.data.translate(translation)
        if sample.bboxes_3d is not None:
            sample.bboxes_3d.translate(translation)

        return sample


@manager.TRANSFORMS.add_component
class ShufflePoint(TransformABC):
    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError("ShufflePoint only supports lidar data!")
        sample.data.shuffle()
        return sample


@manager.TRANSFORMS.add_component
class FilterBBoxOutsideRange(TransformABC):
    def __init__(self, point_cloud_range: Tuple[float]):
        self.point_cloud_range = np.asarray(point_cloud_range, dtype='float32')

    def __call__(self, sample: Sample):
        if sample.bboxes_3d.size == 0:
            return sample
        mask = sample.bboxes_3d.get_mask_of_bboxes_outside_range(
            self.point_cloud_range)
        sample.bboxes_3d = sample.bboxes_3d.masked_select(mask)
        sample.labels = sample.labels[mask]
        return sample


@manager.TRANSFORMS.add_component
class HardVoxelize(TransformABC):
    def __init__(self, point_cloud_range: Tuple[float],
                 voxel_size: Tuple[float], max_points_in_voxel: int,
                 max_voxel_num: int):
        self.max_points_in_voxel = max_points_in_voxel
        self.max_voxel_num = max_voxel_num
        self.voxel_size = np.asarray(voxel_size, dtype='float32')
        self.point_cloud_range = np.asarray(point_cloud_range, dtype='float32')
        self.grid_size = np.round(
            (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) /
            self.voxel_size).astype('int32')

    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError("Voxelize only supports lidar data!")

        # Voxelize
        num_points, num_point_dim = sample.data.shape[0:2]
        voxels = np.zeros(
            (self.max_voxel_num, self.max_points_in_voxel, num_point_dim),
            dtype=sample.data.dtype)
        coords = np.zeros((self.max_voxel_num, 3), dtype=np.int32)
        num_points_per_voxel = np.zeros((self.max_voxel_num, ), dtype=np.int32)
        grid_size_z, grid_size_y, grid_size_x = self.grid_size[::-1]
        grid_idx_to_voxel_idx = np.full((grid_size_z, grid_size_y, grid_size_x),
                                        -1,
                                        dtype=np.int32)

        num_voxels = points_to_voxel(
            sample.data, self.voxel_size, self.point_cloud_range,
            self.grid_size, voxels, coords, num_points_per_voxel,
            grid_idx_to_voxel_idx, self.max_points_in_voxel, self.max_voxel_num)

        voxels = voxels[:num_voxels]
        coords = coords[:num_voxels]
        num_points_per_voxel = num_points_per_voxel[:num_voxels]

        sample.voxels = voxels
        sample.coords = coords
        sample.num_points_per_voxel = num_points_per_voxel

        sample.pop('sweeps', None)
        return sample


@manager.TRANSFORMS.add_component
class RandomObjectPerturb(TransformABC):
    """
    Randomly perturb (rotate and translate) each object.

    Args:
        rotation_range (Union[float, List[float], Tuple[float]], optional):
            Range of random rotation. Defaults to pi / 4.
        translation_std (Union[float, List[float], Tuple[float]], optional):
            Standard deviation of random translation. Defaults to 1.0.
        max_num_attempts (int): Maximum number of perturbation attempts. Defaults to 100.
    """

    def __init__(
            self,
            rotation_range: Union[float, List[float], Tuple[float]] = np.pi / 4,
            translation_std: Union[float, List[float], Tuple[float]] = 1.0,
            max_num_attempts: int = 100):
        if not isinstance(rotation_range, (list, tuple)):
            rotation_range = [-rotation_range, rotation_range]
        self.rotation_range = rotation_range
        if not isinstance(translation_std, (list, tuple)):
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        self.translation_std = translation_std
        self.max_num_attempts = max_num_attempts

    def __call__(self, sample: Sample):
        num_objects = sample.bboxes_3d.shape[0]
        rotation_noises = np.random.uniform(
            self.rotation_range[0],
            self.rotation_range[1],
            size=[num_objects, self.max_num_attempts])
        translation_noises = np.random.normal(
            scale=self.translation_std,
            size=[num_objects, self.max_num_attempts, 3])
        rotation_noises, translation_noises = F.noise_per_box(
            sample.bboxes_3d[:, [0, 1, 3, 4, 6]], sample.bboxes_3d.corners_2d,
            sample.ignored_bboxes_3d.corners_2d, rotation_noises,
            translation_noises)

        # perturb points w.r.t objects' centers (inplace operation)
        normals = F.corner_to_surface_normal(sample.bboxes_3d.corners_3d)
        point_masks = points_in_convex_polygon_3d_jit(sample.data[:, :3],
                                                      normals)
        F.perturb_object_points_(sample.data, sample.bboxes_3d[:, :3],
                                 point_masks, rotation_noises,
                                 translation_noises)

        # perturb bboxes_3d w.r.t to objects' centers (inplace operation)
        F.perturb_object_bboxes_3d_(sample.bboxes_3d, rotation_noises,
                                    translation_noises)

        return sample

@manager.TRANSFORMS.add_component
class PadMultiViewImage(TransformABC):
    """
        only support ModlitySample
    """
    def __init__(self, size=None, size_divisor=None, pad_val=0, padding_mode='constant'):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        border_type = {
            'constant': cv2.BORDER_CONSTANT,
            'edge': cv2.BORDER_REPLICATE,
            'reflect': cv2.BORDER_REFLECT_101,
            'symmetric': cv2.BORDER_REFLECT
        }
        self.padding_mode = border_type[padding_mode]
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def pad_img(self, img, shape, padding=None, pad_val=0, padding_mode='constant'):
        width = max(shape[1] - img.shape[1], 0)
        height = max(shape[0] - img.shape[0], 0)
        padding = (0, 0, width, height)
        img = cv2.copyMakeBorder(
            img,
            padding[1],
            padding[3],
            padding[0],
            padding[2],
            self.padding_mode,
            value=pad_val)
        return img

    def impad_to_multiple(self, img, divisor, pad_val=0):
        pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
        pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
        return self.pad_img(img, shape=(pad_h, pad_w), pad_val=pad_val)

    def _pad_img(self, sample):
        """
        Pad images according to ``self.size``.
        img: [N, H, W, C]
        """
        if self.size is not None:
            padded_img = [self.pad_img(img, shape=self.size, pad_val=self.pad_val) for img in sample.img]
        elif self.size_divisor is not None:
            padded_img = [self.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in sample.img]

        sample.img = padded_img
        updatae_dict = {
            'img_shape' : [img.shape for img in padded_img],
            'pad_shape' : [img.shape for img in padded_img],
            'pad_fixed_size' : self.size,
            'pad_size_divisor' : self.size_divisor
        }
        sample.img_meta[0].update(updatae_dict)
        # sample.img_meta['img_shape'] = [img.shape for img in padded_img]
        # sample.img_meta['pad_shape'] = [img.shape for img in padded_img]
        # sample.img_meta['pad_fixed_size'] = self.size
        # sample.img_meta['pad_size_divisor'] = self.size_divisor
        return sample

    def __call__(self, sample):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        sample = self._pad_img(sample)
        return sample