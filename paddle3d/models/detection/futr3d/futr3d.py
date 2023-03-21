import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.models.base import BaseMultiViewModel
from paddle3d.models.detection.futr3d.futr3d_utils import GridMask, bbox3d2result
from paddle3d.geometries import BBoxes3D
from paddle3d.sample import Sample, SampleMeta

from collections import OrderedDict

import numpy as np

__all__ = ["FUTR3D"]
import pickle

def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename

def load_variavle(filename):
   f=open(filename,'rb')
   r=pickle.load(f)
   f.close()
   return r

@manager.MODELS.add_component
class FUTR3D(BaseMultiViewModel):
    def __init__(self,
                 use_LiDAR=False,
                 use_Cam=True,
                 use_Radar=False,
                 use_grid_mask=True,
                 backbone=None,
                 radar_encoder=None,
                 neck=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_backbone=None,
                 pts_neck=None,
                 head=None,):
        super(FUTR3D, self).__init__()
        self.use_grid_mask = use_grid_mask
        if self.use_grid_mask:
            self.grid_mask = GridMask(use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_LiDAR = use_LiDAR
        self.use_Cam = use_Cam
        self.use_Radar = use_Radar
        self.pts_middle_encoder = pts_middle_encoder
        self.pts_backbone = pts_backbone
        self.pts_neck = pts_neck
        self.head = head
        self.backbone = backbone
        self.neck = neck
        self.radar_encoder = radar_encoder
        self.pts_voxel_encoder = pts_voxel_encoder
        # init weights
        self.init_weights()

    def init_weights(self, bias_lr_factor=0.1):
        for _, param in self.backbone.named_parameters():
            param.optimize_attr['learning_rate'] = bias_lr_factor
        if self.use_LiDAR:
            for _, param in self.pts_backbone.named_parameters():
                param.optimize_attr['learning_rate'] = bias_lr_factor
            for _, param in self.pts_middle_encoder.named_parameters():
                param.optimize_attr['learning_rate'] = bias_lr_factor
            for _, param in self.pts_voxel_encoder.named_parameters():
                param.optimize_attr['learning_rate'] = bias_lr_factor
        self.head.init_weights()

    def train(self):
        super(FUTR3D,self).train()
        self.backbone.train()

    def extract_img_feat(self, img, img_metas=None):
        """
            Extract features of images
        """
        B = img.shape[0]
        if img is not None:
            input_shape = img.shape[-2:]
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.shape[0] == 1:
                img = img.squeeze()
            elif img.dim() == 5 and img.shape[0] > 1:
                B, N, C, H, W = img.shape
                img = img.reshape((B*N, C, H, W))
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        for index, feat in enumerate(img_feats):
            if self.training:
                save_variable(feat.numpy(), '../torch_paddle/paddle_var/b_img_backbone_feats_{}.txt'.format(index))
            else:
                save_variable(feat.numpy(), '../torch_paddle/paddle_var/img_backbone_feats_{}.txt'.format(index))
        if self.with_img_neck:

            img_feats = self.neck(img_feats)
            for index, feat in enumerate(img_feats):
                if self.training:
                    save_variable(feat.numpy(), '../torch_paddle/paddle_var/b_img_neck_feats_{}.txt'.format(index))
                else:
                    save_variable(feat.numpy(), '../torch_paddle/paddle_var/img_neck_feats_{}.txt'.format(index))
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.shape
            img_feats_reshaped.append(img_feat.reshape((B, int(BN / B), C, H, W)))
        return img_feats_reshaped

    def voxelize_process(self, coordinates):

        coors_batch = []
        for i, coor in enumerate(coordinates):
            pad = nn.Pad1D((1, 0), mode='constant', value=i)
            coor_pad = pad(coordinates)
            coors_batch.append(coor_pad)
        coors_batch = paddle.concat(coors_batch, axis=0).squeeze()
        return coors_batch


    def extract_pts_feat(self, input_points):
        if not self.with_pts_bbox:
            return None
        points, voxels, coordinates, num_points = input_points
        voxels = voxels.squeeze()
        num_points = num_points.squeeze()
        coors = self.voxelize_process(coordinates)
        save_variable(voxels.numpy(), '../torch_paddle/paddle_var/voxels.txt')
        save_variable(coors.numpy(), '../torch_paddle/paddle_var/coors.txt')
        save_variable(num_points.numpy(), '../torch_paddle/paddle_var/num_points.txt')

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        save_variable(voxel_features.numpy(), '../torch_paddle/paddle_var/voxel_features.txt')

        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size.item())
        save_variable(x.numpy(), '../torch_paddle/paddle_var/pts_middle_encoder.txt')
        x = self.pts_backbone(x)
        for index, feat in enumerate(x):
            if self.training:
                save_variable(feat.numpy(), '../torch_paddle/paddle_var/b_pts_backbone_{}.txt'.format(index))
            else:
                save_variable(feat.numpy(), '../torch_paddle/paddle_var/pts_backbone_{}.txt'.format(index))
        if self.with_pts_neck:
            x = self.pts_neck(x)

        for index, feat in enumerate(x):
            if self.training:
                save_variable(feat.numpy(), '../torch_paddle/paddle_var/b_pts_neck_{}.txt'.format(index))
            else:
                save_variable(feat.numpy(), '../torch_paddle/paddle_var/pts_neck_{}.txt'.format(index))

        return x

    def extract_feat(self, points, img, radar, img_metas):

        if self.use_Cam:
            img_feats = self.extract_img_feat(img, img_metas)
        else:
            img_feats = None
        if self.use_LiDAR:
            pts_feats = self.extract_pts_feat(points)
        else:
            pts_feats = None
        if self.use_Radar:
            rad_feats = self.radar_encoder(radar)
        else:
            rad_feats = None

        return pts_feats, img_feats, rad_feats

    def _parse_loss(self, losses):

        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, paddle.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        return loss

    def _parse_result_to_sample(self, results, sample):
        num_samples = len(results)
        new_results = []
        for i in range(num_samples):
            data = Sample(None, sample["modality"][i])
            bboxes_3d = results[i]["boxes_3d"]
            labels = results[i]["labels_3d"]
            confidences = results[i]["scores_3d"]
            bottom_center = bboxes_3d[:, :3]
            gravity_center = np.zeros_like(bottom_center)
            gravity_center[:, :2] = bottom_center[:, :2]
            gravity_center[:, 2] = bottom_center[:, 2] + bboxes_3d[:, 5] * 0.5
            bboxes_3d[:, :3] = gravity_center
            data.bboxes_3d = BBoxes3D(bboxes_3d[:, 0:7])
            data.bboxes_3d.coordmode = 'Lidar'
            data.bboxes_3d.origin = [0.5, 0.5, 0.5]
            data.bboxes_3d.rot_axis = 2
            data.bboxes_3d.velocities = bboxes_3d[:, 7:9]
            data['bboxes_3d_numpy'] = bboxes_3d[:, 0:7]
            data['bboxes_3d_coordmode'] = 'Lidar'
            data['bboxes_3d_origin'] = [0.5, 0.5, 0.5]
            data['bboxes_3d_rot_axis'] = 2
            data['bboxes_3d_velocities'] = bboxes_3d[:, 7:9]
            data.labels = labels
            data.confidences = confidences
            data.meta = SampleMeta(id=sample["meta"][i]['id'])
            if "calibs" in sample:
                calib = [calibs.numpy()[i] for calibs in sample["calibs"]]
                data.calibs = calib
            new_results.append(data)
        return new_results

    def forward_mdfs_train(self,
                           pts_feats,
                           img_feats,
                           rad_feats,
                           gt_bboxes_3d=None,
                           gt_labels_3d=None,
                           img_metas=None,
                           gt_bboxes_ignore=None):

        outs = self.head(pts_feats=pts_feats,
                         img_feats=img_feats,
                         rad_feats=rad_feats,
                         img_metas=img_metas)
        out_all_cls_scores = outs['all_cls_scores']
        out_all_bbox_preds = outs['all_bbox_preds']
        save_variable(out_all_cls_scores.numpy(), '../torch_paddle/paddle_var/b_out_all_cls_scores.txt')
        save_variable(out_all_bbox_preds.numpy(), '../torch_paddle/paddle_var/b_out_all_bbox_preds.txt')
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs, gt_bboxes_ignore]
        losses = self.head.loss(*loss_inputs)
        loss = self._parse_loss(losses)
        return losses

    def forward_mdfs_test(self,
                          pts_feats,
                          img_feats,
                          rad_feats,
                          img_metas):
        outs = self.head(
            pts_feats=pts_feats,
            img_feats=img_feats,
            rad_feats=rad_feats,
            img_metas=img_metas
        )
        out_all_cls_scores = outs['all_cls_scores']
        out_all_bbox_preds = outs['all_bbox_preds']
        save_variable(out_all_cls_scores.numpy(), '../torch_paddle/paddle_var/out_all_cls_scores.txt')
        save_variable(out_all_bbox_preds.numpy(), '../torch_paddle/paddle_var/out_all_bbox_preds.txt')
        bbox_list = self.head.get_bboxes(outs, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def train_forward(self, samples):
        if self.use_Cam:
            img_metas = samples['meta']
            img = samples['img']
            gt_labels_3d = samples['gt_labels_3d']
            gt_bboxes_3d = samples['gt_bboxes_3d']
        if self.use_Radar:
            radar = samples['radar']
        else:
            radar = None
        if self.use_LiDAR:
            points = samples['points']
            voxels = samples['voxels']
            coords = samples['coords']
            num_points_per_voxel = samples['num_points_per_voxel']
            input_points = [points, voxels, coords, num_points_per_voxel]
        else:
            input_points = None
        # save_variable(points.numpy(), 'points.txt')
        # save_variable(voxels.numpy(), 'voxels.txt')
        # save_variable(coords.numpy(), 'coords.txt')
        # save_variable(num_points_per_voxel.numpy(), 'num_points_per_voxel.txt')


        points = paddle.to_tensor(load_variavle('points.txt'))
        voxels = paddle.to_tensor(load_variavle('voxels.txt'))
        coords = paddle.to_tensor(load_variavle('coords.txt'))
        num_points_per_voxel = paddle.to_tensor(load_variavle('num_points_per_voxel.txt'))
        input_points = [points, voxels, coords, num_points_per_voxel]
        img = paddle.to_tensor(load_variavle('img.txt'))
        gt_bboxes_3d = paddle.to_tensor(load_variavle('gt_bboxes_3d.txt'))
        gt_labels_3d = paddle.to_tensor(load_variavle('gt_labels_3d.txt'))
        img_metas = load_variavle('img_metas.txt')
        lidar2img = img_metas[0]['lidar2img']
        for i in range(len(lidar2img)):
            img_metas[0]['lidar2img'][i] = paddle.to_tensor(img_metas[0]['lidar2img'][i])

        pts_feats, img_feats, rad_feats = self.extract_feat(points=input_points, img=img, radar=radar, img_metas=img_metas)
        if self.use_LiDAR:
            pts_feats = [feat.unsqueeze(axis=1) for feat in pts_feats]
        mdfs_loss = self.forward_mdfs_train(pts_feats=pts_feats,
                                            img_feats=img_feats,
                                            rad_feats=rad_feats,
                                            img_metas=img_metas,
                                            gt_bboxes_3d=gt_bboxes_3d,
                                            gt_labels_3d=gt_labels_3d)

        return dict(loss=mdfs_loss)

    def test_forward(self, samples):
        if self.use_Cam:
            img_metas = samples['meta']
            img = samples['img']
        if self.use_Radar:
            radar = samples['radar']
        else:
            radar = None
        if self.use_LiDAR:
            points = samples['points']
            voxels = samples['voxels']
            coords = samples['coords']
            num_points_per_voxel = samples['num_points_per_voxel']
            input_points = [points, voxels, coords, num_points_per_voxel]
        else:
            input_points = None
        save_variable(img.numpy(), '../torch_paddle/paddle_var/img.txt')
        save_variable(points.numpy(), '../torch_paddle/paddle_var/points.txt')

        pts_feats, img_feats, rad_feats = self.extract_feat(points=input_points, img=img, radar=radar, img_metas=img_metas)
        if self.use_LiDAR:
            pts_feats = [feat.unsqueeze(axis=1) for feat in pts_feats]
        bbox_results = self.forward_mdfs_test(pts_feats=pts_feats,
                                              img_feats=img_feats,
                                              rad_feats=rad_feats,
                                              img_metas=img_metas)

        return dict(preds=self._parse_result_to_sample(bbox_results, samples))

    def export_forward(self, samples):
        pass

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'head') and self.head is not None