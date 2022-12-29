import paddle
import paddle.nn as nn

from paddle3d.apis import manager
from paddle3d.models.detection.futr3d.futr3d_utils import GridMask, bbox3d2result

from collections import OrderedDict

__all__ = ["FUTR3D"]
# import pickle
# def load_variavle(filename):
#    f=open(filename,'rb')
#    r=pickle.load(f)
#    f.close()
#    return r

@manager.MODELS.add_component
class FUTR3D(nn.Layer):
    def __init__(self,
                 use_grid_mask=True,
                 use_LiDAR=False,
                 use_Cam=True,
                 use_Radar=True,
                 backbone=None,
                 radar_encoder=None,
                 neck=None,
                 head=None):
        super(FUTR3D, self).__init__()
        self.grid_mask = GridMask(use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.use_LiDAR = use_LiDAR
        self.use_Cam = use_Cam
        self.use_Radar = use_Radar
        self.head = head
        self.backbone = backbone
        self.neck = neck
        self.radar_encoder = radar_encoder

    def extract_img_feat(self, img, img_metas=None):
        """
            Extract features of images
        """
        B, N, C, H, W = img.shape
        img = paddle.reshape(img, (B * N, C, H, W))
        if self.use_grid_mask:
            img = self.grid_mask(img)
        # TODO:img.requires_grad = True
        # backbone
        img_feats = self.backbone(img)
        # neck
        img_feats = self.neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.shape
            img_feat = paddle.reshape(img_feat, (B, int(BN / B), C, H, W))
            img_feats_reshaped.append(img_feat)
        return img_feats_reshaped

    def extract_pts_feat(self, points):
        # TODO
        pass

    def extract_feat(self, points, img, radar):

        if self.use_Cam:
            img_feats = self.extract_img_feat(img)
        else:
            img_feats = None

        if self.use_LiDAR:
            pts_feats = self.extract_pts_feat(points)
        else:
            pts_feats = None

        if self.use_Radar:
            radar = radar.squeeze(1)
            rad_feats = self.radar_encoder(radar)
        else:
            rad_feats = None

        return (pts_feats, img_feats, rad_feats)


    def forward_mdfs_train(self,
                           pts_feats,
                           img_feats,
                           rad_feats,
                           gt_bboxes_3d=None,
                           gt_labels_3d=None,
                           gravity_center=None,
                           img_metas=None,
                           gt_bboxes_ignore=None):

        outs = self.head(pts_feats=pts_feats,
                                  img_feats=img_feats,
                                  rad_feats=rad_feats,
                                  img_metas=img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d,gravity_center, outs]
        losses = self.head.loss(*loss_inputs)
        loss = self._parse_loss(losses)
        bbox_list = self.head.get_bboxes(outs, img_metas)
        outputs = {
            'loss':loss,
            'preds':bbox_list
        }
        return outputs

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
        bbox_list = self.head.get_bboxes(outs, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        outputs={
            'preds':bbox_list
        }
        return outputs

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

    def forward(self,sample):
        if self.use_LiDAR:
            points = sample['lidar']
        else:
            points = None
        if self.use_Cam:
            img = sample['img']
            # img = paddle.transpose(img, (0, 1, 4, 2, 3))
            img_metas = sample['img_meta']
        else:
            img = None
        if self.use_Radar:
            radar = sample['radar']
        else:
            radar = None
        gt_labels_3d = sample['labels']
        gt_bboxes_3d = sample['bboxes_3d']
        gravity_center = sample['gravity_center']
        pts_feats, img_feats, rad_feats = self.extract_feat(points=points, img=img, radar=radar)
        if self.training:
            outputs = self.forward_mdfs_train(
                pts_feats=pts_feats,
                img_feats=img_feats,
                rad_feats=rad_feats,
                img_metas=img_metas,
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                gravity_center=gravity_center
            )
        else:
            outputs = self.forward_mdfs_test(
                pts_feats=pts_feats,
                img_feats=img_feats,
                rad_feats=rad_feats,
                img_metas=img_metas
            )
        return outputs
