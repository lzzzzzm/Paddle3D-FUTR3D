import paddle
import paddle.nn as nn

from paddle3d.apis import manager
from paddle3d.models.detection.futr3d.futr3d_utils import GridMask, bbox3d2result

__all__ = ["FUTR3D"]


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
        if self.use_Radar:
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
            rad_feats = self.radar_encoder(radar)
        else:
            rad_feats = None

        return (pts_feats, img_feats, rad_feats)

    def test_forward(self, outs, img_metas):
        bbox_list = self.get_bboxes(outs, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_list

    def forward_mdfs_train(self,
                           pts_feats,
                           img_feats,
                           rad_feats,
                           gt_bboxes_3d=None,
                           gt_labels_3d=None,
                           img_metas=None,
                           gt_bboxes_ignore=None):

        outs = self.pts_bbox_head(pts_feats=pts_feats,
                                  img_feats=img_feats,
                                  rad_feats=rad_feats,
                                  img_metas=img_metas)
        pass

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
        return bbox_results

    def forward(self,sample):
        points = sample['lidar']
        img = sample['img']
        radar = sample['radar']
        img_metas = sample['img_meta']
        pts_feats, img_feats, rad_feats = self.extract_feat(points=points, img=img, radar=radar)
        """
            img_feats[0].shape = [1, 6, 256, 58, 100]
            img_feats[1].shape = [1, 6, 256, 29, 50]
            img_feats[2].shape = [1, 6, 256, 15, 25]
            img_feats[3].shape = [1, 6, 256, 8, 13]

            rad_feats.shape = [1, 1200, 67]
        """

        # # Test forward
        bbox_results = self.forward_mdfs_test(pts_feats=pts_feats,
                                              img_feats=img_feats,
                                              rad_feats=rad_feats,
                                              img_metas=img_metas)
        return bbox_results
