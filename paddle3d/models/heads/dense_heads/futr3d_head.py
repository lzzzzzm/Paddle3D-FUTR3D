import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed import reduce

import copy
from functools import partial

from paddle3d.models.transformer.attention import inverse_sigmoid
from paddle3d.models.detection.futr3d.futr3d_utils import normalize_bbox, nan_to_num
from paddle3d.models.layers.param_init import bias_init_with_prob,constant_
from paddle3d.apis import manager

__all__ = ["DeformableFUTR3DHead"]
import pickle
def load_variavle(filename):
   f=open(filename,'rb')
   r=pickle.load(f)
   f.close()
   return r

def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


@manager.HEADS.add_component
class DeformableFUTR3DHead(nn.Layer):
    def __init__(self,
                 num_query=600,
                 num_classes=10,
                 in_channels=256,
                 embed_dims=256,
                 sync_cls_avg_factor=False,
                 with_box_refine=True,
                 as_two_stage=False,
                 bbox_coder=None,
                 transformer=None,
                 assigner=None,
                 sampler=None,
                 code_size=10,
                 code_weights=None,
                 num_cls_fcs=2,
                 num_reg_fcs=2,
                 pc_range=None,
                 positional_encoding=None,
                 loss_cls=None,
                 loss_bbox=None,
                 loss_iou=None,
                 bg_cls_weight=0.
                 ):
        super(DeformableFUTR3DHead, self).__init__()
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.code_size = code_size
        self.positional_encoding = positional_encoding
        self.assigner = assigner
        self.sampler = sampler
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox

        self.loss_iou = loss_iou

        self.num_cls_fcs = num_cls_fcs - 1
        self.num_reg_fcs = num_reg_fcs
        self.pc_range = pc_range
        self.cls_out_channels = num_classes
        self.bg_cls_weight = bg_cls_weight
        if code_weights is not None:
            self.code_weights = paddle.to_tensor(code_weights)
        else:
            self.code_weights = paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])
        # model layers
        self.bbox_coder = bbox_coder
        self.transformer = transformer
        # cls Layer
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU())
        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)
        # reg Layer
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        # cls branches & reg branches

        def _get_clones(module, N):
            return nn.LayerList([copy.deepcopy(module) for i in range(N)])

        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.LayerList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.LayerList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

        self.init_weights()

    def init_weights(self):
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                constant_(m[-1].bias, bias_init)

    def get_bboxes(self, preds_dicts, img_metas):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            # bboxes = img_metas[i]['box_type_3d'](bboxes, self.code_size-1)
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list

    def forward(self, pts_feats, img_feats, rad_feats, img_metas):
        """
            img_feats: FPN outputs: 4x[B, N, C, W, H]
            img_metas: 'lidar2img': 6x[4, 4]
            rad_feats: [B, N', C']
        """
        query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references = self.transformer(
            pts_feats=pts_feats,
            img_feats=img_feats,
            rad_feats=rad_feats,
            query_embed=query_embeds,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            img_metas=img_metas
        )
        hs = paddle.transpose(hs, (0, 2, 1, 3))
        outputs_classes = []
        outputs_coords = []
        hs = load_variavle('hs.txt')
        init_reference = load_variavle('init_reference.txt')
        inter_references = load_variavle('inter_references.txt')
        hs = paddle.to_tensor(hs)
        init_reference = paddle.to_tensor(init_reference)
        inter_references = paddle.to_tensor(inter_references)
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            assert reference.shape[-1] == 3

            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = F.sigmoid(tmp[..., 0:2])
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = F.sigmoid(tmp[..., 4:5])
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_classes = paddle.stack(outputs_classes)
        outputs_coords = paddle.stack(outputs_coords)
        # save_variable(outputs_coords.numpy(), 'outputs_coords.txt')
        outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }

        return outs

    def multi_apply(self, func, *args, **kwargs):
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)
        return tuple(map(list, zip(*map_results)))

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_bboxes,
                           gt_labels,
                           gt_bboxes_ignore=None):
        num_bboxes = bbox_pred.shape[0]
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        # labels = paddle.full_like(num_bboxes, self.num_classes, dtype='long')
        labels = paddle.full((num_bboxes,), self.num_classes, dtype='int32')

        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        # label_weights = gt_bboxes.new_ones(num_bboxes)
        label_weights = paddle.ones(shape=(num_bboxes,), dtype=gt_bboxes.dtype)

        # bbox targets
        bbox_targets = paddle.zeros_like(bbox_pred)[..., :self.code_size - 1]
        bbox_weights = paddle.zeros_like(bbox_pred)
        updates = paddle.ones(shape=bbox_weights.shape)
        for i in range(len(pos_inds)):
            bbox_weights[pos_inds[i]] = updates[i]
        # bbox_weights[pos_inds] = 1.0

        # DETR
        # for i in range(len(pos_inds)):
        #     bbox_weights[pos_inds[i]] = updates[i]
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = self.multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)
        num_total_pos = sum((paddle.numel(inds) for inds in pos_inds_list))
        num_total_neg = sum((paddle.numel(inds) for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        num_imgs = cls_scores.shape[0]
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = paddle.concat(labels_list, 0)
        label_weights = paddle.concat(label_weights_list, 0)
        bbox_targets = paddle.concat(bbox_targets_list, 0)
        bbox_weights = paddle.concat(bbox_weights_list, 0)
        # classification loss
        cls_scores = cls_scores.reshape((-1, self.cls_out_channels))
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce(paddle.tensor(cls_avg_factor, dtype=cls_scores.dtype))

        cls_avg_factor = max(cls_avg_factor, 1)
        # TODO label_weights, avg_factor
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = paddle.to_tensor(num_total_pos)
        if self.sync_cls_avg_factor:
            clip_total_pos = reduce(num_total_pos, 0)
        else:
            clip_total_pos = num_total_pos
        num_total_pos = paddle.clip(clip_total_pos, min=1)
        # regression L1 loss
        bbox_preds = bbox_preds.reshape((-1, bbox_preds.shape[-1]))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = paddle.isfinite(normalized_bbox_targets).all(axis=-1)
        bbox_weights = bbox_weights * self.code_weights
        isnotnan = isnotnan.unsqueeze(-1).tile([1, self.code_size])
        loss_bbox_preds = paddle.masked_select(bbox_preds[:, :self.code_size], isnotnan).reshape((-1, self.code_size))
        loss_normalized_bbox_targets = paddle.masked_select(normalized_bbox_targets[:, :self.code_size],
                                                            isnotnan).reshape((-1, self.code_size))
        loss_bbox_weights = paddle.masked_select(bbox_weights[:, :self.code_size], isnotnan).reshape(
            (-1, self.code_size))
        loss_bbox = self.loss_bbox(
            loss_bbox_preds,
            loss_normalized_bbox_targets,
            loss_bbox_weights, avg_factor=num_total_pos)

        loss_cls = nan_to_num(loss_cls)
        loss_bbox = nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             gravity_center_list,
             preds_dicts,
             gt_bboxes_ignore=None):

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        num_dec_layers = len(all_cls_scores)
        new_gt_bboxes_list = []
        for i in range(gt_bboxes_list.shape[0]):
            new_gt_bboxes_list.append(paddle.concat((gravity_center_list[i], gt_bboxes_list[i][:, 3:]), axis=1))
        # gt_bboxes_list = [paddle.concat((gravity_center, gt_bboxes[:, 3:]),axis=1) for gravity_center, gt_bboxes in zip(gravity_center_list, gt_bboxes_list)]
        all_gt_bboxes_list = [new_gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        losses_cls, losses_bbox = self.multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        if enc_cls_scores is not None:
            binary_labels_list = [
                paddle.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 new_gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict
