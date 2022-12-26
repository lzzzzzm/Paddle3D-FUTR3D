import paddle
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from paddle3d.models.detection import normalize_bbox

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


from paddle3d.apis import manager

__all__ = ["HungarianAssigner3D"]


@manager.ASSIGNERS.add_component
class HungarianAssigner3D(BaseAssigner):

    def __init__(self,
                 cls_cost,
                 reg_cost,
                 iou_cost,
                 pc_range=None):
        self.cls_cost = cls_cost
        self.reg_cost = reg_cost
        self.iou_cost = iou_cost
        self.pc_range = pc_range

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               gt_bboxes_ignore=None,
               eps=1e-7):
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.shape[0], bbox_pred.shape[0]
        # 1. assign -1 by default

        assigned_gt_inds = paddle.full(shape=(num_bboxes, ), fill_value=-1, dtype='int64')
        assigned_labels = paddle.full(shape=(num_bboxes, ), fill_value=-1, dtype='int64')
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)
        if bbox_pred.shape[-1] > 8:
            reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])
        else:
            reg_cost = self.reg_cost(bbox_pred, normalized_gt_bboxes)
        # weighted sum of above two costs
        cost = cls_cost + reg_cost
        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.numpy()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = paddle.to_tensor(matched_row_inds, place=bbox_pred.place)
        matched_col_inds = paddle.to_tensor(matched_col_inds, place=bbox_pred.place)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)