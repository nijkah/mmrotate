# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh

from mmrotate.core.bbox.iou_calculators.gaussian_dist_calculator import (
    gwd_overlaps, kld, xy_wh_r_2_xy_sigma)


@MATCH_COST.register_module()
class RBoxL1Cost:
    """BBoxL1Cost.
     Args:
         weight (int | float, optional): loss_weight
         box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN
     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import BBoxL1Cost
         >>> import torch
         >>> self = BBoxL1Cost()
         >>> bbox_pred = torch.rand(1, 4)
         >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(bbox_pred, gt_bboxes, factor)
         tensor([[1.6172, 1.6422]])
    """

    def __init__(self, weight=1., box_format='xywha'):
        self.weight = weight
        assert box_format in ['xywha']
        self.box_format = box_format

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                (num_query, 4).
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape (num_gt, 4).
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@MATCH_COST.register_module()
class GWDCost:
    """GWDCost.

     Args:
         dist_mode (str, optional): iou mode such as 'gwd' | 'kld_sym'
         weight (int | float, optional): loss weight
     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import IoUCost
         >>> import torch
         >>> self = IoUCost()
         >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self, dist_mode='gwd', weight=1.):
        self.weight = weight
        self.dist_mode = dist_mode

    def __call__(self, bboxes, gt_bboxes):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape (num_query, 4).
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape (num_gt, 4).
        Returns:
            torch.Tensor: iou_cost value with weight
        """
        from mmrotate.models.losses.gaussian_dist_loss import postprocess

        bboxes = xy_wh_r_2_xy_sigma(bboxes)
        gt_bboxes = xy_wh_r_2_xy_sigma(gt_bboxes)
        # overlaps: [num_bboxes, num_gt]
        dist = gwd_overlaps(bboxes, gt_bboxes)
        gwd_cost = postprocess(dist, 'sqrt', 2.0)

        return gwd_cost * self.weight


@MATCH_COST.register_module()
class KLDCost:
    """KLDCost.

     Args:
         weight (int | float, optional): loss weight
     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import IoUCost
         >>> import torch
         >>> self = IoUCost()
         >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bboxes, gt_bboxes):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape (num_query, 4).
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape (num_gt, 4).
        Returns:
            torch.Tensor: iou_cost value with weight
        """
        from mmrotate.models.losses.gaussian_dist_loss import postprocess

        bboxes = xy_wh_r_2_xy_sigma(bboxes)
        gt_bboxes = xy_wh_r_2_xy_sigma(gt_bboxes)
        # overlaps: [num_bboxes, num_gt]
        dist = kld(bboxes, gt_bboxes, sqrt=False)
        kld_cost = postprocess(dist)

        return kld_cost * self.weight
