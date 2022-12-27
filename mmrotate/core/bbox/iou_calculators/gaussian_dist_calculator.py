# Copyright (c) OpenMMLab. All rights reserved.
# noqa E501, Edited from https://github.com/open-mmlab/mmrotate/blob/main/mmrotate/models/losses/gaussian_dist_loss.py
import torch

from .builder import IOU_CALCULATORS


def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma


def xyxy_2_xy_sigma(xyxy):
    """Convert horizontal bounding box to 2-D Gaussian distribution.

    Args:
        bbox (torch.Tensor): bboxes(xyxy) with shape (N, 4).
    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xyxy.shape
    assert _shape[-1] == 4
    wh = torch.stack(
        [xyxy[..., 2] - xyxy[..., 0], xyxy[..., 3] - xyxy[..., 1]], dim=-1)
    xy = xyxy[..., :2] + wh / 2
    sigma = (0.5 * torch.diag_embed(wh)).square()

    return xy, sigma


def kld(bbox1, bbox2):
    """Kullback-Leibler Divergence Edited from 'kld_loss', 'kld_loss' must have
    the Tensors inputs as same shape and have an output as scalar by
    decorator(weighted_loss) for calculating loss.

    Edited function 'kld',
    Calculate kld for each pair for assign.
    Args:
        bbox1 (Tuple[torch.Tensor]): (xy (n,2), sigma(n,2,2))
        bbox2 (Tuple[torch.Tensor]) : (xy (m,2), sigma(m,2,2))
    Returns:
        kullback leibler divergence (torch.Tensor) shape (n, m)
    """
    xy_1, Sigma_1 = bbox1
    xy_2, Sigma_2 = bbox2

    N, _ = xy_1.shape
    M, _ = xy_2.shape
    xy_1 = xy_1.unsqueeze(1).repeat(1, M, 1).view(-1, 2)
    Sigma_1 = Sigma_1.unsqueeze(1).repeat(1, M, 1, 1).view(-1, 2, 2)
    xy_2 = xy_2.unsqueeze(0).repeat(N, 1, 1).view(-1, 2)
    Sigma_2 = Sigma_2.unsqueeze(0).repeat(N, 1, 1, 1).view(-1, 2, 2)
    return_shape = [N, M]

    Sigma_1_inv = torch.stack((Sigma_1[..., 1, 1], -Sigma_1[..., 0, 1],
                               -Sigma_1[..., 1, 0], Sigma_1[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    Sigma_1_inv = Sigma_1_inv / Sigma_1.det().unsqueeze(-1).unsqueeze(-1)
    dxy = (xy_1 - xy_2).unsqueeze(-1)
    xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigma_1_inv).bmm(dxy).view(-1)

    whr_distance = 0.5 * Sigma_1_inv.bmm(Sigma_2).diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    Sigma_1_det_log = Sigma_1.det().log()
    Sigma_2_det_log = Sigma_2.det().log()
    whr_distance = whr_distance + 0.5 * (Sigma_1_det_log - Sigma_2_det_log)
    whr_distance = whr_distance - 1
    distance = (xy_distance + whr_distance).clamp(0).sqrt()
    return distance.reshape(return_shape)


def gwd_overlaps(pred, target):
    """Gaussian Wasserstein distance.

    Edited from 'gwd_loss',
    'gwd_loss' must have the Tensors inputs as same shape and
    have an output as scalar by decorator(weighted_loss) for calculating loss.
    Edited function 'gwd', Calculate gwd for each pair for assign.
    Args:
        pred (Tuple[torch.Tensor]): (xy (n,2), sigma(n,2,2))
        target (Tuple[torch.Tensor]): (xy (m,2), sigma(m,2,2))
    Returns:
        Gaussian distance with shape (torch.Tensor) (n, m)
    """
    xy_p, Sigma_p = pred
    xy_p = xy_p[..., None, :]
    Sigma_p = Sigma_p[..., None, :, :]

    xy_t, Sigma_t = target
    xy_t = xy_t[..., None, :, :]
    Sigma_t = Sigma_t[..., None, :, :, :]

    xy_distance = (xy_p - xy_t).square().sum(dim=-1)

    whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + Sigma_t.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    _t_tr = (Sigma_p.matmul(Sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (Sigma_p.det() * Sigma_t.det()).clamp(0).sqrt()
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt())

    distance = (xy_distance + whr_distance).clamp(0).sqrt()

    return distance


def kld_sym_overlaps(bbox1, bbox2):
    """
        Args:
            bbox1 (Tuple[torch.Tensor]): (xy (n,2), sigma(n,2,2))
            bbox2 (Tuple[torch.Tensor]) : (xy (m,2), sigma(m,2,2))
        Returns:
            symmetric kullback leibler divergence shape (torch.Tensor) (n, m)
    """
    return (kld(bbox1, bbox2) + kld(bbox2, bbox1).transpose(-1, -2)) * 0.5


@IOU_CALCULATORS.register_module
class GDOverlaps2D:
    """2D Overlaps (e.g. GWD, KLD) Calculator."""

    BAG_GD_OVERLAPS = {
        'gwd': gwd_overlaps,
        'kld_sym': kld_sym_overlaps,
    }
    BAG_PREP = {
        'rbox': xy_wh_r_2_xy_sigma,
        'bbox': xyxy_2_xy_sigma,
    }

    def __init__(self,
                 mode='gwd',
                 representation='rbox',
                 extra_dim=1,
                 is_transpose=False,
                 is_normalize=True,
                 constant=12.7):
        assert mode in self.BAG_GD_OVERLAPS, (
            f'{mode} not in {self.BAG_GD_OVERLAPS.keys()}')
        assert representation in self.BAG_PREP, (
            f'{representation} not in {self.BAG_PREP.keys()}')
        self.overlaps = self.BAG_GD_OVERLAPS[mode]
        self.representation = self.BAG_PREP[representation]
        self.extra_dim = extra_dim
        self.is_transpose = is_transpose
        self.is_normalize = is_normalize
        self.constant = constant
        if not self.is_normalize:
            self.constant = 1.

    def __call__(self, bboxes1, bboxes2):
        """Calculate IoU between 2D bboxes.

        Edited method from https://arxiv.org/abs/2110.13389
            1. We do not normalize kld or gwd because
                it will be used with ATSS.
            2. Kullback-Leibler Divergence.
        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <cx, cy, w, h, r> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <cx, cy, w, h, r> format, or be
                empty.
        Returns:
            Tensor: shape (m, n) if self.is_transpose is False
            else shape (n, m)
        """
        assert bboxes1.size(-1) in [0, 4 + self.extra_dim, 5 + self.extra_dim]
        assert bboxes2.size(-1) in [0, 4 + self.extra_dim, 5 + self.extra_dim]

        bboxes1 = bboxes1[..., :bboxes1.size(-1) - self.extra_dim]
        bboxes2 = bboxes2[..., :bboxes2.size(-1) - self.extra_dim]

        bboxes1 = self.representation(bboxes1)
        bboxes2 = self.representation(bboxes2)

        overlaps = torch.exp(-1. * self.overlaps(bboxes1, bboxes2) /
                             self.constant)

        if self.is_transpose:
            overlaps = overlaps.transpose(-1, -2)
        return overlaps

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = (
            self.__class__.__name__ + f'('
            f'overlaps={self.overlaps}, '
            f'representation={self.representation}), '
            f'extra_dim={self.extra_dim}, is_transpose={self.is_transpose}')
        return repr_str
