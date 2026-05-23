"""Pi3X distillation losses.

Ported verbatim from openpi
(`/scratch/lg154/Research/openpi/src/openpi/models_pytorch/pi0_pytorch.py`
lines 54-235) so loss values are byte-comparable across the two codebases.

Two modes:

* :func:`pi3x_point_loss` — pi3x-style L1 ray + inverse-depth-weighted L1 on
  scale-aligned local 3D points. Default.

* :func:`legacy_conf_mse_loss` — hard sigmoid-confidence-gated L1 / L2 on raw
  (xy, log_z). Cheaper, no scale alignment. Useful as a smoke test.

All inputs are ``(B, V, P, *)`` with ``P = patch_h * patch_w`` (16x16 = 256).
``view_mask`` is ``(B, V)`` and lets padded views (e.g., LIBERO's missing
right-wrist) be excluded — pass all-ones when every view is real.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _local_points_from_xy_logz(xy: Tensor, logz: Tensor) -> Tuple[Tensor, Tensor]:
    """Lift ``(xy, log_z) -> (camera-frame 3D point, depth)``.

    ``points[..., :2] = xy * exp(z)``; ``points[..., 2:] = exp(z)``.
    """
    logz = torch.nan_to_num(logz, nan=0.0, posinf=15.0, neginf=-30.0)
    depth = torch.exp(logz.clamp(max=15.0))
    return torch.cat([xy * depth, depth], dim=-1), depth


def _weighted_mean(
    values: Tensor,
    weights: Tensor,
    dim: Tuple[int, ...],
    *,
    keepdim: bool,
    eps: float = 1e-6,
) -> Tensor:
    denom = weights.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
    return (values * weights).sum(dim=dim, keepdim=keepdim) / denom


def _subsample_alignment_inputs(
    pred_points: Tensor,
    target_points: Tensor,
    weights: Tensor,
    max_points: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    if max_points <= 0 or pred_points.shape[1] <= max_points:
        return pred_points, target_points, weights

    pred_out, target_out, weight_out = [], [], []
    for pred_i, target_i, weight_i in zip(pred_points, target_points, weights, strict=True):
        valid_idx = torch.nonzero(weight_i > 0, as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            idx = torch.zeros(max_points, dtype=torch.long, device=weights.device)
            pred_out.append(pred_i[idx])
            target_out.append(target_i[idx])
            weight_out.append(torch.zeros(max_points, dtype=weight_i.dtype, device=weight_i.device))
            continue

        sample_idx = torch.div(
            torch.arange(max_points, device=weights.device) * valid_idx.numel(),
            max_points,
            rounding_mode="floor",
        ).clamp_max(valid_idx.numel() - 1)
        idx = valid_idx[sample_idx]
        pred_out.append(pred_i[idx])
        target_out.append(target_i[idx])
        weight_out.append(weight_i[idx])

    return torch.stack(pred_out, dim=0), torch.stack(target_out, dim=0), torch.stack(weight_out, dim=0)


@torch.no_grad()
def _align_points_scale_l1(
    pred_points: Tensor, target_points: Tensor, weights: Tensor, max_points: int
) -> Tensor:
    """Per-sample positive-scalar alignment between pred and target points (L1).

    Closed-form via the weighted L1 cumulative-derivative trick: ``argmin_s
    sum_i w_i |s * x_i - y_i|`` is the weighted median of ``y_i / x_i``.
    Returns ``(B,)`` positive scales; falls back to 1.0 for samples with no
    valid weight.
    """
    bsz = pred_points.shape[0]
    pred_points = pred_points.reshape(bsz, -1, 3)
    target_points = target_points.reshape(bsz, -1, 3)
    weights = weights.reshape(bsz, -1)
    pred_points, target_points, weights = _subsample_alignment_inputs(
        pred_points, target_points, weights, max_points
    )

    x = pred_points.flatten(1)
    y = target_points.flatten(1)
    w = weights[..., None].expand_as(pred_points).flatten(1)
    finite = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(w) & (w > 0)
    w = torch.where(finite, w, torch.zeros_like(w))
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    eps = 1e-7
    sign = torch.sign(x)
    x = x * sign
    y = y * sign
    y_div_x = y / x.clamp_min(eps)
    y_div_x, argsort = y_div_x.sort(dim=-1)

    wx = torch.gather(x * w, dim=-1, index=argsort)
    derivatives = 2 * wx.cumsum(dim=-1) - wx.sum(dim=-1, keepdim=True)
    search = torch.searchsorted(
        derivatives,
        torch.zeros_like(derivatives[..., :1]),
        side="left",
    ).clamp_max(derivatives.shape[-1] - 1)
    scale = y_div_x.gather(dim=-1, index=search).squeeze(-1)

    has_weight = weights.sum(dim=-1) > eps
    return torch.where(has_weight, scale.abs().clamp_min(eps), torch.ones_like(scale))


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def pi3x_point_loss(
    xy_pred: Tensor,
    logz_pred: Tensor,
    xy_target: Tensor,
    logz_target: Tensor,
    conf_target: Tensor,
    view_mask: Tensor | None = None,
    *,
    scale_align_num_points: int = 4096,
    depth_weight_min_frac: float = 0.1,
    ray_loss_weight: float = 1.0,
    depth_loss_weight: float = 1.0,
    depth_weighting: str = "pi3x_inverse",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """pi3x-style local-pointmap loss.

    Args
    ----
    xy_pred / logz_pred : ``(B, V, P, 2)`` / ``(B, V, P, 1)``
    xy_target / logz_target : same shapes as preds
    conf_target : ``(B, V, P, 1)`` — pre-sigmoid teacher confidence logits
    view_mask : ``(B, V)`` boolean / 0-1 — ``None`` ⇒ all-ones
    scale_align_num_points : deterministic subsample size for the scale solver
    depth_weight_min_frac : floor for inverse-depth weighting (as a fraction of
        per-sample mean depth) — prevents close-range points from dominating
    depth_weighting : ``"pi3x_inverse"`` (default) or ``"uniform"``

    Returns ``(total, ray_loss, depth_loss, scale)``.
    """
    if view_mask is None:
        B, V = xy_pred.shape[:2]
        view_mask = torch.ones(B, V, device=xy_pred.device, dtype=xy_pred.dtype)

    finite_target = torch.isfinite(xy_target).all(dim=-1, keepdim=True) & torch.isfinite(logz_target)
    xy_target = torch.nan_to_num(xy_target, nan=0.0, posinf=0.0, neginf=0.0)
    xy_pred = torch.nan_to_num(xy_pred, nan=0.0, posinf=0.0, neginf=0.0)

    pred_points, _ = _local_points_from_xy_logz(xy_pred, logz_pred)
    target_points, target_depth = _local_points_from_xy_logz(xy_target, logz_target)

    conf_weights = torch.sigmoid(conf_target)
    conf_weights = torch.where(finite_target, conf_weights, torch.zeros_like(conf_weights))
    valid_weights = (conf_weights > 0.5).to(dtype=xy_pred.dtype)

    ray_weights = view_mask[:, :, None, None].to(dtype=xy_pred.dtype) * finite_target.to(dtype=xy_pred.dtype)
    ray_denom = (ray_weights.sum() * xy_pred.shape[-1]).clamp_min(1.0)
    ray_loss = (torch.abs(xy_pred - xy_target) * ray_weights).sum() / ray_denom

    if depth_weighting == "pi3x_inverse":
        mean_depth = _weighted_mean(target_depth.detach(), valid_weights, dim=(2,), keepdim=True)
        min_depth = (depth_weight_min_frac * mean_depth).clamp_min(1e-6)
        depth_weights = 1.0 / target_depth.detach().clamp_min(min_depth).clamp_min(1e-6)
    elif depth_weighting == "uniform":
        depth_weights = torch.ones_like(target_depth)
    else:
        raise ValueError(f"depth_weighting must be 'pi3x_inverse' or 'uniform', got {depth_weighting!r}")
    point_weights = valid_weights * depth_weights

    scale = _align_points_scale_l1(
        pred_points,
        target_points,
        point_weights.squeeze(-1),
        max_points=scale_align_num_points,
    )
    aligned_pred_points = pred_points * scale.view(-1, 1, 1, 1)

    depth_denom = valid_weights.sum().clamp_min(1.0)
    depth_loss = (
        torch.abs(aligned_pred_points[..., 2:3] - target_points[..., 2:3]) * point_weights
    ).sum() / depth_denom

    total = ray_loss_weight * ray_loss + depth_loss_weight * depth_loss
    return total, ray_loss, depth_loss, scale


def legacy_conf_mse_loss(
    xy_pred: Tensor,
    logz_pred: Tensor,
    xy_target: Tensor,
    logz_target: Tensor,
    conf_target: Tensor,
    view_mask: Tensor | None = None,
    *,
    conf_threshold: float = 0.1,
    order: int = 2,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Hard-confidence-gated L1 (``order=1``) or L2 (``order=2``) loss.

    No scale alignment. Returns ``(total, xy_loss, z_loss, scale=ones)`` so the
    caller can swap between the two losses with the same return contract.
    """
    if view_mask is None:
        B, V = xy_pred.shape[:2]
        view_mask = torch.ones(B, V, device=xy_pred.device, dtype=xy_pred.dtype)

    mask = (torch.sigmoid(conf_target) > conf_threshold).to(xy_pred.dtype)
    mask = mask * view_mask[:, :, None, None].to(xy_pred.dtype)

    if order == 1:
        xy_resid = torch.abs(xy_pred - xy_target)
        z_resid = torch.abs(logz_pred - logz_target)
    elif order == 2:
        xy_resid = (xy_pred - xy_target) ** 2
        z_resid = (logz_pred - logz_target) ** 2
    else:
        raise ValueError(f"order must be 1 (L1) or 2 (MSE), got {order!r}")

    denom = mask.sum().clamp_min(1.0)
    xy_loss = (xy_resid * mask).sum() / denom / xy_pred.shape[-1]
    z_loss = (z_resid * mask).sum() / denom

    total = xy_loss + z_loss
    scale = torch.ones(xy_pred.shape[0], dtype=xy_pred.dtype, device=xy_pred.device)
    return total, xy_loss, z_loss, scale
