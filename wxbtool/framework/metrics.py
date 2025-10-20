# -*- coding: utf-8 -*-
"""
Metrics utilities for wxbtool.

This module provides logger-agnostic, shape-normalized metrics that operate on
PyTorch tensors and return tensors or Python floats suitable for Lightning logging.

Conventions:
- Spatial grid is (H, W).
- Deterministic forecasts and observations are normalized to [B, 1, P, H, W].
- Area weights are broadcastable to [1, 1, 1, H, W].
- All computations occur on the device/dtype of the input tensors.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch as th

from wxbtool.norms.meanstd import denormalizors


def _ensure_5d(x: th.Tensor, pred_span: int) -> th.Tensor:
    """
    Normalize tensor shapes to [B, 1, P, H, W].

    Accepts inputs shaped:
    - [B, P, H, W] -> add channel dim -> [B, 1, P, H, W]
    - [B, 1, P, H, W] -> returned unchanged
    - [B, H, W] -> interpreted as P==1 -> [B, 1, 1, H, W]
    """
    if x.dim() == 5:
        return x
    if x.dim() == 4:
        # Assume [B, P, H, W]
        return x.unsqueeze(1)
    if x.dim() == 3:
        # Assume [B, H, W]
        B, H, W = x.shape
        return x.view(B, 1, 1, H, W)
    raise ValueError(f"Unsupported tensor shape for metrics: {tuple(x.shape)}")


def rmse_weighted(
    forecast: th.Tensor,
    target: th.Tensor,
    *,
    weights: th.Tensor,
    pred_span: int,
    denorm_key: Optional[str] = None,
) -> th.Tensor:
    """
    Compute global area-weighted RMSE over all forecast days.

    Args:
        forecast: Forecast tensor (any of [B, P, H, W], [B,1,P,H,W], [B,H,W]).
        target: Observation tensor (same broadcastable shape as forecast).
        weights: Area weights broadcastable to [1,1,1,H,W].
        pred_span: Prediction span (P).
        denorm_key: If provided, apply denormalization using denormalizors[key].

    Returns:
        Scalar tensor RMSE on the same device/dtype as input tensors.
    """
    device = forecast.device
    dtype = forecast.dtype
    f = _ensure_5d(forecast, pred_span).to(device=device, dtype=dtype)
    t = _ensure_5d(target, pred_span).to(device=device, dtype=dtype)

    w = weights.to(device=device, dtype=dtype)
    if w.dim() == 2:
        H, W = w.shape
        w = w.view(1, 1, 1, H, W)
    elif w.dim() != 5:
        raise ValueError("weights must be [H,W] or broadcastable to [1,1,1,H,W]")

    if denorm_key is not None:
        # Denormalize both forecast/target in-place
        f = denormalizors[denorm_key](f)
        t = denormalizors[denorm_key](t)

    se = (f - t) ** 2
    wse = w * se
    total_se = th.sum(wse)
    # Sum of weights per element = w * ones_like
    total_w = th.sum(w * th.ones_like(wse))
    mse = total_se / (total_w + 1e-12)
    return th.sqrt(mse)


def rmse_by_time(
    forecast: th.Tensor,
    target: th.Tensor,
    *,
    weights: th.Tensor,
    pred_span: int,
    denorm_key: Optional[str] = None,
) -> Tuple[th.Tensor, List[float]]:
    """
    Compute area-weighted RMSE per-day and overall.

    Returns:
        overall_rmse: scalar tensor
        per_day_rmse: list of Python floats length=P
    """
    device = forecast.device
    dtype = forecast.dtype
    f = _ensure_5d(forecast, pred_span).to(device=device, dtype=dtype)
    t = _ensure_5d(target, pred_span).to(device=device, dtype=dtype)

    w = weights.to(device=device, dtype=dtype)
    if w.dim() == 2:
        H, W = w.shape
        w = w.view(1, 1, 1, H, W)
    elif w.dim() != 5:
        raise ValueError("weights must be [H,W] or broadcastable to [1,1,1,H,W]")

    if denorm_key is not None:
        f = denormalizors[denorm_key](f)
        t = denormalizors[denorm_key](t)

    B, C, P, H, W = f.shape
    se = (f - t) ** 2
    wse = w * se

    per_day: List[float] = []
    total_se = th.tensor(0.0, device=device, dtype=dtype)
    total_w = th.tensor(0.0, device=device, dtype=dtype)
    ones = th.ones((B, C, 1, H, W), device=device, dtype=dtype)

    for d in range(P):
        cur = wse[:, :, d : d + 1]  # [B,1,1,H,W]
        cur_se = th.sum(cur)
        cur_w = th.sum(w * ones)
        rmse_d = th.sqrt(cur_se / (cur_w + 1e-12))
        per_day.append(float(rmse_d))

        total_se += cur_se
        total_w += cur_w

    overall = th.sqrt(total_se / (total_w + 1e-12))
    return overall, per_day


def acc_anomaly_by_time(
    f_anomaly: np.ndarray,
    o_anomaly: np.ndarray,
    *,
    weights: np.ndarray,
) -> Tuple[List[float], float, float, float]:
    """
    Compute ACC per-day and return the three aggregated terms for epoch-level ACC.

    Args:
        f_anomaly: Forecast anomalies [B,1,P,H,W] (numpy).
        o_anomaly: Observation anomalies [B,1,P,H,W] (numpy).
        weights: Area weights [H,W] or [1,1,1,H,W] (numpy).

    Returns:
        per_day_acc: list of floats length=P
        prod_sum, fsum_sum, osum_sum: floats for aggregation
    """
    if weights.ndim == 2:
        H, W = weights.shape
        w = weights.reshape(1, 1, 1, H, W)
    else:
        w = weights

    B, C, P, H, W = f_anomaly.shape
    per_day: List[float] = []
    prod_sum = 0.0
    fsum_sum = 0.0
    osum_sum = 0.0

    for d in range(P):
        fa = f_anomaly[:, :, d, :, :]
        oa = o_anomaly[:, :, d, :, :]
        prod = float(np.sum(w * fa * oa))
        fsum = float(np.sum(w * fa**2))
        osum = float(np.sum(w * oa**2))
        acc = prod / (np.sqrt(fsum * osum) + 1e-12)
        per_day.append(acc)

        prod_sum += prod
        fsum_sum += fsum
        osum_sum += osum

    return per_day, prod_sum, fsum_sum, osum_sum


def crps_ensemble(
    predictions: th.Tensor,
    targets: th.Tensor,
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Simplified CRPS on per-pixel ensemble for a single horizon.

    Args:
        predictions: [S, C, H, W] or [B,S,C,H,W] (will be flattened over batch).
        targets: [S, C, H, W] or [B,S,C,H,W] (same shape requirements).

    Returns:
        crps_mean: scalar tensor
        absorb_mean: scalar tensor (0.5*E|Y-Y'| / (E|Y-x|+eps))
    """
    if predictions.dim() == 5:
        # [B,S,C,H,W] -> [B*S, C, H, W]
        B, S, C, H, W = predictions.shape
        preds = predictions.reshape(B * S, C, H, W)
        targs = targets.reshape(B * S, C, H, W)
    elif predictions.dim() == 4:
        preds = predictions
        targs = targets
    else:
        raise ValueError("predictions must be [S,C,H,W] or [B,S,C,H,W]")

    S = preds.shape[0]
    num_pix = preds.shape[1] * preds.shape[2] * preds.shape[3]

    preds_2d = preds.reshape(S, num_pix)
    targs_2d = targs.reshape(S, num_pix)

    abs_errors = th.abs(preds_2d - targs_2d)  # [S, N]
    mean_abs_errors = abs_errors.mean(dim=0)  # [N]

    a = preds_2d.unsqueeze(1)  # [S,1,N]
    b = preds_2d.unsqueeze(0)  # [1,S,N]
    pairwise = th.abs(a - b)  # [S,S,N]
    mean_pairwise = pairwise.mean(dim=(0, 1))  # [N]

    crps = mean_abs_errors - 0.5 * mean_pairwise  # [N]
    absorb = 0.5 * mean_pairwise / (mean_abs_errors + 1e-7)

    return crps.mean(), absorb.mean()
