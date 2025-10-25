from typing import List, Optional, Tuple

import numpy as np
import torch as th

from wxbtool.norms.meanstd import denormalizors


def _ensure_5d(x: th.Tensor, pred_span: int) -> th.Tensor:
    if x.dim() == 5:
        return x
    if x.dim() == 4:
        # Assume [B, C, H, W]
        B, C, H, W = x.shape
        return x.view(B, C, 1, H, W)
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
    vars: List[str]
) -> Tuple[th.Tensor, th.Tensor]:
    if predictions.dim() == 4:
        predictions = predictions.unsqueeze(1)
        targets = targets.unsqueeze(1)
    if predictions.dim() != 5:
        raise ValueError(f"Unsupported predictions dim: {predictions.dim()}")

    B, C, T, H, W = predictions.shape
    crps_results, absb_results = {}, {}
    for cidx, var in enumerate(vars):
        crps_ts, absb_ts = [], []
        for tidx in range(T):
            # (B, C, H, W) -> (B, P) 其中 P=H*W
            pred_t = predictions[:, cidx, tidx, :, :].contiguous().view(B, H * W)
            targ_t = targets[:, cidx, tidx, :, :].contiguous().view(B, H * W)
            pred_t = denormalizors[var](pred_t)
            targ_t = denormalizors[var](targ_t)

            # E|F - O|
            abs_errors = th.abs(pred_t - targ_t)  # (B, P)
            mean_abs_errors = abs_errors.mean(dim=0)  # (P,)

            # E|F - F'|
            pa = pred_t.unsqueeze(1)  # (B, 1, P)
            pb = pred_t.unsqueeze(0)  # (1, B, P)
            pairwise_diff = th.abs(pa - pb)  # (B, B, P)
            mean_pairwise_diff = pairwise_diff.mean(dim=(0, 1))  # (P,)

            crps_vec = mean_abs_errors - 0.5 * mean_pairwise_diff  # (P,)
            absb_vec = 0.5 * mean_pairwise_diff / (mean_abs_errors + 1e-7)  # (P,)
            crps = crps_vec.mean().view(1)
            absb = absb_vec.mean().view(1)
            crps_ts.append(crps)
            absb_ts.append(absb)

        crps = th.cat(crps_ts, dim=0)  # (T)
        absb = th.cat(absb_ts, dim=0)  # (T)
        crps_results[var] = crps.detach().cpu().numpy().tolist()
        absb_results[var] = absb.detach().cpu().numpy().tolist()

    return crps_results, absb_results
