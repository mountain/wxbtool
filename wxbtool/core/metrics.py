from typing import List, Optional, Tuple, Sequence, Dict, Callable, Any, Mapping

import numpy as np
import torch as th
from torch import Tensor
from torchmetrics import Metric

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


class WXBMetric(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False
    plot_lower_bound: float = 0.0

    def __init__(
        self,
        temporal_span: int,
        spatio_weight: Tensor,
        variables: Sequence[str],
        denormalizers: Dict[str, Callable],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(temporal_span, int):
            raise ValueError(f"Expected argument `pred_span` to be an integer but got {temporal_span}")
        self.temporal_span = temporal_span

        if not isinstance(spatio_weight, Tensor):
            raise ValueError(f"Expected argument `weight` to be a tensor but got {spatio_weight}")
        self.spatio_weight = spatio_weight

        if not isinstance(variables, Sequence):
            raise ValueError(f"Expected argument `variables` to be a sequence but got {variables}")
        self.variables = variables

        if not isinstance(denormalizers, Mapping):
            raise ValueError(f"Expected argument `denormalizers` to be a mapping but got {denormalizers}")
        self.denormalizers = denormalizers

    def __repr__(self):
        return repr(self.compute())

    def _get_(self, attr:str) -> Tensor:
        return getattr(self, attr)

    def _set_(self, attr:str, value:Tensor) -> None:
        setattr(self, attr, value)

    def _incr_(self, attr:str, value:Tensor) -> None:
        setattr(self, attr, getattr(self, attr) + value)

    def _sum_(self, value:Tensor) -> Tensor:
        # summarizing among spatio dimensions and batch
        return value.sum(dim=-1).sum(dim=-1).sum(dim=0)
