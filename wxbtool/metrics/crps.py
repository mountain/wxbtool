import torch

from torch import Tensor
from collections.abc import Sequence
from typing import Any, Dict, Callable
from wxbtool.core.types import Data
from wxbtool.core.metrics import WXBMetric


class CRPS(WXBMetric):
    def __init__(
        self,
        temporal_span: int,
        temporal_step: int,
        temporal_shift: int,
        spatio_weight: Tensor,
        variables: Sequence[str],
        denormalizers: Dict[str, Callable],
        **kwargs: Any,
    ) -> None:
        super().__init__(temporal_span, temporal_step, temporal_shift, spatio_weight, variables, denormalizers, **kwargs)

        for variable in variables:
            for t_shift in range(self.temporal_span):
                attr = f"{variable}:crps:{t_shift:03d}"
                self.add_state(attr, default=torch.zeros(1), dist_reduce_fx="sum")
                attr = f"{variable}:crps_sum:{t_shift:03d}"
                self.add_state(attr, default=torch.zeros(1), dist_reduce_fx="sum")
                attr = f"{variable}:crps_total:{t_shift:03d}"
                self.add_state(attr, default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, forecasts: Data, targets: Data) -> None:
        for variable in self.variables:
            for t_shift in range(self.temporal_span):
                denorm = self.denormalizers[variable]
                pred = denorm(forecasts[variable].detach())[:, :, t_shift].to(self.device)
                trgt = denorm(targets[variable].detach())[:, :, t_shift].to(self.device)

                pa = pred.unsqueeze(1)  # (B, 1, P)
                pb = pred.unsqueeze(0)  # (1, B, P)
                mean_pairwise_diff = torch.abs(pa - pb).mean(dim=(0, 1))
                mean_abs_errors = torch.abs(pred - trgt).mean(dim=0)
                crps = (mean_abs_errors - 0.5 * mean_pairwise_diff).mean()

                attr = f"{variable}:crps:{t_shift:03d}"
                self._incr_(attr, crps)
                attr = f"{variable}:crps_total:{t_shift:03d}"
                self._incr_(attr, torch.ones_like(crps).sum())

    def compute(self) -> Tensor:
        crps_list = torch.zeros(len(self.variables), self.temporal_span)
        for index, variable in enumerate(self.variables):
            for t_shift in range(self.temporal_span):
                total = self._get_(f"{variable}:crps_total:{t_shift:03d}")
                crps = self._get_(f"{variable}:crps_sum:{t_shift:03d}") / total
                self._set_(f"{variable}:crps:{t_shift:03d}", crps)
                crps_list[index, t_shift] = crps
        return crps_list.mean()

    def dump(self, path:str) -> None:
        self.compute()

        result = {}
        for variable in self.variables:
            if variable != "data" and variable != "test" and variable != "seed":
                result[variable] = {}
                for t_shift in range(self.temporal_span):
                    crps = self._get_(f"{variable}:crps:{t_shift:03d}")
                    result[variable][f"{t_shift:03d}"] = float(crps.cpu().numpy())

        import json
        with open(path, "w") as f:
            json.dump(result, f)
