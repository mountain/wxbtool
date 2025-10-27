import torch

from torch import Tensor, tensor
from collections.abc import Sequence
from typing import Any, Dict, Callable
from wxbtool.core.types import Data
from wxbtool.core.metrics import WXBMetric


class RMSE(WXBMetric):
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
                self.add_state(f"{variable}:total:{t_shift:03d}", default=tensor(0.0), dist_reduce_fx="sum")
                self.add_state(f"{variable}:sum_weighted_squared_error:{t_shift:03d}", default=torch.zeros(1), dist_reduce_fx="sum")
                self.add_state(f"{variable}:rmse:{t_shift:03d}", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, forecasts: Data, targets: Data) -> None:
        for variable in self.variables:
            denorm = self.denormalizers[variable]
            pred = denorm(forecasts[variable].detach()).to(self.device)
            trgt = denorm(targets[variable].detach()).to(self.device)
            sum_weighted_squared_error = self._sum_(self.spatio_weight * (trgt - pred) ** 2)
            total = self._sum_(self.spatio_weight * torch.ones_like(trgt))

            for t_shift in range(self.temporal_span):
                attr = f"{variable}:sum_weighted_squared_error:{t_shift:03d}"
                self._incr_(attr, sum_weighted_squared_error[:, t_shift].sum())
                attr = f"{variable}:total:{t_shift:03d}"
                self._incr_(attr, total[:, t_shift].sum())

    def compute(self) -> Tensor:
        rmse_list = torch.zeros(len(self.variables), self.temporal_span)
        for index, variable in enumerate(self.variables):
            for t_shift in range(self.temporal_span):
                total = self._get_(f"{variable}:total:{t_shift:03d}")
                mse = self._get_(f"{variable}:sum_weighted_squared_error:{t_shift:03d}") / total
                rmse = torch.sqrt(mse)
                self._set_(f"{variable}:rmse:{t_shift:03d}", rmse)
                rmse_list[index, t_shift] = rmse
        return rmse_list.mean()

    def dump(self, path:str) -> None:
        self.compute()

        result = {}
        for variable in self.variables:
            if variable != "data" and variable != "test" and variable != "seed":
                result[variable] = {}
                for t_shift in range(self.temporal_span):
                    rmse = self._get_(f"{variable}:rmse:{t_shift:03d}")
                    result[variable][f"{t_shift:03d}"] = float(rmse.cpu().numpy())

        import json
        with open(path, "w") as f:
            json.dump(result, f)
