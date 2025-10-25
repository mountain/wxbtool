import torch

from torch import Tensor, tensor
from collections.abc import Sequence, Mapping
from typing import Any, Dict, Callable
from wxbtool.types import Data
from torchmetrics.metric import Metric


class RMSE(Metric):
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

        for variable in variables:
            for t_shift in range(self.temporal_span):
                self.add_state(f"{variable}:total:{t_shift:03d}", default=tensor(0.0), dist_reduce_fx="sum")
                self.add_state(f"{variable}:sum_weighted_squared_error:{t_shift:03d}", default=torch.zeros(1), dist_reduce_fx="sum")
                self.add_state(f"{variable}:rmse:{t_shift:03d}", default=torch.zeros(1), dist_reduce_fx="sum")

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

    def update(self, forecasts: Data, targets: Data) -> None:
        for variable in self.variables:
            denorm = self.denormalizers[variable]
            pred = denorm(forecasts[variable].detach())
            trgt = denorm(targets[variable].detach())
            sum_weighted_squared_error = self._sum_(self.spatio_weight * (trgt - pred) ** 2)
            total = self._sum_(self.spatio_weight * torch.ones_like(trgt))

            for t_shift in range(self.temporal_span):
                attr = f"{variable}:sum_weighted_squared_error:{t_shift:03d}"
                self._incr_(attr, sum_weighted_squared_error[:, t_shift].sum())
                attr = f"{variable}:total:{t_shift:03d}"
                self._incr_(attr, total[:, t_shift].sum())

    def compute(self) -> Tensor:
        rmse_list = torch.zeros(self.temporal_span)
        for variable in self.variables:
            for t_shift in range(self.temporal_span):
                total = self._get_(f"{variable}:total:{t_shift:03d}")
                mse = self._get_(f"{variable}:sum_weighted_squared_error:{t_shift:03d}") / total
                rmse = torch.sqrt(mse)
                self._set_(f"{variable}:rmse:{t_shift:03d}", rmse)
                rmse_list[t_shift] = rmse
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
