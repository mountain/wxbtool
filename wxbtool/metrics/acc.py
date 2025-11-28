import os
import torch

from torch import Tensor, tensor
from collections.abc import Sequence
from typing import Any, Dict, Callable

from wxbtool.data.climatology import ClimatologyAccessor
from wxbtool.core.types import Data, Indexes
from wxbtool.core.metrics import WXBMetric


class ACC(WXBMetric):
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
        super().__init__(temporal_span, temporal_step, temporal_shift, spatio_weight, variables, denormalizers,
                         **kwargs)

        data_home = os.environ.get("WXBHOME", "data")
        self.climatology_accessor = ClimatologyAccessor(home=f"{data_home}/climatology")

        for variable in variables:
            for t_shift in range(self.temporal_span):
                attr = f"{variable}:acc:{t_shift:03d}"
                self.add_state(attr, default=tensor(0.0), dist_reduce_fx="mean")

                attr = f"{variable}:prod:{t_shift:03d}"
                self.add_state(attr, default=tensor(0.0), dist_reduce_fx="sum")
                attr = f"{variable}:fsum:{t_shift:03d}"
                self.add_state(attr, default=tensor(0.0), dist_reduce_fx="sum")
                attr = f"{variable}:osum:{t_shift:03d}"
                self.add_state(attr, default=tensor(0.0), dist_reduce_fx="sum")

    def build_indexers(self, years: Sequence[int]) -> None:
        self.climatology_accessor.build_indexers(years)

    def climatology(self, indexes: Indexes, device: torch.device, dtype: torch.dtype) -> Data:
        batch_size = len(indexes)
        result, data = {}, []
        clean_variables = [var for var in self.variables if var != "data" and var != "seed"]
        for ix in range(self.temporal_span):
            delta = ix * self.temporal_step
            shifts = [idx + delta + self.temporal_shift for idx in indexes]
            clim = self.climatology_accessor.get_climatology(clean_variables, shifts).reshape(
                batch_size, len(clean_variables), 1, self.spatio_height, self.spatio_width
            )
            data.append(torch.as_tensor(clim, device=device, dtype=dtype))
        data = torch.cat(data, dim=2)  # B, C, T, H, W
        for var_index, variable in enumerate(clean_variables):
            result[variable] = data[:, var_index: var_index + 1, :, :, :]
        result["data"] = data
        return result

    def update(self, forecasts: Data, targets: Data, indexes: Indexes, **kwargs) -> None:
        var0 = next(v for v in self.variables if v not in ("data", "test", "seed"))
        ref = forecasts[var0]
        device, dtype = ref.device, ref.dtype
        climatology = self.climatology(indexes, device=device, dtype=dtype)
        weight = self.spatio_weight.to(device=device, dtype=dtype)

        for variable in self.variables:
            if variable != "data" and variable != "test" and variable != "seed":
                for t_shift in range(self.temporal_span):
                    denorm = self.denormalizers[variable]
                    pred = denorm(forecasts[variable].detach()[:, :, t_shift:t_shift + 1])
                    trgt = denorm(targets[variable].detach()[:, :, t_shift:t_shift + 1])
                    clim = denorm(climatology[variable].detach()[:, :, t_shift:t_shift + 1])
                    pred = pred.to(clim.device)
                    trgt = trgt.to(clim.device)

                    if "enable_da" in kwargs and kwargs["enable_da"]:
                        lng_shift = kwargs["lng_shift"]
                        flip_status = kwargs["flip_status"]
                        clim_data = []
                        for ix, (shift, flip) in enumerate(zip(lng_shift, flip_status)):
                            slice = torch.roll(clim[ix:ix+1], shifts=shift, dims=-1)
                            if flip == 1:
                                slice = torch.flip(slice, dims=(-2,-1))
                            clim_data.append(slice)
                        clim = torch.cat(clim_data, dim=0)

                    print(f"Variable: {variable}")
                    print(f"Pred Mean: {pred.mean().item():.4f}, Std: {pred.std().item():.4f}")
                    print(f"Clim Mean: {clim.mean().item():.4f}, Std: {clim.std().item():.4f}")
                    print(f"Diff Mean: {(pred - clim).abs().mean().item():.4f}")

                    anomaly_f = pred - clim
                    anomaly_o = trgt - clim

                    prod = self._sum_(weight * anomaly_f * anomaly_o).sum()
                    fsum = self._sum_(weight * anomaly_f ** 2).sum()
                    osum = self._sum_(weight * anomaly_o ** 2).sum()

                    attr = f"{variable}:prod:{t_shift:03d}"
                    self._incr_(attr, prod)
                    attr = f"{variable}:fsum:{t_shift:03d}"
                    self._incr_(attr, fsum)
                    attr = f"{variable}:osum:{t_shift:03d}"
                    self._incr_(attr, osum)

    def compute(self) -> Tensor:
        acc_list = torch.zeros(len(self.variables), self.temporal_span)
        for index, variable in enumerate(self.variables):
            if variable != "data" and variable != "test" and variable != "seed":
                for t_shift in range(self.temporal_span):
                    prod = self._get_(f"{variable}:prod:{t_shift:03d}")
                    fsum = self._get_(f"{variable}:fsum:{t_shift:03d}")
                    osum = self._get_(f"{variable}:osum:{t_shift:03d}")

                    denominator = torch.sqrt(fsum * osum) + 1e-12

                    acc = prod / denominator
                    self._set_(f"{variable}:acc:{t_shift:03d}", acc)
                    acc_list[index, t_shift] = acc
        return acc_list.mean()

    def dump(self, path: str) -> None:
        result = {}
        for variable in self.variables:
            if variable != "data" and variable != "test" and variable != "seed":
                result[variable] = {}
                for t_shift in range(self.temporal_span):
                    acc = self._get_(f"{variable}:acc:{t_shift:03d}")
                    result[variable][f"{t_shift:03d}"] = float(acc.cpu().numpy())

        import json
        with open(path, "w") as f:
            json.dump(result, f)