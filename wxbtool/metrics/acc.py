import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch import Tensor, tensor
from collections.abc import Sequence
from typing import Any, Dict, Callable

from wxbtool.data.climatology import ClimatologyAccessor
from wxbtool.core.types import Data, Indexes
from wxbtool.core.metrics import WXBMetric


def plot_seasonal_check(pred_phys, target_phys, clim_phys, variable_name, start_index, t_shift, save_dir="debug_plots"):
    os.makedirs(save_dir, exist_ok=True)

    p_map = pred_phys[0, 0].cpu().detach().numpy()
    t_map = target_phys[0, 0].cpu().detach().numpy()
    c_map = clim_phys[0, 0].cpu().detach().numpy()

    pred_anomaly = p_map - c_map
    target_anomaly = t_map - c_map

    vmin_raw, vmax_raw = t_map.min(), t_map.max()
    abs_max = max(np.max(np.abs(pred_anomaly)), np.max(np.abs(target_anomaly)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- Row 1: Forecast Perspective ---
    # 1. Prediction Raw
    im1 = axes[0, 0].imshow(p_map, cmap='coolwarm', vmin=vmin_raw, vmax=vmax_raw)
    axes[0, 0].set_title(f"Forecast (Raw)\nMean: {p_map.mean():.1f}")
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # 2. Climatology (Reference)
    im2 = axes[0, 1].imshow(c_map, cmap='coolwarm', vmin=vmin_raw, vmax=vmax_raw)
    axes[0, 1].set_title(f"Climatology (Day {start_index} + {t_shift})\nMean: {c_map.mean():.1f}")
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 3. Forecast Anomaly
    im3 = axes[0, 2].imshow(pred_anomaly, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
    axes[0, 2].set_title(f"Forecast Anomaly (Pred - Clim)\nMeanAbs: {np.mean(np.abs(pred_anomaly)):.2f}")
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # --- Row 2: Ground Truth Perspective ---
    # 4. Target Raw
    im4 = axes[1, 0].imshow(t_map, cmap='coolwarm', vmin=vmin_raw, vmax=vmax_raw)
    axes[1, 0].set_title(f"Target (Obs)\nMean: {t_map.mean():.1f}")
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 5. Climatology (Copy for easier comparison)
    im5 = axes[1, 1].imshow(c_map, cmap='coolwarm', vmin=vmin_raw, vmax=vmax_raw)
    axes[1, 1].set_title(f"Climatology (Same)\nMean: {c_map.mean():.1f}")
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 6. Target Anomaly
    im6 = axes[1, 2].imshow(target_anomaly, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
    axes[1, 2].set_title(f"Target Anomaly (Obs - Clim)\nMeanAbs: {np.mean(np.abs(target_anomaly)):.2f}")
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # Global Title
    fig.suptitle(f"Variable: {variable_name} | Start Index: {start_index} | Lead Time: {t_shift} days", fontsize=16)

    plt.tight_layout()
    filename = f"{save_dir}/debug_{variable_name}_idx{start_index:04d}_lead{t_shift:03d}.png"
    plt.savefig(filename)
    plt.close()


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

                    is_first_batch = (indexes[0] == 0) if isinstance(indexes[0], int) else (indexes[0].item() == 0)
                    if variable == "t2m" and is_first_batch:
                        start_index = indexes[0]
                        plot_seasonal_check(
                            pred_phys=pred,
                            target_phys=trgt,
                            clim_phys=clim,
                            variable_name=variable,
                            start_index=start_index,
                            t_shift=t_shift * self.temporal_step + self.temporal_shift,
                            save_dir="debug_plots"
                        )


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