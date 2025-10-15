import json
import os
from collections import defaultdict

import lightning as ltn
import numpy as np
import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader

from wxbtool.data.climatology import ClimatologyAccessor
from wxbtool.data.dataset import ensemble_loader
from wxbtool.nn.metrics import (
    rmse_by_time as metrics_rmse_by_time,
    rmse_weighted as metrics_rmse_weighted,
    acc_anomaly_by_time,
)


class LightningModel(ltn.LightningModule):
    def __init__(self, model, opt=None):
        super(LightningModel, self).__init__()
        self.model = model
        self.learning_rate = 1e-3

        self.opt = opt
        # CI flag
        self.ci = (
            True
            if (
                opt
                and hasattr(opt, "test")
                and opt.test == "true"
                and hasattr(opt, "ci")
                and opt.ci
            )
            else False
        )

        if opt and hasattr(opt, "rate"):
            self.learning_rate = float(opt.rate)

        self.climatology_accessors = {}
        self.data_home = os.environ.get("WXBHOME", "data")

        self.labeled_acc_prod_term = {var: 0 for var in self.model.setting.vars_out}
        self.labeled_acc_fsum_term = {var: 0 for var in self.model.setting.vars_out}
        self.labeled_acc_osum_term = {var: 0 for var in self.model.setting.vars_out}

        self.mseByVar = defaultdict()
        self.accByVar = defaultdict()

    def configure_optimizers(self):
        if hasattr(self.model, "configure_optimizers"):
            return self.model.configure_optimizers()

        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, 53)
        return [optimizer], [scheduler]

    def loss_fn(self, input, result, target, indexes=None, mode="train"):
        loss = self.model.lossfun(input, result, target)
        return loss

    def compute_rmse(self, targets, results, variable):
        tgt_data = targets[variable]
        rst_data = results[variable]
        pred_span = self.model.setting.pred_span
        weight = self.model.weight
        return metrics_rmse_weighted(
            rst_data, tgt_data, weights=weight, pred_span=pred_span, denorm_key=variable
        )

    def compute_rmse_by_time(
        self,
        targets: dict[str, th.Tensor],
        results: dict[str, th.Tensor],
        variable: str,
    ) -> th.Tensor:
        """Compute RMSE for each prediction day and overall.

        The per-day RMSE values are stored in ``self.mseByVar`` as a side
        effect so they can later be serialized to JSON for monitoring.

        Args:
            targets: Mapping of variable names to target tensors.
            results: Mapping of variable names to forecast tensors.
            variable: Name of the variable to evaluate.

        Returns:
            A tensor containing the overall RMSE across all prediction days.
        """
        tgt_data = targets[variable]
        rst_data = results[variable]
        pred_span = self.model.setting.pred_span
        weight = self.model.weight
        overall, per_day = metrics_rmse_by_time(
            rst_data, tgt_data, weights=weight, pred_span=pred_span, denorm_key=variable
        )
        epoch = self.current_epoch
        self.mseByVar.setdefault(variable, {}).setdefault(epoch, {})
        for day_idx, rmse_val in enumerate(per_day, start=1):
            self.mseByVar[variable][epoch][day_idx] = rmse_val
        return overall

    def get_climatology_accessor(self, mode):
        # Skip climatology in CI mode
        if self.ci:
            if mode not in self.climatology_accessors:
                self.climatology_accessors[mode] = ClimatologyAccessor(
                    home=f"{self.data_home}/climatology"
                )
            return self.climatology_accessors[mode]

        # Original implementation
        if mode not in self.climatology_accessors:
            self.climatology_accessors[mode] = ClimatologyAccessor(
                home=f"{self.data_home}/climatology"
            )
            years = None
            if mode == "train":
                years = tuple(self.model.setting.years_train)
            if mode == "eval":
                years = tuple(self.model.setting.years_eval)
            if mode == "test":
                years = tuple(self.model.setting.years_test)
            self.climatology_accessors[mode].build_indexers(years)

        return self.climatology_accessors[mode]

    def get_climatology(self, indexies, mode):
        batch_size = len(indexies)
        vars_out = self.model.vars_out
        step = self.model.setting.step
        span = self.model.setting.pred_span
        shift = self.model.setting.pred_shift
        height = self.model.setting.lat_size
        width = self.model.setting.lon_size

        if self.ci:
            return np.zeros((batch_size, len(vars_out), span, height, width))

        # Original implementation
        accessor = self.get_climatology_accessor(mode)
        indexies = indexies.cpu().numpy()

        result = []
        for ix in range(span):
            delta = ix * step
            shifts = list(
                [idx + delta + shift for idx in indexies]
            )  # shift to the forecast time
            data = accessor.get_climatology(vars_out, shifts).reshape(
                batch_size, len(vars_out), 1, height, width
            )
            result.append(data)
        return np.concatenate(result, axis=2)

    def calculate_acc(self, forecast, observation, indexes, variable, mode):
        # Skip plotting and simplify calculations in CI mode
        if self.ci:
            return 1.0, 1.0, 1.0

        batch = forecast.shape[0]
        pred_length = self.model.setting.pred_span
        height = self.model.setting.lat_size
        width = self.model.setting.lon_size

        climatology = self.get_climatology(indexes, mode)
        var_ind = self.model.setting.vars_out.index(variable)
        climatology = climatology[:, var_ind : var_ind + 1, :, :, :]
        forecast = forecast.reshape(batch, 1, pred_length, height, width).cpu().numpy()
        observation = (
            observation.reshape(batch, 1, pred_length, height, width).cpu().numpy()
        )
        climatology = climatology.reshape(batch, 1, pred_length, height, width)
        weight = self.model.weight.reshape(1, 1, 1, height, width).cpu().numpy()

        f_anomaly = forecast - climatology
        o_anomaly = observation - climatology

        # Compute ACC via metrics helper
        per_day_acc, prod_sum, fsum_sum, osum_sum = acc_anomaly_by_time(
            f_anomaly, o_anomaly, weights=weight
        )

        epoch = self.current_epoch
        self.accByVar.setdefault(variable, {}).setdefault(epoch, {})
        for day, acc in enumerate(per_day_acc, start=1):
            self.accByVar[variable][epoch][day] = float(acc)

        # Queue anomaly plots for logging if enabled
        if getattr(self.opt, "plot", "false") == "true":
            # ensure artifacts dict
            if not hasattr(self, "artifacts_to_log") or self.artifacts_to_log is None:
                self.artifacts_to_log = {}
            for day in range(pred_length):
                tag_f = f"anomaly_{variable}_fcs_{day}"
                tag_o = f"anomaly_{variable}_obs_{day}"
                # Use first sample in batch for visualization
                self.artifacts_to_log[tag_f] = {"var": variable, "data": f_anomaly[0, 0, day]}
                self.artifacts_to_log[tag_o] = {"var": variable, "data": o_anomaly[0, 0, day]}

        return prod_sum, fsum_sum, osum_sum

    def forecast_error(self, rmse):
        return rmse

    def forward(self, **inputs):
        return self.model(**inputs)

    def plot(self, inputs, results, targets, indexies, batch_idx, mode):
        # Skip plotting in CI mode or test mode
        if self.ci or mode == "test":
            return
        if getattr(self.opt, "plot", "false") != "true":
            return

        # ensure artifacts dict
        if not hasattr(self, "artifacts_to_log") or self.artifacts_to_log is None:
            self.artifacts_to_log = {}

        # Inputs
        for var in self.model.setting.vars_in:
            inp = inputs[var]
            span = self.model.setting.input_span
            for ix in range(span):
                if inp.dim() == 4:
                    height, width = inp.size(-2), inp.size(-1)
                    dat = inp[0, ix].detach().cpu().numpy().reshape(height, width)
                else:
                    height, width = inp.size(-2), inp.size(-1)
                    dat = inp[0, 0, ix].detach().cpu().numpy().reshape(height, width)
                tag = f"{var}_inp_{ix}"
                self.artifacts_to_log[tag] = {"var": var, "data": dat}

        # Forecast vs Target
        for var in self.model.setting.vars_out:
            fcst = results[var]
            tgrt = targets[var]
            span = self.model.setting.pred_span
            for ix in range(span):
                if fcst.dim() == 4:
                    height, width = fcst.size(-2), fcst.size(-1)
                    fcst_img = fcst[0, ix].detach().cpu().numpy().reshape(height, width)
                    tgrt_img = tgrt[0, ix].detach().cpu().numpy().reshape(height, width)
                else:
                    height, width = fcst.size(-2), fcst.size(-1)
                    fcst_img = fcst[0, 0, ix].detach().cpu().numpy().reshape(height, width)
                    tgrt_img = tgrt[0, 0, ix].detach().cpu().numpy().reshape(height, width)
                self.artifacts_to_log[f"{var}_fcst_{ix}"] = {"var": var, "data": fcst_img}
                self.artifacts_to_log[f"{var}_tgt_{ix}"] = {"var": var, "data": tgrt_img}

        # for bas, var in enumerate(self.model.setting.vars_out):
        #     if inputs[var].dim() == 4:
        #         input_data = inputs[var][0, 0].detach().cpu().numpy()
        #         truth = targets[var][0, 0].detach().cpu().numpy()
        #         forecast = results[var][0, 0].detach().cpu().numpy()
        #         input_data = denormalizors[var](input_data)
        #         forecast = denormalizors[var](forecast)
        #         truth = denormalizors[var](truth)
        #         plot_image(
        #             var,
        #             input_data=input_data,
        #             truth=truth,
        #             forecast=forecast,
        #             title="%s" % var,
        #             year=self.climatology_accessors[mode].yr_indexer[indexies[0]],
        #             doy=self.climatology_accessors[mode].doy_indexer[indexies[0]],
        #             save_path="%s_%02d.png" % (var, batch_idx),
        #         )
        #     if inputs[var].dim() == 5:
        #         input_data = inputs[var][0, 0, 0].detach().cpu().numpy()
        #         truth = targets[var][0, 0, 0].detach().cpu().numpy()
        #         forecast = results[var][0, 0, 0].detach().cpu().numpy()
        #         input_data = denormalizors[var](input_data)
        #         forecast = denormalizors[var](forecast)
        #         truth = denormalizors[var](truth)
        #         plot_image(
        #             var,
        #             input_data=input_data,
        #             truth=truth,
        #             forecast=forecast,
        #             title="%s" % var,
        #             year=self.climatology_accessors[mode].yr_indexer[indexies[0]],
        #             doy=self.climatology_accessors[mode].doy_indexer[indexies[0]],
        #             save_path="%s_%02d.png" % (var, batch_idx),
        #         )

    def training_step(self, batch, batch_idx):
        inputs, targets, indexes = batch
        # self.get_climatology(indexes, "train")

        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        results = self.forward(indexes=indexes, **inputs)

        loss = self.loss_fn(inputs, results, targets, indexes=indexes, mode="train")

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Skip additional batches only in CI mode to keep tests fast
        if self.ci and batch_idx > 0:
            return

        inputs, targets, indexes = batch
        current_batch_size = inputs[self.model.setting.vars[0]].shape[0]

        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        results = self.forward(indexes=indexes, **inputs)
        loss = self.loss_fn(inputs, results, targets, indexes=indexes, mode="eval")
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        total_rmse = 0
        for variable in self.model.setting.vars_out:
            self.mseByVar[variable] = dict()
            self.accByVar[variable] = dict()

            total_rmse += self.compute_rmse_by_time(targets, results, variable)
            prod, fsum, osum = self.calculate_acc(
                results[variable],
                targets[variable],
                indexes=indexes,
                variable=variable,
                mode="eval",
            )
            self.labeled_acc_prod_term[variable] += prod
            self.labeled_acc_fsum_term[variable] += fsum
            self.labeled_acc_osum_term[variable] += osum

        avg_rmse = total_rmse / len(self.model.setting.vars_out)
        self.log(
            "val_rmse",
            avg_rmse,
            on_step=False,
            on_epoch=True,
            batch_size=current_batch_size,
            sync_dist=True,
            prog_bar=True,
        )

        if hasattr(self.trainer, "is_global_zero") and self.trainer.is_global_zero:
            with open(os.path.join(self.logger.log_dir, "val_rmse.json"), "w") as f:
                json.dump(self.mseByVar, f)
            with open(os.path.join(self.logger.log_dir, "val_acc.json"), "w") as f:
                json.dump(self.accByVar, f)

        # Only plot for the first batch in CI mode
        if getattr(self.opt, "plot", "false") == "true" and (not self.ci or batch_idx == 0):
            self.plot(inputs, results, targets, indexes, batch_idx, mode="eval")

    def test_step(self, batch, batch_idx):
        # Skip additional batches only in CI mode to keep tests fast
        if self.ci and batch_idx > 0:
            return

        inputs, targets, indexies = batch
        current_batch_size = inputs[self.model.setting.vars[0]].shape[0]

        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        results = self.forward(indexies=indexies, **inputs)
        loss = self.loss_fn(inputs, results, targets, indexes=indexies, mode="test")
        self.log("test_loss", loss, sync_dist=True, prog_bar=True)

        total_rmse = 0
        for variable in self.model.setting.vars_out:
            self.mseByVar[variable] = dict()
            self.accByVar[variable] = dict()

            total_rmse += self.compute_rmse_by_time(targets, results, variable)
            prod, fsum, osum = self.calculate_acc(
                results[variable],
                targets[variable],
                indexes=indexies,
                variable=variable,
                mode="test",
            )
            self.labeled_acc_prod_term[variable] += prod
            self.labeled_acc_fsum_term[variable] += fsum
            self.labeled_acc_osum_term[variable] += osum

        avg_rmse = total_rmse / len(self.model.setting.vars_out)
        self.log(
            "test_rmse",
            avg_rmse,
            on_step=False,
            on_epoch=True,
            batch_size=current_batch_size,
            sync_dist=True,
            prog_bar=True,
        )

        if hasattr(self.trainer, "is_global_zero") and self.trainer.is_global_zero:
            with open(os.path.join(self.logger.log_dir, "test_rmse.json"), "w") as f:
                json.dump(self.mseByVar, f)
            with open(os.path.join(self.logger.log_dir, "test_acc.json"), "w") as f:
                json.dump(self.accByVar, f)

        self.plot(inputs, results, targets, indexies, batch_idx, mode="test")

    def on_save_checkpoint(self, checkpoint):
        self.labeled_acc_prod_term = {var: 0 for var in self.model.setting.vars_out}
        self.labeled_acc_fsum_term = {var: 0 for var in self.model.setting.vars_out}
        self.labeled_acc_osum_term = {var: 0 for var in self.model.setting.vars_out}

        return checkpoint

    def train_dataloader(self):
        if self.model.dataset_train is None:
            if self.opt.data != "":
                self.model.load_dataset("train", "client", url=self.opt.data)
            else:
                self.model.load_dataset("train", "server")

        # Use a smaller batch size and fewer workers in CI mode
        batch_size = 5 if self.ci else self.opt.batch_size
        num_workers = 2 if self.ci else self.opt.n_cpu

        return DataLoader(
            self.model.dataset_train,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        if self.model.dataset_eval is None:
            if self.opt.data != "":
                self.model.load_dataset("train", "client", url=self.opt.data)
            else:
                self.model.load_dataset("train", "server")

        # Use a smaller batch size and fewer workers in CI mode
        batch_size = 5 if self.ci else self.opt.batch_size
        num_workers = 2 if self.ci else self.opt.n_cpu

        return DataLoader(
            self.model.dataset_eval,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        if self.model.dataset_test is None:
            if self.opt.data != "":
                self.model.load_dataset("train", "client", url=self.opt.data)
            else:
                self.model.load_dataset("train", "server")

        # Use a smaller batch size and fewer workers in CI mode
        batch_size = 5 if self.ci else self.opt.batch_size
        num_workers = 2 if self.ci else self.opt.n_cpu

        return DataLoader(
            self.model.dataset_test,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )


class GANModel(LightningModel):
    def __init__(self, generator, discriminator, opt=None):
        super(GANModel, self).__init__(generator, opt=opt)
        self.generator = generator
        self.discriminator = discriminator
        self.learning_rate = 1e-4  # Adjusted for GANs
        self.automatic_optimization = False
        self.realness = 0
        self.fakeness = 1
        self.alpha = 0.5
        self.crps = None

        if opt and hasattr(opt, "rate"):
            learning_rate = float(opt.rate)
            ratio = float(opt.ratio)
            self.generator.learning_rate = learning_rate
            self.discriminator.learning_rate = learning_rate / ratio

        if opt and hasattr(opt, "alpha"):
            self.alpha = float(opt.alpha)

        self.crpsByVar = {}

    def configure_optimizers(self):
        # Separate optimizers for generator and discriminator
        g_optimizer = th.optim.Adam(
            self.generator.parameters(), lr=self.generator.learning_rate
        )
        d_optimizer = th.optim.Adam(
            self.discriminator.parameters(), lr=self.discriminator.learning_rate
        )
        g_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, 37)
        d_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(d_optimizer, 37)
        return [g_optimizer, d_optimizer], [g_scheduler, d_scheduler]

    def generator_loss(self, fake_judgement):
        # Loss for generator (we want the discriminator to predict all generated images as real)
        return th.nn.functional.binary_cross_entropy_with_logits(
            fake_judgement["data"],
            th.ones_like(fake_judgement["data"], dtype=th.float32),
        )

    def discriminator_loss(self, real_judgement, fake_judgement):
        # Loss for discriminator (real images should be classified as real, fake images as fake)
        real_loss = th.nn.functional.binary_cross_entropy_with_logits(
            real_judgement["data"],
            th.ones_like(real_judgement["data"], dtype=th.float32),
        )
        fake_loss = th.nn.functional.binary_cross_entropy_with_logits(
            fake_judgement["data"],
            th.zeros_like(fake_judgement["data"], dtype=th.float32),
        )
        return (real_loss + fake_loss) / 2

    def forecast_error(self, rmse):
        return self.generator.forecast_error(rmse)

    def compute_crps(self, predictions, targets):
        """
        计算逐时间步的 CRPS，并返回形状为 (B, C, T, H, W) 的张量，兼容后续使用:
          - crps.size(2) 用作时间步个数
          - crps[:, cidx, tidx] 用作 (B, H, W) 的切片
        predictions: (B, T, H, W) 或 (B, C, T, H, W)
        targets:     (B, T, H, W) 或 (B, C, T, H, W)
        其中 B 在 GAN 的 ensemble_loader 中恰好是集合大小。
        """
        # 统一到 5D
        if predictions.dim() == 4:
            predictions5 = predictions.unsqueeze(1)
            targets5 = targets.unsqueeze(1)
        elif predictions.dim() == 5:
            predictions5 = predictions
            targets5 = targets
        else:
            raise ValueError(f"Unsupported predictions dim: {predictions.dim()}")

        B, C, T, H, W = predictions5.shape

        # CI 模式下跳过重计算，返回零张量（形状保持一致，便于后续 .size(2)/索引）
        if self.ci:
            zeros = predictions5.new_zeros((B, C, T, H, W))
            self.crps = zeros
            self.absorb = zeros
            return zeros, zeros

        crps_ts = []
        absb_ts = []

        for t in range(T):
            # (B, C, H, W) -> (B, P) 其中 P=C*H*W
            pred_t = predictions5[:, :, t, :, :].contiguous().view(B, C * H * W)
            targ_t = targets5[:, :, t, :, :].contiguous().view(B, C * H * W)

            # E|F - O|
            abs_errors = th.abs(pred_t - targ_t)            # (B, P)
            mean_abs_errors = abs_errors.mean(dim=0)        # (P,)

            # E|F - F'|
            pa = pred_t.unsqueeze(1)                         # (B, 1, P)
            pb = pred_t.unsqueeze(0)                         # (1, B, P)
            pairwise_diff = th.abs(pa - pb)                  # (B, B, P)
            mean_pairwise_diff = pairwise_diff.mean(dim=(0, 1))  # (P,)

            crps_vec = mean_abs_errors - 0.5 * mean_pairwise_diff     # (P,)
            absb_vec = 0.5 * mean_pairwise_diff / (mean_abs_errors + 1e-7)  # (P,)

            # 回到 (C, H, W)，并扩展 B 维保持 (B, C, H, W) 以兼容下游索引
            crps_map = crps_vec.view(C, H, W)               # (C, H, W)
            absb_map = absb_vec.view(C, H, W)               # (C, H, W)

            crps_exp = crps_map.unsqueeze(0).expand(B, -1, -1, -1).contiguous()  # (B, C, H, W)
            absb_exp = absb_map.unsqueeze(0).expand(B, -1, -1, -1).contiguous()  # (B, C, H, W)

            crps_ts.append(crps_exp)
            absb_ts.append(absb_exp)

        crps = th.stack(crps_ts, dim=2)   # (B, C, T, H, W)
        absb = th.stack(absb_ts, dim=2)   # (B, C, T, H, W)

        # 保存到对象，供需要时使用
        self.crps = crps
        self.absorb = absb

        return crps, absb

    def training_step(self, batch, batch_idx):
        inputs, targets, indexies = batch
        # self.get_climatology(indexies, "train")

        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        data = inputs["data"]
        if data.dim() == 5:
            seed = th.randn_like(data[:, :, :1, :, :], dtype=th.float32)
        elif data.dim() == 4:
            seed = th.randn_like(data[:, :1, :, :], dtype=th.float32)
        else:
            seed = th.randn_like(data, dtype=th.float32)
        inputs["seed"] = seed

        g_optimizer, d_optimizer = self.optimizers()

        self.toggle_optimizer(g_optimizer)
        forecast = self.generator(**inputs)
        real_judgement = self.discriminator(**inputs, target=targets["data"])
        fake_judgement = self.discriminator(**inputs, target=forecast["data"])
        forecast_loss = self.loss_fn(
            inputs, forecast, targets, indexes=indexies, mode="train"
        )
        generate_loss = self.generator_loss(fake_judgement)
        total_loss = self.alpha * forecast_loss + (1 - self.alpha) * generate_loss
        self.manual_backward(total_loss)
        g_optimizer.step()
        g_optimizer.zero_grad()
        realness = real_judgement["data"].mean().item()
        fakeness = fake_judgement["data"].mean().item()
        self.realness = realness
        self.fakeness = fakeness
        self.log("realness", self.realness, prog_bar=True, sync_dist=True)
        self.log("fakeness", self.fakeness, prog_bar=True, sync_dist=True)
        self.log("total", total_loss, prog_bar=True, sync_dist=True)
        self.log("forecast", forecast_loss, prog_bar=True, sync_dist=True)
        self.untoggle_optimizer(g_optimizer)

        self.toggle_optimizer(d_optimizer)
        forecast = self.generator(**inputs)
        forecast["data"] = forecast[
            "data"
        ].detach()  # Detach to avoid generator gradient updates
        real_judgement = self.discriminator(**inputs, target=targets["data"])
        fake_judgement = self.discriminator(**inputs, target=forecast["data"])
        judgement_loss = self.discriminator_loss(real_judgement, fake_judgement)
        self.manual_backward(judgement_loss)
        d_optimizer.step()
        d_optimizer.zero_grad()
        realness = real_judgement["data"].mean().item()
        fakeness = fake_judgement["data"].mean().item()
        self.realness = realness
        self.fakeness = fakeness
        self.log("realness", self.realness, prog_bar=True, sync_dist=True)
        self.log("fakeness", self.fakeness, prog_bar=True, sync_dist=True)
        self.log("judgement", judgement_loss, prog_bar=True, sync_dist=True)
        self.untoggle_optimizer(d_optimizer)

        if self.opt.plot == "true":
            if batch_idx % 10 == 0:
                self.plot(inputs, forecast, targets, indexies, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        # Skip validation for some batches in CI mode or if batch_idx > 2 in any mode
        if (self.ci and batch_idx > 0) or batch_idx > 2:
            return

        inputs, targets, indexies = batch
        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        data = inputs["data"]
        if data.dim() == 5:
            seed = th.randn_like(data[:, :, :1, :, :], dtype=th.float32)
        elif data.dim() == 4:
            seed = th.randn_like(data[:, :1, :, :], dtype=th.float32)
        else:
            seed = th.randn_like(data, dtype=th.float32)
        inputs["seed"] = seed
        forecast = self.generator(**inputs)
        forecast_loss = self.loss_fn(
            inputs, forecast, targets, indexes=indexies, mode="eval"
        )
        real_judgement = self.discriminator(**inputs, target=targets["data"])
        fake_judgement = self.discriminator(**inputs, target=forecast["data"])
        crps, absb = self.compute_crps(forecast["data"], targets["data"])
        self.log("crps", crps.mean(), prog_bar=True, sync_dist=True)
        self.log("absb", absb.mean(), prog_bar=True, sync_dist=True)
        crps = self.forecast_error(crps)

        current_batch_size = forecast["data"].shape[0]
        for cidx, variable in enumerate(self.model.setting.vars_out):
            self.crpsByVar[variable] = dict()
            self.mseByVar[variable] = dict()
            self.accByVar[variable] = dict()

            self.compute_rmse_by_time(targets, forecast, variable)

            for tidx in range(crps.size(2)):
                self.crpsByVar[variable][tidx + 1] = float(crps[:, cidx, tidx].mean().item())

            prod, fsum, osum = self.calculate_acc(
                forecast[variable],
                targets[variable],
                indexes=indexies,
                variable=variable,
                mode="eval",
            )
            self.labeled_acc_prod_term[variable] += prod
            self.labeled_acc_fsum_term[variable] += fsum
            self.labeled_acc_osum_term[variable] += osum
            acc = self.labeled_acc_prod_term[variable] / np.sqrt(
                self.labeled_acc_fsum_term[variable]
                * self.labeled_acc_osum_term[variable]
            )
            self.log(
                f"val_acc_{variable}",
                acc,
                on_step=False,
                on_epoch=True,
                batch_size=current_batch_size,
                sync_dist=True,
            )

        if hasattr(self.trainer, "is_global_zero") and self.trainer.is_global_zero:
            with open(os.path.join(self.logger.log_dir, "val_crps.json"), "w") as f:
                json.dump(self.crpsByVar, f)
            with open(os.path.join(self.logger.log_dir, "val_rmse.json"), "w") as f:
                json.dump(self.mseByVar, f)
            with open(os.path.join(self.logger.log_dir, "val_acc.json"), "w") as f:
                json.dump(self.accByVar, f)

        self.realness = real_judgement["data"].mean().item()
        self.fakeness = fake_judgement["data"].mean().item()
        self.log("realness", self.realness, prog_bar=True, sync_dist=True)
        self.log("fakeness", self.fakeness, prog_bar=True, sync_dist=True)
        self.log("val_forecast", forecast_loss, prog_bar=True, sync_dist=True)
        self.log("val_loss", forecast_loss, prog_bar=True, sync_dist=True)

        if hasattr(self.trainer, "is_global_zero") and self.trainer.is_global_zero:
            if self.opt.plot == "true":
                self.plot(inputs, forecast, targets, indexies, batch_idx, mode="eval")

    def test_step(self, batch, batch_idx):
        # Skip test for some batches in CI mode or if batch_idx > 1 in any mode
        if (self.ci and batch_idx > 0) or batch_idx > 1:
            return

        inputs, targets, indexies = batch
        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        data = inputs["data"]
        if data.dim() == 5:
            seed = th.randn_like(data[:, :, :1, :, :], dtype=th.float32)
        elif data.dim() == 4:
            seed = th.randn_like(data[:, :1, :, :], dtype=th.float32)
        else:
            seed = th.randn_like(data, dtype=th.float32)
        inputs["seed"] = seed
        forecast = self.generator(**inputs)
        forecast_loss = self.loss_fn(
            inputs, forecast, targets, indexes=indexies, mode="test"
        )
        real_judgement = self.discriminator(**inputs, target=targets["data"])
        fake_judgement = self.discriminator(**inputs, target=forecast["data"])
        self.realness = real_judgement["data"].mean().item()
        self.fakeness = fake_judgement["data"].mean().item()
        crps, absb = self.compute_crps(forecast["data"], targets["data"])

        total_rmse = 0
        current_batch_size = forecast["data"].shape[0]
        for variable in self.model.setting.vars_out:
            rmse = self.compute_rmse(targets, forecast, variable)
            self.log(
                f"val_rmse_{variable}",
                rmse,
                on_step=False,
                on_epoch=True,
                batch_size=current_batch_size,
                sync_dist=True,
            )
            total_rmse += rmse

            # Calculate the average RMSE across all variables
            avg_rmse = total_rmse / len(self.model.setting.vars_out)
            self.log(
                "val_rmse",
                avg_rmse,
                on_step=False,
                on_epoch=True,
                batch_size=current_batch_size,
                sync_dist=True,
                prog_bar=True,
            )

        # self.labeled_mse_numerator += mse_numerator
        # self.labeled_mse_denominator += mse_denominator
        # rmse = np.sqrt(self.labeled_mse_numerator / self.labeled_mse_denominator)

        # prod, fsum, osum = self.calculate_acc(
        #     forecast["data"], targets["data"], indexies, mode="test"
        # )
        # self.labeled_acc_prod_term += prod
        # self.labeled_acc_fsum_term += fsum
        # self.labeled_acc_osum_term += osum
        # acc = self.labeled_acc_prod_term / np.sqrt(
        #     self.labeled_acc_fsum_term * self.labeled_acc_osum_term
        # )

        self.log("realness", self.realness, prog_bar=True, sync_dist=True)
        self.log("fakeness", self.fakeness, prog_bar=True, sync_dist=True)
        self.log("forecast", forecast_loss, prog_bar=True, sync_dist=True)
        self.log("crps", crps.mean(), prog_bar=True, sync_dist=True)
        self.log("absb", absb.mean(), prog_bar=True, sync_dist=True)
        # self.log("acc", acc, prog_bar=True, sync_dist=True)
        self.log("rmse", rmse, prog_bar=True, sync_dist=True)

        # Only plot for the first batch in CI mode
        if not self.ci or batch_idx == 0:
            self.plot(inputs, forecast, targets, indexies, batch_idx, mode="test")

    def on_validation_epoch_end(self):
        balance = self.realness - self.fakeness
        self.log("balance", balance)
        if abs(balance - self.opt.balance) < self.opt.tolerance:
            self.trainer.should_stop = True

    def val_dataloader(self):
        if self.model.dataset_eval is None:
            if self.opt.data != "":
                self.model.load_dataset("train", "client", url=self.opt.data)
            else:
                self.model.load_dataset("train", "server")

        # Use a smaller batch size in CI mode
        batch_size = 5 if self.ci else self.opt.batch_size

        return ensemble_loader(
            self.model.dataset_eval,
            batch_size,
            False,
        )

    def test_dataloader(self):
        if self.model.dataset_test is None:
            if self.opt.data != "":
                self.model.load_dataset("train", "client", url=self.opt.data)
            else:
                self.model.load_dataset("train", "server")

        # Use a smaller batch size in CI mode
        batch_size = 5 if self.ci else self.opt.batch_size

        return ensemble_loader(
            self.model.dataset_test,
            batch_size,
            False,
        )
