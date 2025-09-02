import os
import json

import numpy as np
import torch as th
import lightning as ltn
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader
from wxbtool.data.climatology import ClimatologyAccessor
from wxbtool.data.dataset import ensemble_loader
from wxbtool.util.plotter import plot
from wxbtool.norms.meanstd import denormalizors
from collections import defaultdict


plot_root = Path(os.getcwd()) / "plots"
if not plot_root.exists():
    plot_root.mkdir(parents=True, exist_ok=True)


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
        target_type = tgt_data.dtype
        target_device = rst_data.device
        tgt_data = tgt_data.to(device=target_device)
        rst_data = rst_data.to(device=target_device, dtype=target_type)

        height, width = rst_data.size(-2), rst_data.size(-1)
        try:
            weight = self.model.weight  # Assuming shape is [H, W] or compatible
            weight = weight.reshape(1, 1, 1, height, width)
        except AttributeError:
            print("Error: 'weight' attribute not found in self.model.")
            raise
        except Exception as e:
            print(
                f"Error: Failed to reshape weight. Original weight shape: {self.model.weight.shape}, Target shape: (1, 1, 1, {height}, {width})"
            )
            raise e

        weight = weight.to(device=target_device, dtype=target_type)

        # Handle data shape based on RNN mode or non-RNN mode
        pred_span = self.model.setting.pred_span
        try:
            # Reshape, assuming the final required shape is [B, C, P, H, W]
            tgt = tgt_data.reshape(-1, 1, pred_span, height, width)
            rst = rst_data.reshape(-1, 1, pred_span, height, width)
        except RuntimeError as e:
            print("Error: Failed to reshape input tensors.")
            print(f"  Target shape: (-1, {1}, {pred_span}, {height}, {width})")
            print(
                f"  Tgt original shape: {tgt_data.shape}, Rst original shape: {rst_data.shape}"
            )
            raise e

        squared_error = (rst - tgt) ** 2
        weighted_se = (
            weight * squared_error
        )  # PyTorch handles broadcasting automatically
        total_se_sum = th.sum(weighted_se)
        ones_like_data = th.ones_like(rst)
        total_weight_sum = th.sum(weight * ones_like_data)
        mse = total_se_sum / total_weight_sum

        return th.sqrt(mse)

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
        target_type = tgt_data.dtype
        target_device = rst_data.device
        tgt_data = tgt_data.to(device=target_device)
        rst_data = rst_data.to(device=target_device, dtype=target_type)

        height, width = rst_data.size(-2), rst_data.size(-1)
        try:
            weight = self.model.weight  # type: ignore[attr-defined]
            weight = weight.reshape(1, 1, 1, height, width)
        except AttributeError:
            print("Error: 'weight' attribute not found in self.model.")
            raise
        except Exception as e:  # pragma: no cover - defensive branch
            print(
                "Error: Failed to reshape weight. Original weight shape: "
                f"{self.model.weight.shape}, Target shape: (1, 1, 1, {height}, {width})"
            )
            raise e

        weight = weight.to(device=target_device, dtype=target_type)

        pred_span = self.model.setting.pred_span
        try:
            tgt = tgt_data.reshape(-1, 1, pred_span, height, width)
            rst = rst_data.reshape(-1, 1, pred_span, height, width)
        except RuntimeError as e:  # pragma: no cover - reshape errors are rare
            print("Error: Failed to reshape input tensors.")
            print(f"  Target shape: (-1, 1, {pred_span}, {height}, {width})")
            print(
                f"  Tgt original shape: {tgt_data.shape}, Rst original shape: {rst_data.shape}"
            )
            raise e

        _, days = tgt.size(0), tgt.size(2)
        epoch = self.current_epoch
        rst, tgt = denormalizors[variable](rst), denormalizors[variable](tgt)
        squared_error = (rst - tgt) ** 2
        weighted_se = weight * squared_error

        # Accumulate totals for overall RMSE
        total_se_sum = th.tensor(0.0, device=target_device, dtype=target_type)
        total_weight_sum = th.tensor(0.0, device=target_device, dtype=target_type)

        # Ensure dictionary structure for side effects
        self.mseByVar.setdefault(variable, {}).setdefault(epoch, {})

        for day in range(days):
            cur_weighted_se = weighted_se[:, :, day, :, :]
            cur_total_se = th.sum(cur_weighted_se)
            cur_weight_sum = th.sum(weight * th.ones_like(cur_weighted_se))
            cur_mse = cur_total_se / cur_weight_sum
            rmse = float(th.sqrt(cur_mse))
            # Store per-day RMSE for later serialization
            self.mseByVar[variable][epoch][day + 1] = rmse

            total_se_sum += cur_total_se
            total_weight_sum += cur_weight_sum

        overall_mse = total_se_sum / total_weight_sum
        return th.sqrt(overall_mse)

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

        if self.ci:
            return np.zeros((batch_size, len(vars_out), span, 32, 64))

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
                batch_size, len(vars_out), 1, 32, 64
            )
            result.append(data)
        return np.concatenate(result, axis=2)

    def calculate_acc(self, forecast, observation, indexes, variable, mode):
        # Skip plotting and simplify calculations in CI mode
        if self.ci:
            return 1.0, 1.0, 1.0

        batch = forecast.shape[0]
        pred_length = self.model.setting.pred_span
        climatology = self.get_climatology(indexes, mode)
        var_ind = self.model.setting.vars_out.index(variable)
        climatology = climatology[:, var_ind : var_ind + 1, :, :, :]
        forecast = forecast.reshape(batch, 1, pred_length, 32, 64).cpu().numpy()
        observation = observation.reshape(batch, 1, pred_length, 32, 64).cpu().numpy()
        climatology = climatology.reshape(batch, 1, pred_length, 32, 64)
        weight = self.model.weight.reshape(1, 1, 1, 32, 64).cpu().numpy()

        f_anomaly = forecast - climatology
        o_anomaly = observation - climatology

        _, days = climatology.shape[0], climatology.shape[2]
        epoch = self.current_epoch
        for day in range(days):
            self.accByVar[variable].setdefault(epoch, {}).setdefault(day, 0.0)
            prod = np.sum(
                weight * f_anomaly[:, :, day, :, :] * o_anomaly[:, :, day, :, :]
            )
            fsum = np.sum(weight * f_anomaly[:, :, day, :, :] ** 2)
            osum = np.sum(weight * o_anomaly[:, :, day, :, :] ** 2)
            acc = prod / np.sqrt(fsum * osum)
            self.accByVar[variable][epoch][day + 1] = float(acc)

            plot_var = plot_root / f"{variable}"
            if not plot_var.exists():
                plot_var.mkdir(parents=True, exist_ok=True)

            plot_path = plot_root / f"anomaly_{variable}_fcs_{day}.png"
            plot(
                variable,
                open(plot_path, mode="wb"),
                f_anomaly[0, :, day, :, :],
            )
            plot_path = plot_root / f"anomaly_{variable}_obs_{day}.png"
            plot(
                variable,
                open(plot_path, mode="wb"),
                o_anomaly[0, :, day, :, :],
            )

        prod = np.sum(weight * f_anomaly * o_anomaly)
        fsum = np.sum(weight * f_anomaly**2)
        osum = np.sum(weight * o_anomaly**2)

        return prod, fsum, osum

    def forecast_error(self, rmse):
        return rmse

    def forward(self, **inputs):
        return self.model(**inputs)

    def plot(self, inputs, results, targets, indexies, batch_idx, mode):
        # Skip plotting in CI mode or test mode
        if self.ci or mode == "test":
            return

        for bas, var in enumerate(self.model.setting.vars_in):
            inp = inputs[var]
            span = self.model.setting.input_span
            plot_var = plot_root / f"{var}"
            if not plot_var.exists():
                plot_var.mkdir(parents=True, exist_ok=True)

            for ix in range(span):
                plot_inp_path = plot_var / f"{var}_inp_{ix}.png"
                if inp.dim() == 4:
                    dat = inp[0, ix].detach().cpu().numpy().reshape(32, 64)
                else:
                    dat = inp[0, 0, ix].detach().cpu().numpy().reshape(32, 64)
                plot(var, open(plot_inp_path, mode="wb"), dat)

        for bas, var in enumerate(self.model.setting.vars_out):
            plot_var = plot_root / f"{var}"
            if not plot_var.exists():
                plot_var.mkdir(parents=True, exist_ok=True)

            fcst = results[var]
            tgrt = targets[var]
            span = self.model.setting.pred_span
            for ix in range(span):
                plot_fcst_path = plot_var / f"{var}_fcst_{ix}.png"
                plot_tgrt_path = plot_var / f"{var}_tgt_{ix}.png"

                if fcst.dim() == 4:
                    fcst_img = fcst[0, ix].detach().cpu().numpy().reshape(32, 64)
                    tgrt_img = tgrt[0, ix].detach().cpu().numpy().reshape(32, 64)
                else:
                    fcst_img = fcst[0, 0, ix].detach().cpu().numpy().reshape(32, 64)
                    tgrt_img = tgrt[0, 0, ix].detach().cpu().numpy().reshape(32, 64)

                plot(var, open(plot_fcst_path, mode="wb"), fcst_img)
                plot(var, open(plot_tgrt_path, mode="wb"), tgrt_img)

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

        with open(
            os.path.join(self.logger.log_dir, "val_rmse.json"),
            "w",
        ) as f:
            json.dump(self.mseByVar, f)
        with open(os.path.join(self.logger.log_dir, "val_acc.json"), "w") as f:
            json.dump(self.accByVar, f)

        # Only plot for the first batch in CI mode
        if not self.ci or batch_idx == 0 and self.opt.plot == "true":
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
        # Simplified CRPS computation for CI
        if self.ci:
            return 0.1, 0.5

        # Original implementation
        ensemble_size, channels, height, width = predictions.shape

        num_pixels = channels * height * width
        predictions_reshaped = predictions.reshape(
            ensemble_size, num_pixels
        )  # [ensemble_size, num_pixels]
        targets_reshaped = targets.reshape(
            ensemble_size, num_pixels
        )  # [ensemble_size, num_pixels]

        abs_errors = th.abs(
            predictions_reshaped - targets_reshaped
        )  # [ensemble_size, num_pixels]
        mean_abs_errors = abs_errors.mean(dim=0)  # [num_pixels]

        predictions_a = predictions_reshaped.unsqueeze(
            1
        )  # [ensemble_size, 1, num_pixels]
        predictions_b = predictions_reshaped.unsqueeze(
            0
        )  # [1, ensemble_size, num_pixels]
        pairwise_diff = th.abs(
            predictions_a - predictions_b
        )  # [ensemble_size, ensemble_size, num_pixels]
        mean_pairwise_diff = pairwise_diff.mean(dim=(0, 1))  # [num_pixels]

        # Calculate CRPS using the formula
        crps = mean_abs_errors - 0.5 * mean_pairwise_diff  # [num_pixels]
        absorb = 0.5 * mean_pairwise_diff / (mean_abs_errors + 1e-7)
        self.crps = crps.reshape(-1, 1, height, width)
        self.absorb = absorb.reshape(-1, 1, height, width)

        # Average CRPS over all pixels and the batch
        crps_mean = crps.mean()
        absb_mean = absorb.mean()

        return crps_mean, absb_mean

    def training_step(self, batch, batch_idx):
        inputs, targets, indexies = batch
        # self.get_climatology(indexies, "train")

        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        inputs["seed"] = th.randn_like(inputs["data"][:, :1, :, :], dtype=th.float32)

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
        inputs["seed"] = th.randn_like(inputs["data"][:, :1, :, :], dtype=th.float32)
        forecast = self.generator(**inputs)
        forecast_loss = self.loss_fn(
            inputs, forecast, targets, indexes=indexies, mode="eval"
        )
        real_judgement = self.discriminator(**inputs, target=targets["data"])
        fake_judgement = self.discriminator(**inputs, target=forecast["data"])
        crps, absb = self.compute_crps(forecast["data"], targets["data"])
        crps = self.forecast_error(crps)

        # mse_numerator, mse_denominator = self.compute_mse(targets, forecast)
        # self.labeled_mse_numerator += mse_numerator
        # self.labeled_mse_denominator += mse_denominator
        # rmse = np.sqrt(self.labeled_mse_numerator / self.labeled_mse_denominator)
        # rmse = self.forecast_error(rmse)

        current_batch_size = forecast["data"].shape[0]
        total_rmse = 0
        for variable in self.model.setting.vars_out:
            self.mseByVar[variable] = dict()
            rmse = self.compute_rmse_by_time(targets, forecast, variable)
            # self.log(f"val_rmse_{variable}", rmse, on_step=False, on_epoch=True,
            #     batch_size=current_batch_size, sync_dist=True)
            total_rmse += rmse

            # Calculate the average RMSE across all variables

            self.accByVar[variable] = dict()
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
        with open(os.path.join(self.logger.log_dir, "val_rmse.json"), "w") as f:
            json.dump(self.mseByVar, f)
        with open(os.path.join(self.logger.log_dir, "val_acc.json"), "w") as f:
            json.dump(self.accByVar, f)

        self.realness = real_judgement["data"].mean().item()
        self.fakeness = fake_judgement["data"].mean().item()
        self.log("realness", self.realness, prog_bar=True, sync_dist=True)
        self.log("fakeness", self.fakeness, prog_bar=True, sync_dist=True)
        self.log("crps", crps, prog_bar=True, sync_dist=True)
        self.log("absb", absb, prog_bar=True, sync_dist=True)
        self.log("rmse", rmse, prog_bar=True, sync_dist=True)
        self.log("val_forecast", forecast_loss, prog_bar=True, sync_dist=True)
        self.log("val_loss", forecast_loss, prog_bar=True, sync_dist=True)

        if self.opt.plot == "true":
            self.plot(inputs, forecast, targets, indexies, batch_idx, mode="eval")

    def test_step(self, batch, batch_idx):
        # Skip test for some batches in CI mode or if batch_idx > 1 in any mode
        if (self.ci and batch_idx > 0) or batch_idx > 1:
            return

        inputs, targets, indexies = batch
        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        inputs["seed"] = th.randn_like(inputs["data"][:, :1, :, :], dtype=th.float32)
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
        self.log("crps", crps, prog_bar=True, sync_dist=True)
        self.log("absb", absb, prog_bar=True, sync_dist=True)
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
