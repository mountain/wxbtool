import json
import os
import lightning as ltn
import numpy as np
import torch as th
import torch.optim as optim

from collections import defaultdict
from wxbtool.data.climatology import ClimatologyAccessor
from wxbtool.framework.metrics import (
    rmse_by_time as metrics_rmse_by_time,
    rmse_weighted as metrics_rmse_weighted,
    acc_anomaly_by_time,
)
from wxbtool.norms.meanstd import denormalizors
from wxbtool.util.plotter import plot_image


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

        self.artifacts = {}

    def is_rank0(self):
        if hasattr(self.trainer, "is_global_zero"):
            return self.trainer.is_global_zero
        return True

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

    def compute_rmse(self, targets, results, indexes):
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
        return avg_rmse

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
            if not hasattr(self, "artifacts") or self.artifacts is None:
                self.artifacts = {}
            for day in range(pred_length):
                tag_f = f"anomaly_{variable}_fcs_{day}"
                tag_o = f"anomaly_{variable}_obs_{day}"
                # Use first sample in batch for visualization
                self.artifacts[tag_f] = {"var": variable, "data": f_anomaly[0, 0, day]}
                self.artifacts[tag_o] = {"var": variable, "data": o_anomaly[0, 0, day]}

        return prod_sum, fsum_sum, osum_sum

    def forecast_error(self, rmse):
        return rmse

    def forward(self, **inputs):
        return self.model(**inputs)

    def plot_date(self, data, variables, span, key):
        for var in variables:
            item = data[var]
            for ix in range(span):
                if item.dim() == 4:
                    height, width = item.size(-2), item.size(-1)
                    dat = item[0, ix].detach().cpu().numpy().reshape(height, width)
                else:
                    height, width = item.size(-2), item.size(-1)
                    dat = item[0, 0, ix].detach().cpu().numpy().reshape(height, width)
                self.artifacts[f"{var}_{ix:02d}_{key}"] = {"var": var, "data": dat}

    def plot_image(self, inputs, targets, results, indexies, batch_idx, mode):
        if inputs[self.model.setting.vars_out[0]].dim() == 4:
            zero_slice = 0, 0
        else:
            zero_slice = 0, 0, 0

        for bas, var in enumerate(self.model.setting.vars_out):
            input_data = inputs[var][zero_slice].detach().cpu().numpy()
            truth = targets[var][zero_slice].detach().cpu().numpy()
            forecast = results[var][zero_slice].detach().cpu().numpy()
            input_data = denormalizors[var](input_data)
            forecast = denormalizors[var](forecast)
            truth = denormalizors[var](truth)
            plot_image(
                var,
                input_data=input_data,
                truth=truth,
                forecast=forecast,
                title=var,
                year=self.climatology_accessors[mode].yr_indexer[indexies[0]],
                doy=self.climatology_accessors[mode].doy_indexer[indexies[0]],
                save_path="%s_%02d.png" % (var, batch_idx),
            )

    def plot(self, inputs, results, targets, indexies, batch_idx, mode):
        if self.ci or mode == "test":
            return
        if getattr(self.opt, "plot", "false") != "true":
            return

        self.plot_date(inputs, self.model.setting.vars_in, self.model.setting.input_span, "inpt")
        self.plot_date(results, self.model.setting.vars_out, self.model.setting.pred_span, "fcst")
        self.plot_date(targets, self.model.setting.vars_out, self.model.setting.pred_span, "tgrt")

        if mode == "test":
            self.plot_image(inputs, targets, results, indexies, batch_idx, mode)

    def training_step(self, batch, batch_idx):
        inputs, targets, indexes = batch

        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        results = self.forward(indexes=indexes, **inputs)

        loss = self.loss_fn(inputs, results, targets, indexes=indexes, mode="train")
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.ci and batch_idx > 0:
            return

        inputs, targets, indexes = batch
        self.get_climatology(indexes, "eval")

        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        results = self.forward(indexes=indexes, **inputs)

        loss = self.loss_fn(inputs, results, targets, indexes=indexes, mode="eval")
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        rmse = self.compute_rmse(targets, results, indexes)
        self.log("val_rmse", rmse, prog_bar=True, sync_dist=True)

        if self.is_rank0():
            with open(os.path.join(self.logger.log_dir, "val_rmse.json"), "w") as f:
                json.dump(self.mseByVar, f)
            with open(os.path.join(self.logger.log_dir, "val_acc.json"), "w") as f:
                json.dump(self.accByVar, f)

            self.plot(inputs, results, targets, indexes, batch_idx, mode="eval")

    def test_step(self, batch, batch_idx):
        if self.ci and batch_idx > 0:
            return

        inputs, targets, indexes = batch
        self.get_climatology(indexes, "test")

        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        results = self.forward(indexies=indexes, **inputs)

        loss = self.loss_fn(inputs, results, targets, indexes=indexes, mode="test")
        self.log("test_loss", loss, sync_dist=True, prog_bar=True)
        rmse = self.compute_rmse(targets, results, indexes)
        self.log("test_rmse", rmse, prog_bar=True, sync_dist=True)

        if self.is_rank0():
            with open(os.path.join(self.logger.log_dir, "test_rmse.json"), "w") as f:
                json.dump(self.mseByVar, f)
            with open(os.path.join(self.logger.log_dir, "test_acc.json"), "w") as f:
                json.dump(self.accByVar, f)

            self.plot(inputs, results, targets, indexes, batch_idx, mode="test")

    def on_save_checkpoint(self, checkpoint):
        self.labeled_acc_prod_term = {var: 0 for var in self.model.setting.vars_out}
        self.labeled_acc_fsum_term = {var: 0 for var in self.model.setting.vars_out}
        self.labeled_acc_osum_term = {var: 0 for var in self.model.setting.vars_out}

        return checkpoint
