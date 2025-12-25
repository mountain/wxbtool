import os
import lightning as ltn
import torch as th
import torch.optim as optim

from wxbtool.core.decorators import ci_short_circuit
from wxbtool.core.model import Model
from wxbtool.core.plotter import Plotter
from wxbtool.metrics.acc import ACC
from wxbtool.metrics.rmse import RMSE
from wxbtool.norms.meanstd import denormalizors


class LightningModel(ltn.LightningModule):
    def __init__(self, model, opt=None):
        super(LightningModel, self).__init__()
        self.model : Model = model
        self.opt = opt

        self.learning_rate = 1e-3
        if opt and hasattr(opt, "rate"):
            self.learning_rate = float(opt.rate)

        self.train_rmse = self.build_metrics(RMSE)
        self.val_rmse = self.build_metrics(RMSE)
        self.test_rmse = self.build_metrics(RMSE)
        self.train_acc = self.build_metrics(ACC)
        self.val_acc = self.build_metrics(ACC)
        self.test_acc = self.build_metrics(ACC)
        self.train_acc.build_indexers(self.model.setting.years_train)
        self.val_acc.build_indexers(self.model.setting.years_eval)
        self.test_acc.build_indexers(self.model.setting.years_test)

        climateology_accessors = {
            "train": self.train_acc.climatology_accessor,
            "eval": self.val_acc.climatology_accessor,
            "test": self.test_acc.climatology_accessor,
        }

        self.artifacts = {}
        self.plotter = Plotter(self, climateology_accessors)

        self.ci = True if opt and hasattr(opt, "test") and opt.test == "true" else False

    def is_rank0(self):
        if hasattr(self.trainer, "is_global_zero"):
            return self.trainer.is_global_zero
        return True

    def build_metrics(self, metric_class):
        weight = self.model.weight
        variables = ['data'] + self.model.setting.vars_out
        h, w = weight.size(-2), weight.size(-1)
        return metric_class(
            self.model.setting.pred_span,
            self.model.setting.step,
            self.model.setting.pred_shift + self.model.setting.input_span - 1,
            weight.view(1, 1, 1, h, w),
            variables,
            denormalizors
        )

    def configure_optimizers(self):
        if hasattr(self.model, "configure_optimizers"):
            return self.model.configure_optimizers()

        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, 29)
        return [optimizer], [scheduler]

    def loss_fn(self, input, result, target, indexes=None, mode="train"):
        loss = self.model.lossfun(input, result, target)
        return loss

    def forecast_error(self, rmse):
        return rmse

    def forward(self, **inputs):
        return self.model(**inputs)

    def plot(self, inputs, results, targets, indexes, mode):
        self.plotter.plot_date(inputs, self.model.setting.vars_in, self.model.setting.input_span, "inpt")
        self.plotter.plot_date(results, self.model.setting.vars_out, self.model.setting.pred_span, "fcst")
        self.plotter.plot_date(targets, self.model.setting.vars_out, self.model.setting.pred_span, "tgrt")
        if mode == "test":
            self.plotter.plot_map(inputs, targets, results, indexes, mode)

    @ci_short_circuit
    def training_step(self, batch, batch_idx):
        inputs, targets, indexes = batch

        if self.model.enable_da:
            key0 = list(inputs.keys())[0]
            self.model.update_da_status(batch=inputs[key0].size(0))

        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        results = self.forward(indexes=indexes, **inputs)

        loss = self.loss_fn(inputs, results, targets, indexes=indexes, mode="train")
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.train_rmse(results, targets)
        self.log("train_rmse", self.train_rmse, prog_bar=True, sync_dist=True)
        self.train_acc(results, targets, indexes,
                enable_da=self.model.enable_da,
                lng_shift=self.model.lng_shift,
                flip_status=self.model.flip_status)
        self.log("train_acc", self.train_acc, prog_bar=True, sync_dist=True)

        if self.is_rank0():
            self.train_rmse.dump(os.path.join(self.logger.log_dir, "train_rmse.json"))
            self.train_acc.dump(os.path.join(self.logger.log_dir, "train_acc.json"))

        return loss

    @ci_short_circuit
    def validation_step(self, batch, batch_idx):
        self.model.clear_da_status()
        inputs, targets, indexes = batch
        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        results = self.forward(indexes=indexes, **inputs)

        loss = self.loss_fn(inputs, results, targets, indexes=indexes, mode="eval")
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.val_rmse(results, targets)
        self.log("val_rmse", self.val_rmse, prog_bar=True, sync_dist=True)
        self.val_acc(results, targets, indexes)
        self.log("val_acc", self.val_acc, prog_bar=True, sync_dist=True)

        if self.is_rank0():
            self.val_rmse.dump(os.path.join(self.logger.log_dir, "val_rmse.json"))
            self.val_acc.dump(os.path.join(self.logger.log_dir, "val_acc.json"))
            if batch_idx % 10 == 0:
                self.plot(inputs, results, targets, indexes, mode="eval")

    @ci_short_circuit
    def test_step(self, batch, batch_idx):
        self.model.clear_da_status()
        inputs, targets, indexes = batch
        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        results = self.forward(indexes=indexes, **inputs)

        loss = self.loss_fn(inputs, results, targets, indexes=indexes, mode="test")
        self.log("test_loss", loss, sync_dist=True, prog_bar=True)
        self.test_rmse(results, targets)
        self.log("test_rmse", self.test_rmse, prog_bar=True, sync_dist=True)
        self.test_acc(results, targets, indexes)
        self.log("test_acc", self.test_acc, prog_bar=True, sync_dist=True)

        if self.is_rank0():
            self.test_rmse.dump(os.path.join(self.logger.log_dir, "test_rmse.json"))
            self.test_acc.dump(os.path.join(self.logger.log_dir, "test_acc.json"))
            self.plot(inputs, results, targets, indexes, mode="test")

    def on_fit_start(self):
        self.model.to(self.device)
        for n, b in self.named_buffers():
            if b.device.type == "cpu":
                print(f"[WARN] CPU buffer before train: {n}, shape={tuple(b.shape)}, dtype={b.dtype}")

    def on_train_epoch_end(self):
        if self.is_rank0():
            log_dir = self.logger.log_dir
            self.train_rmse.dump(os.path.join(log_dir, "train_rmse.json"))
            self.train_acc.dump(os.path.join(log_dir, "train_acc.json"))

        self.train_rmse.reset()
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        if self.is_rank0():
            log_dir = self.logger.log_dir
            self.val_rmse.dump(os.path.join(log_dir, "val_rmse.json"))
            self.val_acc.dump(os.path.join(log_dir, "val_acc.json"))

        self.val_rmse.reset()
        self.val_acc.reset()

    def on_test_epoch_end(self):
        if self.is_rank0():
            log_dir = self.logger.log_dir
            self.test_rmse.dump(os.path.join(log_dir, "test_rmse.json"))
            self.test_acc.dump(os.path.join(log_dir, "test_acc.json"))

        self.test_rmse.reset()
        self.test_acc.reset()
