import numpy as np
import torch as th
import lightning as ltn

from torch.utils.data import DataLoader
from wxbtool.util.plotter import plot


class LightningModel(ltn.LightningModule):
    def __init__(self, model, opt=None):
        super(LightningModel, self).__init__()
        self.model = model
        self.learning_rate = 1e-3

        self.counter = 0
        self.labeled_loss = 0
        self.labeled_rmse = 0

        self.opt = opt

    def configure_optimizers(self):
        optimizer = th.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, 53)
        return [optimizer], [scheduler]

    def loss_fn(self, input, result, target):
        return self.model.lossfun(input, result, target)

    def compute_rmse(self, targets, results):
        _, tgt = self.model.get_targets(**targets)
        _, rst = self.model.get_results(**results)
        tgt = (
            tgt.detach().cpu().numpy().reshape(-1, self.model.setting.pred_span, 32, 64)
        )
        rst = (
            rst.detach().cpu().numpy().reshape(-1, self.model.setting.pred_span, 32, 64)
        )
        rmse = np.sqrt(np.mean(self.model.weight.cpu().numpy() * (rst - tgt) ** 2))
        return rmse

    def forward(self, **inputs):
        return self.model(**inputs)

    def plot(self, inputs, results, targets):
        vars_in, _ = self.model.get_inputs(**inputs)
        for bas, var in enumerate(self.model.setting.vars_in):
            for ix in range(self.model.setting.input_span):
                img = vars_in[var][0, ix].detach().cpu().numpy().reshape(32, 64)
                plot(var, open("%s_inp_%d.png" % (var, ix), mode="wb"), img)

        vars_fc, _ = self.model.get_results(**results)
        vars_tg, _ = self.model.get_targets(**targets)
        for bas, var in enumerate(self.model.setting.vars_out):
            for ix in range(self.model.setting.pred_span):
                fcst = vars_fc[var][0, ix].detach().cpu().numpy().reshape(32, 64)
                tgrt = vars_tg[var][0, ix].detach().cpu().numpy().reshape(32, 64)
                plot(var, open("%s_fcs_%d.png" % (var, ix), mode="wb"), fcst)
                plot(var, open("%s_tgt_%d.png" % (var, ix), mode="wb"), tgrt)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = {v: inputs[v].float() for v in self.model.setting.vars}
        targets = {v: targets[v].float() for v in self.model.setting.vars}
        results = self.forward(**inputs)

        loss = self.loss_fn(inputs, results, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = {v: inputs[v].float() for v in self.model.setting.vars}
        targets = {v: targets[v].float() for v in self.model.setting.vars}
        results = self.forward(**inputs)
        loss = self.loss_fn(inputs, results, targets)
        rmse = self.compute_rmse(targets, results)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True)

        batch_len = inputs[self.model.setting.vars[0]].shape[0]
        self.labeled_loss += loss.item() * batch_len
        self.labeled_rmse += rmse * batch_len
        self.counter += batch_len

        self.plot(inputs, results, targets)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = {v: inputs[v].float() for v in self.model.setting.vars}
        targets = {v: targets[v].float() for v in self.model.setting.vars}
        results = self.forward(**inputs)
        loss = self.loss_fn(inputs, results, targets)
        rmse = self.compute_rmse(targets, results)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True)

        self.plot(inputs, results, targets)

    def on_save_checkpoint(self, checkpoint) -> None:
        import glob
        import os

        rmse = self.labeled_rmse / self.counter
        loss = self.labeled_loss / self.counter
        record = "%2.5f-%03d-%1.5f.ckpt" % (rmse, checkpoint["epoch"], loss)
        fname = "best-%s" % record
        with open(fname, "bw") as f:
            th.save(checkpoint, f)
        for ix, ckpt in enumerate(sorted(glob.glob("best-*.ckpt"), reverse=True)):
            if ix > 5:
                os.unlink(ckpt)

        self.counter = 0
        self.labeled_loss = 0
        self.labeled_correct = 0

        print()

    def train_dataloader(self):
        if self.model.dataset_train is None:
            if self.opt.data != "":
                self.model.load_dataset("train", "client", url=self.opt.data)
            else:
                self.model.load_dataset("train", "server")
        return DataLoader(
            self.model.dataset_train,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.n_cpu,
            shuffle=True,
        )

    def val_dataloader(self):
        if self.model.dataset_eval is None:
            if self.opt.data != "":
                self.model.load_dataset("train", "client", url=self.opt.data)
            else:
                self.model.load_dataset("train", "server")
        return DataLoader(
            self.model.dataset_eval,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.n_cpu,
            shuffle=False,
        )

    def test_dataloader(self):
        if self.model.dataset_test is None:
            if self.opt.data != "":
                self.model.load_dataset("train", "client", url=self.opt.data)
            else:
                self.model.load_dataset("train", "server")
        return DataLoader(
            self.model.dataset_test,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.n_cpu,
            shuffle=False,
        )
