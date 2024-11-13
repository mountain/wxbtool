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

        if opt and hasattr(opt, 'rate'):
            self.learning_rate = float(opt.rate)

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
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
        mname = self.model.name
        fname = f"trains/{mname}/best-{record}"
        os.makedirs(
            f"trains/{mname}", exist_ok=True
        )
        with open(fname, "bw") as f:
            th.save(checkpoint, f)
        for ix, ckpt in enumerate(sorted(glob.glob(f"trains/{mname}/best-*.ckpt"), reverse=False)):
            if ix > 5:
                os.unlink(ckpt)

        self.counter = 0
        self.labeled_loss = 0
        self.labeled_rmse = 0

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
            shuffle=True,
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
            shuffle=True,
        )


class GANModel(LightningModel):
    def __init__(self, generator, discriminator, opt=None):
        super(GANModel, self).__init__(generator, opt=opt)
        self.generator = generator
        self.discriminator = discriminator
        self.learning_rate = 1e-4  # Adjusted for GANs
        self.automatic_optimization = False

        if opt and hasattr(opt, 'rate'):
            learning_rate = float(opt.rate)
            ratio = float(opt.ratio)
            self.generator.learning_rate = learning_rate
            self.discriminator.learning_rate = learning_rate / ratio

    def configure_optimizers(self):
        # Separate optimizers for generator and discriminator
        g_optimizer = th.optim.Adam(self.generator.parameters(), lr=self.generator.learning_rate)
        d_optimizer = th.optim.Adam(self.discriminator.parameters(), lr=self.discriminator.learning_rate)
        return [g_optimizer, d_optimizer]

    def generator_loss(self, fake_judgement):
        # Loss for generator (we want the discriminator to predict all generated images as real)
        return th.nn.functional.binary_cross_entropy_with_logits(fake_judgement["data"], th.ones_like(fake_judgement["data"], dtype=th.float32))

    def discriminator_loss(self, real_judgement, fake_judgement):
        # Loss for discriminator (real images should be classified as real, fake images as fake)
        real_loss = th.nn.functional.binary_cross_entropy_with_logits(real_judgement["data"], th.ones_like(real_judgement["data"], dtype=th.float32))
        fake_loss = th.nn.functional.binary_cross_entropy_with_logits(fake_judgement["data"], th.zeros_like(fake_judgement["data"], dtype=th.float32))
        return (real_loss + fake_loss) / 2

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs, _ = self.model.get_inputs(**inputs)
        targets, _ = self.model.get_targets(**targets)
        inputs['noise'] = th.randn_like(inputs['data'][:, :1, :, :], dtype=th.float32)

        g_optimizer, d_optimizer = self.optimizers()

        self.toggle_optimizer(g_optimizer)
        forecast = self.generator(**inputs)
        judgement = self.discriminator(**inputs, target=forecast["data"])
        forecast_loss = self.loss_fn(inputs, forecast, targets)
        generate_loss = self.generator_loss(judgement)
        total_loss = forecast_loss + generate_loss
        self.log("total", total_loss, prog_bar=True)
        self.log("forecast", forecast_loss, prog_bar=True)
        self.manual_backward(total_loss)
        g_optimizer.step()
        g_optimizer.zero_grad()
        self.untoggle_optimizer(g_optimizer)

        self.toggle_optimizer(d_optimizer)
        forecast = self.generator(**inputs)
        forecast["data"].detach()  # Detach to avoid generator gradient updates
        real_judgement = self.discriminator(**inputs, target=targets['data'])
        fake_judgement = self.discriminator(**inputs, target=forecast["data"])
        judgement_loss = self.discriminator_loss(real_judgement, fake_judgement)
        self.log("judgement", judgement_loss, prog_bar=True)
        self.manual_backward(judgement_loss)
        d_optimizer.step()
        d_optimizer.zero_grad()
        self.untoggle_optimizer(d_optimizer)

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs, _ = self.model.get_inputs(**inputs)
        targets, _ = self.model.get_targets(**targets)
        inputs['noise'] = th.randn_like(inputs['data'][:, :1, :, :], dtype=th.float32)
        forecast = self.generator(**inputs)
        judgement = self.discriminator(**inputs, target=forecast["data"])
        forecast_loss = self.loss_fn(inputs, forecast, targets)
        realness = judgement["data"].mean().item()
        self.log("realness", realness, prog_bar=True)
        self.log("val_forecast", forecast_loss, prog_bar=True)
        self.log("val_loss", forecast_loss, prog_bar=True)

        rmse = self.compute_rmse(targets, forecast)
        batch_len = inputs['data'].shape[0]
        self.labeled_loss += forecast_loss.item() * batch_len
        self.labeled_rmse += rmse * batch_len
        self.counter += batch_len

        self.plot(inputs, forecast, targets)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs, _ = self.model.get_inputs(**inputs)
        targets, _ = self.model.get_targets(**targets)
        inputs['noise'] = th.randn_like(inputs['data'][:, :1, :, :], dtype=th.float32)
        forecast = self.generator(**inputs)
        judgement = self.discriminator(**inputs, target=forecast["data"])
        forecast_loss = self.loss_fn(inputs, forecast, targets)
        realness = judgement["data"].mean().item()
        self.log("realness", realness, on_step=False, on_epoch=True, prog_bar=True)
        self.log("forecast", forecast_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.plot(inputs, forecast, targets)
