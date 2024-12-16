import os
import numpy as np
import torch as th
import lightning as ltn

from torch.utils.data import DataLoader
from wxbtool.data.climatology import ClimatologyAccessor
from wxbtool.data.dataset import ensemble_loader
from wxbtool.util.plotter import plot, plot_image
from wxbtool.norms.meanstd import denormalizors
from torch.optim import Optimizer
from typing import Iterable, Optional, Callable, List, Dict


class LightningModel(ltn.LightningModule):
    def __init__(self, model, opt=None):
        super(LightningModel, self).__init__()
        self.model = model
        self.learning_rate = 1e-3

        self.opt = opt

        if opt and hasattr(opt, 'rate'):
            self.learning_rate = float(opt.rate)

        self.climatology_accessors = {}
        self.data_home = os.environ.get("WXBHOME", "/data/climatology")

        self.counter = 0
        self.labeled_loss = 0
        self.labeled_mse_numerator = 0
        self.labeled_mse_denominator = 0
        self.labeled_acc_prod_term = 0
        self.labeled_acc_fsum_term = 0
        self.labeled_acc_osum_term = 0

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, 53)
        return [optimizer], [scheduler]

    def loss_fn(self, input, result, target):
        return self.model.lossfun(input, result, target)

    def compute_mse(self, targets, results):
        tgt = targets['data']
        rst = results['data']
        weight = self.model.weight.cpu().numpy()
        tgt = (
            tgt.detach().cpu().numpy().reshape(-1, self.model.setting.pred_span, 32, 64)
        )
        rst = (
            rst.detach().cpu().numpy().reshape(-1, self.model.setting.pred_span, 32, 64)
        )
        weight = (
            weight.reshape(1, 1, 32, 64)
        )
        se_sum, weight_sum = np.sum(weight * (rst - tgt) ** 2, axis=(2, 3)), np.sum(weight, axis=(2, 3))
        return se_sum.sum(), (weight_sum * np.ones_like(se_sum)).sum()

    def get_climatology_accessor(self, mode):
        if mode not in self.climatology_accessors:
            self.climatology_accessors[mode] = ClimatologyAccessor(home=f"{self.data_home}/climatology")
            if mode == 'train':
                years = tuple(self.model.setting.years_train)
            if mode == 'eval':
                years = tuple(self.model.setting.years_eval)
            if mode == 'test':
                years = tuple(self.model.setting.years_test)
            self.climatology_accessors[mode].build_indexers(years)

        return self.climatology_accessors[mode]

    def get_climatology(self, vars, indexies, mode):
        accessor = self.get_climatology_accessor(mode)
        return accessor.get_climatology(vars, indexies)

    def calculate_acc(self, forecast, observation, indexies, mode):
        vars_out = self.model.vars_out
        climatology = self.get_climatology(vars_out, indexies, mode)

        indexies = list(indexies.cpu().numpy())
        weight = self.model.weight.cpu().numpy()
        forecast = forecast.cpu().numpy()
        observation = observation.cpu().numpy()

        weight = weight.reshape(1, 1, 32, 64)
        climatology = climatology.reshape(-1, 1, 32, 64)
        forecast = forecast.reshape(-1, self.model.setting.pred_span, 32, 64)
        observation = observation.reshape(-1, self.model.setting.pred_span, 32, 64)

        f_anomaly = forecast - climatology
        o_anomaly = observation - climatology
        plot(vars_out[0], open("anomaly_%s_fcs.png" % vars_out[0], mode="wb"), f_anomaly[0])
        plot(vars_out[0], open("anomaly_%s_obs.png" % vars_out[0], mode="wb"), o_anomaly[0])

        # f_anomaly_bar = np.sum(weight * f_anomaly) / np.sum(weight)
        # o_anomaly_bar = np.sum(weight * o_anomaly) / np.sum(weight)
        # f_anomaly = f_anomaly - f_anomaly_bar
        # o_anomaly = o_anomaly - o_anomaly_bar

        prod = np.sum(weight * f_anomaly * o_anomaly, axis=(2, 3))
        fsum = np.sum(weight * f_anomaly**2, axis=(2, 3))
        osum = np.sum(weight * o_anomaly**2, axis=(2, 3))

        return prod.sum(), fsum.sum(), osum.sum()

 
    def forecast_error(self, rmse):
        return rmse

    def forward(self, **inputs):
        return self.model(**inputs)

    def plot(self, inputs, results, targets, indexies, batch_idx, mode):
        for bas, var in enumerate(self.model.setting.vars_in):
            for ix in range(self.model.setting.input_span):
                dat = inputs[var][0, ix].detach().cpu().numpy().reshape(32, 64)
                plot(var, open("%s_inp_%d.png" % (var, ix), mode="wb"), dat)

        for bas, var in enumerate(self.model.vars_out):
            for ix in range(self.model.setting.pred_span):
                fcst = results[var][0, ix].detach().cpu().numpy().reshape(32, 64)
                tgrt = targets[var][0, ix].detach().cpu().numpy().reshape(32, 64)
                plot(var, open("%s_fcs_%d.png" % (var, ix), mode="wb"), fcst)
                plot(var, open("%s_tgt_%d.png" % (var, ix), mode="wb"), tgrt)
        
        input_data = inputs['data'][0, 0].detach().cpu().numpy()
        truth = targets['data'][0, 0].detach().cpu().numpy()
        forecast = results['data'][0, 0].detach().cpu().numpy()
        input_data = denormalizors[var](input_data)
        forecast = denormalizors[var](forecast)
        truth = denormalizors[var](truth)
        plot_image(
            self.model.vars_out[0],
            input_data=input_data,
            truth=truth,
            forecast=forecast,
            title="%s" % var,
            year=self.climatology_accessors[mode].yr_indexer[indexies[0]],
            doy=self.climatology_accessors[mode].doy_indexer[indexies[0]],
            save_path="%s_%02d.png" % (var, batch_idx)
        )

    def training_step(self, batch, batch_idx):
        inputs, targets, indexies = batch
        self.get_climatology(self.model.setting.vars, indexies, 'train')

        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        results = self.forward(**inputs)

        loss = self.loss_fn(inputs, results, targets)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, indexies = batch
        batch_len = inputs[self.model.setting.vars[0]].shape[0]

        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        results = self.forward(**inputs)
        loss = self.loss_fn(inputs, results, targets)

        mse_numerator, mse_denominator = self.compute_mse(targets, results)
        self.labeled_mse_numerator += mse_numerator
        self.labeled_mse_denominator += mse_denominator
        rmse = np.sqrt(self.labeled_mse_numerator / self.labeled_mse_denominator)

        prod, fsum, osum = self.calculate_acc(results["data"], targets["data"], indexies=indexies, mode='eval')
        self.labeled_acc_prod_term += prod
        self.labeled_acc_fsum_term += fsum
        self.labeled_acc_osum_term += osum
        acc = self.labeled_acc_prod_term / np.sqrt(self.labeled_acc_fsum_term * self.labeled_acc_osum_term)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_rmse", rmse, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        self.plot(inputs, results, targets, indexies, batch_idx, mode='eval')

        self.labeled_loss += loss.item() * batch_len
        self.counter += batch_len

    def test_step(self, batch, batch_idx):
        inputs, targets, indexies = batch
        batch_len = inputs[self.model.setting.vars[0]].shape[0]

        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        results = self.forward(**inputs)
        loss = self.loss_fn(inputs, results, targets)

        mse_numerator, mse_denominator = self.compute_mse(targets, results)
        self.labeled_mse_numerator += mse_numerator
        self.labeled_mse_denominator += mse_denominator
        rmse = np.sqrt(self.labeled_mse_numerator / self.labeled_mse_denominator)
        
        prod, fsum, osum = self.calculate_acc(results["data"], targets["data"], indexies=indexies, mode='test')
        self.labeled_acc_prod_term += prod
        self.labeled_acc_fsum_term += fsum
        self.labeled_acc_osum_term += osum
        acc = self.labeled_acc_prod_term / np.sqrt(self.labeled_acc_fsum_term * self.labeled_acc_osum_term)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_rmse", rmse, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        self.plot(inputs, results, targets, indexies, batch_idx, mode='test')
        self.counter += batch_len

    def on_save_checkpoint(self, checkpoint):
        import glob
        import os

        rmse = np.sqrt(self.labeled_mse_numerator / self.labeled_mse_denominator)
        error = self.forecast_error(rmse)
        loss = self.labeled_loss / self.counter
        if self.trainer.global_rank == 0:
            print()
            record = "%2.5f-%03d-%1.5f.ckpt" % (error, checkpoint["epoch"], loss)
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
        self.labeled_mse_numerator = 0
        self.labeled_mse_denominator = 0
        self.labeled_acc_prod_term = 0
        self.labeled_acc_fsum_term = 0
        self.labeled_acc_osum_term = 0

        return checkpoint

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

        if opt and hasattr(opt, 'rate'):
            learning_rate = float(opt.rate)
            ratio = float(opt.ratio)
            self.generator.learning_rate = learning_rate
            self.discriminator.learning_rate = learning_rate / ratio

        if opt and hasattr(opt, 'alpha'):
            self.alpha = float(opt.alpha)

    def configure_optimizers(self):
        # Separate optimizers for generator and discriminator
        g_optimizer = th.optim.Adam(self.generator.parameters(), lr=self.generator.learning_rate)
        d_optimizer = th.optim.Adam(self.discriminator.parameters(), lr=self.discriminator.learning_rate)
        g_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, 37)
        d_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(d_optimizer, 37)
        return [g_optimizer, d_optimizer], [g_scheduler, d_scheduler]

    def generator_loss(self, fake_judgement):
        # Loss for generator (we want the discriminator to predict all generated images as real)
        return th.nn.functional.binary_cross_entropy(
            fake_judgement["data"], th.ones_like(fake_judgement["data"], dtype=th.float32))

    def discriminator_loss(self, real_judgement, fake_judgement):
        # Loss for discriminator (real images should be classified as real, fake images as fake)
        real_loss = th.nn.functional.binary_cross_entropy(
            real_judgement["data"], th.ones_like(real_judgement["data"], dtype=th.float32))
        fake_loss = th.nn.functional.binary_cross_entropy(
            fake_judgement["data"], th.zeros_like(fake_judgement["data"], dtype=th.float32))
        return (real_loss + fake_loss) / 2

    def forecast_error(self, rmse):
        return self.generator.forecast_error(rmse)

    def compute_crps(self, predictions, targets):
        ensemble_size, channels, height, width = predictions.shape

        num_pixels = channels * height * width
        predictions_reshaped = predictions.reshape(ensemble_size, num_pixels)  # [ensemble_size, num_pixels]
        targets_reshaped = targets.reshape(ensemble_size, num_pixels)  # [ensemble_size, num_pixels]

        abs_errors = th.abs(predictions_reshaped - targets_reshaped)  # [ensemble_size, num_pixels]
        mean_abs_errors = abs_errors.mean(dim=0)  # [num_pixels]

        predictions_a = predictions_reshaped.unsqueeze(1)  # [ensemble_size, 1, num_pixels]
        predictions_b = predictions_reshaped.unsqueeze(0)  # [1, ensemble_size, num_pixels]
        pairwise_diff = th.abs(predictions_a - predictions_b)  # [ensemble_size, ensemble_size, num_pixels]
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
        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        inputs['seed'] = th.randn_like(inputs['data'][:, :1, :, :], dtype=th.float32)

        g_optimizer, d_optimizer = self.optimizers()

        self.toggle_optimizer(g_optimizer)
        forecast = self.generator(**inputs)
        real_judgement = self.discriminator(**inputs, target=targets['data'])
        fake_judgement = self.discriminator(**inputs, target=forecast["data"])
        forecast_loss = self.loss_fn(inputs, forecast, targets)
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
        forecast["data"] = forecast["data"].detach()  # Detach to avoid generator gradient updates
        real_judgement = self.discriminator(**inputs, target=targets['data'])
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

        if batch_idx % 10 == 0:
            self.plot(inputs, forecast, targets, indexies, mode='train')

    def validation_step(self, batch, batch_idx):
        inputs, targets, indexies = batch
        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        inputs['seed'] = th.randn_like(inputs['data'][:, :1, :, :], dtype=th.float32)
        forecast = self.generator(**inputs)
        forecast_loss = self.loss_fn(inputs, forecast, targets)
        real_judgement = self.discriminator(**inputs, target=targets['data'])
        fake_judgement = self.discriminator(**inputs, target=forecast["data"])
        crps, absb = self.compute_crps(forecast["data"], targets["data"])

        mse_numerator, mse_denominator = self.compute_mse(targets, forecast)
        self.labeled_mse_numerator += mse_numerator
        self.labeled_mse_denominator += mse_denominator
        rmse = np.sqrt(self.labeled_mse_numerator / self.labeled_mse_denominator)

        prod, fsum, osum = self.calculate_acc(forecast["data"], targets["data"], indexies=indexies, mode='eval')
        self.labeled_acc_prod_term += prod
        self.labeled_acc_fsum_term += fsum
        self.labeled_acc_osum_term += osum
        acc = self.labeled_acc_prod_term / np.sqrt(self.labeled_acc_fsum_term * self.labeled_acc_osum_term)
        
        self.realness = real_judgement["data"].mean().item()
        self.fakeness = fake_judgement["data"].mean().item()
        self.log("realness", self.realness, prog_bar=True, sync_dist=True)
        self.log("fakeness", self.fakeness, prog_bar=True, sync_dist=True)
        self.log("crps", crps, prog_bar=True, sync_dist=True)
        self.log("absb", absb, prog_bar=True, sync_dist=True)
        self.log("acc", acc, prog_bar=True, sync_dist=True)
        self.log("rmse", rmse, prog_bar=True, sync_dist=True)
        self.log("val_forecast", forecast_loss, prog_bar=True, sync_dist=True)
        self.log("val_loss", forecast_loss, prog_bar=True, sync_dist=True)

        batch_len = inputs['data'].shape[0]
        self.labeled_loss += forecast_loss.item() * batch_len
        self.counter += batch_len

        self.plot(inputs, forecast, targets, indexies, mode='eval')

    def test_step(self, batch, batch_idx):
        inputs, targets, indexies = batch
        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        inputs['seed'] = th.randn_like(inputs['data'][:, :1, :, :], dtype=th.float32)
        forecast = self.generator(**inputs)
        forecast_loss = self.loss_fn(inputs, forecast, targets)
        real_judgement = self.discriminator(**inputs, target=targets['data'])
        fake_judgement = self.discriminator(**inputs, target=forecast["data"])
        self.realness = real_judgement["data"].mean().item()
        self.fakeness = fake_judgement["data"].mean().item()
        crps, absb = self.compute_crps(forecast["data"], targets["data"])

        mse_numerator, mse_denominator = self.compute_mse(targets, forecast)
        self.labeled_mse_numerator += mse_numerator
        self.labeled_mse_denominator += mse_denominator
        rmse = np.sqrt(self.labeled_mse_numerator / self.labeled_mse_denominator)

        prod, fsum, osum = self.calculate_acc(forecast["data"], targets["data"], indexies, mode='test')
        self.labeled_acc_prod_term += prod
        self.labeled_acc_fsum_term += fsum
        self.labeled_acc_osum_term += osum
        acc = self.labeled_acc_prod_term / np.sqrt(self.labeled_acc_fsum_term * self.labeled_acc_osum_term)

        self.log("realness", self.realness, prog_bar=True, sync_dist=True)
        self.log("fakeness", self.fakeness, prog_bar=True, sync_dist=True)
        self.log("forecast", forecast_loss, prog_bar=True, sync_dist=True)
        self.log("crps", crps, prog_bar=True, sync_dist=True)
        self.log("absb", absb, prog_bar=True, sync_dist=True)
        self.log("acc", acc, prog_bar=True, sync_dist=True)
        self.log("rmse", rmse, prog_bar=True, sync_dist=True)
        self.plot(inputs, forecast, targets, indexies, mode='test')

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
        return ensemble_loader(
            self.model.dataset_eval,
            self.opt.batch_size,
            False,
        )

    def test_dataloader(self):
        if self.model.dataset_test is None:
            if self.opt.data != "":
                self.model.load_dataset("train", "client", url=self.opt.data)
            else:
                self.model.load_dataset("train", "server")
        return ensemble_loader(
            self.model.dataset_test,
            self.opt.batch_size,
            False,
        )
