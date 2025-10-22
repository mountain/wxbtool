import json
import os
import numpy as np
import torch as th

from torch.utils.data import DataLoader
from wxbtool.data.dataset import ensemble_loader
from wxbtool.lightning.base import LightningModel


class GANModel(LightningModel):
    def __init__(self, generator, discriminator, opt=None):
        super(GANModel, self).__init__(generator, opt=opt)
        self.generator = generator
        self.discriminator = discriminator
        self.automatic_optimization = False
        self.crps = None
        self.alpha = 0.5

        self.learning_rate = 1e-4
        self.generator.learning_rate = 1e-4
        self.discriminator.learning_rate = 1e-4

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
            self.generator.parameters(), lr=self.generator.learning_rate, weight_decay=0.0, betas=(0.0, 0.9),
        )
        d_optimizer = th.optim.Adam(
            self.discriminator.parameters(), lr=self.discriminator.learning_rate, weight_decay=0.0, betas=(0.0, 0.9),
        )
        return [g_optimizer, d_optimizer], []

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
        if predictions.dim() == 4:
            predictions5 = predictions.unsqueeze(1)
            targets5 = targets.unsqueeze(1)
        elif predictions.dim() == 5:
            predictions5 = predictions
            targets5 = targets
        else:
            raise ValueError(f"Unsupported predictions dim: {predictions.dim()}")

        B, C, T, H, W = predictions5.shape

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

        self.crps = crps
        self.absorb = absb

        return crps, absb

    def training_step(self, batch, batch_idx):
        inputs, targets, indexies = batch
        g_optimizer, d_optimizer = self.optimizers()

        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)

        # seed
        data = inputs["data"]
        if data.dim() == 5:
            seed = th.randn_like(data[:, :, :1, :, :], dtype=th.float32)
        elif data.dim() == 4:
            seed = th.randn_like(data[:, :1, :, :], dtype=th.float32)
        else:
            seed = th.randn_like(data, dtype=th.float32)
        inputs["seed"] = seed

        # --- Generator Update ---
        self.toggle_optimizer(g_optimizer)

        forecast = self.generator(**inputs)

        fake_judgement_for_g = self.discriminator(**inputs, target=forecast["data"])
        generate_loss = self.generator_loss(fake_judgement_for_g)
        forecast_loss = self.loss_fn(inputs, forecast, targets, indexes=indexies, mode="train")
        total_g_loss = self.alpha * forecast_loss + (1 - self.alpha) * generate_loss

        self.manual_backward(total_g_loss)
        g_optimizer.step()
        g_optimizer.zero_grad()
        self.untoggle_optimizer(g_optimizer)

        self.log("total", total_g_loss, prog_bar=True)
        self.log("forecast", forecast_loss, prog_bar=True)

        # --- Discriminator Update ---
        self.toggle_optimizer(d_optimizer)

        fake_data_detached = forecast["data"].detach()

        real_judgement = self.discriminator(**inputs, target=targets["data"])
        fake_judgement_for_d = self.discriminator(**inputs, target=fake_data_detached)

        judgement_loss = self.discriminator_loss(real_judgement, fake_judgement_for_d)

        self.manual_backward(judgement_loss)
        d_optimizer.step()
        d_optimizer.zero_grad()
        self.untoggle_optimizer(d_optimizer)

        realness = real_judgement["data"].mean()
        fakeness = fake_judgement_for_d["data"].mean()
        self.log("realness", realness, prog_bar=True, sync_dist=True)
        self.log("fakeness", fakeness, prog_bar=True, sync_dist=True)
        self.log("judgement", judgement_loss, prog_bar=True, sync_dist=True)

        if self.opt.plot == "true" and batch_idx % 10 == 0:
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

        if self.is_rank0():
            with open(os.path.join(self.logger.log_dir, "val_crps.json"), "w") as f:
                json.dump(self.crpsByVar, f)
            with open(os.path.join(self.logger.log_dir, "val_rmse.json"), "w") as f:
                json.dump(self.mseByVar, f)
            with open(os.path.join(self.logger.log_dir, "val_acc.json"), "w") as f:
                json.dump(self.accByVar, f)

        realness = real_judgement["data"].mean().item()
        fakeness = fake_judgement["data"].mean().item()
        self.log("realness", realness, prog_bar=True, sync_dist=True)
        self.log("fakeness", fakeness, prog_bar=True, sync_dist=True)
        self.log("val_forecast", forecast_loss, prog_bar=True, sync_dist=True)
        self.log("val_loss", forecast_loss, prog_bar=True, sync_dist=True)

        if self.is_rank0():
            self.plot(inputs, forecast, targets, indexies, batch_idx, mode="eval")

    def test_step(self, batch, batch_idx):
        # Skip test for some batches in CI mode or if batch_idx > 1 in any mode
        if (self.ci and batch_idx > 0) or batch_idx > 1:
            return

        inputs, targets, indexies = batch
        current_batch_size = inputs[self.model.setting.vars[0]].shape[0]

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
        realness = real_judgement["data"].mean().item()
        fakeness = fake_judgement["data"].mean().item()
        crps, absb = self.compute_crps(forecast["data"], targets["data"])

        total_rmse = 0
        for variable in self.model.setting.vars_out:
            self.mseByVar[variable] = dict()
            self.accByVar[variable] = dict()

            total_rmse += self.compute_rmse_by_time(targets, forecast, variable)
            prod, fsum, osum = self.calculate_acc(
                forecast[variable],
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

        if self.is_rank0():
            with open(os.path.join(self.logger.log_dir, "test_rmse.json"), "w") as f:
                json.dump(self.mseByVar, f)
            with open(os.path.join(self.logger.log_dir, "test_acc.json"), "w") as f:
                json.dump(self.accByVar, f)

        self.log("realness", realness, prog_bar=True, sync_dist=True)
        self.log("fakeness", fakeness, prog_bar=True, sync_dist=True)
        self.log("forecast", forecast_loss, prog_bar=True, sync_dist=True)
        self.log("crps", crps.mean(), prog_bar=True, sync_dist=True)
        self.log("absb", absb.mean(), prog_bar=True, sync_dist=True)

        # Only plot for the first batch in CI mode
        if not self.ci or batch_idx == 0:
            self.plot(inputs, forecast, targets, indexies, batch_idx, mode="test")

    def on_fit_start(self):
        self.discriminator.to(self.device)
        self.generator.to(self.device)

    def train_dataloader(self):
        if self.model.dataset_train is None:
            if self.opt.data != "":
                self.model.load_dataset("train", "client", url=self.opt.data)
            else:
                self.model.load_dataset("train", "server")

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
