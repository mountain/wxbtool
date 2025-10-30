import os
from typing import Tuple

import torch as th

from torch.utils.data import DataLoader

from wxbtool.core.decorators import ci_short_circuit, ci_batch_injection
from wxbtool.metrics.crps import CRPS
from wxbtool.core.types import Data, Indexes, Batch, Tensor
from wxbtool.data.dataset import ensemble_loader
from wxbtool.paradigm.base import LightningModel


class GANModel(LightningModel):
    def __init__(self, generator, discriminator, opt=None):
        super(GANModel, self).__init__(generator, opt=opt)
        self.phase = None
        self.generator = generator
        self.discriminator = discriminator
        self.automatic_optimization = False

        self.learning_rate = 1e-4
        self.generator.learning_rate = 1e-4
        self.discriminator.learning_rate = 1e-4
        if opt and hasattr(opt, "rate"):
            learning_rate = float(opt.rate)
            ratio = float(opt.ratio)
            self.generator.learning_rate = learning_rate
            self.discriminator.learning_rate = learning_rate / ratio

        self.alpha = 0.5
        if opt and hasattr(opt, "alpha"):
            self.alpha = float(opt.alpha)

        self.train_crps = self.build_metrics(CRPS)
        self.val_crps = self.build_metrics(CRPS)
        self.test_crps = self.build_metrics(CRPS)

    def configure_optimizers(self):
        g_optimizer = th.optim.Adam(
            self.generator.parameters(), lr=self.generator.learning_rate, weight_decay=0.0, betas=(0.0, 0.9),
        )
        d_optimizer = th.optim.Adam(
            self.discriminator.parameters(), lr=self.discriminator.learning_rate, weight_decay=0.0, betas=(0.0, 0.9),
        )
        g_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, 53)
        d_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(d_optimizer, 53)
        return [g_optimizer, d_optimizer], [g_scheduler, d_scheduler]

    def generator_loss(self, fake_judgement):
        return th.nn.functional.binary_cross_entropy_with_logits(
            fake_judgement["data"],
            th.ones_like(fake_judgement["data"], dtype=th.float32),
        )

    def discriminator_loss(self, real_judgement, fake_judgement):
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

    def seed(self, data: Tensor):
        if data.dim() == 5:
            seed = th.randn_like(data[:, :, :1, :, :], dtype=th.float32)
        elif data.dim() == 4:
            seed = th.randn_like(data[:, :1, :, :], dtype=th.float32)
        else:
            seed = th.randn_like(data, dtype=th.float32)
        return seed

    def with_optimizer(self, optimizer, compute_loss_fn):
        self.toggle_optimizer(optimizer)
        loss = compute_loss_fn()
        self.manual_backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        self.untoggle_optimizer(optimizer)

    def log_loss(self, forecast_loss, total_loss):
        prefix = "" if self.phase == "train" else f"{self.phase}_"
        if total_loss is not None:
            self.log(f"{prefix}loss", total_loss, prog_bar=True)
        if forecast_loss is not None:
            self.log(f"{prefix}forecast", forecast_loss, prog_bar=True)

    def log_judgement(self, real_judgement, fake_judgement, judgement_loss):
        realness = th.sigmoid(real_judgement["data"]).mean().item()
        fakeness = th.sigmoid(fake_judgement["data"]).mean().item()
        self.log("realness", realness, prog_bar=True, sync_dist=True)
        self.log("fakeness", fakeness, prog_bar=True, sync_dist=True)
        self.log("judgement", judgement_loss, prog_bar=True, sync_dist=True)

    def prepare_for_step(self, inputs:Data, targets:Data) -> Tuple[Data, Data, Data]:
        inputs = self.model.get_inputs(**inputs)
        targets = self.model.get_targets(**targets)
        data = inputs["data"]
        inputs["seed"] = self.seed(data)
        local_data = {}
        return inputs, targets, local_data

    def compute_generator(self, inputs:Data, targets:Data, local_data:Data, indexes:Indexes, batch_idx:int) -> Data:
        forecast = self.generator(**inputs)
        fake_data = forecast["data"].detach()
        local_data['fake_data'] = fake_data
        local_data['forecast'] = forecast
        return forecast

    def compute_generator_loss(self, inputs:Data, targets:Data, local_data:Data, indexes:Indexes, batch_idx:int) -> Tensor:
        forecast = self.compute_generator(inputs, targets, local_data, indexes, batch_idx)
        fake_judgement = self.discriminator(**inputs, target=forecast["data"])
        generate_loss = self.generator_loss(fake_judgement)
        forecast_loss = self.loss_fn(inputs, forecast, targets, indexes=indexes, mode="train")
        total_loss = self.alpha * forecast_loss + (1 - self.alpha) * generate_loss
        self.log_loss(forecast_loss, total_loss)
        return total_loss

    def compute_discriminator_loss(self, inputs:Data, targets:Data, local_data:Data) -> Tensor:
        fake_data = local_data['fake_data']
        real_judgement = self.discriminator(**inputs, target=targets["data"])
        fake_judgement = self.discriminator(**inputs, target=fake_data)
        judgement_loss = self.discriminator_loss(real_judgement, fake_judgement)
        self.log_judgement(real_judgement, fake_judgement, judgement_loss)
        return judgement_loss

    def compute_all(self, inputs:Data, targets:Data, local_data:Data, indexes:Indexes, batch_idx:int):
        self.compute_generator_loss(inputs, targets, local_data, indexes, batch_idx)
        self.compute_discriminator_loss(inputs, targets, local_data)
        self.log_all(inputs, targets, local_data, indexes, batch_idx)

    def log_all(self, inputs:Data, targets:Data, local_data:Data, indexes:Indexes, batch_idx: int):
        forecast = local_data['forecast']
        {"train": self.train_rmse, "val": self.val_rmse, "test": self.test_rmse}[self.phase](forecast, targets)
        {"train": self.train_acc, "val": self.val_acc, "test": self.test_acc}[self.phase](forecast, targets, indexes)
        {"train": self.train_crps, "val": self.val_crps, "test": self.test_crps}[self.phase](forecast, targets)
        if self.is_rank0():
            getattr(self, f"{self.phase}_rmse").dump(os.path.join(self.logger.log_dir, f"{self.phase}_rmse.json"))
            getattr(self, f"{self.phase}_acc").dump(os.path.join(self.logger.log_dir, f"{self.phase}_acc.json"))
            getattr(self, f"{self.phase}_crps").dump(os.path.join(self.logger.log_dir, f"{self.phase}_crps.json"))
            phase = self.phase if self.phase != "val" else "eval"
            if batch_idx % 10 == 0:
                self.plot(inputs, forecast, targets, indexes, mode=phase)

    @ci_short_circuit
    def training_step(self, batch: Batch, batch_idx: int) -> None:
        self.phase = "train"
        inputs, targets, indexes = batch
        inputs, targets, local_data = self.prepare_for_step(inputs, targets)
        g_block = lambda : self.compute_generator_loss(inputs, targets, local_data, indexes, batch_idx)
        d_block = lambda : self.compute_discriminator_loss(inputs, targets, local_data)
        g_optimizer, d_optimizer = self.optimizers()
        self.with_optimizer(g_optimizer, g_block)
        self.with_optimizer(d_optimizer, d_block)
        self.log_all(inputs, targets, local_data, indexes, batch_idx)

    @ci_short_circuit
    def validation_step(self, batch, batch_idx):
        self.phase = "val"
        inputs, targets, indexes = batch
        inputs, targets, local_data = self.prepare_for_step(inputs, targets)
        self.compute_all(inputs, targets, local_data, indexes, batch_idx)

    @ci_short_circuit
    def test_step(self, batch, batch_idx):
        self.phase = "test"
        inputs, targets, indexes = batch
        inputs, targets, local_data = self.prepare_for_step(inputs, targets)
        self.compute_all(inputs, targets, local_data, indexes, batch_idx)

    def on_fit_start(self):
        super().on_fit_start()
        self.discriminator.to(self.device)
        self.generator.to(self.device)

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        if self.is_rank0():
            log_dir = self.logger.log_dir
            self.train_crps.dump(os.path.join(log_dir, "train_crps.json"))

        self.val_crps.reset()

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if self.is_rank0():
            log_dir = self.logger.log_dir
            self.val_crps.dump(os.path.join(log_dir, "val_crps.json"))

        self.val_crps.reset()

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        if self.is_rank0():
            log_dir = self.logger.log_dir
            self.test_crps.dump(os.path.join(log_dir, "test_crps.json"))

        self.test_crps.reset()

    @ci_batch_injection(batch_size_ci=2)
    def train_dataloader(self):
        if self.model.dataset_train is None:
            if self.opt.data != "":
                self.model.load_dataset("train", "client", url=self.opt.data)
            else:
                self.model.load_dataset("train", "server")

        batch_size = self.opt.batch_size
        num_workers = self.opt.n_cpu
        return DataLoader(
            self.model.dataset_train,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

    @ci_batch_injection(batch_size_ci=2)
    def val_dataloader(self):
        if self.model.dataset_eval is None:
            if self.opt.data != "":
                self.model.load_dataset("train", "client", url=self.opt.data)
            else:
                self.model.load_dataset("train", "server")

        batch_size = self.opt.batch_size
        return ensemble_loader(
            self.model.dataset_eval,
            batch_size,
            False,
        )

    @ci_batch_injection(batch_size_ci=2)
    def test_dataloader(self):
        if self.model.dataset_test is None:
            if self.opt.data != "":
                self.model.load_dataset("train", "client", url=self.opt.data)
            else:
                self.model.load_dataset("train", "server")

        batch_size = self.opt.batch_size
        return ensemble_loader(
            self.model.dataset_test,
            batch_size,
            False,
        )

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint['state_dict']
        keys_to_reshape = [
            'model.phi', 'model.theta', 'model.weight',
            'generator.phi', 'generator.theta', 'generator.weight'
        ]

        for key in keys_to_reshape:
            if key in state_dict:
                param = state_dict[key]
                if param.dim() == 2 and param.shape == th.Size([32, 64]):
                    state_dict[key] = param.view(1, 1, 1, 32, 64)
