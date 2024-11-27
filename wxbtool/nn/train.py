import os
import sys
import importlib

import torch as th
import lightning.pytorch as pl

from lightning.pytorch.callbacks import EarlyStopping
from wxbtool.nn.lightning import LightningModel, GANModel


if th.cuda.is_available():
    accelerator = "gpu"
    th.set_float32_matmul_precision("medium")
elif th.backends.mps.is_available():
    accelerator = "cpu"
else:
    accelerator = "cpu"


def main(context, opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    try:
        sys.path.insert(0, os.getcwd())
        mdm = importlib.import_module(opt.module, package=None)

        if opt.gan == "true":
            learning_rate = float(opt.rate)
            ratio = float(opt.ratio)
            generator_lr, discriminator_lr = learning_rate, learning_rate / ratio
            model = GANModel(mdm.generator, mdm.discriminator, opt=opt)
            model.generator.learning_rate = generator_lr
            model.discriminator.learning_rate = discriminator_lr
            callbacks = []
        else:
            learning_rate = float(opt.rate)
            model = LightningModel(mdm.model, opt=opt)
            model.learning_rate = learning_rate
            callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=30)]

        n_epochs = 1 if opt.test == "true" else opt.n_epochs

        if opt.gpu is not None and opt.gpu != "":
            devices = [int(idx) for idx in opt.gpu.split(",")]
        else:
            devices = 1

        trainer = pl.Trainer(
            devices=devices,
            accelerator=accelerator,
            precision=32,
            max_epochs=n_epochs,
            callbacks=callbacks,
        )

        if opt.load:
            model.load_from_checkpoint(opt.load)

        trainer.fit(model)
        trainer.test(model=model, dataloaders=model.test_dataloader())
    except ImportError as e:
        exc_info = sys.exc_info()
        print(e)
        print("failure when training model")
        import traceback

        traceback.print_exception(*exc_info)
        del exc_info
        sys.exit(-1)
