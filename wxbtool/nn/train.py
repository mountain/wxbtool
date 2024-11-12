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
        else:
            learning_rate = float(opt.rate)
            model = LightningModel(mdm.model, opt=opt)
            model.learning_rate = learning_rate

        n_epochs = 1 if opt.test == "true" else opt.n_epochs
        trainer = pl.Trainer(
            accelerator=accelerator,
            precision=32,
            max_epochs=n_epochs,
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=30)],
        )

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
