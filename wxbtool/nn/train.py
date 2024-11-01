import os
import sys
import importlib

import torch as th
import lightning.pytorch as pl

from lightning.pytorch.callbacks import EarlyStopping
from wxbtool.nn.lightning import LightningModel


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

        model = LightningModel(mdm.model, opt=opt)
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
