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
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
        sys.path.insert(0, os.getcwd())
        mdm = importlib.import_module(opt.module, package=None)

        is_optimized = hasattr(opt, "optimize") and opt.optimize

        if opt.gan == "true":
            model = GANModel(mdm.generator, mdm.discriminator, opt=opt)
        else:
            model = LightningModel(mdm.model, opt=opt)

        n_epochs = 1

        # Use optimized settings for CI mode
        if is_optimized:
            patience = 2  # More aggressive early stopping
            limit_val_batches = 3  # Limit validation to first 3 batches
            limit_test_batches = 2  # Limit testing to first 2 batches
            trainer = pl.Trainer(
                accelerator=accelerator,
                precision=32,
                max_epochs=n_epochs,
                callbacks=[
                    EarlyStopping(monitor="val_loss", mode="min", patience=patience)
                ],
                limit_val_batches=limit_val_batches,
                limit_test_batches=limit_test_batches,
            )
        else:
            trainer = pl.Trainer(
                accelerator=accelerator,
                precision=32,
                max_epochs=n_epochs,
                callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=30)],
            )

        if opt.load is None or opt.load == "":
            trainer.fit(model, model.train_dataloader(), model.val_dataloader())
            trainer.test(model=model, dataloaders=model.test_dataloader())
        else:
            trainer.test(
                ckpt_path=opt.load, model=model, dataloaders=model.test_dataloader()
            )

    except ImportError as e:
        exc_info = sys.exc_info()
        print(e)
        print("failure when loading model")
        import traceback

        traceback.print_exception(*exc_info)
        del exc_info
        sys.exit(-1)
