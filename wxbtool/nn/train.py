import os
import sys
import importlib

import torch as th
import lightning.pytorch as pl

from lightning.pytorch.callbacks import EarlyStopping
from wxbtool.nn.lightning import LightningModel, GANModel


if th.cuda.is_available():
    accelerator = "gpu"
    th.set_float32_matmul_precision("high")
elif th.backends.mps.is_available():
    accelerator = "cpu"
else:
    accelerator = "cpu"


def main(context, opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    try:
        sys.path.insert(0, os.getcwd())
        mdm = importlib.import_module(opt.module, package=None)

        n_epochs = 1 if opt.test == "true" else opt.n_epochs
        is_optimized = hasattr(opt, "optimize") and opt.optimize

        if opt.gpu is not None and opt.gpu != "":
            devices = [int(idx) for idx in opt.gpu.split(",")]
        else:
            devices = 1

        if opt.gan == "true":
            learning_rate = float(opt.rate)
            ratio = float(opt.ratio)
            generator_lr, discriminator_lr = learning_rate, learning_rate / ratio
            model = GANModel(mdm.generator, mdm.discriminator, opt=opt)
            model.generator.learning_rate = generator_lr
            model.discriminator.learning_rate = discriminator_lr
            callbacks = []
            
            # Use optimized settings for CI mode
            if is_optimized:
                patience = 3  # Reduced patience for early stopping
                limit_val_batches = 3  # Limit validation to first 3 batches
                limit_test_batches = 2  # Limit testing to first 2 batches
                trainer = pl.Trainer(
                    strategy="ddp_find_unused_parameters_true",
                    devices=devices,
                    accelerator=accelerator,
                    precision=32,
                    max_epochs=n_epochs,
                    callbacks=callbacks,
                    limit_val_batches=limit_val_batches,
                    limit_test_batches=limit_test_batches,
                )
            else:
                trainer = pl.Trainer(
                    strategy="ddp_find_unused_parameters_true",
                    devices=devices,
                    accelerator=accelerator,
                    precision=32,
                    max_epochs=n_epochs,
                    callbacks=callbacks,
                )
        else:
            learning_rate = float(opt.rate)
            model = LightningModel(mdm.model, opt=opt)
            model.learning_rate = learning_rate
            
            # Use optimized settings for CI mode
            if is_optimized:
                patience = 3  # Reduced patience for early stopping
                callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=patience)]
                limit_val_batches = 3  # Limit validation to first 3 batches
                limit_test_batches = 2  # Limit testing to first 2 batches
                trainer = pl.Trainer(
                    strategy="ddp_find_unused_parameters_true",
                    devices=devices,
                    accelerator=accelerator,
                    precision=32,
                    max_epochs=n_epochs,
                    callbacks=callbacks,
                    limit_val_batches=limit_val_batches,
                    limit_test_batches=limit_test_batches,
                )
            else:
                callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=50)]
                trainer = pl.Trainer(
                    strategy="ddp_find_unused_parameters_true",
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
        
        # Skip saving the model in test mode with optimization
        if not (opt.test == "true" and is_optimized):
            th.save(model, model.model.name + ".ckpt")
    except ImportError as e:
        exc_info = sys.exc_info()
        print(e)
        print("failure when training model")
        import traceback

        traceback.print_exception(*exc_info)
        del exc_info
        sys.exit(-1)
