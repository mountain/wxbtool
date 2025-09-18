import importlib
import os
import sys

import torch as th
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from wxbtool.nn.lightning import GANModel, LightningModel
from wxbtool.nn.config import configure_trainer, detect_torchrun
from wxbtool.nn.callbacks import UniversalLoggingCallback

if th.cuda.is_available():
    accelerator = "gpu"
    th.set_float32_matmul_precision("high")
elif th.backends.mps.is_available():
    accelerator = "mps"
else:
    accelerator = "cpu"


def main(context, opt):
    global accelerator
    ctx = detect_torchrun()
    if opt.gpu == "-1":
        accelerator = "cpu"
    elif not ctx["is_torchrun"]:
        # Only apply CUDA_VISIBLE_DEVICES in single-node mode
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    try:
        sys.path.insert(0, os.getcwd())
        mdm = importlib.import_module(opt.module, package=None)

        n_epochs = 1 if opt.test == "true" else opt.n_epochs
        is_optimized = hasattr(opt, "optimize") and opt.optimize

        precision = "bf16-mixed" if accelerator == "gpu" else "32"

        if opt.gan == "true":
            learning_rate = float(opt.rate)
            ratio = float(opt.ratio)
            generator_lr, discriminator_lr = learning_rate, learning_rate / ratio
            if opt.load:
                model = GANModel.load_from_checkpoint(
                    opt.load, mdm.generator, mdm.discriminator, opt=opt
                )
            else:
                model = GANModel(mdm.generator, mdm.discriminator, opt=opt)
            model.generator.learning_rate = generator_lr
            model.discriminator.learning_rate = discriminator_lr

            trainer = configure_trainer(
                opt,
                callbacks=[UniversalLoggingCallback()],
                precision=precision,
                max_epochs=n_epochs,
                strategy="ddp_find_unused_parameters_true",
            )
        else:
            learning_rate = float(opt.rate)
            if opt.load:
                model = LightningModel.load_from_checkpoint(
                    opt.load, model=mdm.model, opt=opt
                )
            else:
                model = LightningModel(mdm.model, opt=opt)
            model.learning_rate = learning_rate
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                filename="best-{epoch:03d}-{val_loss:.3f}",
                save_top_k=5,
                mode="min",
                dirpath=f"trains/{model.model.name}",
                save_weights_only=False,
            )

            callbacks = [
                EarlyStopping(monitor="val_loss", mode="min", patience=50),
                checkpoint_callback,
                UniversalLoggingCallback(),
            ]
            trainer = configure_trainer(
                opt,
                callbacks=callbacks,
                precision=precision,
                max_epochs=n_epochs,
                strategy="ddp_find_unused_parameters_true",
            )

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
