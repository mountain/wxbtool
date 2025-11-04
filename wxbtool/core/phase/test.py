import importlib
import os
import sys

import torch as th
from lightning.pytorch.callbacks import EarlyStopping

from wxbtool.paradigm.seq2seq import Seq2SeqModel
from wxbtool.paradigm.gan import GANModel
from wxbtool.core.config import configure_trainer, detect_torchrun
from wxbtool.core.callbacks import UniversalLoggingCallback

if th.cuda.is_available():
    accelerator = "gpu"
    th.set_float32_matmul_precision("medium")
elif th.backends.mps.is_available():
    accelerator = "cpu"
else:
    accelerator = "cpu"


def main(context, opt):
    try:
        ctx = detect_torchrun()
        if getattr(opt, "gpu", None) != "-1" and not ctx["is_torchrun"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
        sys.path.insert(0, os.getcwd())
        mdm = importlib.import_module(opt.module, package=None)

        is_optimized = hasattr(opt, "optimize") and opt.optimize

        if opt.gan == "true":
            model = GANModel(mdm.generator, mdm.discriminator, opt=opt)
        else:
            model = Seq2SeqModel(mdm.model, opt=opt)

        n_epochs = 1

        precision = "bf16-mixed" if accelerator == "gpu" else "32"

        # Use optimized settings for CI mode
        if is_optimized:
            patience = 2  # More aggressive early stopping
            limit_val_batches = 3  # Limit validation to first 3 batches
            limit_test_batches = 2  # Limit testing to first 2 batches
            trainer = configure_trainer(
                opt,
                precision=precision,
                max_epochs=n_epochs,
                callbacks=[
                    UniversalLoggingCallback(),
                ],
                limit_val_batches=limit_val_batches,
                limit_test_batches=limit_test_batches,
            )
        else:
            trainer = configure_trainer(
                opt,
                precision=precision,
                max_epochs=n_epochs,
                callbacks=[
                    UniversalLoggingCallback(),
                ],
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
