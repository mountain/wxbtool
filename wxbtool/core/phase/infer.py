import importlib
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch as th
import xarray as xr

import wxbtool.core.config as config
from wxbtool.data.dataset import WxDataset
from wxbtool.paradigm.seq2seq import Seq2SeqModel
from wxbtool.util.plot import plot
from wxbtool.core.config import get_runtime_device, detect_torchrun, is_rank_zero

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
        device = get_runtime_device(opt)
        mdm = importlib.import_module(opt.module, package=None)
        model = Seq2SeqModel(mdm.model, opt=opt)

        # Load the model checkpoint if provided
        if opt.load:
            checkpoint = th.load(opt.load, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])

        # Move to device and set eval mode
        model.to(device)
        model.eval()

        parser_time = datetime.strptime(opt.datetime, "%Y-%m-%d")

        # Load the dataset
        dataset = WxDataset(
            root=model.model.setting.root,  # WXBHOME_PATH
            resolution=model.model.setting.resolution,
            years=[
                parser_time.year
            ],  # the year thar your date belongs to, must be Iterable
            vars=model.model.setting.vars,
            levels=model.model.setting.levels,
            input_span=model.model.setting.input_span,
            pred_shift=model.model.setting.pred_shift,
            pred_span=model.model.setting.pred_span,
            step=model.model.setting.step,
        )

        # Find the index of the specific datetime
        sample_var = model.model.setting.vars[0]
        sample_path = os.path.join(
            config.root,
            sample_var,
            f"{sample_var}_{parser_time.year}_{model.model.setting.resolution}.nc",
        )
        sample_data = xr.open_dataarray(sample_path)
        day_of_year = int(parser_time.strftime("%j"))
        datetime_index = sample_data.time.values.tolist().index(float(day_of_year - 1))

        # Get the input data for the specific datetime
        inputs, _, _ = dataset[datetime_index]

        # Convert inputs to torch tensors
        inputs = {
            k: th.tensor(
                v[: model.model.setting.input_span, :, :], dtype=th.float32, device=device
            ).unsqueeze(0)
            for k, v in inputs.items()
        }  # 只取input_span范围

        inputs = model.model.get_inputs(**inputs)

        # Perform inference
        with th.no_grad():
            results = model(**inputs)

        results = model.model.get_forcast(**results)

        # Only rank-0 writes outputs in distributed runs
        if not is_rank_zero():
            return

        # Save the output
        output_dir = f"output/{opt.datetime}"
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"{opt.module.split('.')[-1]}.{opt.output}"
        output_path = os.path.join(output_dir, file_name)

        if opt.output.endswith("png"):
            for var, data in results.items():
                if var == "t2m":
                    plot(
                        var, open(output_path, mode="wb"), data.squeeze().cpu().numpy()
                    )
        elif opt.output.endswith("nc"):
            if len(model.model.setting.vars_out) > 1:
                ds = xr.DataArray(
                    None,
                    dims=("vars", "time", "lat", "lon"),
                    coords={
                        "time": pd.date_range(
                            start=opt.datetime,
                            periods=model.model.setting.input_span
                            + model.model.setting.pred_span,
                            freq="D",
                        ),
                        "vars": model.model.setting.vars_out,
                        "lat": model.model.setting.get_latitude_array(),
                        "lon": model.model.setting.get_longitude_array(),
                    },
                )
                for var in model.model.setting.vars_out:
                    lat_size, lon_size = model.model.setting.spatial_shape
                    ds.loc[var] = results[var].reshape(-1, lat_size, lon_size)
            else:
                lat_size, lon_size = model.model.setting.spatial_shape
                ds = xr.DataArray(
                    results["t2m"].reshape(lat_size, lon_size),
                    coords={
                        "lat": model.model.setting.get_latitude_array(),
                        "lon": model.model.setting.get_longitude_array(),
                    },
                    dims=["lat", "lon"],
                )
            ds.to_netcdf(output_path)
        else:
            raise ValueError("Unsupported output format. Use either png or nc.")

    except ImportError as e:
        exc_info = sys.exc_info()
        print(e)
        print("failure when loading model")
        import traceback

        traceback.print_exception(*exc_info)
        del exc_info
        sys.exit(-1)


def main_gan(context, opt):
    try:
        ctx = detect_torchrun()
        if getattr(opt, "gpu", None) != "-1" and not ctx["is_torchrun"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
        sys.path.insert(0, os.getcwd())
        device = get_runtime_device(opt)
        mdm = importlib.import_module(opt.module, package=None)
        generator = mdm.generator

        # Load the model checkpoint if provided
        if opt.load:
            checkpoint = th.load(opt.load, map_location=device)
            generator.load_state_dict(checkpoint["state_dict"])

        # Move to device and set eval mode
        generator.to(device)
        generator.eval()

        # Load the dataset
        dataset = WxDataset(
            root=generator.setting.root,
            resolution=generator.setting.resolution,
            years=generator.setting.years_test,
            vars=generator.setting.vars,
            levels=generator.setting.levels,
            input_span=generator.setting.input_span,
            pred_shift=generator.setting.pred_shift,
            pred_span=generator.setting.pred_span,
            step=generator.setting.step,
        )

        # Find the index of the specific datetime
        datetime_index = np.where(dataset.time == np.datetime64(opt.datetime))[0][0]

        # Get the input data for the specific datetime
        inputs, _ = dataset[datetime_index]

        # Convert inputs to torch tensors
        inputs = {
            k: th.tensor(v, dtype=th.float32, device=device).unsqueeze(0) for k, v in inputs.items()
        }

        # Perform GAN inference
        results = []
        with th.no_grad():
            for _ in range(opt.samples):
                noise = th.randn_like(inputs["data"][:, :1, :, :], dtype=th.float32)
                inputs["noise"] = noise
                result = generator(**inputs)
                results.append(result["data"].cpu().numpy())

        # Only rank-0 writes outputs in distributed runs
        if not is_rank_zero():
            return

        # Save the output
        if opt.output.endswith(".png"):
            for i, data in enumerate(results):
                plot(
                    f"sample_{i}",
                    open(f"{opt.output}_{i}.png", mode="wb"),
                    data.squeeze(),
                )
        elif opt.output.endswith(".nc"):
            ds = xr.Dataset(
                {
                    f"sample_{i}": (("time", "lat", "lon"), data.squeeze())
                    for i, data in enumerate(results)
                }
            )
            ds.to_netcdf(opt.output)
        else:
            raise ValueError("Unsupported output format. Use either png or nc.")

    except ImportError as e:
        exc_info = sys.exc_info()
        print(e)
        print("failure when loading model")
        import traceback

        traceback.print_exception(*exc_info)
        del exc_info
        sys.exit(-1)
