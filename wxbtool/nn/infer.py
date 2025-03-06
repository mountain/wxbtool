import os
import sys
import importlib
import torch as th
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from wxbtool.nn.lightning import LightningModel
from wxbtool.data.dataset import WxDataset
from wxbtool.util.plotter import plot
from datetime import datetime
import wxbtool.config as config
import logging

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

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
        model = LightningModel(mdm.model)

        # Load the model checkpoint if provided
        if opt.load:
            checkpoint = th.load(opt.load)
            model.load_state_dict(checkpoint['state_dict'])

        # Set the model to evaluation mode
        model.eval()

        parser_time = datetime.strptime(opt.datetime, "%Y-%m-%d")

        # Load the dataset
        dataset = WxDataset(
            root=model.model.setting.root,        # WXBHOME_PATH
            resolution=model.model.setting.resolution,
            years=[parser_time.year],  # 目标文件的年份,要求是iterable
            vars=model.model.setting.vars,
            levels=model.model.setting.levels,
            input_span=model.model.setting.input_span,
            pred_shift=model.model.setting.pred_shift,
            pred_span=model.model.setting.pred_span,
            step=model.model.setting.step,
        )

        # Find the index of the specific datetime
        sample_data = xr.open_dataarray(f"{config.root}/geopotential/geopotential_{parser_time.year}_5.625deg.nc")
        day_of_year = int(parser_time.strftime('%j'))
        datetime_index = sample_data.time.values.tolist().index(float(day_of_year-1))

        # Get the input data for the specific datetime
        inputs, _, _ = dataset[datetime_index]

        # Convert inputs to torch tensors
        inputs = {k: th.tensor(v, dtype=th.float32).unsqueeze(0) for k, v in inputs.items()}

        inputs = model.model.get_inputs(**inputs)

        # Perform inference
        with th.no_grad():
            results = model(**inputs)
        
        results = model.model.get_forcast(**results)

        # Save the output
        output_dir = f"output/{opt.datetime}"
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"{opt.module.split('.')[-1]}.{opt.output}"
        output_path = os.path.join(output_dir, file_name)

        if opt.output.endswith('png'):
            for var, data in results.items():
                if var == "t2m":
                    plot(var, open(output_path, mode="wb"), data.squeeze().cpu().numpy())
        elif opt.output.endswith('nc'):
            ds = xr.DataArray(results["t2m"].reshape(32,64), coords={'lat': np.linspace(-87.1875, 87.1875, 32), 'lon':np.linspace(0, 354.375, 64)}, dims=['lat', 'lon'])
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
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
        sys.path.insert(0, os.getcwd())
        mdm = importlib.import_module(opt.module, package=None)
        generator = mdm.generator

        # Load the model checkpoint if provided
        if opt.load:
            checkpoint = th.load(opt.load)
            generator.load_state_dict(checkpoint['state_dict'])

        # Set the model to evaluation mode
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
        inputs = {k: th.tensor(v, dtype=th.float32).unsqueeze(0) for k, v in inputs.items()}

        # Perform GAN inference
        results = []
        with th.no_grad():
            for _ in range(opt.samples):
                noise = th.randn_like(inputs['data'][:, :1, :, :], dtype=th.float32)
                inputs['noise'] = noise
                result = generator(**inputs)
                results.append(result['data'].cpu().numpy())

        # Save the output
        if opt.output.endswith('.png'):
            for i, data in enumerate(results):
                plot(f"sample_{i}", open(f"{opt.output}_{i}.png", mode="wb"), data.squeeze())
        elif opt.output.endswith('.nc'):
            ds = xr.Dataset({f"sample_{i}": (("time", "lat", "lon"), data.squeeze()) for i, data in enumerate(results)})
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
