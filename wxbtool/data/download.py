import importlib
import sys

import cdsapi
import xarray as xr
import numpy as np
import os
import shutil
import datetime
import logging

from wxbtool.nn.lightning import GANModel, LightningModel


def download_latest_hourly_era5_data(variables, start_date, end_date, output_folder):
    c = cdsapi.Client()
    for variable in variables:
        variable_folder = os.path.join(output_folder, variable)
        if not os.path.exists(variable_folder):
            os.makedirs(variable_folder)

        for date in (start_date + datetime.timedelta(n) for n in range((end_date - start_date).days + 1)):
            year_folder = os.path.join(variable_folder, date.strftime("%Y"))
            if not os.path.exists(year_folder):
                os.makedirs(year_folder)

            month_folder = os.path.join(year_folder, date.strftime("%m"))
            if not os.path.exists(month_folder):
                os.makedirs(month_folder)

            filename = os.path.join(month_folder, date.strftime("%Y%m%d_%H") + ".nc")
            if not os.path.exists(filename):
                c.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": variable,
                        "year": date.strftime("%Y"),
                        "month": date.strftime("%m"),
                        "day": date.strftime("%d"),
                        "time": date.strftime("%H:00"),
                        "format": "netcdf",
                    },
                    filename,
                )


def organize_downloaded_data(output_folder):
    for variable in os.listdir(output_folder):
        variable_folder = os.path.join(output_folder, variable)
        for year in os.listdir(variable_folder):
            year_folder = os.path.join(variable_folder, year)
            for month in os.listdir(year_folder):
                month_folder = os.path.join(year_folder, month)
                for filename in os.listdir(month_folder):
                    file_path = os.path.join(month_folder, filename)
                    file_date = datetime.datetime.strptime(filename, "%Y%m%d_%H.nc")
                    new_filename = file_date.strftime("%Y%m%d_%H") + ".nc"
                    new_file_path = os.path.join(month_folder, new_filename)
                    if file_path != new_file_path:
                        os.rename(file_path, new_file_path)


def handle_coverage_option(coverage, output_folder, variables):
    now = datetime.datetime.utcnow()
    if coverage == "daily":
        start_date = now - datetime.timedelta(days=1)
    elif coverage == "weekly":
        start_date = now - datetime.timedelta(weeks=1)
    elif coverage == "monthly":
        start_date = now - datetime.timedelta(weeks=4)
    else:
        start_date = now - datetime.timedelta(days=1)

    download_latest_hourly_era5_data(variables, start_date, now, output_folder)
    organize_downloaded_data(output_folder)


def handle_retention_option(retention, output_folder, variables):
    now = datetime.datetime.utcnow()
    retention_period = {
        "daily": datetime.timedelta(days=1),
        "weekly": datetime.timedelta(weeks=1),
        "monthly": datetime.timedelta(weeks=4),
    }[retention]

    for variable in variables:
        variable_folder = os.path.join(output_folder, variable)
        for year_folder in os.listdir(variable_folder):
            year_folder_path = os.path.join(variable_folder, year_folder)
            for month_folder in os.listdir(year_folder_path):
                month_folder_path = os.path.join(year_folder_path, month_folder)
                for filename in os.listdir(month_folder_path):
                    file_path = os.path.join(month_folder_path, filename)
                    file_date = datetime.datetime.strptime(filename, "%Y%m%d_%H.nc")
                    if now - file_date > retention_period:
                        os.remove(file_path)


def main(context, opt):
    output_folder = "era5"
    try:
        sys.path.insert(0, os.getcwd())
        mdm = importlib.import_module(opt.module, package=None)
        if opt.gan == "true":
            model = GANModel(mdm.generator, mdm.discriminator, opt=opt)
        else:
            model = LightningModel(mdm.model, opt=opt)
        setting = model.setting
        variables = setting["vars"]

        # check .cdsapirc file in the home directory, if not exist, create one by prompting user to input the key
        if not os.path.exists(os.path.expanduser("~/.cdsapirc")):
            key = input("Please enter your ECMWF API key: ")
            with open(os.path.expanduser("~/.cdsapirc"), "w") as f:
                f.write("url: https://cds.climate.copernicus.eu/api/v2\n")
                f.write(f"key: {key}\n")

        handle_coverage_option(opt.coverage, output_folder, variables)
        handle_retention_option(opt.retention, output_folder, variables)
    except ImportError as e:
        exc_info = sys.exc_info()
        print(e)
        print("failure when downloading data")
        import traceback

        traceback.print_exception(*exc_info)
        del exc_info
        sys.exit(-1)
    except Exception as e:
        exc_info = sys.exc_info()
        print(e)
        print("failure when downloading data")
        import traceback

        traceback.print_exception(*exc_info)
        del exc_info
        sys.exit(-1)
