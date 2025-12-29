import datetime
import importlib
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import cdsapi

DEFAULT_DATAPATH_FORMAT = "{year}/{month:02d}/{day:02d}/{year}{month:02d}{day:02d}_{hour:02d}.nc"

from wxbtool.paradigm.seq2seq import Seq2SeqModel
from wxbtool.paradigm.gan import GANModel

# Configure logging to display information and errors
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


@dataclass
class Config:
    output_folder: str
    variables: List[str]
    vars2d: List[str]
    vars3d: List[str]
    levels: List[str]
    coverage: str  # 'daily', 'weekly', 'monthly'
    retention_days: int = 7  # Number of days to retain data
    retention_action: str = "delete"  # Options: "delete", "archive"
    archive_folder: Optional[str] = None  # Required if action is "archive"
    reference_time_delta: datetime.timedelta = field(
        default_factory=lambda: datetime.timedelta(weeks=1)
    )
    grid: List[float] = field(
        default_factory=lambda: [5.625, 5.625]
    )  # Spatial resolution
    area: List[float] = field(
        default_factory=lambda: [90, -180, -90, 180]
    )  # Global coverage


class ERA5Downloader:
    def __init__(self, config: Config):
        self.c = cdsapi.Client()
        self.config = config
        self.ensure_output_dirs()

    def ensure_output_dirs(self):
        """Create the base directories for each variable."""
        for variable in self.config.variables:
            var_path = os.path.join(self.config.output_folder, variable)
            os.makedirs(var_path, exist_ok=True)
        logging.info("Base output directories are set up.")
        # If archiving is enabled, ensure the archive folder exists
        if self.config.retention_action == "archive" and self.config.archive_folder:
            os.makedirs(self.config.archive_folder, exist_ok=True)
            logging.info("Archive directory is set up.")

    def get_time_span(self):
        """Determine the start and end dates based on the coverage type."""
        end_date = datetime.datetime.utcnow() - self.config.reference_time_delta
        if self.config.coverage == "daily":
            start_date = end_date - datetime.timedelta(days=1)
        elif self.config.coverage == "weekly":
            start_date = end_date - datetime.timedelta(weeks=1)
        elif self.config.coverage == "monthly":
            start_date = end_date - datetime.timedelta(
                days=30
            )  # Approximate month as 30 days
        elif self.config.coverage.isdigit():
            self.config.coverage = int(self.config.coverage)
            start_date = end_date - datetime.timedelta(days=self.config.coverage)
        else:
            raise ValueError(
                "Unsupported coverage type. Use 'daily', 'weekly', or 'monthly'."
            )
        return start_date, end_date

    def generate_datetime_list(
        self, start_date: datetime.datetime, end_date: datetime.datetime
    ):
        """Generate a list of (date, hour) tuples for each hour within the time span."""
        current_datetime = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_datetime = end_date.replace(hour=23, minute=0, second=0, microsecond=0)
        while current_datetime <= end_datetime:
            yield current_datetime, f"{current_datetime.hour:02}:00"
            current_datetime += datetime.timedelta(hours=1)

    def build_filename(
        self, variable: str, date: datetime.datetime, time_str: str
    ) -> str:
        """Construct the filename for the NetCDF file."""
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")
        hour = time_str.split(":")[0]
        filename = f"{date.strftime('%Y%m%d')}_{hour}.nc"
        return os.path.join(
            self.config.output_folder, variable, year, month, day, filename
        )

    def ensure_variable_dirs(self, variable: str, date: datetime.datetime):
        """Ensure that the directory for a specific variable, year, and month exists."""
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")
        var_year_path = os.path.join(self.config.output_folder, variable, year)
        var_month_path = os.path.join(var_year_path, month)
        var_day_path = os.path.join(var_month_path, day)
        # os.makedirs(var_month_path, exist_ok=True)
        os.makedirs(var_day_path, exist_ok=True)

    def retrieve_data(
        self, variable: str, date: datetime.datetime, time_str: str, filename: str
    ):
        """Retrieve data for a specific variable, date, and time."""
        request_params = {
            "product_type": "reanalysis",
            "variable": variable,
            "year": date.strftime("%Y"),
            "month": date.strftime("%m"),
            "day": date.strftime("%d"),
            "time": time_str,
            "format": "netcdf",
            "grid": self.config.grid,  # Spatial resolution
            "area": self.config.area,  # Global coverage
        }

        if variable in self.config.vars3d:
            request_params["pressure_level"] = self.config.levels
            dataset = "reanalysis-era5-pressure-levels"
        elif variable in self.config.vars2d:
            dataset = "reanalysis-era5-single-levels"
        else:
            logging.warning(
                f"Variable '{variable}' is not recognized as single or pressure level."
            )
            return

        try:
            self.c.retrieve(dataset, request_params, filename)
            logging.info(f"Downloaded: {filename}")
        except Exception as e:
            logging.error(f"Failed to download {filename}: {e}")

    def manage_retention(self):
        """Manage data retention by deleting or archiving old files."""
        retention_threshold = datetime.datetime.utcnow() - datetime.timedelta(
            days=self.config.retention_days + 7
        )
        logging.info(
            f"Managing retention: deleting or archiving files older than {self.config.retention_days} days."
        )

        for variable in self.config.variables:
            var_path = os.path.join(self.config.output_folder, variable)
            for root, dirs, files in os.walk(var_path):
                for file in files:
                    if file.endswith(".nc"):
                        file_path = os.path.join(root, file)
                        try:
                            file_mtime = datetime.datetime.utcfromtimestamp(
                                os.path.getmtime(file_path)
                            )
                            if file_mtime < retention_threshold:
                                if self.config.retention_action == "delete":
                                    os.remove(file_path)
                                    logging.info(f"Deleted old file: {file_path}")
                                elif (
                                    self.config.retention_action == "archive"
                                    and self.config.archive_folder
                                ):
                                    # Determine archive subdirectory based on variable, year, month
                                    relative_path = os.path.relpath(
                                        root, self.config.output_folder
                                    )
                                    archive_dir = os.path.join(
                                        self.config.archive_folder, relative_path
                                    )
                                    os.makedirs(archive_dir, exist_ok=True)
                                    shutil.move(
                                        file_path, os.path.join(archive_dir, file)
                                    )
                                    logging.info(
                                        f"Archived old file: {file_path} to {archive_dir}"
                                    )
                        except Exception as e:
                            logging.error(f"Error processing file {file_path}: {e}")

    def download(self):
        """Execute the download process based on the configuration."""
        start_date, end_date = self.get_time_span()
        logging.info(
            f"Downloading data from {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}."
        )

        for variable in self.config.variables:
            for date, time_str in self.generate_datetime_list(start_date, end_date):
                self.ensure_variable_dirs(variable, date)
                filename = self.build_filename(variable, date, time_str)
                abs_filename = os.path.abspath(filename)
                if not os.path.exists(abs_filename):
                    self.retrieve_data(variable, date, time_str, abs_filename)
                else:
                    logging.info(f"File already exists: {abs_filename}")

        # After downloading, manage retention
        # self.manage_retention() # Commented out to avoid deletion


def main(context, opt):
    try:
        sys.path.insert(0, os.getcwd())
        mdm = importlib.import_module(opt.module, package=None)
        if opt.gan == "true":
            model = GANModel(mdm.generator, mdm.discriminator, opt=opt)
        else:
            model = Seq2SeqModel(mdm.model, opt=opt)
        setting = model.model.setting
        resolution = float(setting.resolution.replace("deg", ""))
        variables = setting.vars
        vars2d = setting.vars2d
        vars3d = setting.vars3d
        levels = setting.levels

        # check .cdsapirc file in the home directory, if not exist, create one by prompting user to input the key
        if not os.path.exists(os.path.expanduser("~/.cdsapirc")):
            key = input("Please enter your ECMWF API key: ")
            with open(os.path.expanduser("~/.cdsapirc"), "w") as f:
                f.write("url: https://cds.climate.copernicus.eu/api\n")
                f.write(f"key: {key}\n")

        config = Config(
            output_folder="era5/",  # Replace with your desired output path
            variables=variables,
            vars2d=vars2d,
            vars3d=vars3d,
            levels=levels,
            grid=[resolution, resolution],  # Spatial resolution
            area=[90, -180, -90, 180],  # Global coverage
            coverage=opt.coverage,  # Options: "daily", "weekly", "monthly"
            # retention_days={"daily": 1, "weekly": 7, "monthly": 30}[opt.retention],
            # retention_action="delete",  # Options: "delete", "archive"
        )

        # Initialize and run the downloader
        downloader = ERA5Downloader(config)
        downloader.download()
    except Exception as e:
        exc_info = sys.exc_info()
        print(e)
        print("failure when downloading data")
        import traceback

        traceback.print_exception(*exc_info)
        del exc_info
        sys.exit(-1)
