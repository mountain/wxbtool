from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from wxbtool.datagen.util import generate_time_label_array_gray


def generate_climate_dataset(start_year=1980, years=3, output_folder="./"):
    lat = np.linspace(-90, 90, 32)
    lon = np.linspace(0, 360 - 5.625, 64)
    days_per_year = 365

    for year in range(start_year, start_year + years):
        print(f"Generating data for year {year}...")

        time = [datetime(year, 1, 1) + timedelta(days=i) for i in range(days_per_year)]

        data = np.zeros((len(time), len(lat), len(lon)))  # shape: (time, lat, lon)
        for i, day in enumerate(time):
            label = day.strftime("%Y-%m-%d")
            image = generate_time_label_array_gray(label, image_size=(32, 64))
            data[i] = image / 255.0

        ds = xr.Dataset(
            {"test": (["time", "lat", "lon"], data)},
            coords={"time": time, "lat": lat, "lon": lon},
        )

        output_file = f"{output_folder}test_variable_{year}_5.625deg.nc"
        ds.to_netcdf(output_file)
        print(f"Saved NetCDF file: {output_file}")


def generate_climatology_dataset(output_folder="./"):
    lat = np.linspace(-90, 90, 32)
    lon = np.linspace(0, 360 - 5.625, 64)
    days_per_year = 365

    year = 1970
    print(f"Generating data for year {year}...")

    time = [datetime(year, 1, 1) + timedelta(days=i) for i in range(days_per_year)]

    data = np.zeros((len(time), len(lat), len(lon)))  # shape: (time, lat, lon)
    for i, day in enumerate(time):
        label = day.strftime("%m-%d")
        image = generate_time_label_array_gray(label, image_size=(32, 64))
        data[i] = image / 255.0

    ds = xr.Dataset(
        {"test": (["time", "lat", "lon"], data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )

    output_file = f"{output_folder}test_variable.nc"
    ds.to_netcdf(output_file)
    print(f"Saved NetCDF file: {output_file}")


if __name__ == "__main__":
    generate_climate_dataset(start_year=1980, years=3, output_folder="./")
    generate_climatology_dataset(output_folder="./")
