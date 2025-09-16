# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr


# Land-sea mask
def load_lsm(resolution, root):
    data_path = "%s/constants/constants_%s.nc" % (root, resolution)
    ds = xr.open_dataset(data_path)
    ds = ds.transpose("lat", "lon")
    dt = np.array(ds["lsm"].data, dtype=np.float32)
    return dt


# Soil type
def load_slt(resolution, root):
    data_path = "%s/constants/constants_%s.nc" % (root, resolution)
    ds = xr.open_dataset(data_path)
    ds = ds.transpose("lat", "lon")
    dt = np.array(ds["slt"].data, dtype=np.float32)
    return dt


# Orography
def load_orography(resolution, root):
    data_path = "%s/constants/constants_%s.nc" % (root, resolution)
    ds = xr.open_dataset(data_path)
    ds = ds.transpose("lat", "lon")
    dt = np.array(ds["orography"].data, dtype=np.float32)
    return dt


# Latitude
def load_lat2d(resolution, root):
    data_path = "%s/constants/constants_%s.nc" % (root, resolution)
    ds = xr.open_dataset(data_path)
    ds = ds.transpose("lat", "lon")
    dt = np.array(ds["lat2d"].data, dtype=np.float64)
    return dt


# Longitude
def load_lon2d(resolution, root):
    data_path = "%s/constants/constants_%s.nc" % (root, resolution)
    ds = xr.open_dataset(data_path)
    ds = ds.transpose("lat", "lon")
    dt = np.array(ds["lon2d"].data, dtype=np.float64)
    return dt


# Area weight
# Calculate the area weight of each grid point
# Note the difference of grid and its dual grid
# Reference: formula (1) in https://arxiv.org/pdf/2308.15560
def load_area_weight(resolution, root):
    res = float(resolution[:-3])
    n_lat, n_lng = int(180 // res), int(360 // res)
    lat_edges = np.linspace(-90.0, +90.0, n_lat + 1)
    lat_edges = np.radians(lat_edges)
    sin_lat_edges = np.sin(lat_edges)
    interval_areas = sin_lat_edges[1:] - sin_lat_edges[:-1]
    avg_area = np.sum(interval_areas) / n_lat
    lat_weights = interval_areas / avg_area
    weight = np.tile(lat_weights[:, np.newaxis], (1, n_lng))

    assert np.isclose(weight.sum(), n_lat * n_lng)

    return weight
