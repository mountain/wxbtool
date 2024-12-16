# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import torch
import torch.nn.functional as F

from scipy.ndimage import zoom
from threading import local
from wxbtool.data.variables import split_name
from wxbtool.util.cmaps import cmaps, var2cmap


data = local()


def imgdata():
    if "img" in dir(data):
        return data.img
    data.img = np.zeros([32, 64, 4], dtype=np.uint8)
    return data.img


def colorize(data, out, cmap):
    data = data.reshape(32, 64)
    data = (data - data.min() + 0.0001) / (data.max() - data.min() + 0.0001)
    data = (data * (data >= 0) * (data < 1) + (data >= 1)) * 255
    fliped = (data[::-1, :]).astype(np.uint8)
    return np.take(cmaps[cmap], fliped, axis=0, out=out)


def imsave(fileobj, data):
    is_success, img = cv2.imencode(".png", data)
    buffer = img.tobytes()
    fileobj.write(buffer)


def plot(var, fileobj, data):
    code, _ = split_name(var)
    imsave(fileobj, colorize(data, imgdata(), var2cmap[code]))


class Ploter:
    def __init__(self, lon, lat):
        self.lon = lon
        self.lat = lat
        self.proj = ccrs.PlateCarree(central_longitude=180)  # 0~360 投影

    def plot(self, input_data, truth, forecast, title="Input vs Truth vs Forecast", year=2000, doy=0, save_path=None):
        vmin = min(np.nanmin(input_data), np.nanmin(truth), np.nanmin(forecast))
        vmax = max(np.nanmax(input_data), np.nanmax(truth), np.nanmax(forecast))

        fig, axes = plt.subplots(1, 3, figsize=(20, 6),
                                 subplot_kw={'projection': self.proj},
                                 constrained_layout=True)

        self._plot_map(axes[0], input_data, vmin, vmax, title="Input")
        self._plot_map(axes[1], truth, vmin, vmax, title="Truth")
        mesh = self._plot_map(axes[2], forecast, vmin, vmax, title="Forecast")

        cbar = fig.colorbar(mesh, ax=axes, location='bottom', shrink=0.7, pad=0.1, orientation='horizontal')
        cbar.set_label("Value", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        title = f"{title} ({year}-{doy})"
        fig.suptitle(title, fontsize=16, y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_map(self, ax, data, vmin, vmax, title=""):
        mesh = ax.pcolormesh(self.lon, self.lat, data, cmap="coolwarm",
                             transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(title, fontsize=14)
        ax.coastlines(resolution='110m', color='black', linewidth=1)
        ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray', alpha=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        return mesh


def bicubic_upsample(data, scale=(8, 16)):
    return zoom(data, scale, order=3)


def adjust_longitude(lon):
    lon = np.where(lon > 180, lon - 360, lon)
    return lon


def plot_image(input_data, truth, forecast, title="", save_path=None):
    input_data_high = bicubic_upsample(input_data, scale=(8, 8))
    truth_high = bicubic_upsample(truth, scale=(8, 8))
    forecast_high = bicubic_upsample(forecast, scale=(8, 8))

    lon_high = np.linspace(0, 360, 512)
    lat_high = np.linspace(-90, 90, 256)
    lon_grid, lat_grid = np.meshgrid(lon_high, lat_high)

    ploter = Ploter(lon_grid, lat_grid)
    ploter.plot(input_data_high, truth_high, forecast_high, title=title, save_path=save_path)
