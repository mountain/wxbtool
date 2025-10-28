# -*- coding: utf-8 -*-

from threading import local

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

import wxbtool.norms.meanstd as meanstd
from wxbtool.core.resolution import ResolutionConfig
from wxbtool.util.cmaps import cmaps, var2cmap

data = local()


def imgdata(spatial_shape=(32, 64)):
    """Get image data buffer with specified spatial shape"""
    cache_key = f"img_{spatial_shape[0]}_{spatial_shape[1]}"
    if cache_key in dir(data):
        return getattr(data, cache_key)
    img = np.zeros([spatial_shape[0], spatial_shape[1], 4], dtype=np.uint8)
    setattr(data, cache_key, img)
    return img


def colorize(data, out, cmap, spatial_shape=None):
    """Colorize data with automatic spatial shape detection"""
    if spatial_shape is None:
        # Try to infer spatial shape from data
        if hasattr(data, "shape") and len(data.shape) >= 2:
            spatial_shape = data.shape[-2:]
        else:
            # Fallback to default resolution
            spatial_shape = ResolutionConfig.get_spatial_shape("5.625deg")

    data = data.reshape(spatial_shape)
    data = (data - data.min() + 0.0001) / (data.max() - data.min() + 0.0001)
    data = (data * (data >= 0) * (data < 1) + (data >= 1)) * 255
    fliped = (data[::-1, :]).astype(np.uint8)
    return np.take(cmaps[cmap], fliped, axis=0, out=out)


def imsave(fileobj, data):
    is_success, img = cv2.imencode(".png", data)
    buffer = img.tobytes()
    fileobj.write(buffer)


def plot(var, fileobj, data, spatial_shape=None):
    """Plot variable data with automatic spatial shape detection"""
    import wxbtool.data.variables as variables

    code, _ = variables.split_name(var)
    if spatial_shape is None and hasattr(data, "shape") and len(data.shape) >= 2:
        spatial_shape = data.shape[-2:]
    elif spatial_shape is None:
        spatial_shape = ResolutionConfig.get_spatial_shape("5.625deg")

    cmap_id = var2cmap.get(code, "coolwarm")
    imsave(fileobj, colorize(data, imgdata(spatial_shape), cmap_id, spatial_shape))


class Plotter:
    def __init__(self, lon, lat):
        self.lon = lon
        self.lat = lat
        self.proj = ccrs.PlateCarree(central_longitude=180)  # 0~360 投影

    def plot(
        self,
        code,
        input_data,
        truth,
        forecast,
        title="Input vs Truth vs Forecast",
        year=2000,
        doy=0,
        save_path=None,
        vmin=None,
        vmax=None,
        artifacts=None,
    ):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(20, 6),
            subplot_kw={"projection": self.proj},
            constrained_layout=True,
        )

        self._plot_map(axes[0], input_data, vmin, vmax, title="Input")
        self._plot_map(axes[1], truth, vmin, vmax, title="Truth")
        mesh = self._plot_map(axes[2], forecast, vmin, vmax, title="Forecast")

        cbar = fig.colorbar(
            mesh,
            ax=axes,
            location="bottom",
            shrink=0.7,
            pad=0.1,
            orientation="horizontal",
        )
        cbar.set_label("Value", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        title = f"{title} ({year}-{doy})"
        fig.suptitle(title, fontsize=16, y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_map(self, ax, data, vmin, vmax, title=""):
        mesh = ax.pcolormesh(
            self.lon,
            self.lat,
            data,
            cmap="coolwarm",
            transform=ccrs.PlateCarree(),
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        ax.set_title(title, fontsize=14)
        ax.coastlines(resolution="110m", color="black", linewidth=1)
        ax.add_feature(
            cfeature.LAND, edgecolor="black", facecolor="lightgray", alpha=0.5
        )
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
        return mesh


def bicubic_upsample(data, scale=(8, 16)):
    return zoom(data, scale, order=3)


def adjust_longitude(lon):
    lon = np.where(lon > 180, lon - 360, lon)
    return lon


def plot_image(
    code,
    input_data,
    truth,
    forecast,
    title="",
    year=2000,
    doy=0,
    save_path=None,
    setting=None,
):
    """Plot high-resolution images with resolution-aware upsampling"""
    # Determine base spatial shape
    if setting is not None:
        base_shape = setting.spatial_shape
    elif hasattr(input_data, "shape") and len(input_data.shape) >= 2:
        base_shape = input_data.shape[-2:]
    else:
        base_shape = ResolutionConfig.get_spatial_shape("5.625deg")

    # Calculate appropriate upsampling scale based on base resolution
    lat_scale = 256 // base_shape[0]
    lon_scale = 512 // base_shape[1]

    input_data_high = bicubic_upsample(input_data, scale=(lat_scale, lon_scale))
    truth_high = bicubic_upsample(truth, scale=(lat_scale, lon_scale))
    forecast_high = bicubic_upsample(forecast, scale=(lat_scale, lon_scale))
    mean = meanstd.means[code]
    std = meanstd.stds[code]

    lon_high = np.linspace(0, 360, base_shape[1] * lon_scale)
    lat_high = np.linspace(-90, 90, base_shape[0] * lat_scale)
    lon_grid, lat_grid = np.meshgrid(lon_high, lat_high)

    plotter = Plotter(lon_grid, lat_grid)
    plotter.plot(
        code,
        input_data_high,
        truth_high,
        forecast_high,
        title=title,
        year=year,
        doy=doy,
        save_path=save_path,
        vmin=mean - 3 * std,
        vmax=mean + 3 * std,
    )
