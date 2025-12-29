import os
from typing import Tuple

import numpy as np

import wxbtool.core.config as config
from wxbtool.core.resolution import ResolutionConfig


class Setting:
    def __init__(self):
        self.root = os.environ.get("WXBHOME", config.root)  # The root path of WeatherBench Dataset, inject from config
        self.resolution = "5.625deg"  # The spatial resolution of the model

        # Dataset organization configuration (Flexible Granularity & Path)
        # granularity controls the date_range frequency used for file discovery.
        # Supported values: "yearly", "quarterly", "monthly", "weekly", "daily", "hourly".
        self.granularity = "yearly"
        # data_path_format is formatted relative to the variable directory: <root>/<var>/<formatted>
        # Supported placeholders: {var}, {resolution}, {year}, {month}, {day}, {hour}, {week}, {quarter}
        self.data_path_format = "{var}_{year}_{resolution}.nc"

        # Load spatial configuration based on resolution
        self._load_spatial_config()

        self.name = "test"  # The name of the model

        self.step = 4
        self.input_span = 3
        self.pred_span = 1
        self.pred_shift = 72

        self.levels = ["300", "500", "700", "850", "1000"]
        self.height = len(self.levels)

        self.vars = [
            "geopotential",
            "toa_incident_solar_radiation",
            "2m_temperature",
            "temperature",
            "total_cloud_cover",
        ]
        self.vars2d = [
            "toa_incident_solar_radiation",
            "total_cloud_cover",
            "2m_temperature",
        ]
        self.vars3d = ["geopotential", "temperature"]
        self.vars_in = ["z500", "z1000", "tau", "t850", "tcc", "t2m", "tisr"]
        self.vars_out = ["t850", "z500"]

        self.years_train = [
            1980,
            1981,
            1982,
            1983,
            1984,
            1985,
            1986,
            1987,
            1988,
            1989,
            1990,
            1991,
            1992,
            1993,
            1994,
            1995,
            1996,
            1997,
            1998,
            1999,
            2000,
            2001,
            2002,
            2003,
            2004,
            2005,
            2006,
            2007,
            2008,
            2009,
            2010,
            2011,
            2012,
            2013,
            2014,
        ]
        self.years_test = [2015]
        self.years_eval = [2016, 2017, 2018]

    def _load_spatial_config(self) -> None:
        """Load spatial dimensions and ranges based on resolution"""
        spatial_config = ResolutionConfig.get_config(self.resolution)

        # Core spatial dimensions
        self.lat_size = spatial_config["lat_size"]
        self.lon_size = spatial_config["lon_size"]

        # Grid boundaries and steps
        self.lat_range = spatial_config["lat_range"]
        self.lon_range = spatial_config["lon_range"]
        self.lat_step = spatial_config["lat_step"]
        self.lon_step = spatial_config["lon_step"]

        # Derived properties
        self.spatial_shape = (self.lat_size, self.lon_size)
        self.total_spatial_size = self.lat_size * self.lon_size

    def get_latitude_array(self) -> np.ndarray:
        """Generate latitude coordinate array for this resolution

        Returns:
            NumPy array of latitude values from north to south
        """
        lat_north, lat_south = self.lat_range
        return np.linspace(lat_north, lat_south, self.lat_size)

    def get_longitude_array(self) -> np.ndarray:
        """Generate longitude coordinate array for this resolution

        Returns:
            NumPy array of longitude values from west to east
        """
        lon_west, lon_east = self.lon_range
        return np.linspace(lon_west, lon_east, self.lon_size)

    def get_meshgrid(self, normalized: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Generate coordinate meshgrid for this resolution

        Args:
            normalized: If True, return coordinates normalized to [0,1] range

        Returns:
            Tuple of (longitude_grid, latitude_grid) meshgrid arrays
        """
        if normalized:
            x = np.linspace(0, 1, num=self.lat_size)
            y = np.linspace(0, 1, num=self.lon_size)
        else:
            x = self.get_latitude_array()
            y = self.get_longitude_array()
        return np.meshgrid(y, x)  # Note: meshgrid returns (Y, X) order
