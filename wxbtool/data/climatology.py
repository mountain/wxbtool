import logging
import os

import numpy as np
import torch
import xarray as xr

import wxbtool.core.config as config
from wxbtool.core.setting import Setting
from wxbtool.norms.meanstd import normalizors

logger = logging.getLogger()


class ClimatologyAccessor:
    """
    A class to handle climatology data retrieval with caching for reindexer and climatology data.
    """

    def __init__(self, home="/path/to/climatology", setting=None):
        """
        Initialize the ClimatologyAccessor with the path to climatology data files.

        Parameters:
        - home (str): Root directory path where climatology `.nc` files are stored.
        - setting (Setting, optional): Setting instance to use. If None, a new one will be created when needed.
        """
        self.home = home
        self.climatology_data = {}  # Cache for climatology DataArrays
        self.doy_indexer = []
        self.yr_indexer = []

        # Cache for levels data
        self._all_levels = None
        self.setting = setting

    @staticmethod
    def is_leap_year(year):
        """
        Determine if a given year is a leap year.

        Parameters:
        - year (int): The year to check.

        Returns:
        - bool: True if leap year, False otherwise.
        """
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    def build_indexers(self, years_tuple):
        """
        Build indexers list that maps each batch index to a day-of-year (DOY) index.
        For leap years, the 366th day is mapped to the 365th day (index 364).

        Parameters:
        - years_tuple (tuple of int): Tuple of years to consider.

        Returns:
        - list of int: Reindexer list mapping batch_idx to DOY index.
        """
        for yr in years_tuple:
            # Add DOY indices 0 to 364 for each year
            self.doy_indexer.extend(range(365))
            self.yr_indexer.extend([yr] * 365)
            if ClimatologyAccessor.is_leap_year(yr):
                # Map the 366th day to index 364
                self.doy_indexer.append(364)
                self.yr_indexer.append(yr)

    def get_all_levels(self):
        """
        Get all available levels from the first 3D variable file.

        Returns:
        - list: List of available levels.
        """
        if self._all_levels is not None:
            return self._all_levels

        setting = self.setting if self.setting is not None else Setting()

        var3d_path = os.path.join(config.root, setting.vars3d[0])
        try:
            any_file = os.listdir(var3d_path)[0]
            sample_data = xr.open_dataarray(f"{var3d_path}/{any_file}")
            self._all_levels = sample_data.level.values.tolist()
        except (FileNotFoundError, IndexError, AttributeError) as e:
            logger.warning(f"Could not determine levels automatically: {e}")
            # Fallback to default levels if we can't determine them automatically
            self._all_levels = [
                50.0,
                100.0,
                150.0,
                200.0,
                250.0,
                300.0,
                400.0,
                500.0,
                600.0,
                700.0,
                850.0,
                925.0,
                1000.0,
            ]
            logger.info(f"Using default levels: {self._all_levels}")

        return self._all_levels

    def load_climatology_var(self, var):
        """
        Load and cache climatology data for a given variable.

        Parameters:
        - var (str): Variable name.

        Raises:
        - FileNotFoundError: If the climatology file for the variable is not found.
        - ValueError: If the climatology data does not contain a 'time' dimension.
        """
        import wxbtool.data.variables as variables

        if var not in self.climatology_data:
            code, lvl = variables.split_name(var)
            vname = variables.code2var.get(code, None)
            if vname is None:
                raise ValueError(
                    f"Variable '{var}' is not supported for climatology data."
                )

            file_path = os.path.join(self.home, f"{vname}.nc")
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Climatology data file not found: {file_path}")

            if vname in variables.vars2d:
                with xr.open_dataset(file_path) as ds:
                    if "time" in ds.dims:
                        ds = ds.transpose("time", "lat", "lon")
                    else:
                        ds = ds.transpose("dayofyear", "lat", "lon")

                    data = np.array(ds[variables.codes[vname]].data, dtype=np.float32)
                    # add a channel dimension at the 1 position
                    data = np.expand_dims(data, axis=1)
                    self.climatology_data[var] = data
            else:
                all_levels = self.get_all_levels()
                lvl_idx = all_levels.index(float(lvl))
                with xr.open_dataset(file_path) as ds:
                    ds = ds.transpose("time", "level", "lat", "lon")
                    data = np.array(ds[variables.codes[vname]].data, dtype=np.float32)[
                        :, lvl_idx
                    ]
                    # add a channel dimension at 1 position
                    data = np.expand_dims(data, axis=1)
                    self.climatology_data[var] = data

    def get_climatology(self, vars, indexes):
        """
        Retrieve climatology data for specified variables based on batch indices.

        Parameters:
        - vars (list of str): List of variable names.
        - indexes (int or list of int): Batch index or list of batch indices.

        Returns:
        - dict: Dictionary containing climatology data for each variable.
                Format: {var: data_array}
        """
        # Convert indexes to a list
        if isinstance(indexes, int):
            indexes = [indexes]
        elif isinstance(indexes, (list, tuple, np.ndarray)):
            indexes = list(indexes)
        elif isinstance(indexes, torch.Tensor):
            indexes = indexes.tolist()
        else:
            raise TypeError(
                f"`indexes` should be an integer or a list/tuple of integers, but got: {type(indexes)}"
            )

        total_days = len(self.doy_indexer)

        # Validate indexes
        for idx in indexes:
            if idx < 0 or idx >= total_days:
                raise IndexError(f"indexes {idx} is out of range (0-{total_days - 1}).")

        # Map batch_idx to DOY indices using the doy indexer
        doy_indices = [self.doy_indexer[idx] for idx in indexes]
        seq_len = len(doy_indices)

        climatology_dict = {}
        for var in vars:
            # Load and cache climatology data if not already loaded
            self.load_climatology_var(var)
            climatology_var = self.climatology_data[var]
            selected_data = climatology_var[doy_indices]
            h = selected_data.shape[2]
            w = selected_data.shape[3]
            climatology_dict[var] = normalizors[var](selected_data).reshape(
                1, 1, seq_len, h, w
            )

        return np.concatenate([climatology_dict[v] for v in vars], axis=1)


# Example Usage
if __name__ == "__main__":
    # Create a setting instance
    setting = Setting()

    # Initialize the accessor with the path to climatology data and the setting
    climatology_accessor = ClimatologyAccessor(
        home="/data/climatology", setting=setting
    )

    # Define the years and variables
    years = [2000, 2001, 2002, 2003, 2004]  # Includes both leap and non-leap years
    variables = ["temperature", "precipitation"]

    # Build the indexers
    climatology_accessor.build_indexers(years)

    # Example 1: Retrieve climatology data for a single batch_idx
    batch_index = 0  # Corresponds to January 1st of the first year
    climatology_single = climatology_accessor.get_climatology(variables, batch_index)
    print("Single batch_idx:")
    print(f"Data shape: {climatology_single.shape}")

    # Example 2: Retrieve climatology data for multiple batch_idx values
    batch_indices = [0, 365, 730, 1095, 1460]  # Corresponds to January 1st of each year
    climatology_multiple = climatology_accessor.get_climatology(
        variables, batch_indices
    )
    print("\nMultiple batch_idx:")
    print(f"Data shape: {climatology_multiple.shape}")
