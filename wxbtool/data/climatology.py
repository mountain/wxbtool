import xarray as xr
import os
import numpy as np
import torch

from wxbtool.data.dataset import all_levels
from wxbtool.norms.meanstd import normalizors


class ClimatologyAccessor:
    """
    A class to handle climatology data retrieval with caching for reindexer and climatology data.
    """

    def __init__(self, home="/path/to/climatology"):
        """
        Initialize the ClimatologyAccessor with the path to climatology data files.

        Parameters:
        - home (str): Root directory path where climatology `.nc` files are stored.
        """
        self.home = home
        self.climatology_data = {}  # Cache for climatology DataArrays
        self.doy_indexer = []
        self.yr_indexer = []

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
                    ds = ds.transpose("time", "lat", "lon")
                    data = np.array(ds[variables.codes[vname]].data, dtype=np.float32)
                    # add a channel dimension at the 1 position
                    data = np.expand_dims(data, axis=1)
                    self.climatology_data[var] = data
            else:
                lvl_idx = all_levels.index(lvl)
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
        - years (list of int): List of years.
        - vars (list of str): List of variable names.
        - batch_idx (int or list of int): Batch index or list of batch indices.

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

        climatology_dict = {}

        for var in vars:
            # Load and cache climatology data if not already loaded
            self.load_climatology_var(var)
            climatology_var = self.climatology_data[var]
            selected_data = climatology_var[doy_indices]
            climatology_dict[var] = normalizors[var](selected_data)

        return np.concatenate([climatology_dict[v] for v in vars], axis=1)


# Example Usage
if __name__ == "__main__":
    # Initialize the accessor with the path to climatology data
    climatology_accessor = ClimatologyAccessor(home="/data/climatology")

    # Define the years and variables
    years = [2000, 2001, 2002, 2003, 2004]  # Includes both leap and non-leap years
    variables = ["temperature", "precipitation"]

    # Example 1: Retrieve climatology data for a single batch_idx
    batch_index = 0  # Corresponds to January 1st of the first year
    climatology_single = climatology_accessor.get_climatology(
        years, variables, batch_index
    )
    print("Single batch_idx:")
    for var, data in climatology_single.items():
        print(f"Variable: {var}, Data: {data}")

    # Example 2: Retrieve climatology data for multiple batch_idx values
    batch_indices = [0, 365, 730, 1095, 1460]  # Corresponds to January 1st of each year
    climatology_multiple = climatology_accessor.get_climatology(
        years, variables, batch_indices
    )
    print("\nMultiple batch_idx:")
    for var, data in climatology_multiple.items():
        print(f"Variable: {var}, Data: {data}")
