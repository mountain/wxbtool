import xarray as xr
import os
import numpy as np

from functools import lru_cache
from wxbtool.data.dataset import all_levels
from wxbtool.data.variables import split_name, code2var, codes, vars3d, vars2d


class ClimatologyAccessor:
    """
    A class to handle climatology data retrieval with caching for reindexer and climatology data.
    """

    def __init__(self, home='/path/to/climatology'):
        """
        Initialize the ClimatologyAccessor with the path to climatology data files.

        Parameters:
        - home (str): Root directory path where climatology `.nc` files are stored.
        """
        self.home = home
        self.climatology_data = {}  # Cache for climatology DataArrays

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

    @staticmethod
    @lru_cache(maxsize=None)
    def build_reindexer(years_tuple):
        """
        Build a reindexer list that maps each batch index to a day-of-year (DOY) index.
        For leap years, the 366th day is mapped to the 365th day (index 364).

        Parameters:
        - years_tuple (tuple of int): Tuple of years to consider.

        Returns:
        - list of int: Reindexer list mapping batch_idx to DOY index.
        """
        reindexer = []
        for yr in years_tuple:
            # Add DOY indices 0 to 364 for each year
            reindexer.extend(range(365))
            if ClimatologyAccessor.is_leap_year(yr):
                # Map the 366th day to index 364
                reindexer.append(364)
        return reindexer

    def load_climatology_var(self, var):
        """
        Load and cache climatology data for a given variable.

        Parameters:
        - var (str): Variable name.

        Raises:
        - FileNotFoundError: If the climatology file for the variable is not found.
        - ValueError: If the climatology data does not contain a 'time' dimension.
        """
        if var not in self.climatology_data:
            code, lvl = split_name(var)
            vname = code2var.get(code, None)
            if vname is None:
                raise ValueError(f"Variable '{var}' is not supported for climatology data.")
            
            file_path = os.path.join(self.home, f"{vname}.nc")
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Climatology data file not found: {file_path}")

            if vname in vars2d:
                with xr.open_dataset(file_path) as ds:
                    ds = ds.transpose("time", "lat", "lon")
                    self.climatology_data[var] = np.array(ds[codes[vname]].data, dtype=np.float32)
            else:
                all_levels.index(lvl)
                with xr.open_dataset(file_path) as ds:
                    ds = ds.transpose("time", "level", "lat", "lon")
                    self.climatology_data[var] = np.array(ds[codes[vname]].data, dtype=np.float32)[:, all_levels.index(lvl)]

    def get_climatology(self, years, vars, batch_idx):
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
        # Convert batch_idx to a list if it's a single integer
        if isinstance(batch_idx, int):
            batch_indices = [batch_idx]
            single_index = True
        elif isinstance(batch_idx, (list, tuple, np.ndarray)):
            batch_indices = list(batch_idx)
            single_index = False
        else:
            raise TypeError("`batch_idx` should be an integer or a list/tuple of integers.")

        # Convert years list to a tuple for caching
        years_tuple = tuple(years)

        # Build or retrieve the cached reindexer
        reindexer = self.build_reindexer(years_tuple)
        total_days = len(reindexer)

        # Validate batch_indices
        for idx in batch_indices:
            if idx < 0 or idx >= total_days:
                raise IndexError(f"batch_idx {idx} is out of range (0-{total_days-1}).")

        # Map batch_idx to DOY indices using the reindexer
        doy_indices = [reindexer[idx] for idx in batch_indices]

        climatology_dict = {}

        for var in vars:
            # Load and cache climatology data if not already loaded
            self.load_climatology_var(var)
            climatology_var = self.climatology_data[var]
            selected_data = climatology_var[doy_indices]
            climatology_dict[var] = selected_data

        return np.concatenate([climatology_dict[v] for v in vars], axis=0)


# Example Usage
if __name__ == "__main__":
    # Initialize the accessor with the path to climatology data
    climatology_accessor = ClimatologyAccessor(home='/data/climatology')

    # Define the years and variables
    years = [2000, 2001, 2002, 2003, 2004]  # Includes both leap and non-leap years
    variables = ['temperature', 'precipitation']

    # Example 1: Retrieve climatology data for a single batch_idx
    batch_index = 0  # Corresponds to January 1st of the first year
    climatology_single = climatology_accessor.get_climatology(years, variables, batch_index)
    print("Single batch_idx:")
    for var, data in climatology_single.items():
        print(f"Variable: {var}, Data: {data}")

    # Example 2: Retrieve climatology data for multiple batch_idx values
    batch_indices = [0, 365, 730, 1095, 1460]  # Corresponds to January 1st of each year
    climatology_multiple = climatology_accessor.get_climatology(years, variables, batch_indices)
    print("\nMultiple batch_idx:")
    for var, data in climatology_multiple.items():
        print(f"Variable: {var}, Data: {data}")
