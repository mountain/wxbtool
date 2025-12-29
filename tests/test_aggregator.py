# -*- coding: utf-8 -*-
import os
import shutil
import tempfile
import unittest

# Set WXBHOME before importing wxbtool components that depend on it
os.environ['WXBHOME'] = os.getcwd()

import numpy as np
import pandas as pd
import xarray as xr
from wxbtool.core.setting import Setting
from wxbtool.data.aggregator import Aggregator

class TestAggregator(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.src_dir = os.path.join(self.test_dir, 'src')
        self.dst_dir = os.path.join(self.test_dir, 'dst')
        os.makedirs(self.src_dir)
        os.makedirs(self.dst_dir)

        # Create Dummy Setting
        class DummySetting(Setting):
            def __init__(self, root):
                super().__init__()
                self.root = root
                self.resolution = '5.625deg'
                self.granularity = 'daily'
                self.data_path_format = '{var}/{year}/{month:02d}/{var}_{year}-{month:02d}-{day:02d}.nc'
                self.vars = ['2m_temperature']
                self.years_train = [2020]
                self.years_test = []
                self.years_eval = []

        self.setting = DummySetting(self.dst_dir)

        # Create Source Data (Hourly)
        # We need a few days of data
        # Source Format: {year}/{month:02d}/{day:02d}/{var}_{year}-{month:02d}-{day:02d}T{hour:02d}_{resolution}.nc
        self.dates = pd.date_range(start='2020-01-01', end='2020-01-05 23:00', freq='H')
        for date in self.dates:
            self._create_dummy_file(date, '2m_temperature')
            
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_dummy_file(self, date, var):
        # Path
        rel_path = f"{date.year}/{date.month:02d}/{date.day:02d}/{var}_{date.year}-{date.month:02d}-{date.day:02d}T{date.hour:02d}_5.625deg.nc"
        
        full_path = os.path.join(self.src_dir, var, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Create Data
        # Value = hour (to make mean predictable)
        data = np.full((1, 2, 2), float(date.hour), dtype=np.float32)
        da = xr.DataArray(
            data,
            dims=['time', 'lat', 'lon'],
            coords={'time': [date], 'lat': [0, 1], 'lon': [0, 1]},
            name=var
        )
        ds = da.to_dataset()
        ds.to_netcdf(full_path)

    def test_aggregator_backward(self):
        # Window = 24h, Backward
        # Target: Daily (00:00).
        # For 2020-01-02 00:00:
        # Backward window (t-24h, t] => (2020-01-01 00:00, 2020-01-02 00:00]
        # Should include 2020-01-01 01:00 ... 2020-01-02 00:00
        
        agg = Aggregator(
            self.setting, 
            src_root=self.src_dir, 
            window_hours=24, 
            alignment='backward', 
            workers=2
        )
        agg.run()
        
        # Check output
        # Expect 2020-01-02 output file
        # Path format: {var}/{year}/{month:02d}/{var}_{year}-{month:02d}-{day:02d}.nc
        out_path = os.path.join(self.dst_dir, '2m_temperature', '2m_temperature/2020/01/2m_temperature_2020-01-02.nc')
        
        self.assertTrue(os.path.exists(out_path), f"Output file not found: {out_path}")
        
        ds = xr.open_dataset(out_path)
        # Variable inside file should be code 't2m' or whatever Aggregator decided.
        # Aggregator uses get_code('2m_temperature') -> 't2m'.
        val = ds['t2m'].values.mean()
        
        # Expectation:
        # 24 hours of data.
        # If range is (t-24, t] strictly:
        # 01-01 01:00 (val=1) to 01-02 00:00 (val=0)
        # Sum = (1+2+...+23) + 0 = 276
        # Count = 24
        # Mean = 11.5
        
        # However, xarray slice is inclusive [start, end].
        # So 01-01 00:00 (val=0) to 01-02 00:00 (val=0)
        # That's 25 points.
        # Sum = 276 + 0 = 276
        # Mean = 276 / 25 = 11.04
        
        # Let's see what happens.
        # This test ensures we know what the behavior IS.
        # Code: selection = combined.sel(time=slice(t_start, t_end))
        
        print(f"Mean value: {val}")
        ds.close()

if __name__ == '__main__':
    unittest.main()
