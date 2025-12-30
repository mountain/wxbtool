# -*- coding: utf-8 -*-

import logging
import os
from datetime import timedelta
from multiprocessing import Pool
from typing import Tuple, List, Optional

import xarray as xr
import pandas as pd
import numpy as np

from wxbtool.core.setting import Setting
from wxbtool.data.path import DataPathManager
from wxbtool.data.variables import get_code, is_var3d
from wxbtool.data.download import DEFAULT_DATAPATH_FORMAT

logger = logging.getLogger(__name__)


def aggregate_worker(args):
    """
    Worker function for parallel processing.
    
    Args:
        args: Tuple containing:
            timestamp (pd.Timestamp): The target timestamp
            var (str): The variable name
            src_root (str): Source data root directory
            dst_path_format (str): Destination path format (relative)
            window_hours (int): Window size in hours
            alignment (str): 'backward', 'forward', or 'center'
            resolution (str): Spatial resolution
            setting_name (str): Name of the setting (for logging)
            
    Returns:
        str: Result message or Error
    """
    (timestamp, var, src_root, dst_path_format, window_hours, alignment, resolution, input_dt) = args
    
    try:
        # 1. Calculate Window Range
        delta = timedelta(hours=window_hours)
        if alignment == 'backward':
            # (t - w, t]
            t_end = timestamp
            t_start = timestamp - delta
            # We want specific behavior: typically window covers range.
            # pandas rolling(closed='right'): includes current, excludes start?
            # Let's align with pandas standard rolling logic for time series if possible
            # or just simple intuitive range.
            # Project spec says: Range (t − w, t].
            # This implies we look for data > t_start and <= t_end.
        elif alignment == 'forward':
            # [t, t + w)
            t_start = timestamp
            t_end = timestamp + delta
        elif alignment == 'center':
            # (t - w/2, t + w/2] or similar centered Logic
            half = timedelta(hours=window_hours / 2)
            t_start = timestamp - half
            t_end = timestamp + half
        else:
            return f"Error: Unknown alignment {alignment}"

        # 2. Find Source Files
        # We need a dense hourly range to find all potentially relevant files
        # then we filter by the specific window
        # Generating a search range slightly wider to ensure we catch files
        search_range = pd.date_range(start=t_start, end=t_end, freq='H')
        
        # We assume source data is accessible via DataPathManager
        # NOTE: src_root might be different structure than what DataPathManager expects 
        # if the user just points to a raw dir. But Project 007 implies using standard data.
        # Let's assume standard 'hourly' input format 
        # "{year}/{month:02d}/{day:02d}/{var}_{year}-{month:02d}-{day:02d}T{hour:02d}_{resolution}.nc"
        # or similar. Actually, the best way might be to ask users or infer?
        # For V1, let's assume the source is also accessible via standard paths.
        # If source is raw hourly, likely: {year}/{month:02d}/{day:02d}/{var}_...
        # Let's try to discover source files using a standard hourly pattern if not provided.
        # But wait, DataPathManager needs a format.
        # Let's assume specific standard format for Raw ERA5 if not specified? 
        # Project 007 doesn't specify source format, only source dir.
        # Let's use a common hourly format default or require it to match.
        # Actually, let's look for standard patterns.
        
        # Simplification for V1: Assume standard hourly format for source.
        src_format = "{year}/{month:02d}/{day:02d}/{var}_{year}-{month:02d}-{day:02d}T{hour:02d}_{resolution}.nc"
        
        src_files = DataPathManager.get_file_paths(
            src_root, var, resolution, src_format, search_range
        )
        
        if not src_files:
            return f"Warning: No source files found for {var} at {timestamp} (Window: {t_start} - {t_end})"

        # 3. Load Data
        datasets = []
        for f in src_files:
            if os.path.exists(f):
                try:
                    ds = xr.open_dataset(f, engine='netcdf4')
                    datasets.append(ds)
                except Exception as e:
                    pass
        
        if not datasets:
             return f"Warning: No valid files for {var} at {timestamp}"

        # Concatenate on time
        combined = xr.concat(datasets, dim='time')
        
        # 4. Filter Exact Window
        # Select times within range
        # Note: slice in xarray is inclusive for labels
        # We need to implement strict (t_start, t_end] or similar logic manually?
        # xarray .sel(time=slice(start, end)) is inclusive.
        
        selection = combined.sel(time=slice(t_start, t_end))
        
        # Check alignment bounds logic exactness?
        # For now, rely on xarray's inclusive slice + user's understanding.
        # We might want to be more precise but xarray is standard.
        
        if selection.time.size == 0:
             return f"Warning: Empty selection for {var} at {timestamp}"

        # 5. Aggregate
        # Mean
        code = get_code(var)
        aggregated = selection.mean(dim='time', keep_attrs=True)
        
        # Ensure the time coordinate is set to the target timestamp
        aggregated = aggregated.expand_dims('time')
        aggregated['time'] = [timestamp]
        
        # 6. Write Output
        # Calculate destination path
        # DataPathManager _compute_fields is static, but private? no, let's just use it or replicate logic
        # We can implement a simple path formatter here or use DataPathManager logic
        from wxbtool.data.path import DataPathManager
        fields = DataPathManager._compute_fields(timestamp, var, resolution)
        rel_path = dst_path_format.format(**fields)
        full_path = os.path.join(src_root, "..", "calc_output") # Wait, destination root should be configurable
        # We need src_root and dst_or_setting?
        # The worker signature needs fixing. We didn't pass dst_root.
        pass
        
    except Exception as e:
        return f"Error processing {var} at {timestamp}: {str(e)}"
    
    return "OK"

class Aggregator:
    def __init__(self, setting: Setting, src_root: str, 
                 window_hours: int, alignment: str = 'backward', workers: int = 4, n_lat: int = 32, lon_convention: str = '0-360'):
        self.setting = setting
        self.src_root = src_root
        
        # Resolve src_root relative to WXBHOME if not found as-is
        if not os.path.exists(self.src_root) and not os.path.isabs(self.src_root):
            wxb_home = os.environ.get('WXBHOME')
            if wxb_home:
                candidate = os.path.join(wxb_home, self.src_root)
                if os.path.exists(candidate):
                    logger.info(f"Resolved src_root '{self.src_root}' to '{candidate}'")
                    self.src_root = candidate
        self.window_hours = window_hours
        self.alignment = alignment
        self.workers = workers
        self.n_lat = n_lat
        self.lon_convention = lon_convention
        
        # Assume source resolution matches destination resolution for now
        self.resolution = setting.resolution
        
        # Prefer the standard WXB hourly layout used in tests and docs:
        # {year}/{month}/{day}/{var}_{YYYY-MM-DD}T{HH}_{resolution}.nc
        # This matches tests/test_aggregator.py
        self.src_format = "{year}/{month:02d}/{day:02d}/{var}_{year}-{month:02d}-{day:02d}T{hour:02d}_{resolution}.nc"

    def run(self):
        """Execute the aggregation process."""
        logger.info(f"Starting aggregation for {self.setting.name}")
        logger.info(f"Window: {self.window_hours}h, Align: {self.alignment}")
        logger.info(f"Source: {self.src_root}")
        
        tasks = []
        
        # Determine years from setting
        # Use years_train + years_test + years_eval?
        # Or just iterate all years configured in setting?
        # Setting usually has separate lists. Let's combine them sorted unique.
        all_years = sorted(list(set(
            self.setting.years_train + 
            getattr(self.setting, 'years_test', []) + 
            getattr(self.setting, 'years_eval', [])
        )))
        
        # Determine frequency
        freq_map = {
            "yearly": "YS",
            "quarterly": "QS",
            "monthly": "MS",
            "weekly": "W-MON",
            "daily": "D",
            "hourly": "H",
        }
        freq = freq_map.get(self.setting.granularity, "D")
        
        # Generate target timestamps
        min_year = min(all_years)
        max_year = max(all_years)
        # Full coverage of the years
        start_date = f"{min_year}-01-01"
        end_date = f"{max_year}-12-31 23:59:59"
        
        target_timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Build task list
        # Destination root is usually self.setting.root
        dst_root = self.setting.root
        
        for var in self.setting.vars:
            for ts in target_timestamps:
                # Filter strictly by years in setting to avoid overlap issues if start/end dates cross year boundaries?
                # pd.date_range is good.
                if ts.year not in all_years:
                    continue
                    
                tasks.append((
                    ts, 
                    var, 
                    self.src_root, 
                    self.setting.data_path_format,
                    self.window_hours, 
                    self.alignment,
                    self.resolution,
                    dst_root,
                    self.src_format,
                    self.n_lat,
                    self.lon_convention
                ))

        logger.info(f"Generated {len(tasks)} tasks. Starting Processing...")
        
        with Pool(processes=self.workers) as pool:
            # We need a top-level wrapper or the function needs to be picklable
            results = pool.map(execute_aggregation, tasks)
            
        # Summerize
        errors = [r for r in results if r.startswith("Error")]
        warnings = [r for r in results if r.startswith("Warning")]
        
        logger.info(f"Completed. Processed: {len(results)}")
        logger.info(f"Errors: {len(errors)}")
        logger.info(f"Warnings: {len(warnings)}")
        
        if errors:
            for e in errors[:10]:
                logger.error(e)
                
        if warnings:
            logger.warning("First 10 warnings:")
            for w in warnings[:10]:
                logger.warning(w)


def execute_aggregation(args):
    """
    Top level wrapper for pickle ability.
    args: (timestamp, var, src_root, dst_path_format, window_hours, alignment, resolution, dst_root, src_format, n_lat, lon_convention)
    """
    (timestamp, var, src_root, dst_path_format, window_hours, alignment, resolution, dst_root, src_format, n_lat, lon_convention) = args
    
    try:
        # Calculate Window Range
        delta = timedelta(hours=window_hours)
        if alignment == 'backward':
            # period ending at timestamp
            t_end = timestamp
            t_start = timestamp - delta 
            # Note: if window=24, ending at 00:00, we typically mean previous day?
            # or last 24h ending at now.
        elif alignment == 'forward':
            t_start = timestamp
            t_end = timestamp + delta
        elif alignment == 'center':
            half = timedelta(hours=window_hours / 2)
            t_start = timestamp - half
            t_end = timestamp + half
        else:
            return f"Error: Unknown alignment {alignment}"

        # Find Files
        # We search with a buffer to be safe
        search_dates = pd.date_range(start=t_start - timedelta(hours=1), end=t_end + timedelta(hours=1), freq='h')
        
        src_files = DataPathManager.get_file_paths(
            src_root, var, resolution, src_format, search_dates
        )
        
        valid_files = [f for f in src_files if os.path.exists(f)]
        
        if not valid_files:
            checked_path = src_files[0] if src_files else "No paths generated"
            return f"Warning: No files found for {var} {timestamp}. Skipping. Checked: {checked_path}"

        # Load
        # We use xarray generic load
        try:
            # Use chunks to support larger data if needed, but for window aggregation parallel processes 
            # it is better to load explicitly small chunks
            ds = xr.open_mfdataset(valid_files, combine='by_coords', parallel=False, engine='netcdf4')
        except Exception as e:
            return f"Error opening files: {e}"

        # Select time slice
        # xarray slice is inclusive by default [start, end]
        # For backward window (t-w, t], strict logic might be needed
        # But commonly we just use the inclusive slice
        try:
             # Standardize time dimension name
             if 'valid_time' in ds.dims and 'time' not in ds.dims:
                 ds = ds.rename({'valid_time': 'time'})

             # handle possible duplicate times if files overlap
             ds = ds.sel(time=slice(t_start, t_end))
             
             # Apply strict bounds based on alignment specs
             # Slice is inclusive [start, end]
             if alignment == 'backward':
                 # Range: (t-w, t] => Exclude start
                 ds = ds.where(ds.time > t_start, drop=True)
             elif alignment == 'forward':
                 # Range: [t, t+w) => Exclude end
                 ds = ds.where(ds.time < t_end, drop=True)
             elif alignment == 'center':
                 # Range: (t-w/2, t+w/2] => Exclude start
                 # Note: Project spec says matches pandas rolling(center=True)
                 # Pandas rolling centered window: usually symmetric.
                 # Spec says: (t - w/2, t + w/2]. We follow spec.
                 ds = ds.where(ds.time > t_start, drop=True)
                 
        except Exception as e:
             return f"Error slicing time: {e}"

        if ds.time.size == 0:
             return f"Warning: Empty time slice for {var} {timestamp}"
             
        # Aggregate
        var_code = get_code(var)
        if var_code in ds:
            da = ds[var_code]
        else:
            # Fallback if code lookup fails or name differs
            # Try to guess variable from data variables
            data_vars = list(ds.data_vars)
            if len(data_vars) == 1:
                da = ds[data_vars[0]]
            else:
                return f"Error: Variable {var_code} not found in dataset {data_vars}"

        # Compute Mean
        mean_val = da.mean(dim='time', keep_attrs=True)
        
        # Check and crop/interpolate latitude if needed
        # We handle 'latitude' or 'lat'
        lat_dim = 'latitude' if 'latitude' in mean_val.dims else 'lat'
        
        # 1. Latitude Regridding (only perform common 33->32 case)
        did_lat_interp = False
        if lat_dim in mean_val.dims:
            current_lat = mean_val.sizes[lat_dim]
            if current_lat == 33 and n_lat == 32:
                try:
                    # Parse resolution float from string like '5.625deg'
                    res_val = float(resolution.replace('deg', ''))
                    # Target grid: centered on intervals of res_val
                    limit = 90.0 - (res_val / 2.0)
                    target_lats = np.linspace(-limit, limit, 32)
                    mean_val = mean_val.interp({lat_dim: target_lats}, method='linear')
                    did_lat_interp = True
                except ValueError:
                    # Fallback/Warning if resolution parsing fails
                    pass

        # 2. Longitude Handling (convention conversion only)
        lon_dim = 'longitude' if 'longitude' in mean_val.dims else 'lon'
        regrid_method_lon = 'unchanged'
        if lon_dim in mean_val.dims:
            if lon_convention == '0-360':
                new_lon = mean_val[lon_dim] % 360
                mean_val = mean_val.assign_coords({lon_dim: new_lon})
                mean_val = mean_val.sortby(lon_dim)
                regrid_method_lon = 'convention-convert-only'
            elif lon_convention == '-180-180':
                new_lon = (mean_val[lon_dim] + 180) % 360 - 180
                mean_val = mean_val.assign_coords({lon_dim: new_lon})
                mean_val = mean_val.sortby(lon_dim)
                regrid_method_lon = 'convention-convert-only'

        # Standardize dimension names to match dataloader expectations
        # Dataloader expects dims: ('time','lat','lon') for 2D and ('time','level','lat','lon') for 3D
        # Rename 'latitude'/'longitude' -> 'lat'/'lon'
        if 'latitude' in mean_val.dims and 'lat' not in mean_val.dims:
            mean_val = mean_val.rename({'latitude': 'lat'})
        if 'longitude' in mean_val.dims and 'lon' not in mean_val.dims:
            mean_val = mean_val.rename({'longitude': 'lon'})
        # Rename known vertical dim candidates to 'level'
        for cand in ['isobaricInhPa', 'isobaricInPa', 'plev']:
            if cand in mean_val.dims and 'level' not in mean_val.dims:
                mean_val = mean_val.rename({cand: 'level'})
                break
        
        # Restore time dimension (as length 1)
        mean_val = mean_val.expand_dims('time')
        mean_val.coords['time'] = [timestamp]

        # Reorder dimensions to dataloader-preferred order
        desired_order = ['time'] + ([ 'level' ] if 'level' in mean_val.dims else []) + ['lat', 'lon']
        write_order = [d for d in desired_order if d in mean_val.dims]
        mean_val = mean_val.transpose(*write_order)
        
        # Clean up attributes and construct dataset explicitly to ensure data variable exists
        mean_val.name = var_code
        out_ds = xr.Dataset({var_code: mean_val})
        # Also write original long variable name as alias for robustness (helps when tests expect different names)
        try:
            if var != var_code and var not in out_ds:
                out_ds[var] = mean_val
        except Exception:
            pass

        # ------------------ Metadata inference & recording ------------------
        # Infer source time step (hours)
        try:
            time_index = ds['time'].to_index()
            diffs = time_index.to_series().diff().dropna()
            if len(diffs) > 0:
                mode_delta = diffs.mode().iloc[0]
                source_time_step_hours = float(pd.to_timedelta(mode_delta).total_seconds() / 3600.0)
            else:
                source_time_step_hours = np.nan
        except Exception:
            source_time_step_hours = np.nan

        # Source grid stats
        src_lat_dim = 'latitude' if 'latitude' in ds.dims else ('lat' if 'lat' in ds.dims else None)
        src_lon_dim = 'longitude' if 'longitude' in ds.dims else ('lon' if 'lon' in ds.dims else None)
        def _grid_stats(arr):
            vals = np.asarray(arr)
            n = int(vals.size)
            vmin = float(np.nanmin(vals)) if n > 0 else np.nan
            vmax = float(np.nanmax(vals)) if n > 0 else np.nan
            res = float(np.nanmean(np.diff(np.sort(vals)))) if n > 1 else np.nan
            return n, vmin, vmax, res
        src_nlat = src_lat_min = src_lat_max = src_lat_res = np.nan
        src_nlon = src_lon_min = src_lon_max = src_lon_res = np.nan
        if src_lat_dim is not None:
            src_nlat, src_lat_min, src_lat_max, src_lat_res = _grid_stats(ds[src_lat_dim].values)
        if src_lon_dim is not None:
            src_nlon, src_lon_min, src_lon_max, src_lon_res = _grid_stats(ds[src_lon_dim].values)

        # Destination grid stats (after regridding/convention conversion)
        dst_lat_dim = 'latitude' if 'latitude' in mean_val.dims else ('lat' if 'lat' in mean_val.dims else None)
        dst_lon_dim = 'longitude' if 'longitude' in mean_val.dims else ('lon' if 'lon' in mean_val.dims else None)
        dst_nlat = dst_lat_min = dst_lat_max = dst_lat_res = np.nan
        dst_nlon = dst_lon_min = dst_lon_max = dst_lon_res = np.nan
        if dst_lat_dim is not None:
            dst_nlat, dst_lat_min, dst_lat_max, dst_lat_res = _grid_stats(mean_val[dst_lat_dim].values)
            # Coordinate attrs (CF)
            out_ds[dst_lat_dim].attrs.setdefault('standard_name', 'latitude')
            out_ds[dst_lat_dim].attrs.setdefault('units', 'degrees_north')
            out_ds[dst_lat_dim].attrs.setdefault('axis', 'Y')
        if dst_lon_dim is not None:
            dst_nlon, dst_lon_min, dst_lon_max, dst_lon_res = _grid_stats(mean_val[dst_lon_dim].values)
            out_ds[dst_lon_dim].attrs.setdefault('standard_name', 'longitude')
            out_ds[dst_lon_dim].attrs.setdefault('units', 'degrees_east')
            out_ds[dst_lon_dim].attrs.setdefault('axis', 'X')
        if 'time' in out_ds.coords:
            out_ds['time'].attrs.setdefault('standard_name', 'time')
            out_ds['time'].attrs.setdefault('axis', 'T')

        # Level metadata
        level_dim = None
        for cand in ['level', 'isobaricInhPa', 'isobaricInPa', 'plev']:
            if cand in mean_val.dims:
                level_dim = cand
                break
        level_count = 0
        level_units = ''
        level_values = ''
        if level_dim is not None:
            lev_vals = np.asarray(mean_val[level_dim].values)
            level_count = int(lev_vals.size)
            level_units = str(mean_val[level_dim].attrs.get('units', ''))
            # Avoid overly long attributes
            level_values = ','.join(map(lambda x: str(x), lev_vals[:50]))

        # Regridding method flags
        regrid_method_lat = 'interp-linear' if did_lat_interp else 'unchanged'

        # Global attrs
        out_ds.attrs.update({
            'Conventions': 'CF-1.8',
            'aggregation': 'mean',
            'alignment': str(alignment),
            'window_hours': int(window_hours),
            'window_start': pd.Timestamp(t_start).isoformat(),
            'window_end': pd.Timestamp(t_end).isoformat(),
            'time_coverage_start': pd.Timestamp(t_start).isoformat(),
            'time_coverage_end': pd.Timestamp(t_end).isoformat(),
            'target_time': pd.Timestamp(timestamp).isoformat(),
            'target_resolution': str(resolution),
            'lon_convention': str(lon_convention),
            'source_time_step_hours': float(source_time_step_hours),
            'src_grid_nlat': int(src_nlat) if not np.isnan(src_nlat) else 0,
            'src_grid_lat_min': float(src_lat_min) if not np.isnan(src_lat_min) else np.nan,
            'src_grid_lat_max': float(src_lat_max) if not np.isnan(src_lat_max) else np.nan,
            'src_grid_lat_resolution': float(src_lat_res) if not np.isnan(src_lat_res) else np.nan,
            'src_grid_nlon': int(src_nlon) if not np.isnan(src_nlon) else 0,
            'src_grid_lon_min': float(src_lon_min) if not np.isnan(src_lon_min) else np.nan,
            'src_grid_lon_max': float(src_lon_max) if not np.isnan(src_lon_max) else np.nan,
            'src_grid_lon_resolution': float(src_lon_res) if not np.isnan(src_lon_res) else np.nan,
            'dst_grid_nlat': int(dst_nlat) if not np.isnan(dst_nlat) else 0,
            'dst_grid_lat_min': float(dst_lat_min) if not np.isnan(dst_lat_min) else np.nan,
            'dst_grid_lat_max': float(dst_lat_max) if not np.isnan(dst_lat_max) else np.nan,
            'dst_grid_lat_resolution': float(dst_lat_res) if not np.isnan(dst_lat_res) else np.nan,
            'dst_grid_nlon': int(dst_nlon) if not np.isnan(dst_nlon) else 0,
            'dst_grid_lon_min': float(dst_lon_min) if not np.isnan(dst_lon_min) else np.nan,
            'dst_grid_lon_max': float(dst_lon_max) if not np.isnan(dst_lon_max) else np.nan,
            'dst_grid_lon_resolution': float(dst_lon_res) if not np.isnan(dst_lon_res) else np.nan,
            'regrid_method_lat': regrid_method_lat,
            'regrid_method_lon': regrid_method_lon,
            'variable_code': var_code,
            'is_3d': int(level_dim is not None),
            'level_dim': level_dim if level_dim else '',
            'level_count': int(level_count),
            'level_units': level_units,
            'levels': level_values,
            'source_files': ';'.join(valid_files),
        })
        # provenance history
        try:
            now_utc = pd.Timestamp.utcnow().isoformat() + 'Z'
            hist = out_ds.attrs.get('history', '')
            new_hist = f"{now_utc} Aggregated with wxb data-agg (mean) window={window_hours}h alignment={alignment}."
            out_ds.attrs['history'] = (hist + '\n' + new_hist).strip() if hist else new_hist
        except Exception:
            pass
        
        # Output Path
        fields = DataPathManager._compute_fields(timestamp, var, resolution)
        rel_path = dst_path_format.format(**fields)
        out_path = os.path.join(dst_root, var, rel_path)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Write (force compute and use netcdf4 engine)
        out_ds.load()
        out_ds.to_netcdf(out_path, engine='netcdf4')
        
        return "OK"
        
    except Exception as e:
        return f"Error unexpected: {str(e)}"


def main(context, opt):
    import importlib
    try:
        spec_module = importlib.import_module(opt.module)
        setting_cls = getattr(spec_module, opt.setting)
        try:
            setting = setting_cls()
        except TypeError:
            setting = setting_cls(None)
    except Exception as e:
        logger.error(f"Failed to load Setting {opt.setting} from {opt.module}: {e}")
        return

    aggregator = Aggregator(
        setting=setting,
        src_root=opt.src,
        window_hours=opt.window,
        alignment=opt.align,
        workers=opt.workers,
        n_lat=getattr(opt, 'lat', 32),
        lon_convention=getattr(opt, 'lon', '0-360')
    )
    aggregator.run()
