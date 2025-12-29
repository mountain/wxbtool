# -*- coding: utf-8 -*-

import hashlib
import http.client
import json
import logging
import os
import os.path as path
import random
import socket

import msgpack
import msgpack_numpy as m
import numpy as np
import pandas as pd
import requests
import xarray as xr

from wxbtool.data.path import DataPathManager
from wxbtool.core.setting import Setting

m.patch()

from torch.utils.data import DataLoader, Dataset, Sampler  # noqa: E402

logger = logging.getLogger()


class WindowArray(type(np.zeros(0, dtype=np.float32))):
    def __new__(subtype, orig, shift=0, step=1):
        shape = [orig.shape[_] for _ in range(len(orig.shape))]
        self = np.ndarray.__new__(
            subtype, shape, dtype=np.float32, buffer=orig.tobytes()
        )[shift::step]
        return self


class WxDataset(Dataset):
    def __init__(
        self,
        root=None,
        resolution=None,
        years=None,
        vars=None,
        levels=None,
        step=None,
        input_span=None,
        pred_shift=None,
        pred_span=None,
        granularity=None,
        data_path_format=None,
        setting=None,
    ):
        self.setting = setting if setting is not None else Setting()

        # Use values from setting if not explicitly provided
        self.root = root if root is not None else self.setting.root
        self.resolution = (
            resolution if resolution is not None else self.setting.resolution
        )
        self.input_span = (
            input_span if input_span is not None else self.setting.input_span
        )
        self.step = step if step is not None else self.setting.step
        self.pred_shift = (
            pred_shift if pred_shift is not None else self.setting.pred_shift
        )
        self.pred_span = pred_span if pred_span is not None else self.setting.pred_span
        self.years = years if years is not None else self.setting.years_train
        self.vars = vars if vars is not None else self.setting.vars
        self.levels = levels if levels is not None else self.setting.levels
        self.granularity = (
            granularity if granularity is not None else self.setting.granularity
        )
        self.data_path_format = (
            data_path_format
            if data_path_format is not None
            else self.setting.data_path_format
        )
        self.inputs = {}
        self.targets = {}
        self.shapes = {
            "data": {},
        }
        self.accumulated = {}
        self.active_vars = []

        if resolution == "5.625deg":
            self.height = 13
            self.width = 32
            self.length = 64

        code = "%s:%s:%s:%s:%s:%s:%s:%s:%s:%s" % (
            self.resolution,
            self.years,
            self.vars,
            self.levels,
            self.step,
            self.input_span,
            self.pred_shift,
            self.pred_span,
            self.granularity,
            self.data_path_format,
        )
        hashstr = hashlib.md5(code.encode("utf-8")).hexdigest()
        self.hashcode = hashstr

        dumpdir = path.abspath("%s/.cache/%s" % (self.root, hashstr))
        if not path.exists(dumpdir):
            os.makedirs(dumpdir, exist_ok=True)
        self.load(dumpdir)

        self.memmap(dumpdir)

    def load(self, dumpdir):
        import wxbtool.data.variables as v  # noqa: E402

        # Load existing shapes metadata if present (for cache hit/skip)
        shapes_path = f"{dumpdir}/shapes.json"
        if os.path.exists(shapes_path):
            try:
                with open(shapes_path) as fp:
                    self.shapes = json.load(fp)
                logger.info("Loaded existing shapes metadata from %s", shapes_path)
                if "data" in self.shapes and all(
                    var in self.shapes["data"] for var in self.vars
                ):
                    logger.info("cache found, skip loading")
                    self.active_vars.extend(
                        [var for var in self.vars if var in self.shapes["data"]]
                    )
                    return
            except Exception as e:
                logger.warning("Failed to load shapes metadata %s: %s", shapes_path, e)

        # Determine available levels only if any of the requested variables are 3D
        all_levels = []
        # Identify 3D variables present in the current dataset variables (self.vars)
        var3d_list = [var for var in self.vars if v.is_var3d(var)]
        if var3d_list:
            first3d = var3d_list[0]
            try:
                var3d_path = os.path.join(self.root, first3d)
                # Filter out hidden files like .DS_Store
                files = [
                    f
                    for f in os.listdir(var3d_path)
                    if not f.startswith(".")
                    and os.path.isfile(os.path.join(var3d_path, f))
                ]
                files.sort()  # Ensure deterministic order
                if not files:
                    raise FileNotFoundError(f"No valid data files found in {var3d_path}")
                any_file = files[0]
                sample_data = xr.open_dataarray(f"{var3d_path}/{any_file}", engine="netcdf4")
                all_levels = sample_data.level.values.tolist()
            except (FileNotFoundError, IndexError, AttributeError) as e:
                logger.error(
                    f"Failed to read levels from 3D variable '{first3d}'. "
                    f"Please check data integrity and paths. Original error: {e}"
                )
                raise ValueError(
                    f"Configuration requires 3D variables in dataset (found {var3d_list}), "
                    f"but failed to load level info from '{first3d}'."
                )

        # Find indices of requested levels in the available levels
        levels_selector = []
        for lvl in self.levels:
            try:
                levels_selector.append(all_levels.index(float(lvl)))
            except ValueError:
                logger.error(f"Level {lvl} not found in available levels: {all_levels}")
                raise ValueError(
                    f"Level {lvl} not found in available levels: {all_levels}"
                )

        selector = np.array(levels_selector, dtype=np.int64)

        size = 0
        last_loaded_var = None

        # Construct a date range covering selected years according to granularity
        min_year, max_year = min(self.years), max(self.years)
        start_date = f"{min_year}-01-01"
        end_date = f"{max_year}-12-31"
        freq_map = {
            "yearly": "YS",
            "quarterly": "QS",
            "monthly": "MS",
            "weekly": "W-MON",
            "daily": "D",
            "hourly": "H",
        }
        freq = freq_map.get(self.setting.granularity, "D")
        if self.setting.granularity not in freq_map:
            logger.warning(
                f"Unknown granularity '{self.setting.granularity}', defaulting to daily frequency 'D'"
            )
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

        # Diagnostics: show effective granularity/format and variable list
        logger.info(
            "Dataset discovery with granularity=%s, data_path_format='%s', years=%s",
            self.setting.granularity,
            self.setting.data_path_format,
            self.years,
        )
        logger.info("Variables requested: %s", self.vars)

        for var in self.vars:
            # Flush previously loaded variable (if any) before switching vars
            if (
                last_loaded_var
                and last_loaded_var in self.accumulated
                and last_loaded_var != var
            ):
                self.inputs[last_loaded_var] = WindowArray(
                    self.accumulated[last_loaded_var],
                    shift=self.input_span * self.step,
                    step=self.step,
                )
                self.targets[last_loaded_var] = WindowArray(
                    self.accumulated[last_loaded_var],
                    shift=self.pred_span * self.step + self.pred_shift,
                    step=self.step,
                )
                self.dump_var(dumpdir, last_loaded_var)

            # If cache file exists and shapes metadata already has this var, skip rebuild
            cached_npy = os.path.join(dumpdir, f"{var}.npy")
            if (
                isinstance(self.shapes, dict)
                and "data" in self.shapes
                and var in self.shapes["data"]
                and os.path.exists(cached_npy)
            ):
                logger.info(
                    "Cache hit for %s: using existing cache %s", var, cached_npy
                )
                self.active_vars.append(var)
                last_loaded_var = var
                continue

            # Try loading using multiple directory/token candidates while keeping the
            # data variable key resolved from the original var via v.get_code(var).
            candidates = [var]
            try:
                alias_name = v.resolve_name(var)
                if alias_name != var:
                    candidates.append(alias_name)
            except Exception:
                pass
            try:
                code_name = v.get_code(var)
                if code_name not in candidates:
                    candidates.append(code_name)
            except Exception:
                pass

            found_any = False
            tried_total = 0
            examples_missing = []

            for cand in candidates:
                file_paths = DataPathManager.get_file_paths(
                    self.root,
                    cand,
                    self.resolution,
                    self.setting.data_path_format,
                    date_range,
                )
                tried_total += len(file_paths)
                for data_path in file_paths:
                    if not os.path.exists(data_path):
                        if len(examples_missing) < 3:
                            examples_missing.append(data_path)
                        logger.debug(f"Missing data file skipped: {data_path}")
                        continue

                    if v.is_var3d(var):
                        length = self.load_3ddata(
                            data_path, var, selector, self.accumulated
                        )
                    else:
                        # Treat as 2D if we can resolve a code; this is robust to alias/renamed variables.
                        try:
                            _ = v.get_code(var)
                            length = self.load_2ddata(data_path, var, self.accumulated)
                        except KeyError:
                            raise ValueError(f"variable {var} does not supported!")
                    size += length
                    found_any = True

                if found_any:
                    break  # No need to try further candidates once loaded

            if not found_any:
                msg = (
                    f"No existing files found for variable '{var}' under root '{self.root}'. "
                    f"Tried {tried_total} candidate path(s) across {len(candidates)} name forms {candidates}. "
                )
                if examples_missing:
                    msg += "Examples (first 3): " + "; ".join(examples_missing)
                logger.warning(msg)

            if var in self.accumulated:
                self.active_vars.append(var)
                last_loaded_var = var

        if last_loaded_var and last_loaded_var in self.accumulated:
            self.inputs[last_loaded_var] = WindowArray(
                self.accumulated[last_loaded_var],
                shift=self.input_span * self.step,
                step=self.step,
            )
            self.targets[last_loaded_var] = WindowArray(
                self.accumulated[last_loaded_var],
                shift=self.pred_span * self.step + self.pred_shift,
                step=self.step,
            )
            self.dump_var(dumpdir, last_loaded_var)

        # If no in-memory accumulation was done but we already have active_vars from cache hits,
        # defer loading to memmap() instead of failing here.
        if not self.accumulated and len(self.active_vars) == 0:
            logger.error("No data accumulated. Please check the data loading process.")
            raise ValueError(
                "No data accumulated. Ensure that data is correctly loaded and accumulated."
            )

        # Only check in-memory lengths consistency when we actually accumulated data in-memory.
        # When we rely entirely on cache hits (active_vars > 0) and no new accumulation, defer to memmap().
        if self.accumulated:
            lengths = {var: acc.shape[0] for var, acc in self.accumulated.items()}
            unique_lengths = set(lengths.values())

            if len(unique_lengths) != 1:
                max_length = max(unique_lengths)

                inconsistent_vars = {
                    var: length
                    for var, length in lengths.items()
                    if length != max_length
                }

                if inconsistent_vars:
                    for var, length in inconsistent_vars.items():
                        logger.error(
                            f"Variable {var} has inconsistent length {length}. Expected length: {max_length}."
                        )

                    raise ValueError(
                        "Inconsistent data lengths across variables detected. Please check the data loading process."
                    )
            else:
                logger.info("All variables have consistent data lengths.")
        else:
            logger.info(
                "No in-memory accumulation; relying on cache for vars: %s",
                self.active_vars,
            )

        with open("%s/shapes.json" % dumpdir, mode="w") as fp:
            json.dump(self.shapes, fp)

        self.size = size // max(1, len(self.active_vars))
        logger.info("total %s items loaded!", self.size)

        for var in list(self.accumulated.keys()):
            del self.accumulated[var]

    def load_2ddata(self, data_path, var, accumulated):
        import wxbtool.data.variables as v  # noqa: E402

        with xr.open_dataset(data_path, engine="netcdf4") as ds:
            ds = ds.transpose("time", "lat", "lon")
            if var not in accumulated:
                accumulated[var] = np.array(ds[v.get_code(var)].data, dtype=np.float32)
            else:
                accumulated[var] = np.concatenate(
                    [
                        accumulated[var],
                        np.array(ds[v.get_code(var)].data, dtype=np.float32),
                    ],
                    axis=0,
                )
            logger.info(
                "%s[%s]: %s",
                var,
                os.path.basename(data_path),
                str(accumulated[var].shape),
            )

        return accumulated[var].shape[0]

    def load_3ddata(self, data_path, var, selector, accumulated):
        import wxbtool.data.variables as v  # noqa: E402

        with xr.open_dataset(data_path, engine="netcdf4") as ds:
            ds = ds.transpose("time", "level", "lat", "lon")
            if var not in accumulated:
                accumulated[var] = np.array(ds[v.get_code(var)].data, dtype=np.float32)[
                    :, selector, :, :
                ]
            else:
                accumulated[var] = np.concatenate(
                    [
                        accumulated[var],
                        np.array(ds[v.get_code(var)].data, dtype=np.float32)[
                            :, selector, :, :
                        ],
                    ],
                    axis=0,
                )
            logger.info(
                "%s[%s]: %s",
                var,
                os.path.basename(data_path),
                str(accumulated[var].shape),
            )

        return accumulated[var].shape[0]

    def dump_var(self, dumpdir, var):
        file_dump = "%s/%s.npy" % (dumpdir, var)
        # Ensure dict structure exists
        if "data" not in self.shapes:
            self.shapes["data"] = {}
        # Update shape metadata and write array
        self.shapes["data"][var] = self.accumulated[var].shape
        np.save(file_dump, self.accumulated[var])
        # Persist shapes metadata incrementally after each variable
        try:
            with open(f"{dumpdir}/shapes.json", mode="w") as fp:
                json.dump(self.shapes, fp)
        except Exception as e:
            logger.warning("Failed to write shapes metadata for %s: %s", var, e)

    def memmap(self, dumpdir):
        import wxbtool.data.variables as v  # noqa: E402

        with open("%s/shapes.json" % dumpdir) as fp:
            shapes = json.load(fp)

        self.active_vars = [var for var in self.vars if var in shapes["data"]]
        for var in self.active_vars:
            file_dump = "%s/%s.npy" % (dumpdir, var)

            # load data from memmap, and skip the first shift elements of mmap data header
            shape = shapes["data"][var]
            total_size = np.prod(shape)
            data = np.memmap(file_dump, dtype=np.float32, mode="r")
            shift = data.shape[0] - total_size
            self.accumulated[var] = np.reshape(data[shift:], shape)

            if v.is_var2d(var) or v.is_var3d(var):
                self.inputs[var] = self.accumulated[var]
                self.targets[var] = self.accumulated[var]

    def __len__(self):
        length = (
            self.accumulated[self.active_vars[0]].shape[0]
            - (self.input_span - 1) * self.step
            - (self.pred_span - 1) * self.step
            - self.pred_shift
        )
        logger.info(f"Dataset length: {length}")
        return length

    def __getitem__(self, item):
        import wxbtool.data.variables as v  # noqa: E402

        inputs, targets = {}, {}
        for var in self.active_vars:
            if v.is_var2d(var) or v.is_var3d(var):
                input_slice = self.inputs[var][item :: self.step][: self.input_span]
                target_slice = self.targets[var][
                    item
                    + self.step * (self.input_span - 1)
                    + self.pred_shift :: self.step
                ][: self.pred_span]
                inputs[var] = input_slice
                targets[var] = target_slice
                if input_slice.shape[0] != self.input_span:
                    logger.warning(
                        f"Input slice for var {var} at index {item} has shape {input_slice.shape}"
                    )
                if target_slice.shape[0] != self.pred_span:
                    logger.warning(
                        f"Target slice for var {var} at index {item} has shape {target_slice.shape}"
                    )

        return inputs, targets, item


class WxDatasetClient(Dataset):
    def __init__(
        self,
        url,
        phase,
        resolution=None,
        years=None,
        vars=None,
        levels=None,
        step=None,
        input_span=None,
        pred_shift=None,
        pred_span=None,
        granularity="daily",
        data_path_format="default",
        setting=None,
    ):
        self.url = url
        self.phase = phase
        self.setting = setting if setting is not None else Setting()

        # Use values from setting if not explicitly provided
        self.resolution = (
            resolution if resolution is not None else self.setting.resolution
        )
        self.step = step if step is not None else self.setting.step
        self.input_span = (
            input_span if input_span is not None else self.setting.input_span
        )
        self.pred_shift = (
            pred_shift if pred_shift is not None else self.setting.pred_shift
        )
        self.pred_span = pred_span if pred_span is not None else self.setting.pred_span
        self.years = years if years is not None else self.setting.years_train
        self.vars = vars if vars is not None else self.setting.vars
        self.levels = levels if levels is not None else self.setting.levels
        self.granularity = granularity
        self.data_path_format = data_path_format

        code = "%s:%s:%s:%s:%s:%s:%s:%s:%s:%s" % (
            self.resolution,
            self.years,
            self.vars,
            self.levels,
            self.step,
            self.input_span,
            self.pred_shift,
            self.pred_span,
            self.granularity,
            self.data_path_format,
        )
        self.hashcode = hashlib.md5(code.encode("utf-8")).hexdigest()

        if self.url.startswith("unix:"):
            self.url = self.url.replace("/", "%2F")
            self.url = self.url.replace("unix:", "http+unix://")

    def __len__(self):
        url = "%s/%s/%s" % (self.url, self.hashcode, self.phase)
        if self.url.startswith("http+unix://"):
            sock_path = self.url.replace("http+unix://", "").replace("%2F", "/")
            endpoint = f"{self.hashcode}/{self.phase}"
            sock_path = "/" + sock_path
            conn = http.client.HTTPConnection("localhost")
            conn.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.sock.connect(sock_path)
            conn.request(
                "GET",
                "/" + endpoint,
                headers={"Host": "localhost", "Connection": "close"},
            )
            r = conn.getresponse()
            if r.status != 200:
                raise Exception("http error %s: %s" % (r.status, r.reason))
            data = msgpack.loads(r.read())
        else:
            r = requests.get(url)
            if r.status_code != 200:
                raise Exception("http error %s: %s" % (r.status_code, r.text))
            data = msgpack.loads(r.content)

        return data["size"]

    def __getitem__(self, item):
        url = "%s/%s/%s/%d" % (self.url, self.hashcode, self.phase, item)
        if self.url.startswith("http+unix://"):
            sock_path = self.url.replace("http+unix://", "").replace("%2F", "/")
            endpoint = f"{self.hashcode}/{self.phase}/{item}"
            sock_path = "/" + sock_path
            conn = http.client.HTTPConnection("localhost")
            conn.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.sock.connect(sock_path)
            conn.request(
                "GET",
                "/" + endpoint,
                headers={"Host": "localhost", "Connection": "close"},
            )
            r = conn.getresponse()
            if r.status != 200:
                raise Exception("http error %s: %s" % (r.status, r.reason))
            data = msgpack.loads(r.read())
        else:
            r = requests.get(url)
            if r.status_code != 200:
                raise Exception("http error %s: %s" % (r.status_code, r.text))
            data = msgpack.loads(r.content)

        for key, val in data.items():
            if key != "inputs" and key != "targets":
                continue
            for var, blk in val.items():
                val[var] = np.array(np.copy(blk), dtype=np.float32)

        return data["inputs"], data["targets"], item


class EnsembleBatchSampler(Sampler):
    def __init__(self, dataset, ensemble_size, shuffle=True):
        super().__init__()
        self.dataset = dataset
        self.ensemble_size = ensemble_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        for idx in self.indices:
            for _ in range(self.ensemble_size):
                yield idx

    def __len__(self):
        return len(self.dataset) * self.ensemble_size


def ensemble_loader(dataset, ensemble_size, shuffle=True):
    sampler = EnsembleBatchSampler(dataset, ensemble_size, shuffle)
    return DataLoader(
        dataset,
        batch_size=ensemble_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )
