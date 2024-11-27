# -*- coding: utf-8 -*-

import os
import os.path as path
import hashlib
import requests
import logging
import json
import http.client
import socket
import random

import xarray as xr
import numpy as np

import msgpack
import msgpack_numpy as m

m.patch()

from itertools import product  # noqa: E402
from torch.utils.data import Dataset, DataLoader, Sampler  # noqa: E402

from wxbtool.data.variables import codes, vars2d, vars3d  # noqa: E402

logger = logging.getLogger()

all_levels = [
    "50",
    "100",
    "150",
    "200",
    "250",
    "300",
    "400",
    "500",
    "600",
    "700",
    "850",
    "925",
    "1000",
]


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
        root,
        resolution,
        years,
        vars,
        levels,
        step=1,
        input_span=2,
        pred_shift=24,
        pred_span=1,
    ):
        self.root = root
        self.resolution = resolution
        self.input_span = input_span
        self.step = step
        self.pred_shift = pred_shift
        self.pred_span = pred_span
        self.years = years
        self.vars = vars
        self.levels = levels
        self.inputs = {}
        self.targets = {}
        self.shapes = {
            "data": {},
        }
        self.accumulated = {}

        if resolution == "5.625deg":
            self.height = 13
            self.width = 32
            self.length = 64

        code = "%s:%s:%s:%s:%s:%s:%s:%s" % (
            resolution,
            years,
            vars,
            levels,
            step,
            input_span,
            pred_shift,
            pred_span,
        )
        hashstr = hashlib.md5(code.encode("utf-8")).hexdigest()
        self.hashcode = hashstr

        dumpdir = path.abspath("%s/.cache/%s" % (self.root, hashstr))
        if not path.exists(dumpdir):
            os.makedirs(dumpdir)
            self.load(dumpdir)

        self.memmap(dumpdir)

    def load(self, dumpdir):
        levels_selector = [all_levels.index(lvl) for lvl in self.levels]
        selector = np.array(levels_selector, dtype=np.int64)

        size = 0
        lastvar = None
        for var, yr in product(self.vars, self.years):
            if var in vars3d:
                length = self.load_3ddata(yr, var, selector, self.accumulated)
            elif var in vars2d:
                length = self.load_2ddata(yr, var, self.accumulated)
            else:
                raise ValueError("variable %s does not supported!" % var)
            size += length

            if lastvar and lastvar != var:
                self.inputs[lastvar] = WindowArray(
                    self.accumulated[lastvar],
                    shift=self.input_span * self.step,
                    step=self.step,
                )
                self.targets[lastvar] = WindowArray(
                    self.accumulated[lastvar],
                    shift=self.pred_span * self.step + self.pred_shift,
                    step=self.step,
                )
                self.dump_var(dumpdir, lastvar)

            lastvar = var

        if lastvar:
            self.inputs[lastvar] = WindowArray(
                self.accumulated[lastvar],
                shift=self.input_span * self.step,
                step=self.step,
            )
            self.targets[lastvar] = WindowArray(
                self.accumulated[lastvar],
                shift=self.pred_span * self.step + self.pred_shift,
                step=self.step,
            )
            self.dump_var(dumpdir, lastvar)

        if not self.accumulated:
            logger.error("No data accumulated. Please check the data loading process.")
            raise ValueError(
                "No data accumulated. Ensure that data is correctly loaded and accumulated."
            )

        lengths = {var: acc.shape[0] for var, acc in self.accumulated.items()}
        unique_lengths = set(lengths.values())

        if len(unique_lengths) != 1:
            max_length = max(unique_lengths)

            inconsistent_vars = {
                var: length for var, length in lengths.items() if length != max_length
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

        with open("%s/shapes.json" % dumpdir, mode="w") as fp:
            json.dump(self.shapes, fp)

        self.size = size // len(self.vars)
        logger.info("total %s items loaded!", self.size)

        for var in list(self.accumulated.keys()):
            del self.accumulated[var]

    def load_2ddata(self, year, var, accumulated):
        data_path = "%s/%s/%s_%d_%s.nc" % (self.root, var, var, year, self.resolution)
        with xr.open_dataset(data_path) as ds:
            ds = ds.transpose("time", "lat", "lon")
            if var not in accumulated:
                accumulated[var] = np.array(ds[codes[var]].data, dtype=np.float32)
            else:
                accumulated[var] = np.concatenate(
                    [accumulated[var], np.array(ds[codes[var]].data, dtype=np.float32)],
                    axis=0,
                )
            logger.info("%s[%d]: %s", var, year, str(accumulated[var].shape))

        return accumulated[var].shape[0]

    def load_3ddata(self, year, var, selector, accumulated):
        data_path = "%s/%s/%s_%d_%s.nc" % (self.root, var, var, year, self.resolution)
        with xr.open_dataset(data_path) as ds:
            ds = ds.transpose("time", "level", "lat", "lon")
            if var not in accumulated:
                accumulated[var] = np.array(ds[codes[var]].data, dtype=np.float32)[
                    :, selector, :, :
                ]
            else:
                accumulated[var] = np.concatenate(
                    [
                        accumulated[var],
                        np.array(ds[codes[var]].data, dtype=np.float32)[
                            :, selector, :, :
                        ],
                    ],
                    axis=0,
                )
            logger.info("%s[%d]: %s", var, year, str(accumulated[var].shape))

        return accumulated[var].shape[0]

    def dump_var(self, dumpdir, var):
        file_dump = "%s/%s.npy" % (dumpdir, var)
        self.shapes["data"][var] = self.accumulated[var].shape
        np.save(file_dump, self.accumulated[var])

    def memmap(self, dumpdir):
        with open("%s/shapes.json" % dumpdir) as fp:
            shapes = json.load(fp)

        for var in self.vars:
            file_dump = "%s/%s.npy" % (dumpdir, var)

            # load data from memmap, and skip the first shift elements of mmap data header
            shape = shapes["data"][var]
            total_size = np.prod(shape)
            data = np.memmap(file_dump, dtype=np.float32, mode="r")
            shift = data.shape[0] - total_size
            self.accumulated[var] = np.reshape(data[shift:], shape)

            if var in vars2d or var in vars3d:
                self.inputs[var] = self.accumulated[var]
                self.targets[var] = self.accumulated[var]

    def __len__(self):
        length = (
            self.accumulated[self.vars[0]].shape[0]
            - (self.input_span - 1) * self.step
            - (self.pred_span - 1) * self.step
            - self.pred_shift
        )
        logger.info(f"Dataset length: {length}")
        return length

    def __getitem__(self, item):
        inputs, targets = {}, {}
        for var in self.vars:
            if var in vars2d or var in vars3d:
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

        return inputs, targets


class WxDatasetClient(Dataset):
    def __init__(
        self,
        url,
        phase,
        resolution,
        years,
        vars,
        levels,
        step=1,
        input_span=2,
        pred_shift=24,
        pred_span=1,
    ):
        self.url = url
        self.phase = phase

        code = "%s:%s:%s:%s:%s:%s:%s:%s" % (
            resolution,
            years,
            vars,
            levels,
            step,
            input_span,
            pred_shift,
            pred_span,
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
            conn.request("GET", "/" + endpoint, headers={"Host": "localhost", "Connection": "close"})
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
            conn.request("GET", "/" + endpoint, headers={"Host": "localhost", "Connection": "close"})
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
            for var, blk in val.items():
                val[var] = np.array(np.copy(blk), dtype=np.float32)

        return data["inputs"], data["targets"]


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
