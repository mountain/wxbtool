# -*- coding: utf-8 -*-

import os
import os.path as path
import hashlib
import requests
import logging
import json

import xarray as xr
import numpy as np

import msgpack
import msgpack_numpy as m

m.patch()

from itertools import product  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

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

        with open("%s/shapes.json" % dumpdir, mode="w") as fp:
            json.dump(self.shapes, fp)

        self.size = size // len(self.vars)
        logger.info("total %s items loaded!", self.size)

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
        del self.accumulated[var]

    def memmap(self, dumpdir):
        with open("%s/shapes.json" % dumpdir) as fp:
            shapes = json.load(fp)

        for var in self.vars:
            file_dump = "%s/%s.npy" % (dumpdir, var)

            shape = shapes["data"][var]
            total_size = np.prod(shape)
            data = np.memmap(file_dump, dtype=np.float32, mode="r")
            shift = data.shape[0] - total_size
            self.accumulated[var] = np.reshape(data[shift:], shape)

            if var in vars2d or var in vars3d:
                self.inputs[var] = WindowArray(
                    self.accumulated[var],
                    shift=self.input_span * self.step,
                    step=self.step,
                )
                self.targets[var] = WindowArray(
                    self.accumulated[var],
                    shift=self.pred_span * self.step + self.pred_shift,
                    step=self.step,
                )

    def __len__(self):
        return (
            self.accumulated[self.vars[0]].shape[0]
            - self.input_span * self.step
            - self.pred_shift
        )

    def __getitem__(self, item):
        inputs, targets = {}, {}
        for var in self.vars:
            if var in vars2d or var in vars3d:
                inputs[var] = self.inputs[var][item : item + self.input_span]
                targets[var] = self.targets[var][item : item + self.pred_span]
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

    def __len__(self):
        url = "%s/%s/%s" % (self.url, self.hashcode, self.phase)
        r = requests.get(url)
        if r.status_code != 200:
            raise Exception("http error %s: %s" % (r.status_code, r.text))

        data = msgpack.loads(r.content)

        return data["size"]

    def __getitem__(self, item):
        url = "%s/%s/%s/%d" % (self.url, self.hashcode, self.phase, item)
        r = requests.get(url)
        if r.status_code != 200:
            raise Exception("http error %s: %s" % (r.status_code, r.text))

        data = msgpack.loads(r.content)
        for key, val in data.items():
            for var, blk in val.items():
                val[var] = np.copy(blk)

        return data["inputs"], data["targets"]
