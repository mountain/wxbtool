# -*- coding: utf-8 -*-

import random

import numpy as np
import torch as th
import torch.nn as nn

from wxbtool.data.constants import (
    load_lat2d,
    load_lon2d,
    load_lsm,
    load_orography,
    load_slt,
)
from wxbtool.data.dataset import WxDataset, WxDatasetClient


def cast(element):
    element = np.array(element, dtype=np.float32)
    tensor = th.FloatTensor(element)
    return tensor


class Model(nn.Module):
    def __init__(self, setting, enable_da=False):
        super().__init__()
        self.setting = setting

        self.enable_da = enable_da
        self.lng_shift = []
        self.flip_status = []

        self.dataset_train, self.dataset_test, self.dataset_eval = None, None, None
        self.train_size = -1
        self.test_size = -1
        self.eval_size = -1

        # Assume lsm, slt, oro as constant inputs
        self._constant_size = 3
        self.prepare_constant()

    def prepare_constant(self):
        lsm = cast(load_lsm(self.setting.resolution, self.setting.root))
        slt = cast(load_slt(self.setting.resolution, self.setting.root))
        oro = cast(load_orography(self.setting.resolution, self.setting.root))
        phi = cast(load_lat2d(self.setting.resolution, self.setting.root)) * np.pi / 180
        theta = (
            cast(load_lon2d(self.setting.resolution, self.setting.root)) * np.pi / 180
        )

        dt = th.cos(phi)
        lsm = ((lsm - 0.33707827) / 0.45900375).view(1, 1, 32, 64)
        slt = ((slt - 0.67920434) / 1.1688842).view(1, 1, 32, 64)
        oro = ((oro - 379.4976) / 859.87225).view(1, 1, 32, 64)
        phi = phi.view(1, 1, 32, 64)
        theta = theta.view(1, 1, 32, 64)

        self.register_buffer("weight", dt / dt.mean())
        self.register_buffer("constant", th.cat((lsm, slt, oro), dim=1))
        self.register_buffer("phi", phi)
        self.register_buffer("theta", theta)

    def get_weight(self, device):
        return self.weight.to(device)

    def load_dataset(self, phase, mode, **kwargs):
        if mode == "server":
            self.dataset_train, self.dataset_eval, self.dataset_test = (
                WxDataset(
                    self.setting.root,
                    self.setting.resolution,
                    self.setting.years_train,
                    self.setting.vars,
                    self.setting.levels,
                    input_span=self.setting.input_span,
                    pred_shift=self.setting.pred_shift,
                    pred_span=self.setting.pred_span,
                    step=self.setting.step,
                    granularity=self.setting.granularity,
                    data_path_format=self.setting.data_path_format,
                    setting=self.setting,
                ),
                WxDataset(
                    self.setting.root,
                    self.setting.resolution,
                    self.setting.years_eval,
                    self.setting.vars,
                    self.setting.levels,
                    input_span=self.setting.input_span,
                    pred_shift=self.setting.pred_shift,
                    pred_span=self.setting.pred_span,
                    step=self.setting.step,
                    granularity=self.setting.granularity,
                    data_path_format=self.setting.data_path_format,
                    setting=self.setting,
                ),
                WxDataset(
                    self.setting.root,
                    self.setting.resolution,
                    self.setting.years_test,
                    self.setting.vars,
                    self.setting.levels,
                    input_span=self.setting.input_span,
                    pred_shift=self.setting.pred_shift,
                    pred_span=self.setting.pred_span,
                    step=self.setting.step,
                    granularity=self.setting.granularity,
                    data_path_format=self.setting.data_path_format,
                    setting=self.setting,
                ),
            )
        else:
            ds_url = kwargs["url"]
            self.dataset_train, self.dataset_eval, self.dataset_test = (
                WxDatasetClient(
                    ds_url,
                    "train",
                    self.setting.resolution,
                    self.setting.years_train,
                    self.setting.vars,
                    self.setting.levels,
                    input_span=self.setting.input_span,
                    pred_shift=self.setting.pred_shift,
                    pred_span=self.setting.pred_span,
                    step=self.setting.step,
                    granularity=self.setting.granularity,
                    data_path_format=self.setting.data_path_format,
                ),
                WxDatasetClient(
                    ds_url,
                    "eval",
                    self.setting.resolution,
                    self.setting.years_eval,
                    self.setting.vars,
                    self.setting.levels,
                    input_span=self.setting.input_span,
                    pred_shift=self.setting.pred_shift,
                    pred_span=self.setting.pred_span,
                    step=self.setting.step,
                    granularity=self.setting.granularity,
                    data_path_format=self.setting.data_path_format,
                ),
                WxDatasetClient(
                    ds_url,
                    "test",
                    self.setting.resolution,
                    self.setting.years_test,
                    self.setting.vars,
                    self.setting.levels,
                    input_span=self.setting.input_span,
                    pred_shift=self.setting.pred_shift,
                    pred_span=self.setting.pred_span,
                    step=self.setting.step,
                    granularity=self.setting.granularity,
                    data_path_format=self.setting.data_path_format,
                ),
            )

        self.train_size = len(self.dataset_train)
        self.eval_size = len(self.dataset_eval)
        self.test_size = len(self.dataset_test)

        import logging

        logger = logging.getLogger()
        if self.dataset_train:
            logger.info("train dataset key: %s", self.dataset_train.hashcode)
        if self.dataset_eval:
            logger.info("eval dataset key: %s", self.dataset_eval.hashcode)
        if self.dataset_test:
            logger.info("test dataset key: %s", self.dataset_test.hashcode)

    def constant_size(self):
        if not hasattr(self, "_constant_size"):
            self.prepare_constant()
        return self._constant_size

    def update_da_status(self, batch):
        if self.enable_da and self.training:
            self.lng_shift = []
            self.flip_status = []
            for _ in range(batch):
                self.lng_shift.append(random.randint(0, 64))
                self.flip_status.append(random.randint(0, 1))

    def clear_da_status(self):
        self.lng_shift = []
        self.flip_status = []

    def augment_data(self, data):
        raise NotImplementedError()

    def get_augmented_constant(self, input_data):
        raise NotImplementedError()

    def get_inputs(self, **kwargs):
        raise NotImplementedError()

    def get_targets(self, **kwargs):
        raise NotImplementedError()

    def get_results(self, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def lossfun(self, inputs, result, target):
        raise NotImplementedError()


class Base2d(Model):
    def __init__(self, setting, enable_da=False):
        super().__init__(setting, enable_da)

    def augment_data(self, data):
        if self.enable_da and self.training:
            augmented = []
            b, c, w, h = data.size()
            for _ in range(b):
                slice = data[_ : _ + 1]
                shift = self.lng_shift[_]
                flip = self.flip_status[_]
                slice = slice.roll(shift, dims=(3,))
                if flip == 1:
                    slice = th.flip(slice, dims=(2, 3))
                augmented.append(slice)
            data = th.cat(augmented, dim=0)
        return data

    def get_augmented_constant(self, input_data):
        constant = self.constant.repeat(
            input_data.size()[0], 1, 1, 1
        )
        constant = self.augment_data(constant)
        phi = self.phi.repeat(input_data.size()[0], 1, 1, 1)
        theta = self.theta.repeat(input_data.size()[0], 1, 1, 1)
        cos_phi = th.cos(phi)
        sin_phi = th.sin(phi)
        cos_theta = th.cos(theta)
        sin_theta = th.sin(theta)
        constant = th.cat((constant, sin_phi, cos_phi, sin_theta, cos_theta), dim=1)
        return constant


class Base3d(Model):
    def __init__(self, setting, enable_da=False):
        super().__init__(setting, enable_da)

    def augment_data(self, data):
        if self.enable_da and self.training:
            augmented = []
            b, c, t, w, h = data.size()
            for _ in range(b):
                slice = data[_ : _ + 1]
                shift = self.lng_shift[_]
                flip = self.flip_status[_]
                slice = slice.roll(shift, dims=(4,))
                if flip == 1:
                    slice = th.flip(slice, dims=(3, 4))
                augmented.append(slice)
            data = th.cat(augmented, dim=0)
        return data

    def get_augmented_constant(self, input_data):
        b, c, t, w, h = input_data.size()
        constant = self.constant.view(1, self._constant_size, 1, w, h).repeat(b, 1, t, 1, 1)
        constant = self.augment_data(constant)
        phi = self.phi.view(1, 1, 1, w, h).repeat(b, 1, t, 1, 1)
        theta = self.theta.view(1, 1, 1, w, h).repeat(b, 1, t, 1, 1)
        cos_phi = th.cos(phi)
        sin_phi = th.sin(phi)
        cos_theta = th.cos(theta)
        sin_theta = th.sin(theta)
        constant = th.cat((constant, sin_phi, cos_phi, sin_theta, cos_theta), dim=1)
        return constant
