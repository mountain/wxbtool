# -*- coding: utf-8 -*-

"""
A modeling spec for t850

This spec follows basic settings and discussions in

  Improving Data‐Driven Global Weather Prediction Using Deep Convolutional Neural Networks on a Cubed Sphere
  by Jonathan A. Weyn, Dale R. Durran, Rich Caruana
  https://doi.org/10.1029/2020MS002109

But specialized for recurrence neural networks

"""

import torch as th
import torch.nn as nn
from wxbtool.nn.model import Base2d
from wxbtool.nn.setting import Setting
from wxbtool.norms.meanstd import (
    norm_z500,
    norm_z1000,
    norm_tau,
    norm_t850,
    norm_t2m,
    norm_tisr,
)


mse = nn.MSELoss()


class SettingRecur(Setting):
    def __init__(self):
        super().__init__()
        self.resolution = "5.625deg"  # The spatial resolution of the model

        self.levels = [
            "300",
            "500",
            "700",
            "850",
            "1000",
        ]  # Which vertical levels to choose
        self.height = len(self.levels)  # How many vertical levels to choose

        # The name of variables to choose, for both input features and output
        self.vars = [
            "temperature",
            "geopotential",
            "toa_incident_solar_radiation",
            "2m_temperature",
        ]

        # The code of variables in input features
        self.vars_in = ["z500", "z1000", "tau", "t850", "t2m", "tisr"]
        # The code of variables in output
        self.vars_out = ["z500", "z1000", "tau", "t850", "t2m", "tisr"]

        # temporal scopes for train
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
        # temporal scopes for evaluation
        self.years_eval = [2015, 2016]
        # temporal scopes for test
        self.years_test = [2017, 2018]

        self.step = 8  # How many hours of a hourly step which all features in organized temporally
        self.input_span = 3  # How many hourly steps for an input
        self.pred_span = 3  # How many hourly steps for a prediction
        self.pred_shift = 24  # How many hours between the end of the input span and the beginning of prediction span


class Spec(Base2d):
    def __init__(self, setting):
        super().__init__(setting)
        self.name = "t850_recur"

    def get_inputs(self, **kwargs):
        z500 = norm_z500(
            kwargs["geopotential"].view(
                -1, self.setting.input_span, self.setting.height, 32, 64
            )[:, :, self.setting.levels.index("500")]
        )
        z1000 = norm_z1000(
            kwargs["geopotential"].view(
                -1, self.setting.input_span, self.setting.height, 32, 64
            )[:, :, self.setting.levels.index("1000")]
        )
        tau = norm_tau(
            kwargs["geopotential"].view(
                -1, self.setting.input_span, self.setting.height, 32, 64
            )[:, :, self.setting.levels.index("300")]
            - kwargs["geopotential"].view(
                -1, self.setting.input_span, self.setting.height, 32, 64
            )[:, :, self.setting.levels.index("700")]
        )
        t850 = norm_t850(
            kwargs["temperature"].view(
                -1, self.setting.input_span, self.setting.height, 32, 64
            )[:, :, self.setting.levels.index("850")]
        )
        t2m = norm_t2m(
            kwargs["2m_temperature"].view(-1, self.setting.input_span, 32, 64)
        )
        tisr = norm_tisr(
            kwargs["toa_incident_solar_radiation"].view(
                -1, self.setting.input_span, 32, 64
            )
        )

        z500 = self.augment_data(z500)
        z1000 = self.augment_data(z1000)
        tau = self.augment_data(tau)
        t850 = self.augment_data(t850)
        t2m = self.augment_data(t2m)
        tisr = self.augment_data(tisr)

        return {
            "z500": z500,
            "z1000": z1000,
            "tau": tau,
            "t850": t850,
            "t2m": t2m,
            "tisr": tisr,
        }, th.cat(
            (
                z500,
                z1000,
                tau,
                t850,
                t2m,
                tisr,
            ),
            dim=1,
        )

    def get_targets(self, **kwargs):
        z500 = norm_z500(
            kwargs["geopotential"].view(
                -1, self.setting.input_span, self.setting.height, 32, 64
            )[:, :, self.setting.levels.index("500")]
        )
        z1000 = norm_z1000(
            kwargs["geopotential"].view(
                -1, self.setting.input_span, self.setting.height, 32, 64
            )[:, :, self.setting.levels.index("1000")]
        )
        tau = norm_tau(
            kwargs["geopotential"].view(
                -1, self.setting.input_span, self.setting.height, 32, 64
            )[:, :, self.setting.levels.index("300")]
            - kwargs["geopotential"].view(
                -1, self.setting.input_span, self.setting.height, 32, 64
            )[:, :, self.setting.levels.index("700")]
        )
        t850 = norm_t850(
            kwargs["temperature"].view(
                -1, self.setting.input_span, self.setting.height, 32, 64
            )[:, :, self.setting.levels.index("850")]
        )
        t2m = norm_t2m(
            kwargs["2m_temperature"].view(-1, self.setting.input_span, 32, 64)
        )
        tisr = norm_tisr(
            kwargs["toa_incident_solar_radiation"].view(
                -1, self.setting.input_span, 32, 64
            )
        )

        z500 = self.augment_data(z500)
        z1000 = self.augment_data(z1000)
        tau = self.augment_data(tau)
        t850 = self.augment_data(t850)
        t2m = self.augment_data(t2m)
        tisr = self.augment_data(tisr)

        return {
            "z500": z500,
            "z1000": z1000,
            "tau": tau,
            "t850": t850,
            "t2m": t2m,
            "tisr": tisr,
        }, th.cat(
            (
                z500,
                z1000,
                tau,
                t850,
                t2m,
                tisr,
            ),
            dim=1,
        )

    def get_results(self, **kwargs):
        z500 = kwargs["z500"]
        z1000 = kwargs["z1000"]
        tau = kwargs["tau"]
        t850 = kwargs["t850"]
        t2m = kwargs["t2m"]
        tisr = kwargs["tisr"]
        return {
            "z500": z500,
            "z1000": z1000,
            "tau": tau,
            "t850": t850,
            "t2m": t2m,
            "tisr": tisr,
        }, th.cat(
            (
                z500,
                z1000,
                tau,
                t850,
                t2m,
                tisr,
            ),
            dim=1,
        )

    def forward(self, **kwargs):
        raise NotImplementedError("Spec is abstract and can not be initialized")

    def lossfun(self, inputs, result, target):
        ch = len(self.setting.vars_out) * self.setting.pred_span
        _, rst = self.get_results(**result)
        _, tgt = self.get_targets(**target)
        weight = self.weight.to(rst.device)
        rst = weight * rst.view(-1, ch, 32, 64)
        tgt = weight * tgt.view(-1, ch, 32, 64)

        lossall = mse(rst, tgt)

        return lossall
