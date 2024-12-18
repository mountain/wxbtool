"""
A modeling spec for t2m
"""

import torch as th
import torch.nn as nn

import wxbtool.data.variables as v
import wxbtool.norms.meanstd as n

from wxbtool.nn.model import Base2d
from wxbtool.nn.setting import Setting
from wxbtool.data.variables import split_name

mse = nn.MSELoss()


class SettingSimple(Setting):
    def __init__(self):
        super().__init__()
        self.resolution = "5.625deg"  # The spatial resolution of the model

        # Which vertical levels to choose
        self.levels = []
        # How many vertical levels to choose
        self.height = len(self.levels)

        # The name of variables to choose, for both input features and output
        self.vars = ["2m_temperature"]

        # The code of variables in input features
        self.vars_in = ["t2m"]

        # The code of variables in output
        self.vars_out = ["t2m"]

        # temporal scopes for train
        self.years_train = [1980, 1981, 1982]
        # temporal scopes for evaluation
        self.years_eval = [1980, 1981, 1982]
        # temporal scopes for test
        self.years_test = [1980, 1981, 1982]


class Setting3d(SettingSimple):
    def __init__(self):
        super().__init__()
        self.step = 5  # How many days of a daily step which all features in organized temporally
        self.input_span = 2  # How many daily steps for an input
        self.pred_span = 1  # How many daily steps for a prediction
        self.pred_shift = 3  # How many days between the end of the input span and the beginning of prediction span


class Setting6d(SettingSimple):
    def __init__(self):
        super().__init__()
        self.step = 5  # How many days of a daily step which all features in organized temporally
        self.input_span = 2  # How many daily steps for an input
        self.pred_span = 2  # How many daily steps for a prediction
        self.pred_shift = 6  # How many days between the end of the input span and the beginning of prediction span


class Setting10d(SettingSimple):
    def __init__(self):
        super().__init__()
        self.step = 10  # How many days of a daily step which all features in organized temporally
        self.input_span = 2  # How many daily steps for an input
        self.pred_span = 2  # How many daily steps for a prediction
        self.pred_shift = 10  # How many days between the end of the input span and the beginning of prediction span


class Setting30d(SettingSimple):
    def __init__(self):
        super().__init__()
        self.step = 30  # How many days of a daily step which all features in organized temporally
        self.input_span = 1  # How many daily steps for an input
        self.pred_span = 1  # How many daily steps for a prediction
        self.pred_shift = 30  # How many days between the end of the input span and the beginning of prediction span
        self.vars = ["test_variable"]
        self.vars_in = ["test"]
        self.vars_out = ["test"]


class Setting60d(SettingSimple):
    def __init__(self):
        super().__init__()
        self.step = 60  # How many days of a daily step which all features in organized temporally
        self.input_span = 1  # How many daily steps for an input
        self.pred_span = 1  # How many daily steps for a prediction
        self.pred_shift = 60  # How many days between the end of the input span and the beginning of prediction span
        self.vars = ["test_variable"]
        self.vars_in = ["test"]
        self.vars_out = ["test"]


class Setting90d(SettingSimple):
    def __init__(self):
        super().__init__()
        self.step = 90  # How many days of a daily step which all features in organized temporally
        self.input_span = 1  # How many daily steps for an input
        self.pred_span = 1  # How many daily steps for a prediction
        self.pred_shift = 90  # How many days between the end of the input span and the beginning of prediction span
        self.vars = ["test_variable"]
        self.vars_in = ["test"]
        self.vars_out = ["test"]


class Spec(Base2d):
    def __init__(self, setting):
        super().__init__(setting)
        self.vars_out = [
            "t2m",
        ]

    def get_inputs(self, **kwargs):
        vdic, vlst = {}, []
        for nm in self.setting.vars_in:
            code, lvl = split_name(nm)
            var = v.code2var[code]
            if var in v.vars3d:
                d = (
                    kwargs[var]
                    .view(-1, self.setting.input_span, self.setting.height, 32, 64)[
                        :, :, self.setting.levels.index(lvl)
                    ]
                    .float()
                )
            else:
                d = kwargs[var].view(-1, self.setting.input_span, 32, 64).float()
            d = n.normalizors[nm](d)
            d = self.augment_data(d)
            vdic[nm] = d
            vlst.append(d)

        data = th.cat(vlst, dim=1)
        vdic["data"] = data
        for key, val in kwargs.items():
            if type(val) is th.Tensor:
                vdic[key] = val.float()

        return vdic

    def get_targets(self, **kwargs):
        t2m = n.norm_t2m(
            kwargs["2m_temperature"].view(-1, self.setting.pred_span, 32, 64)
        ).float()

        return {
            "data": t2m,
            "t2m": t2m,
        }

    def get_results(self, **kwargs):
        t2m = kwargs["t2m"].float()
        return {
            "data": t2m,
            "t2m": t2m,
        }

    def forward(self, **kwargs):
        raise NotImplementedError("Spec is abstract and can not be initialized")

    def lossfun(self, inputs, result, target):
        rst = result["data"]
        tgt = target["data"]
        weight = self.get_weight(rst.device).float()
        rst = weight * rst.view(-1, self.setting.pred_span, 32, 64)
        tgt = weight * tgt.view(-1, self.setting.pred_span, 32, 64)

        losst = mse(rst[:, 0], tgt[:, 0])

        return losst


class ClimateSpec(Base2d):
    def __init__(self, setting):
        super().__init__(setting)
        self.vars_in = ["test"]
        self.vars_out = ["test"]

    def get_inputs(self, **kwargs):
        data = kwargs["test_variable"].view(-1, self.setting.pred_span, 32, 64).float()
        return {
            "data": data,
            "test": data,
        }

    def get_targets(self, **kwargs):
        data = kwargs["test_variable"].view(-1, self.setting.pred_span, 32, 64).float()
        return {
            "data": data,
            "test": data,
        }

    def get_results(self, **kwargs):
        data = kwargs["test"].float()
        return {
            "data": data,
            "test": data,
        }

    def forward(self, **kwargs):
        raise NotImplementedError("Spec is abstract and can not be initialized")

    def lossfun(self, inputs, result, target):
        rst = result["data"]
        tgt = target["data"]
        weight = self.get_weight(rst.device).float()
        rst = rst.view(-1, self.setting.pred_span, 32, 64)
        tgt = tgt.view(-1, self.setting.pred_span, 32, 64)
        return (weight * (rst - tgt) ** 2).mean()
