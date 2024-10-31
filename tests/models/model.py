# -*- coding: utf-8 -*-

import torch as th

from leibniz.nn.net.simple import SimpleCNN2d
from tests.spec.spec import Spec, Setting3d


class Model(Spec):
    def __init__(self, setting):
        super().__init__(setting)
        self.name = "model"
        self.mlp = SimpleCNN2d(
            self.setting.input_span * len(self.setting.vars_in)
            + self.constant_size
            + 2,
            self.setting.pred_span * len(self.setting.vars_out),
        )

    def forward(self, **kwargs):
        batch_size = kwargs["2m_temperature"].size()[0]
        self.update_da_status(batch_size)

        _, input = self.get_inputs(**kwargs)
        cnst = self.get_augmented_constant(input)
        input = th.cat((input, cnst), dim=1)

        output = self.mlp(input)

        return {"t2m": output.view(batch_size, self.setting.pred_span, 32, 64)}


setting = Setting3d()
model = Model(setting)
