# -*- coding: utf-8 -*-

"""
Demo model in wxbtool package

This model predict t850 3-days in the future
The performance is relative weak, but it can be easily fitted into one normal gpu
the weighted rmse is 2.41 K
"""

import torch as th
from leibniz.nn.activation import CappingRelu
from leibniz.nn.layer.senet import SEBottleneck
from leibniz.nn.net import resunet

from wxbtool.specs.res5_625.t850weyn import Setting3d, Spec


class ResUNetModel(Spec):
    def __init__(self, setting):
        super().__init__(setting)
        self.name = "t850d3sm-weyn"
        lat_size, lon_size = setting.spatial_shape
        self.resunet = resunet(
            setting.input_span * (len(setting.vars) + 2) + self.constant_size() + 2,
            1,
            spatial=(lat_size, lon_size + 2),
            layers=5,
            ratio=-1,
            vblks=[2, 2, 2, 2, 2],
            hblks=[1, 1, 1, 1, 1],
            scales=[-1, -1, -1, -1, -1],
            factors=[1, 1, 1, 1, 1],
            block=SEBottleneck,
            relu=CappingRelu(),
            final_normalized=False,
        )

    def forward(self, **kwargs):
        batch_size = kwargs["temperature"].size()[0]
        self.update_da_status(batch_size)

        _, input = self.get_inputs(**kwargs)
        constant = self.get_augmented_constant(input)
        input = th.cat((input, constant), dim=1)

        # Periodic boundary conditions using dynamic dimensions
        lat_size, lon_size = self.setting.spatial_shape
        lon_last_idx = lon_size - 1
        input = th.cat(
            (input[:, :, :, lon_last_idx:], input, input[:, :, :, :1]), dim=3
        )

        output = self.resunet(input)

        # Remove padding: keep only the original longitude range
        output = output[:, :, :, 1 : lon_size + 1]
        return {"t850": output}


setting = Setting3d()
model = ResUNetModel(setting)
