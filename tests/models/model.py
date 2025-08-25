# -*- coding: utf-8 -*-

"""
Base model for testing - Optimized for faster testing
"""

import torch as th

from leibniz.nn.net.simple import SimpleCNN2d
from tests.spec.spec import Spec, Setting3d


class Model(Spec):
    """
    Optimized base model for testing with a simplified architecture.
    """

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "model"

        # Use a smaller CNN for faster computation
        self.cnn = SimpleCNN2d(
            self.setting.input_span * len(self.setting.vars_in)
            + self.constant_size()
            + 2,
            self.setting.pred_span * len(self.setting.vars_out),
        )

    def forward(self, **kwargs):
        batch_size = kwargs["data"].size()[0]
        self.update_da_status(batch_size)

        inputs = kwargs["data"]
        cnst = self.get_augmented_constant(inputs)
        inputs = th.cat((inputs, cnst), dim=1)

        # Forward pass through the CNN
        output = self.cnn(inputs).view(batch_size, self.setting.pred_span, 32, 64)

        return {"t2m": output, "data": output}


setting = Setting3d()
model = Model(setting)
