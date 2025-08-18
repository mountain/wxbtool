"""
Test climate model in wxbtool package - Optimized for faster testing
"""

import torch as th

from leibniz.nn.net import SimpleCNN2d
from tests.spec.spec import Setting60d, ClimateSpec


setting = Setting60d()


class Mdl(ClimateSpec):
    """
    Optimized climate model for testing with a simplified architecture.
    """

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "climate-60d"  # Fixed name to match the actual model

        # Use a smaller CNN for faster computation
        self.cnn = SimpleCNN2d(
            self.setting.input_span * len(self.setting.vars_in)
            + self.constant_size
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
        return {"test": output, "data": output}


model = Mdl(setting)
