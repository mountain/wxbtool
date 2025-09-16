"""
Test model in wxbtool package - Optimized for faster testing
"""

import numpy as np
import torch as th
from leibniz.nn.net import SimpleCNN2d
from torch.utils.data import Dataset

from tests.spec.spec import Setting3d, Spec
from wxbtool.data.variables import vars3d

setting = Setting3d()


class TestDataset(Dataset):
    """
    Optimized test dataset that generates minimal data for testing.

    This dataset creates small dummy data arrays to reduce memory usage
    and computation time during tests.
    """

    def __len__(self):
        # Further reduced dataset size for even faster testing
        return 10  # Reduced from 20

    def __getitem__(self, item):
        inputs, targets = {}, {}
        for var in setting.vars:
            if var in vars3d:
                # Use zeros instead of ones to reduce memory initialization time
                inputs.update(
                    {
                        var: np.zeros(
                            (1, setting.input_span, setting.height, 32, 64),
                            dtype=np.float32,
                        )
                    }
                )
                targets.update(
                    {
                        var: np.zeros(
                            (1, setting.pred_span, setting.height, 32, 64),
                            dtype=np.float32,
                        )
                    }
                )
            else:
                inputs.update(
                    {var: np.zeros((1, setting.input_span, 32, 64), dtype=np.float32)}
                )
                targets.update(
                    {var: np.zeros((1, setting.pred_span, 32, 64), dtype=np.float32)}
                )

        # Add a small amount of random data to ensure non-zero gradients
        for key in inputs:
            inputs[key][0, 0, 0, 0] = 1.0
        for key in targets:
            targets[key][0, 0, 0, 0] = 1.0

        return inputs, targets, item


class Mdl(Spec):
    """
    Optimized model for testing with a simplified architecture.
    """

    def __init__(self, setting):
        super().__init__(setting)
        self.name = "fast"

        # Use a smaller CNN for faster computation
        input_channels = (
            self.setting.input_span * len(self.setting.vars_in)
            + self.constant_size()
            + 2
        )
        output_channels = self.setting.pred_span * len(self.setting.vars_out)

        # SimpleCNN2d only accepts input and output channels
        self.cnn = SimpleCNN2d(input_channels, output_channels)

    def load_dataset(self, phase, mode, **kwargs):
        self.phase = phase
        self.mode = mode

        # Use the same dataset instance for all splits to save memory
        dataset = TestDataset()
        self.dataset_train = dataset
        self.dataset_eval = dataset
        self.dataset_test = dataset

        self.train_size = len(self.dataset_train)
        self.eval_size = len(self.dataset_eval)
        self.test_size = len(self.dataset_test)

    def forward(self, **kwargs):
        batch_size = kwargs["data"].size()[0]
        self.update_da_status(batch_size)

        inputs = kwargs["data"]
        cnst = self.get_augmented_constant(inputs)
        inputs = th.cat((inputs, cnst), dim=1)

        # Forward pass through the CNN
        output = self.cnn(inputs).view(batch_size, self.setting.pred_span, 32, 64)

        return {"t2m": output, "data": output}


model = Mdl(setting)
