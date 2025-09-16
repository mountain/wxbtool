import logging

import numpy as np
from leibniz.nn.net import SimpleCNN2d
from torch.utils.data import Dataset

from tests.spec.spec import SettingRNN12d, SpecRNN

setting = SettingRNN12d()
logger = logging.getLogger(__name__)


class TestDataset(Dataset):
    """
    Optimized test dataset that generates minimal data for testing.

    This dataset creates small dummy data arrays to reduce memory usage
    and computation time during tests.
    """

    def __len__(self):
        # Reduced dataset size for faster testing
        return 20  # Reduced from 60

    def __getitem__(self, item):
        inputs, targets = {}, {}
        for var in setting.vars:
            inputs.update(
                {var: np.zeros((setting.input_span, 32, 64), dtype=np.float32)}
            )
            targets.update(
                {var: np.zeros((setting.pred_span, 32, 64), dtype=np.float32)}
            )

        # Add a small amount of random data to ensure non-zero gradients
        for key in inputs:
            inputs[key][0, 0, 0] = 1.0
        for key in targets:
            targets[key][0, 0, 0] = 1.0

        return inputs, targets, item


class TestModel(SpecRNN):
    def __init__(self, setting):
        super().__init__(setting)
        self.name = "climate-rnn-test"
        self.cnn = SimpleCNN2d(1, 1)

    def forecast_error(self, rmse):
        return rmse

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
        inputs = kwargs["data"]
        assert (
            inputs.dim() == 5
        ), "Input data must be 5D tensor (batch_size, channels, time_steps, height, width)"
        frame = self.cnn(inputs[:, :, 0, :, :])
        frame = frame.view(-1, 1, 1, inputs.size(3), inputs.size(4))
        frames = frame.repeat(1, 1, self.setting.pred_span // self.setting.step, 1, 1)
        data = frames
        return {
            "data": data,
            "t2m": data,
        }


model = TestModel(SettingRNN12d())
