"""
Test model in wxbtool package
"""

import numpy as np
import torch as th

from leibniz.nn.net import SimpleCNN2d
from torch.utils.data import Dataset

from tests.spec.spec import Setting6d, Spec
from wxbtool.data.variables import vars3d


setting = Setting6d()


class TestDataset(Dataset):
    def __len__(self):
        return 60

    def __getitem__(self, item):
        inputs, targets = {}, {}
        for var in setting.vars:
            if var in vars3d:
                inputs.update(
                    {var: np.ones((1, setting.input_span, setting.height, 32, 64))}
                )
                targets.update(
                    {var: np.ones((1, setting.pred_span, setting.height, 32, 64))}
                )
            else:
                inputs.update({var: np.ones((1, setting.input_span, 32, 64))})
                targets.update({var: np.ones((1, setting.pred_span, 32, 64))})
        return inputs, targets, item


class Mdl(Spec):
    def __init__(self, setting):
        super().__init__(setting)
        self.name = "fast"
        self.cnn = SimpleCNN2d(
            self.setting.input_span * len(self.setting.vars_in)
            + self.constant_size
            + 2,
            self.setting.pred_span * len(self.setting.vars_out),
        )

    def load_dataset(self, phase, mode, **kwargs):
        self.phase = phase
        self.mode = mode

        self.dataset_train = TestDataset()
        self.dataset_eval = TestDataset()
        self.dataset_test = TestDataset()

        self.train_size = len(self.dataset_train)
        self.eval_size = len(self.dataset_eval)
        self.test_size = len(self.dataset_test)

    def forward(self, **kwargs):
        batch_size = kwargs["data"].size()[0]
        self.update_da_status(batch_size)

        inputs = kwargs["data"]
        cnst = self.get_augmented_constant(inputs)
        inputs = th.cat((inputs, cnst), dim=1)

        output = self.cnn(inputs).view(batch_size, self.setting.pred_span, 32, 64)

        return {"t2m": output, "data": output}


model = Mdl(setting)
