import os
import pytest
import sys
from pathlib import Path

pytest.importorskip("numpy")
th = pytest.importorskip("torch")

root = Path(__file__).resolve().parents[1] / "tests"
sys.path.insert(0, str(root))
os.environ.setdefault("WXBHOME", str(root))

from wxbtool.nn.lightning import LightningModel  # noqa: E402


class DummySetting:
    vars_out = ["test"]
    pred_span = 2


class DummyModel:
    def __init__(self):
        self.setting = DummySetting()
        self.weight = th.ones(2, 2)

    def __call__(self, **inputs):  # pragma: no cover - unused
        return {}


def test_compute_rmse_by_time_side_effect():
    model = DummyModel()
    lightning = LightningModel(model)

    targets = {"test": th.zeros(1, 2, 2, 2)}
    results = {"test": th.ones(1, 2, 2, 2)}

    rmse = lightning.compute_rmse_by_time(targets, results, "test")

    assert th.isclose(rmse, th.tensor(1.0))
    assert lightning.mseByVar["test"][0][1] == 1.0
    assert lightning.mseByVar["test"][0][2] == 1.0
