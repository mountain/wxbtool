from typing import Sequence, Dict, Callable, Any, Mapping
from torch import Tensor
from torchmetrics import Metric


class WXBMetric(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False
    plot_lower_bound: float = 0.0

    def __init__(
        self,
        temporal_span: int,
        temporal_step: int,
        temporal_shift: int,
        spatio_weight: Tensor,
        variables: Sequence[str],
        denormalizers: Dict[str, Callable],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(temporal_span, int):
            raise ValueError(f"Expected argument `pred_span` to be an integer but got {temporal_span}")
        self.temporal_span = temporal_span

        if not isinstance(temporal_step, int):
            raise ValueError(f"Expected argument `temporal_step` to be an integer but got {temporal_step}")
        self.temporal_step = temporal_step

        if not isinstance(temporal_shift, int):
            raise ValueError(f"Expected argument `temporal_shift` to be an integer but got {temporal_shift}")
        self.temporal_shift = temporal_shift

        if not isinstance(spatio_weight, Tensor):
            raise ValueError(f"Expected argument `weight` to be a tensor but got {spatio_weight}")
        self.spatio_weight = spatio_weight
        self.spatio_height = spatio_weight.size(-2)
        self.spatio_width = spatio_weight.size(-1)

        if not isinstance(variables, Sequence):
            raise ValueError(f"Expected argument `variables` to be a sequence but got {variables}")
        self.variables = variables

        if not isinstance(denormalizers, Mapping):
            raise ValueError(f"Expected argument `denormalizers` to be a mapping but got {denormalizers}")
        self.denormalizers = denormalizers

    def __repr__(self):
        return repr(self.compute())

    def _get_(self, attr:str) -> Tensor:
        return getattr(self, attr)

    def _set_(self, attr:str, value:Tensor) -> None:
        setattr(self, attr, value)

    def _incr_(self, attr:str, value:Tensor) -> None:
        setattr(self, attr, getattr(self, attr) + value)

    def _sum_(self, value:Tensor) -> Tensor:
        # summarizing among spatio dimensions and batch
        return value.sum(dim=-1).sum(dim=-1).sum(dim=0)
