import torch

from typing import Dict, List, Tuple


Tensor = torch.Tensor
Data = Dict[str, Tensor]
Indexes = List[int]
Batch = Tuple[Data, Data, Indexes]
