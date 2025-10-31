from dataclasses import dataclass


@dataclass
class Context:
    name: str
    root: str
    spatio_resolution: float
    temporal_resolution: float
    granularity: float
    levels: list
    variables: list
    vars2d: list
    vars3d: list
    codes: dict
    years: list
    path_format: str


