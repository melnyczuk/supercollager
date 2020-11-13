from dataclasses import dataclass
from typing import Generic, Tuple, TypeVar, Union

from numpy import ndarray


@dataclass
class AnalysedImage:
    img: ndarray
    mask: ndarray
    label: Union[str, None] = None


@dataclass
class Bounds:
    left: int
    top: int
    right: int
    bottom: int


@dataclass
class MaskBox:
    frame: Tuple[int, int]
    classId: int
    score: float
    mask: ndarray
    bounds: Bounds
