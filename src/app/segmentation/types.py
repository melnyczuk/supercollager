from dataclasses import dataclass
from typing import Tuple

from numpy import ndarray


@dataclass
class AnalysedImage:
    np_img: ndarray
    mask: ndarray
    label: str


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
