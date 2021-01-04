from dataclasses import dataclass
from typing import Tuple

from numpy import ndarray
from PIL import Image  # type: ignore


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


@dataclass(frozen=True)
class LabelImage:
    pil_img: Image
    label: str = ""


@dataclass
class MaskBox:
    frame: Tuple[int, ...]
    classId: int
    score: float
    mask: ndarray
    bounds: Bounds
