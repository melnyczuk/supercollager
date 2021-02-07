from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from PIL import Image  # type: ignore


class ImageType:
    def __init__(self: "ImageType", img: Union[Image.Image, np.ndarray]):
        self.pil = (
            img
            if isinstance(img, Image.Image)
            else Image.fromarray(img.astype(np.uint8))
        )
        self.np = np.array(self.pil)
        self.dimensions = self.pil.size
        self.channels = self.np.shape[2] if len(self.np.shape) > 2 else 1


@dataclass
class AnalysedImage:
    img: ImageType
    mask: np.ndarray
    label: str


@dataclass
class Bounds:
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True)
class LabelImage:
    img: ImageType
    label: str = ""


@dataclass
class MaskBox:
    frame: Tuple[int, ...]
    classId: int
    score: float
    mask: np.ndarray
    bounds: Bounds
