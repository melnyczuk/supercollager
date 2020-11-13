from dataclasses import dataclass
from typing import Generic, TypeVar, Union

from numpy import ndarray

T = TypeVar("T")


@dataclass
class AnalysedImage:
    img: ndarray
    mask: ndarray
    label: Union[str, None] = None


@dataclass
class Bounds(Generic[T]):
    left: T
    top: T
    right: T
    bottom: T


@dataclass
class MaskBox(Generic[T]):
    classId: T
    score: T
    mask: ndarray
    bounds: Bounds[T]
