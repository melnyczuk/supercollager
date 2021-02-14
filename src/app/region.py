from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    right: int
    bottom: int

    def crop(self: "Region", array: np.ndarray) -> np.ndarray:
        return array[
            self.top : self.bottom,  # noqa: E203
            self.left : self.right,  # noqa: E203
        ]

    def fill(self: "Region", array: np.ndarray, roi: np.ndarray) -> np.ndarray:
        array[
            self.top : self.bottom,  # noqa: E203
            self.left : self.right,  # noqa: E203
        ] = roi
        return array
