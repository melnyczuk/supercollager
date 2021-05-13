from typing import List, Tuple

import numpy as np

from src.app.region import Region
from src.app.transform import Transform


class MaskBox:
    mask: np.ndarray

    def __init__(
        self: "MaskBox",
        shape: Tuple[int, int],
        box: np.ndarray,
        masks: np.ndarray,
    ) -> None:
        (id, score, *edges) = box
        bounds = self.__calc_bounds(shape, edges)
        mask = np.zeros(shape)
        roi = self.__process_mask(masks[int(id)], bounds)
        self.mask = bounds.fill(mask, roi)
        return

    def __process_mask(
        self: "MaskBox",
        detection_mask: np.ndarray,
        bounds: Region,
    ) -> np.ndarray:
        dsize = (bounds.right - bounds.left, bounds.bottom - bounds.top)
        mask = (detection_mask * 255).astype(np.uint8)
        return Transform.resize(mask, dsize=dsize)

    def __calc_bounds(
        self: "MaskBox",
        shape: Tuple[int, int],
        edges: List[float],
    ) -> Region:
        (left, top, right, bottom) = edges
        return Region(
            left=self.__min_max(shape[1], left),
            top=self.__min_max(shape[0], top),
            right=self.__min_max(shape[1], right),
            bottom=self.__min_max(shape[0], bottom),
        )

    def __min_max(
        self: "MaskBox",
        shape_dim: int,
        box_dim: float,
    ) -> int:
        scaled_dim = int(shape_dim * box_dim)
        return int(max(0, min(scaled_dim, shape_dim - 1)))
