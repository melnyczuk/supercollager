from random import randint
from typing import Callable, Tuple

import cv2  # type:ignore
import numpy as np
from PIL import Image  # type: ignore

from .roi import ROI, Region


class Transform:
    @staticmethod
    def rotate(np_img: np.ndarray, factor: float) -> np.ndarray:
        return np.array(
            Image.fromarray(np_img)
            .rotate(factor)
            .resize(np_img.shape[:2][::-1])
        )

    @staticmethod
    def warp(np_img: np.ndarray, factor: float) -> np.ndarray:
        box = ROI.get_bounding_box(np_img)
        pts = cv2.getAffineTransform(*_get_corner_points(box, factor))
        return cv2.warpAffine(np_img, pts, np_img.shape[:2][::-1])

    @staticmethod
    def kaleidoscope(np_img: np.ndarray, factor: float) -> np.ndarray:
        return np.dstack(  # type: ignore
            [
                Transform.warp(np_img[:, :, ch], factor)
                if ch % 2 == 0
                else np_img[:, :, ch]
                for ch in range(np_img.shape[2])
            ]
        )


def _get_corner_points(
    region: Region,
    factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    (x0, y0, x1, y1) = region
    offset = int((((x1 - x0) ** 2 + (y1 - y0) ** 2) ** (1 / 2)) * factor)

    def warp_fn(n: int) -> int:
        return randint(n - offset, n + offset)

    warp_vectors = np.vectorize(warp_fn)  # type: ignore

    src = np.array(((x0, y0), (x0, y1), (x1, y0)))
    trg = warp_vectors(src)

    return (src.astype(np.float32), trg.astype(np.float32))
