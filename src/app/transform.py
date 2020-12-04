from random import randint
from typing import Tuple

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
        grey = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        box = ROI.get_bounding_box(grey)
        offset = int(min(*np_img.shape[:2]) * factor)
        pts = cv2.getAffineTransform(*_get_corner_points(box, offset))
        return cv2.warpAffine(np_img, pts, np_img.shape[:2][::-1])

    @staticmethod
    def kaleidoscope(np_img: np.ndarray, factor: float) -> np.ndarray:
        return np.dstack(  # type: ignore
            [
                (Transform.warp(np_img[:, :, ch], factor) + np_img[:, :, ch])
                if i % 2 == 0
                else np_img[:, :, ch]
                for (i, ch) in enumerate(range(np_img.shape[2]))
            ]
        )


def _get_corner_points(
    region: Region,
    offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    warp_vectors = np.vectorize(lambda n: randint(n - offset, n + offset))  # type: ignore # noqa: E501
    (y0, x0, y1, x1) = region
    src = np.array(((x0, y0), (x0, y1), (x1, y0)))
    trg = warp_vectors(src)
    return (src.astype(np.float32), trg.astype(np.float32))
