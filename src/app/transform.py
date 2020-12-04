from random import randint
from typing import Tuple

import cv2  # type:ignore
import numpy as np
from PIL import Image  # type: ignore

from src.app.composition import Composition


class Transform:
    @staticmethod
    def warp(
        np_img: np.ndarray,
        factor: int,
        kaleidoscope: bool = False,
    ) -> np.ndarray:
        bgr = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)
        threshold = _get_threshold(bgr)
        roi = Composition.get_roi(bgr, threshold)
        true_factor = int(min(*np_img.shape[:2]) * factor)
        return (
            _kaleidoscope(np_img, roi, true_factor)
            if kaleidoscope
            else cv2.warpAffine(
                np_img,
                cv2.getAffineTransform(*_pts(roi, true_factor)),
                np_img.shape[:2][::-1],
            )
        )

    @staticmethod
    def rotate(mask: np.ndarray, rotation: float = 0.0) -> np.ndarray:
        return np.array(
            Image.fromarray(mask).rotate(rotation).resize(mask.shape[::-1])
        )


def _pts(
    roi: Tuple[int, int, int, int],
    o: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    (x0, y0, w, h) = roi
    (x1, y1) = (x0 + w, y0 + h)

    warp_vectors = np.vectorize(lambda n: randint(n - o, n + o))  # type: ignore

    src = np.array(((x0, y0), (x0, y1), (x1, y0)))
    trg = warp_vectors(src)
    return (src.astype(np.float32), trg.astype(np.float32))


def _get_threshold(np_img: np.ndarray) -> int:
    histogram = Image.fromarray(np_img).convert("L").histogram()
    mode = max(*histogram)
    return histogram.index(mode)


def _kaleidoscope(
    np_img: np.ndarray, roi: Tuple[int, int, int, int], factor: int
) -> np.ndarray:
    wrp = _channel_warper(np_img.shape[:2], roi, factor)
    return np.dstack(
        [
            wrp(np_img[:, :, ch]) if i % 2 == 0 else np_img[:, :, ch]
            for (i, ch) in enumerate(range(np_img.shape[2]))
        ]
    )


def _channel_warper(
    size: Tuple[int, int], roi: Tuple[int, int, int, int], factor: int
):
    pts = lambda: cv2.getAffineTransform(*_pts(roi, factor))
    return lambda channel: cv2.warpAffine(channel, pts(), size[::-1])
