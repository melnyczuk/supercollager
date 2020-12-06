from typing import Tuple, Union

import cv2  # type: ignore
import numpy as np
from PIL import Image  # type: ignore

Region = Tuple[int, int, int, int]


class ROI:
    @staticmethod
    def crop_np(
        np_img: np.ndarray,
        threshold: Union[int, None] = None,
    ) -> np.ndarray:
        (x0, y0, x1, y1) = ROI.get_bounding_box(np_img, threshold)
        return np_img[y0:y1, x0:x1, :]

    @staticmethod
    def crop_pil(
        pil_img: Image,
        threshold: Union[int, None] = None,
    ) -> Image:
        box = ROI.get_bounding_box(np.array(pil_img), threshold)
        return pil_img.crop(box)

    @staticmethod
    def get_bounding_box(
        np_img: np.ndarray,
        threshold: Union[int, None] = None,
    ) -> Region:
        grey = _get_grey(np_img)
        (x0, y0, w, h) = _get_roi(grey, threshold)
        (x1, y1) = (x0 + w, y0 + h)
        return (x0, y0, x1, y1)


def _get_roi(
    np_grey: np.ndarray,
    threshold: Union[int, None] = None,
) -> Region:
    t = (threshold if threshold else _get_threshold(np_grey)) + 1
    (_, binary) = cv2.threshold(np_grey, t + 1, 255, cv2.THRESH_BINARY)
    (ctrs, _) = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not ctrs:
        return (0, 0, 0, 0)
    sorted_contours = sorted(ctrs, key=lambda ctr: ctr.size, reverse=True)
    return cv2.boundingRect(sorted_contours[0])


def _get_grey(np_img: np.ndarray) -> np.ndarray:
    if n_channels := np_img.shape[2] == 4:
        return np_img[:, :, 3]
    if n_channels == 3:
        return cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    if n_channels == 2:
        return cv2.cvtColor(np_img, cv2.COLOR_BAYER_GR2GRAY)
    return np_img


def _get_threshold(np_img: np.ndarray) -> int:
    histogram = Image.fromarray(np_img).convert("L").histogram()
    mode = max(*histogram)
    return histogram.index(mode)
