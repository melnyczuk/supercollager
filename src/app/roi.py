from typing import Tuple, Union

import cv2  # type: ignore
import numpy as np
from PIL import Image  # type: ignore

Region = Tuple[int, int, int, int]


class ROI:
    @staticmethod
    def get_bounding_box(
        np_grey: np.ndarray,
        threshold: Union[int, None] = None,
    ) -> Region:
        (x0, y0, w, h) = _get_roi(np_grey, threshold)
        (x1, y1) = (x0 + w, y0 + h)
        return (x0, y0, x1, y1)

    @staticmethod
    def crop_roi(pil_img: Image, threshold: Union[int, None] = None) -> Image:
        grey = cv2.cvtColor(np.array(pil_img), cv2.COLOR_BGR2GRAY)
        box = ROI.get_bounding_box(grey, threshold)
        return pil_img.crop(box)


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
    sorted_contours = sorted(ctrs, key=lambda ctr: ctr.size, reverse=True)
    return (
        cv2.boundingRect(sorted_contours[0])
        if (len(sorted_contours))
        else [0, 0, 0, 0]
    )


def _get_threshold(np_img: np.ndarray) -> int:
    histogram = Image.fromarray(np_img).convert("L").histogram()
    mode = max(*histogram)
    return histogram.index(mode)
