from typing import Tuple

import cv2  # type: ignore

from src.app.types import ImageType

Region = Tuple[int, int, int, int]


class ROI:
    @staticmethod
    def crop(img: ImageType, threshold: int = None) -> ImageType:
        (x0, y0, x1, y1) = _get_bounding_box(img, threshold)
        return ImageType(img.np[y0:y1, x0:x1, :])


def _get_bounding_box(img: ImageType, threshold: int = None) -> Region:
    (x0, y0, w, h) = _get_roi(img, threshold)
    (x1, y1) = (x0 + w, y0 + h)
    return (x0, y0, x1, y1)


def _get_roi(img: ImageType, threshold: int = None) -> Region:
    grey = _get_grey(img)
    t = (threshold if threshold else _get_threshold(grey)) + 1
    (_, binary) = cv2.threshold(grey.np, t + 1, 255, cv2.THRESH_BINARY)
    (ctrs, _) = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not ctrs:
        return (0, 0, 0, 0)
    sorted_contours = sorted(ctrs, key=lambda ctr: ctr.size, reverse=True)
    return cv2.boundingRect(sorted_contours[0])


def _get_grey(img: ImageType) -> ImageType:
    if img.channels == 4:
        return ImageType(img.np[:, :, 3])
    if img.channels == 3:
        return ImageType(cv2.cvtColor(img.np, cv2.COLOR_RGB2GRAY))
    return img


def _get_threshold(img: ImageType) -> int:
    histogram = img.pil.histogram()
    mode = max(*histogram)
    return histogram.index(mode)
