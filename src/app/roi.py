from typing import Optional

import cv2
import numpy as np
from PIL import Image

from src.app.region import Region


class ROI:
    @staticmethod
    def crop(img: np.ndarray, threshold: int = None) -> np.ndarray:
        if not (region := ROI.__get_roi(img, threshold)):
            return img
        return region.crop(img)

    @staticmethod
    def __get_roi(img: np.ndarray, threshold: int = None) -> Optional[Region]:
        grey = ROI.__get_grey(img)
        thresh = threshold if threshold else ROI.__get_threshold(grey)
        (_, bin) = cv2.threshold(grey, thresh, 255, cv2.THRESH_BINARY)
        (ctrs, _) = cv2.findContours(
            bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not ctrs:
            return None
        (left, top, w, h) = cv2.boundingRect(
            sorted(ctrs, key=lambda ctr: ctr.size, reverse=True)[0]
        )
        return Region(left=left, right=left + w, top=top, bottom=top + h)

    @staticmethod
    def __get_grey(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                return img[:, :, 3]
            if img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    @staticmethod
    def __get_threshold(img: np.ndarray) -> int:
        histogram = Image.fromarray(img).histogram()
        mode = max(*histogram)
        return histogram.index(mode)
