from typing import Tuple, Union

import cv2
import numpy as np
from PIL import Image


class Transform:
    @staticmethod
    def resize(img: np.ndarray, dsize: Tuple[int, int]) -> np.ndarray:
        return cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def rotate(
        img: np.ndarray,
        rotate: Union[float, bool] = True,
    ) -> np.ndarray:
        if not rotate:
            return img
        rotation = 90.0 if type(rotate) == bool else rotate
        rotated = Image.fromarray(img.astype(np.uint8)).rotate(rotation)
        return np.array(rotated, dtype=np.uint8)
