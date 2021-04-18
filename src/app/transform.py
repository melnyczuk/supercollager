from typing import Union

import numpy as np
from PIL import Image  # type: ignore


class Transform:
    @staticmethod
    def rotate(
        img: np.ndarray,
        rotate: Union[float, bool] = False,
    ) -> np.ndarray:

        if not rotate:
            return img

        rotation = 90.0 if type(rotate) == bool else rotate
        dimensions = img.shape[:2][::-1]
        rotated = (
            Image.fromarray(img.astype(np.uint8))
            .rotate(rotation)
            .resize(dimensions)
        )
        return np.array(rotated, dtype=np.uint8)
