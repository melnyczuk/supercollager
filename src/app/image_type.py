from typing import Union

import numpy as np
from PIL import Image  # type: ignore


class ImageType:
    def __init__(self: "ImageType", img: Union[Image.Image, np.ndarray]):
        self.pil = (
            img
            if isinstance(img, Image.Image)
            else Image.fromarray(img.astype(np.uint8))
        )
        self.np = np.array(self.pil)
        self.dimensions = self.pil.size
        self.channels = self.np.shape[2] if len(self.np.shape) > 2 else 1
