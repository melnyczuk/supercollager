from typing import List

import numpy as np
from numpy.random import randint
from PIL import Image  # type: ignore

from ..app import Composition
from .segment import segment


def collage(
    uris: List[str] = [],
    smooth: bool = False,
) -> List[Image.Image]:
    transparencies = [np.array(img) for img in segment(uris, smooth)]
    lum = randint(5, 50)
    return [Composition.layer_to_image(transparencies, (lum, lum, lum))]
