from typing import List

from numpy.random import randint
from PIL import Image  # type:ignore

from ..app import Composition, Masking, Segmentation, Transformation
from ..logger import logger


def collage(
    uris: List[str] = [],
    aspect: float = 1.0,
    blocky: bool = False,
    deform: bool = False,
    rotation: float = 0.0,
) -> List[Image.Image]:
    rgbas = [
        Masking.stack_alpha(
            ai.img,
            Transformation.rotate(
                ai.mask,
                rotation=(rotation if rotation != 0.0 else 90.0)
                if (deform or (rotation != 0.0))
                else 0.0,
            ),
        )
        for ai in Segmentation.mask_rcnn(uris, blocky=blocky)
    ]
    logger.log(f"segmented {len(rgbas)} images from {len(uris)} URIs")
    lum = randint(5, 50)
    return [Composition.layer_to_image(rgbas, (lum, lum, lum), aspect=aspect)]
