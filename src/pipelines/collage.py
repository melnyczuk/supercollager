from src.app.transformation import Transformation
from src.app.segmentation.types import AnalysedImage
from typing import List

from numpy.random import randint
from PIL import Image  # type: ignore

from ..app import Composition, Masking, Segmentation, Transformation
from ..logger import logger


def collage(
    uris: List[str] = [],
    smooth: bool = False,
    deform: bool = False,
    aspect: float = 1.0,
) -> List[Image.Image]:
    analysed_imgs = Segmentation.mask_rcnn(uris, smooth=smooth)
    rgbas = [
        Masking.stack_alpha(
            ai.img,
            Transformation.rotate(ai.mask, rotation=90 if deform else 0),
        )
        for ai in analysed_imgs
    ]
    logger.log(f"segmented {len(rgbas)} images from {len(uris)} URIs")
    lum = randint(5, 50)
    return [Composition.layer_to_image(rgbas, (lum, lum, lum), aspect=aspect)]
