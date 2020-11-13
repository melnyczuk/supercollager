from typing import List

from PIL import Image  # type: ignore

from ..app import Masking, Segmentation
from ..logger import logger


def segment(uris: List[str], smooth: bool = False) -> List[Image.Image]:
    imgs = Segmentation.mask_rcnn(uris, smooth)
    rgbas = [
        Masking.stack_alpha(analysed_img.img, analysed_img.mask)
        for analysed_img in imgs
    ]
    logger.log(f"segmented {len(rgbas)} images from {len(uris)} URIs")
    return [Image.fromarray(rgba) for rgba in rgbas]
