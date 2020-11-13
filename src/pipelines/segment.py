from typing import List

from PIL import Image  # type:ignore

from ..app import Masking, Segmentation
from ..logger import logger


def segment(
    uris: List[str],
    blocky: bool = True,
) -> List[Image.Image]:
    imgs = [
        Masking.stack_alpha(analysed_img.img, analysed_img.mask)
        for analysed_img in Segmentation.mask_rcnn(uris, blocky=blocky)
    ]
    logger.log(f"segmented {len(imgs)} images from {len(uris)} URIs")
    return [(Image.fromarray(img) for img in imgs)]
