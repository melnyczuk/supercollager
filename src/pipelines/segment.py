from PIL import Image  # type: ignore

from typing import List

from src.app import Masking, Segmentation
from src.logger import logger


def segment(uris: List[str], block_size: int = 0) -> List[Image.Image]:
    imgs = [Segmentation.gluoncv(uri, block_size) for uri in uris]
    rgbas = [
        Masking.draw_transparency(analysed_img.img, analysed_img.mask)
        for analysed_imgs in imgs
        for analysed_img in analysed_imgs
    ]
    logger.log(f"segmented {len(rgbas)} images from {len(uris)} URIs")
    return [Image.fromarray(rgba) for rgba in rgbas]
