from typing import List

from PIL import Image  # type: ignore
from tqdm.std import tqdm  # type: ignore

from ..app import (
    Colors,
    Composition,
    Masking,
    Noise,
    Segmentation,
    Transformation,
)
from ..logger import logger


def blocks(
    uris: List[str] = [],
    smooth: bool = False,
    n: int = 5,
    aspect: float = 1.0,
) -> List[Image.Image]:
    masks = [
        analysed_img.mask
        for analysed_img in Segmentation.mask_rcnn(uris, smooth=smooth)
    ]
    masks.sort(key=lambda x: x.size)
    [background, *color_list] = Colors.as_list(n + 1)
    logger.log(f"segmented {len(masks)} masks from {len(uris)} URIs")
    blocks = [
        Masking.to_rgba(mask, color_list[i])
        for i, mask in tqdm(enumerate(masks[:n]))
    ]
    comp = Composition.layer_to_np(blocks, background, aspect)
    big = Transformation.scale_up_nparray(comp, 2)
    return [Noise.salt_pepper(big, 0.05)]
