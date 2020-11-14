from typing import List

from PIL import Image  # type: ignore

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
    aspect: float = 1.0,
    blocky: bool = False,
    limit: int = 0,
    rotate: float = 0.0,
) -> List[Image.Image]:
    masks = [
        Transformation.rotate(mask=analysed_img.mask, rotation=rotate)
        for analysed_img in Segmentation.mask_rcnn(uris=uris, blocky=blocky)
    ]
    masks.sort(key=lambda x: x.size)
    [background, *color_list] = Colors.as_list(limit + 1)
    logger.log(f"segmented {len(masks)} masks from {len(uris)} URIs")
    blocks = [
        Masking.to_rgba(mask=mask, color=color_list[i])
        for i, mask in enumerate(masks[:limit])
    ]
    comp = Composition.layer_to_np(
        layers=blocks, background=background, aspect=aspect
    )
    big = Transformation.scale_up_nparray(arr=comp, scalar=2)
    return [Noise.salt_pepper(big, 0.05)]
