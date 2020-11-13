from datetime import datetime
from typing import List

from ..app import (
    IO,
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
    dir: str = "./dump/output",
    n: int = 5,
    smooth: bool = False,
) -> None:
    masks = [
        analysed_img.mask
        for analysed_img in Segmentation.mask_rcnn(uris, smooth=smooth)
    ]
    masks.sort(key=lambda x: x.size)
    [background, *color_list] = Colors.as_list(n + 1)
    logger.log(f"segmented {len(masks)} masks from {len(uris)} URIs")
    blocks = [
        Masking.to_rgba(mask, color_list[i]) for i, mask in enumerate(masks[:n])
    ]
    comp = Composition.layer_to_np(blocks, background)
    big = Transformation.scale_up_nparray(comp, 2)

    output = Noise.salt_pepper(big, 0.05)

    now = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    IO.save.np_array(output, fname=now, dir=dir, ext=".jpg")
