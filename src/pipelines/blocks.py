import os
from datetime import datetime
from src.app.transformation import Transformation

from src.logger import logger
from src.app import (
    Colors,
    Composition,
    IO,
    Masking,
    Noise,
    Segmentation,
)

from typing import List


def blocks(
    uris: List[str] = [],
    dir: str = "./dump/output",
    n: int = 5,
    smooth: bool = False,
) -> None:
    masks = [
        analysed_img.mask
        for uri in uris
        if os.path.exists(uri)
        for analysed_img in Segmentation.gluoncv(uri, block_size=int(smooth))
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

    now = datetime.now().strftime("%Y%d%m-%H:%M:%S")
    logger.log(f"saving image to {dir}/{now}.jpg")
    IO.save.np_array(output, fname=now, dir=dir, ext=".jpg")
