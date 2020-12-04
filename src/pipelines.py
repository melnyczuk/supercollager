from dataclasses import dataclass
from random import random
from typing import List

import numpy as np
from numpy.random import randint
from PIL import Image  # type: ignore
from tqdm.std import tqdm  # type: ignore

from src.app.segmentation.types import AnalysedImage

from .app import Composition, Masking, Post, Segmentation, Transform


@dataclass(frozen=True)
class LabelImage:
    pil_img: Image
    label: str = ""


def segment(
    uris: List[str],
    blocky: bool = True,
    deform: bool = False,
    rotation: float = 0.0,
    warp: float = 0.0,
) -> List[LabelImage]:
    real_rotation = (
        (rotation if rotation != 0.0 else (90.0 * random()))
        if (deform or (rotation != 0.0))
        else 0.0
    )

    analysed_images = Segmentation.mask_rcnn(uris=uris, blocky=blocky)

    return [
        LabelImage(
            label=ai.label,
            pil_img=_make_segment(ai, warp, real_rotation),
        )
        for ai in tqdm(analysed_images)
    ]


def _make_segment(ai: AnalysedImage, warp: float, rotation: float) -> Image:
    kaleidoscope = bool(int(random() * 1.2))
    rgb = Transform.warp(ai.np_img, warp, kaleidoscope)
    alpha = Transform.rotate(ai.mask, rotation=rotation)
    return Image.fromarray(Masking.stack_alpha(rgb=rgb, alpha=alpha))


def collage(**kwargs) -> List[LabelImage]:
    comp = Composition.layer_images(
        imgs=[label_image.pil_img for label_image in segment(**kwargs)],
        background=randint(5, 50),
    ).convert("RGB")

    return [LabelImage(Post.run(comp))]
