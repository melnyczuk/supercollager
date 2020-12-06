from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.random import randint
from PIL import Image  # type: ignore
from tqdm.std import tqdm  # type: ignore

from .app import ROI, AnalysedImage, Composition, Post, Segmentation, Transform


@dataclass(frozen=True)
class LabelImage:
    pil_img: Image
    label: str = ""


def collage(**kwargs) -> List[LabelImage]:
    comp = Composition.layer_images(
        imgs=[label_image.pil_img for label_image in segment(**kwargs)],
        background=randint(5, 15),
    ).convert("RGB")

    return [LabelImage(Post.run(comp))]


def segment(
    uris: List[str],
    blocky: bool = False,
    deform: bool = False,
    rotation: float = 0.0,
    warp: float = 0.0,
) -> List[LabelImage]:
    real_rotation = (
        0.0 if not deform and not rotation else rotation if rotation else 90.0
    )

    analysed_images = Segmentation.mask_rcnn(uris=uris, blocky=blocky)

    return [
        LabelImage(
            label=ai.label,
            pil_img=_make_segment(ai, warp, real_rotation),
        )
        for ai in tqdm(analysed_images)
    ]


def _make_segment(
    ai: AnalysedImage,
    warp_factor: float = 0.0,
    rotation: float = 0.0,
) -> Image:
    mask = Transform.rotate(ai.mask, factor=rotation) if rotation else ai.mask
    rgba = np.dstack((ai.np_img, mask))  # type: ignore
    warp = Transform.warp(rgba, warp_factor) if warp_factor else rgba
    pil_img = Image.fromarray(warp)
    return ROI.crop_pil(pil_img)
