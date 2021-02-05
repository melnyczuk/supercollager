from typing import List

from numpy.random import randint
from tqdm.std import tqdm  # type: ignore

from src.app import ROI, Composition, LabelImage, Masking, Post, Segmentation


def collage(
    uris: List[str], blocky: bool = False, **kwargs
) -> List[LabelImage]:
    analysed_images = Segmentation.mask_rcnn(uris=uris, blocky=blocky)

    imgs = [
        Masking.apply_mask(ai.np_img, ai.mask, **kwargs)
        for ai in tqdm(analysed_images)
    ]

    comp = Composition.layer_images(
        imgs=imgs,
        background=randint(5, 15),
    ).convert("RGB")

    return [LabelImage(Post.process(comp))]


def segment(
    uris: List[str], blocky: bool = False, **kwargs
) -> List[LabelImage]:
    analysed_images = Segmentation.mask_rcnn(uris=uris, blocky=blocky)

    return [
        LabelImage(
            label=ai.label,
            pil_img=ROI.crop(Masking.apply_mask(ai.np_img, ai.mask, **kwargs)),
        )
        for ai in tqdm(analysed_images)
    ]
