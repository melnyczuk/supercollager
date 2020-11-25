from dataclasses import dataclass
from typing import List

from numpy.random import randint
from PIL import Image  # type:ignore

from .app import Composition, Masking, Post, Segmentation, Transformation


@dataclass(frozen=True)
class LabelImage:
    image: Image.Image
    label: str = ""


def segment(
    uris: List[str],
    blocky: bool = True,
    deform: bool = False,
    rotation: float = 0.0,
) -> List[LabelImage]:
    return [
        (
            LabelImage(
                label=ai.label,
                image=Image.fromarray(
                    Masking.stack_alpha(
                        rgb=ai.img,
                        alpha=Transformation.rotate(
                            ai.mask,
                            rotation=(rotation if rotation != 0.0 else 90.0)
                            if (deform or (rotation != 0.0))
                            else 0.0,
                        ),
                    ),
                ),
            )
        )
        for ai in Segmentation.mask_rcnn(uris=uris, blocky=blocky)
    ]


def collage(**kwargs) -> List[LabelImage]:
    comp = Composition.layer_images(
        imgs=[li.image for li in segment(**kwargs)],
        background=randint(5, 50),
    ).convert("RGB")

    return [LabelImage(Post.run(comp))]
