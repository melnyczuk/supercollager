from dataclasses import dataclass
from io import BytesIO
from os import mkdir, path
from typing import Union

import numpy as np
import requests
from PIL import Image  # type: ignore
from PIL.Image import Image as ImageType  # type: ignore

EXTS = ["jpg", "jpeg", "png", "tif", "tiff"]


@dataclass(frozen=True)
class IO:
    @staticmethod
    def load(path: str) -> ImageType:
        resource = (
            path
            if not path.startswith("http")
            else BytesIO(requests.get(path).content)
        )
        return Image.open(resource).convert("RGB")

    @staticmethod
    def save(
        img: Union[ImageType, np.ndarray],
        fname: str = None,
        dir: str = None,
    ) -> None:
        if not dir:
            raise ValueError("pls provide dir to save to")

        if not fname:
            raise ValueError("pls provide file name to save as")

        if not path.isdir(dir):
            mkdir(dir)

        pil_img = img if isinstance(img, ImageType) else Image.fromarray(img)
        ext = "png" if pil_img.mode == "RGBA" else "jpg"
        out_path = f"{path.join(dir, _remove_ext(fname))}.{ext}"
        pil_img.save(out_path)


def _remove_ext(fname: str) -> str:
    for ext in EXTS:
        if ext in fname:
            return fname.split(f".{ext}")[0]
    return fname
