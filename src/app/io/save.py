from os import path, mkdir
from PIL import Image  # type:ignore

import numpy as np

from src.logger import logger


class Save:
    @staticmethod
    def np_array(
        arr: np.ndarray,
        fname: str = None,
        dir: str = None,
        ext: str = "png",
    ) -> None:
        if not dir:
            raise ValueError("pls provide dir to save to")
        if not fname:
            raise ValueError("pls provide file name to save as")

        if not path.isdir(dir):
            mkdir(dir)

        out_path = _get_out_path(dir, fname, ext)
        mode = _get_mode(ext)
        logger.log(f"saving image to {out_path}")
        img = Image.fromarray(arr, mode=mode)
        img.save(out_path)

    def image(
        img: Image,
        fname: str = None,
        dir: str = None,
        ext: str = None,
    ) -> None:
        if not dir:
            raise ValueError("pls provide dir to save to")

        if not fname:
            raise ValueError("pls provide file name to save as")

        if not ext:
            raise ValueError("pls provide ext to save")

        if not path.isdir(dir):
            mkdir(dir)

        out_path = _get_out_path(dir, fname, ext)
        mode = _get_mode(ext)
        logger.log(f"saving image to {out_path}.{ext}")
        out = img.convert(mode)
        out.save(out_path)


def _get_out_path(dir: str, fname: str, ext: str) -> str:
    dotext = f".{ext}"
    fname_noext = fname.split(dotext)[0]
    full_path = path.join(dir, fname_noext)
    return f"{full_path}.{ext}"


def _get_mode(ext: str) -> str:
    return "RGB" if ext == "jpg" else "RGBA"
