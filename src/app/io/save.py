from os import mkdir, path

import numpy as np
from PIL import Image  # type:ignore


class Save:
    @staticmethod
    def np_img(
        np_img: np.ndarray,
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
        mode = "RGBA" if np_img.shape[2] == 4 and ext == "png" else "RGB"
        img = Image.fromarray(np_img, mode=mode)
        img.save(out_path)

    def pil_img(
        pil_img: Image,
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
        mode = "RGBA" if ext == "png" else "RGB"
        out = pil_img.convert(mode)
        out.save(out_path)


def _get_out_path(dir: str, fname: str, ext: str) -> str:
    dotext = f".{ext}"
    fname_noext = fname.split(dotext)[0]
    full_path = path.join(dir, fname_noext)
    return f"{full_path}.{ext}"
