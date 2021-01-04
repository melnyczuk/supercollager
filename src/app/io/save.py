from os import mkdir, path

import numpy as np
from PIL import Image  # type:ignore


class Save:
    @staticmethod
    def np_img(
        np_img: np.ndarray,
        fname: str = None,
        dir: str = None,
    ) -> None:
        if not dir:
            raise ValueError("pls provide dir to save to")

        if not fname:
            raise ValueError("pls provide file name to save as")

        if not path.isdir(dir):
            mkdir(dir)

        real_ext = "png" if np_img.shape[2] == 4 else "jpg"
        out_path = _get_out_path(dir, fname, real_ext)
        img = Image.fromarray(np_img)
        img.save(out_path)

    def pil_img(
        pil_img: Image,
        fname: str = None,
        dir: str = None,
    ) -> None:
        if not dir:
            raise ValueError("pls provide dir to save to")

        if not fname:
            raise ValueError("pls provide file name to save as")

        if not path.isdir(dir):
            mkdir(dir)

        real_ext = "png" if pil_img.mode == "RGBA" else "jpg"
        out_path = _get_out_path(dir, fname, real_ext)
        pil_img.save(out_path)


def _get_out_path(dir: str, fname: str, ext: str) -> str:
    dotext = f".{ext.replace('.', '')}"
    fname_noext = fname.split(dotext)[0]
    full_path = path.join(dir, fname_noext)
    return f"{full_path}.{ext}"
