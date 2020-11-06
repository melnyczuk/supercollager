from os import path, mkdir
from PIL import Image  # type:ignore

import numpy as np


class Save:
    @staticmethod
    def np_array(
        arr: np.ndarray,
        fname: str = None,
        dir: str = None,
        mode: str = None,
        ext: str = ".png",
    ) -> None:
        if not dir:
            raise ValueError("pls provide dir to save to")
        if not fname:
            raise ValueError("pls provide file name to save as")

        if not path.isdir(dir):
            mkdir(dir)

        out_path = path.join(dir, fname.split(ext)[0])
        Image.fromarray(arr, mode=mode).save(f"{out_path}{ext}")

    def image(
        img: Image,
        fname: str = None,
        dir: str = None,
        ext: str = ".png",
    ) -> None:
        if not dir:
            raise ValueError("pls provide dir to save to")
        if not fname:
            raise ValueError("pls provide file name to save as")

        if not path.isdir(dir):
            mkdir(dir)

        out_path = path.join(dir, fname.split(ext)[0])
        img.save(f"{out_path}{ext}")
