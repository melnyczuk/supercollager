from os import path, mkdir
from PIL import Image  # type:ignore
import numpy as np
from gluoncv import data, model_zoo, utils  # type: ignore

from typing import Any, List, Tuple

from src.logger import logger


class Load:
    @staticmethod
    def gluoncv_model(model_name: str, pretrained: bool = True) -> Any:
        logger.log(f"loading {model_name} gluoncv model...")
        return model_zoo.get_model(model_name, pretrained=pretrained)

    @staticmethod
    def mxnet_array_from_url(url: str) -> Tuple[List, np.ndarray, str]:
        dump_fname = url.split("/")[-1]
        logger.log(f"downloading {url}")
        im_fname = utils.download(url, path=f"./dump/{dump_fname}")
        logger.log(f"creating mxnet_array from {dump_fname}")
        mx, img = data.transforms.presets.rcnn.load_test(im_fname)
        return (mx, img, dump_fname.split(".")[0])


class Save:
    @staticmethod
    def np_to_png(
        arr: np.ndarray,
        fname: str = None,
        dir: str = None,
        mode: str = None,
    ) -> None:
        if not dir:
            raise ValueError("pls provide dir to save to")
        if not fname:
            raise ValueError("pls provide file name to save as")

        if not path.isdir(dir):
            mkdir(dir)

        out_path = path.join(dir, fname.split(".png")[0])
        Image.fromarray(arr, mode=mode).save(f"{out_path}.png")
