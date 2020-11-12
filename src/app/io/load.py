from PIL import Image  # type: ignore
import numpy as np
from gluoncv import data, model_zoo, utils  # type: ignore

from typing import Any, List, Tuple

from src.logger import logger


class Load:
    @staticmethod
    def image(path: str) -> Image:
        return Image.open(path)

    @staticmethod
    def np_array(path: str) -> np.ndarray:
        img = Load.image(path)
        return np.array(img.getdata())

    @staticmethod
    def gluoncv_model(model_name: str, pretrained: bool = True) -> Any:
        logger.log(f"loading {model_name} gluoncv model...")
        return model_zoo.get_model(model_name, pretrained=pretrained)

    @staticmethod
    def mxnet_array_from_url(url: str) -> Tuple[List, np.ndarray, str]:
        dump_fname = url.split("/")[-1]
        logger.log(f"downloading {url}")
        im_fname = utils.download(url, path=f"./dump/input/{dump_fname}")
        logger.log(f"creating mxnet_array from {dump_fname}")
        mx, img = data.transforms.presets.rcnn.load_test(im_fname)
        return (mx, img, dump_fname.split(".")[0])

    @staticmethod
    def mxnet_array_from_path(path: str) -> Tuple[List, np.ndarray, str]:
        logger.log(f"creating mxnet_array from {path}")
        mx, img = data.transforms.presets.rcnn.load_test(path)
        return (mx, img, path.split("/")[-1].split(".")[0])
