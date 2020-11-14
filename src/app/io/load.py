from io import BytesIO
from typing import Any, List, Tuple

import numpy as np
import requests
from gluoncv import data, model_zoo, utils  # type: ignore
from PIL import Image  # type: ignore

from ...logger import logger


def is_url(uri: str) -> bool:
    return uri.startswith("http")


class Load:
    @staticmethod
    def image(path: str) -> Image:
        resource = (
            path if not is_url(path) else BytesIO(requests.get(path).content)
        )
        return Image.open(resource)

    @staticmethod
    def np_array(path: str) -> np.ndarray:
        img = Load.image(path)
        return np.array(img)

    @staticmethod
    def gluoncv_model(model_name: str, pretrained: bool = True) -> Any:
        logger.log(f"loading {model_name} gluoncv model...")
        return model_zoo.get_model(model_name, pretrained=pretrained)

    @staticmethod
    def mxnet_array(uri: str) -> Tuple[List, np.ndarray, str]:
        dump_fname = uri.split("/")[-1]

        if is_url(uri):
            logger.log(f"downloading {uri} to {dump_fname}")
            uri = utils.download(uri, path=f"./dump/input/{dump_fname}")

        logger.log(f"creating mxnet_array from {dump_fname}")
        (mx, img) = data.transforms.presets.rcnn.load_test(uri)
        return (mx, img, dump_fname.split(".")[0])
