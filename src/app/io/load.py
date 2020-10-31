from gluoncv import data, model_zoo, utils  # type: ignore

from typing import Any, List, Tuple
import numpy as np

from src.logger import logger


def gluoncv_model(model_name: str, pretrained: bool = True) -> Any:
    logger.log(f"loading {model_name} gluoncv model...")
    return model_zoo.get_model(model_name, pretrained=pretrained)


def mxnet_array_from_url(url: str) -> Tuple[List, np.ndarray, str]:
    dump_fname = url.split("/")[-1]
    logger.log(f"downloading {url}")
    im_fname = utils.download(url, path=f"./dump/{dump_fname}")
    logger.log(f"creating mxnet_array from {dump_fname}")
    mx, img = data.transforms.presets.rcnn.load_test(im_fname)
    return (mx, img, dump_fname.split(".")[0])
