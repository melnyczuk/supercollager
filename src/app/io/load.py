from gluoncv import data, model_zoo, utils  # type: ignore

from typing import List, Tuple
import numpy as np


def gluoncv_model(model_name: str, pretrained=True):
    print(f"loading {model_name} gluoncv model...")
    return model_zoo.get_model(model_name, pretrained=pretrained)


def mxnet_array_from_url(url: str) -> Tuple[List, np.ndarray, str]:
    tmp_fname = url.split("/")[-1]
    print(f"downloading {url} to ./tmp/{tmp_fname}")
    im_fname = utils.download(url, path=f"./tmp/{tmp_fname}")
    print(f"creating mxnet_array from {tmp_fname}")
    mx, img = data.transforms.presets.rcnn.load_test(im_fname)
    return (mx, img, tmp_fname.split(".")[0])
