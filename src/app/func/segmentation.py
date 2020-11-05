import pickle
from random import randint
from os import path, mkdir
from gluoncv import utils  # type: ignore

import numpy as np
from numpy import dstack as np_dstack  # type: ignore

from typing import Callable, List, Tuple

from src.app.io import Load
from src.app.func import masking


def from_url(url: str) -> List:
    (mxnet_array, img, fname) = Load.mxnet_array_from_url(url)
    dimensions = _type_safe_dimensions(img)
    dump_path = _get_safe_dump_dir(fname)
    return [
        {
            "label": label,
            "img": _draw_transparency(img, mask, randint(3, 12)),
        }
        for (mask, label) in _segment(
            mxnet_array, dimensions, dump_path=dump_path
        )
    ]


def _segment(
    mxnet_array: List,
    dimensions: Tuple[int, int],
    model_name: str = "mask_rcnn_resnet50_v1b_coco",
    dump_path: str = "./dump/segment_data",
) -> List[Tuple[np.ndarray, str]]:
    (ids, scores, bboxes, masks, entities) = _pickle_memo(
        lambda: _run_net(mxnet_array, model_name),
        dump_path,
    )
    exp_masks, id_indexes = utils.viz.expand_mask(
        masks, bboxes, dimensions, scores
    )
    labels = (f"{entities[int(ids[i])]}-{i}" for i in id_indexes)
    return [*zip(exp_masks, labels)]


# Computationally intense
def _run_net(
    mxnet_array: List,
    model_name: str = "mask_rcnn_resnet50_v1b_coco",
) -> Tuple:
    net = load.gluoncv_model(model_name)
    data = (mx.asnumpy() for [mx] in net(mxnet_array))
    return (*data, net.classes)


def _draw_transparency(
    rgb: np.ndarray, mask: np.ndarray, edge_grain: int = 1
) -> np.ndarray:
    downsampled_blocky_mask = masking.to_block_mat(mask, edge_grain)
    alpha = downsampled_blocky_mask * np.full(mask.shape, 255.0, dtype=np.uint8)
    return np_dstack((rgb, alpha))


# pickle memoization - replace with something more elegant
def _pickle_memo(fn: Callable, dump_path: str) -> List:
    if not path.exists(dump_path):
        dump = fn()
        with open(dump_path, "wb") as fp:
            pickle.dump(dump, fp)
    else:
        with open(dump_path, "rb") as fp:
            dump = pickle.load(fp)
    return dump


def _get_safe_dump_dir(fname: str) -> str:
    if not path.isdir(f"./dump/{fname}"):
        mkdir(f"./dump/{fname}")

    return f"./dump/{fname}/dump"


def _type_safe_dimensions(img: np.ndarray) -> Tuple[int, int]:
    (width, height) = img.shape[:2][::-1]
    return (width, height)


if __name__ == "__main__":
    from sys import argv
    from src.logger import logger
    from src.app.io import Save

    url = argv[1]
    dir = f"./dump/{url.split('/')[-1].split('.')[0]}"
    data = from_url(url)
    logger.log(f"saving {len(data)} imgs")
    for d in data:
        Save.np_to_png(d["img"], fname=d["label"], dir=dir)
