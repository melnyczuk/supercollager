import pickle
from os import path, mkdir
from gluoncv import utils  # type: ignore

import numpy as np
from numpy import dstack  # type:ignore

from src.app.io import save, load

from typing import Callable, List, Tuple


def from_url(url: str) -> List:
    (mxnet_array, img, fname) = load.mxnet_array_from_url(url)
    dimensions = _type_safe_dimensions(img)
    dump_path = _get_safe_dump_dir(fname)
    return [
        {"label": label, "img": _draw_transparency(img, mask)}
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
    exp_masks, _ = utils.viz.expand_mask(masks, bboxes, dimensions, scores)
    labels = (
        f"{entities[int(id)]}-{i}"
        for i, ([id], [score]) in enumerate(zip(ids, scores))
        if (score > 0.5)
    )
    return [*zip(exp_masks, labels)]


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


# Computationally intense
def _run_net(
    mxnet_array: List,
    model_name: str = "mask_rcnn_resnet50_v1b_coco",
) -> Tuple:
    net = load.gluoncv_model(model_name)
    data = (xx[0].asnumpy() for xx in net(mxnet_array))
    return (*data, net.classes)


def _draw_transparency(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    alpha = mask * np.full(mask.shape, 255.0, dtype=np.uint8)
    return dstack((rgb, alpha))


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

    url = argv[1]
    dir = f"./dump/{url.split('/')[-1].split('.')[0]}"
    data = from_url(url)
    logger.log(f"saving {len(data)} imgs")
    for d in data:
        save.np_to_png(d["img"], fname=d["label"], dir=dir)
