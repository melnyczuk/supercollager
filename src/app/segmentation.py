import pickle
from random import randint
from os import path, mkdir
import numpy as np
from gluoncv import utils  # type: ignore

from typing import Any, Callable, List, Tuple, Union

from src.app.io import IO
from src.app.masking import Masking


class Segmentation:
    @staticmethod
    def from_url(url: str) -> List:
        (mxnet_array, img, fname) = IO.load.mxnet_array_from_url(url)
        dimensions = _type_safe_dimensions(img)
        dump_path = _get_safe_dump_dir(fname)
        return [
            {
                "label": label,
                "img": Masking.draw_transparency(
                    img, Masking.to_block_mat(mask, randint(3, 12))
                ),
            }
            for (mask, label) in _segment(
                mxnet_array,
                dimensions,
                dump_path=dump_path,
            )
        ]

    @staticmethod
    def masks_from_url(url: str) -> List:
        (mxnet_array, img, fname) = IO.load.mxnet_array_from_url(url)
        dimensions = _type_safe_dimensions(img)
        dump_path = _get_safe_dump_dir(fname)
        return [
            Masking.to_block_mat(mask, randint(3, 12))
            for (mask, _) in _segment(
                mxnet_array,
                dimensions,
                dump_path=dump_path,
            )
        ]


# Computationally intense
def _run_net(net: Any, mxnet_array: List) -> Tuple:
    data = (mx.asnumpy() for [mx] in net(mxnet_array))
    return (*data, net.classes)


def _segment(
    mxnet_array: List,
    dimensions: Tuple[int, int],
    dump_path: str = "./dump/segment_data",
) -> List[Tuple[np.ndarray, str]]:
    net = IO.load.gluoncv_model("mask_rcnn_resnet50_v1b_coco")

    (ids, scores, bboxes, masks, entities) = _pickle_memo(
        lambda: _run_net(net, mxnet_array),
        dump_path,
    )

    exp_masks, id_indexes = utils.viz.expand_mask(
        masks,
        bboxes,
        dimensions,
        scores,
    )

    labels = (f"{entities[int(ids[i])]}-{i}" for i in id_indexes)

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


def _get_safe_dump_dir(fname: str) -> str:
    if not path.isdir(f"./dump/{fname}"):
        mkdir(f"./dump/{fname}")

    return f"./dump/{fname}/dump"


def _type_safe_dimensions(img: np.ndarray) -> Tuple[int, int]:
    (width, height) = img.shape[:2][::-1]
    return (width, height)
