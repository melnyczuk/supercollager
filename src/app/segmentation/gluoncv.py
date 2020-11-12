import os
from dataclasses import dataclass
import pickle
from random import randint
import numpy as np
from gluoncv import utils  # type: ignore

from typing import Any, Callable, List, Tuple

from src.app.io import IO
from src.app.masking import Masking


@dataclass
class AnalysedImage:
    label: str
    img: np.ndarray
    mask: np.ndarray


class GluonCVSegmentation:
    @staticmethod
    def run(resource: str, block_size: int = 0) -> List[AnalysedImage]:
        fn = (
            IO.load.mxnet_array_from_url
            if resource.startswith("http")
            else IO.load.mxnet_array_from_path
        )
        return _get_masks_and_labels(*fn(resource), block_size=block_size)


def _get_masks_and_labels(
    mxnet_array: List, img: np.ndarray, fname: str, block_size: int = 0
) -> List[AnalysedImage]:
    dimensions = _type_safe_dimensions(img)
    dump_path = f"./dump/pickles/{fname}.dump"

    return [
        AnalysedImage(
            label=label,
            img=img,
            mask=Masking.to_block_mat(
                mask,
                (lambda: block_size if block_size != 0 else randint(3, 12))(),
            ),
        )
        for (mask, label) in _segment(
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
    dump_path: str = "./dump/dump.dump",
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
    if not os.path.exists(dump_path):
        dump = fn()
        with open(dump_path, "wb") as fp:
            pickle.dump(dump, fp)
    else:
        with open(dump_path, "rb") as fp:
            dump = pickle.load(fp)
    return dump


def _type_safe_dimensions(img: np.ndarray) -> Tuple[int, int]:
    (width, height) = img.shape[:2][::-1]
    return (width, height)
