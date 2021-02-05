import os
import pickle
from typing import Any, Callable, List, Tuple

import numpy as np
from gluoncv import data, model_zoo, utils  # type: ignore

from src.app.masking import Masking
from src.app.types import AnalysedImage
from src.logger import logger


class GluonCVSegmentation:
    @staticmethod
    def run(resources: List[str], blocky: bool = False) -> List[AnalysedImage]:
        return [
            analysed_image
            for resource in resources
            for analysed_image in _get_masks_and_labels(
                resource,
                blocky=blocky,
            )
        ]


def _get_masks_and_labels(
    uri: str,
    blocky: bool = False,
) -> List[AnalysedImage]:
    dump_fname = uri.split("/")[-1]
    fname = dump_fname.split(".")[0]

    if uri.startswith("http"):
        logger.log(f"downloading {uri} to {dump_fname}")
        uri = utils.download(uri, path=f"./dump/input/{dump_fname}")

    logger.log(f"creating mxnet_array from {dump_fname}")
    (mxnet_array, np_img) = data.transforms.presets.rcnn.load_test(uri)

    dimensions = _type_safe_dimensions(np_img)
    dump_path = f"./dump/pickles/{fname}.dump"

    return [
        AnalysedImage(
            label=label,
            np_img=np_img,
            mask=Masking.to_block_mat(mask, blocky=blocky),
        )
        for (mask, label) in _segment(mxnet_array, dimensions, dump_path)
    ]


def _segment(
    mxnet_array: List,
    dimensions: Tuple[int, int],
    dump_path: str = "./dump/dump.dump",
) -> List[Tuple[np.ndarray, str]]:
    model_name = "mask_rcnn_resnet50_v1b_coco"
    logger.log(f"loading {model_name} gluoncv model...")
    net = model_zoo.get_model(model_name, pretrained=True)

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


# Computationally intense
def _run_net(net: Any, mxnet_array: List) -> Tuple:
    data = (mx.asnumpy() for [mx] in net(mxnet_array))
    return (*data, net.classes)


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


def _type_safe_dimensions(np_img: np.ndarray) -> Tuple[int, int]:
    (width, height) = np_img.shape[:2][::-1]
    return (width, height)
