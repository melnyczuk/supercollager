from os import path, mkdir
import pickle
import numpy as np
from gluoncv import utils  # type: ignore
from matplotlib import pyplot as plt

from src.app.io import save, load

from typing import List, Tuple


# Computationally intense
def _run_net(
    mxnet_array: List,
    model_name: str = "mask_rcnn_resnet50_v1b_coco",
) -> Tuple[List[np.ndarray], List[str]]:
    # "faster_rcnn_resnet50_v1b_voc"
    net = load.gluoncv_model(model_name)
    data = (xx[0].asnumpy() for xx in net(mxnet_array))
    return [*data], net.classes


def _segment(
    mxnet_array: List,
    dimensions: Tuple[int, int],
    model_name: str = "mask_rcnn_resnet50_v1b_coco",
    dump_path: str = "./tmp/segment_data",
) -> List[Tuple[np.ndarray, str]]:
    # pickle memoization - replace with something more elegant
    if not path.exists(dump_path):
        data, entities = _run_net(mxnet_array, model_name)
        with open(dump_path, "wb") as fp:
            pickle.dump([data, entities], fp)
    else:
        with open("dump", "rb") as file_dump:
            dump = pickle.load(file_dump)
            [data, entities] = dump

    (ids, scores, bboxes, masks) = data

    exp_masks, _ = utils.viz.expand_mask(masks, bboxes, dimensions, scores)
    labels = (
        entities[int(id)]
        for ([id], [score]) in zip(ids, scores)
        if (score > 0.5)
    )

    return [*zip(exp_masks, labels)]


def _draw_img(img, mask):
    return utils.viz.plot_mask(img, np.array([mask]))


def _get_safe_dump_dir(fname: str) -> str:
    if not path.isdir(f"./tmp/{fname}"):
        mkdir(f"./tmp/{fname}")

    return f"./tmp/{fname}/dump"


def _type_safe_dimensions(img: np.ndarray) -> Tuple[int, int]:
    (width, height) = img.shape[:2][::-1]
    return (width, height)


def from_url(url: str) -> List:
    (x, img, fname) = load.mxnet_array_from_url(url)
    dimensions = _type_safe_dimensions(img)
    dump_path = _get_safe_dump_dir(fname)

    return [
        {"label": label, "img": _draw_img(img, mask)}
        for (mask, label) in _segment(x, dimensions, dump_path=dump_path)
    ]


if __name__ == "__main__":
    from sys import argv

    url = argv[1]
    dir = f"./tmp/{url.split('/')[-1].split('.')[0]}"
    data = from_url(url)
    print(f"saving {len(data)} imgs...")
    for d in data:
        save.np_to_png(
            d["img"],
            fname=d["label"],
            dir=dir,
        )
