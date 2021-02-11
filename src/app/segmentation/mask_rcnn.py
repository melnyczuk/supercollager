import os
from typing import List, Tuple

import cv2  # type: ignore
import numpy as np
from tqdm.std import tqdm  # type:ignore

from src.app.masking import Masking
from src.app.types import AnalysedImage, Bounds, ImageType, MaskBox

WEIGHTS_DIR = os.path.abspath("weights")
CONFIG = "conf.pbtxt"
CLASS_NAMES = "classes.txt"
WEIGHTS = [
    "mask_rcnn_inception_v2_coco_2018_01_28.pb",
    "mask_rcnn_pyimagesearch_coco.pb",
    "mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.pb",
    "mask_rcnn_resnet50_atrous_coco_2018_01_28.pb",
]


def _load_net(m: int = 0) -> cv2.dnn_Net:
    weights = os.path.join(WEIGHTS_DIR, WEIGHTS[m])
    conf = os.path.join(WEIGHTS_DIR, CONFIG)
    net = cv2.dnn.readNetFromTensorflow(weights, conf)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


with open(os.path.join(WEIGHTS_DIR, CLASS_NAMES), "r") as file:
    coco_class_names = file.read().split("\n")

net = _load_net()
# model = cv2.dnn_SegmentationModel(net)


class MaskRCNNSegmentation:
    @staticmethod
    def run(imgs: List[ImageType]) -> List[AnalysedImage]:
        return [
            AnalysedImage(
                img=img,
                mask=_process_mask(mb),
                label=coco_class_names[mb.classId],
            )
            for img in tqdm(imgs)
            for mb in _get_maskboxes(img)
        ]


def _get_maskboxes(img: ImageType) -> List[MaskBox]:
    blob = cv2.dnn.blobFromImage(img.np)
    net.setInput(blob)
    ([[boxes]], masks) = net.forward(["detection_out_final", "detection_masks"])
    return _match_masks_to_boxes(
        img.np.shape[:2], list(zip(boxes[:, 1:], masks))
    )


def _match_masks_to_boxes(
    frame: Tuple[int, ...],
    detection_masks: List[Tuple[np.ndarray, np.ndarray]],
    thresh: float = 0.0,
) -> List[MaskBox]:
    return [
        MaskBox(
            frame=frame,
            classId=int(box[0]),
            mask=(masks[int(box[0])] * 255).astype(np.uint8),
            score=box[1],
            bounds=_calc_bounds(frame, box),
        )
        for box, masks in detection_masks
        if box[1] > thresh
    ]


def _process_mask(maskbox: MaskBox) -> np.ndarray:
    arr = np.zeros(maskbox.frame)
    target_size = (
        maskbox.bounds.right - maskbox.bounds.left,
        maskbox.bounds.bottom - maskbox.bounds.top,
    )
    arr[
        maskbox.bounds.top : maskbox.bounds.bottom,  # noqa: E203
        maskbox.bounds.left : maskbox.bounds.right,  # noqa: E203
    ] = Masking.upscale(maskbox.mask, target_size)
    return arr


def _calc_bounds(frame: Tuple[int, ...], box: np.ndarray) -> Bounds:
    def min_max(shape_dim: int, box_dim: float) -> int:
        scaled_dim = int(shape_dim * box_dim)
        return int(max(0, min(scaled_dim, shape_dim - 1)))

    return Bounds(
        left=min_max(frame[1], box[2]),
        top=min_max(frame[0], box[3]),
        right=min_max(frame[1], box[4]),
        bottom=min_max(frame[0], box[5]),
    )
