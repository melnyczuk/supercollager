import os
from typing import List, Tuple

import cv2  # type: ignore
import numpy as np
from PIL import Image  # type: ignore
from tqdm.std import tqdm  # type:ignore

from ...logger import logger
from ..io import IO
from ..masking import Masking
from ..types import AnalysedImage, Bounds, MaskBox

WEIGHTS_DIR = os.path.abspath("weights")
CONFIG = "conf.pbtxt"
CLASS_NAMES = "classes.txt"
WEIGHTS = [
    "mask_rcnn_inception_v2_coco_2018_01_28.pb",
    "mask_rcnn_pyimagesearch_coco.pb",
    "mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.pb",
    "mask_rcnn_resnet50_atrous_coco_2018_01_28.pb",
]


def _load_net(m: int = 0) -> cv2.dnn_SegmentationModel:
    weights = os.path.join(WEIGHTS_DIR, WEIGHTS[m])
    conf = os.path.join(WEIGHTS_DIR, CONFIG)
    net = cv2.dnn.readNetFromTensorflow(weights, conf)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


net = _load_net()

with open(os.path.join(WEIGHTS_DIR, CLASS_NAMES), "r") as file:
    coco_class_names = file.read().split("\n")


class MaskRCNNSegmentation:
    @staticmethod
    def run(
        uris: List[str],
        blocky: bool = False,
    ) -> List[AnalysedImage]:
        pil_imgs = [IO.load.pil_img(uri) for uri in uris]

        analysed_imgs = [
            AnalysedImage(
                np_img=np.array(img, dtype=np.uint8),
                mask=Masking.to_block_mat(
                    _process_mask(mb),
                    blocky,
                ).astype(np.uint8),
                label=coco_class_names[mb.classId],
            )
            for img in tqdm(pil_imgs)
            for mb in _get_maskboxes(img)
        ]

        logger.log(
            f"segmented {len(analysed_imgs)} images from {len(uris)} URIs finding {len(set(ai.label for ai in analysed_imgs))} different types of object"  # noqa: E501
        )

        return analysed_imgs


def _get_maskboxes(pil_img: Image.Image) -> List[MaskBox]:
    cv_img = np.array(pil_img)
    blob = cv2.dnn.blobFromImage(cv_img)
    net.setInput(blob)
    ([[boxes]], masks) = net.forward(["detection_out_final", "detection_masks"])
    return _match_masks_to_boxes(
        cv_img.shape[:2], list(zip(boxes[:, 1:], masks))
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
            mask=masks[int(box[0])],
            score=box[1],
            bounds=_calc_bounds(frame, box),
        )
        for box, masks in detection_masks
        if box[1] > thresh
    ]


def _process_mask(maskbox: MaskBox) -> np.ndarray:
    arr = np.zeros(maskbox.frame)
    arr[
        maskbox.bounds.top : maskbox.bounds.bottom,  # noqa: E203
        maskbox.bounds.left : maskbox.bounds.right,  # noqa: E203
    ] = _resize_mask(maskbox)
    return arr


def _resize_mask(maskbox: MaskBox) -> np.ndarray:
    target_size = (
        maskbox.bounds.right - maskbox.bounds.left,
        maskbox.bounds.bottom - maskbox.bounds.top,
    )
    return (
        cv2.resize(maskbox.mask, target_size, interpolation=cv2.INTER_NEAREST)
        * 255
    ).astype(np.uint8)


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
