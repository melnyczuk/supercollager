import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

import cv2  # type:ignore
import numpy as np
from PIL import Image  # type: ignore
from tqdm.std import tqdm  # type:ignore

from ..masking import Masking
from .types import AnalysedImage, Bounds, MaskBox

WEIGHTS_DIR = os.path.abspath("weights/mask_rcnn_inception_v2_coco_2018_01_28")
WEIGHTS = os.path.join(WEIGHTS_DIR, "frozen_inference_graph.pb")
CONFIG = os.path.join(WEIGHTS_DIR, "conf.pbtxt")


def _load_net():
    net = cv2.dnn.readNetFromTensorflow(WEIGHTS, CONFIG)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


net = _load_net()


@dataclass
class MaskRCNNSegmentation:
    @staticmethod
    def run(
        uris: List[str],
        smooth: bool = False,
    ) -> List[AnalysedImage]:
        imgs = [Image.open(uri) for uri in uris]
        return [
            AnalysedImage(
                img=np.array(img, dtype=np.uint8),
                mask=Masking.to_block_mat(
                    _process_mask(img.size, mb), smooth
                ).astype(np.uint8),
            )
            for img in tqdm(imgs)
            for mb in _get_maskboxes(img)
        ]


def _get_maskboxes(img: Image) -> List[MaskBox]:
    blob = _get_blob(img)
    net.setInput(blob)
    ([[boxes]], masks) = net.forward(["detection_out_final", "detection_masks"])
    return _match_masks_to_boxes(img.size, list(zip(boxes[:, 1:], masks)))


def _get_blob(img: Image) -> Any:
    img_rgb = np.array(img)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    return cv2.dnn.blobFromImage(img_gray, swapRB=True, crop=False)


def _match_masks_to_boxes(
    frame_shape: Tuple[int, int],
    detection_masks: Iterable[Tuple[np.ndarray, np.ndarray]],
    thresh: float = 0.0,
) -> List[MaskBox]:
    return [
        MaskBox(
            classId=int(box[0]),
            mask=masks[int(box[0])],
            score=box[1],
            bounds=_calc_bounds(frame_shape, box),
        )
        for box, masks in detection_masks
        if box[1] > thresh
    ]


def _process_mask(shape: Tuple[int, int], maskbox: MaskBox) -> np.ndarray:
    mask = _resize_mask(maskbox)
    inverted = _invert_mask(mask)
    arr = np.zeros(shape)
    arr[
        int(maskbox.bounds.top) : int(maskbox.bounds.bottom + 1),  # noqa: E203
        int(maskbox.bounds.left) : int(maskbox.bounds.right + 1),  # noqa: E203
    ] = inverted
    return arr.transpose()


def _resize_mask(maskbox: MaskBox) -> np.ndarray:
    return cv2.resize(
        maskbox.mask,
        (
            maskbox.bounds.right - maskbox.bounds.left + 1,
            maskbox.bounds.bottom - maskbox.bounds.top + 1,
        ),
    )


def _invert_mask(mask: np.ndarray) -> np.ndarray:
    int_mask = (mask * 255).astype(np.uint8)
    return np.full(mask.shape, 255, dtype=np.uint8) - int_mask


def _calc_bounds(shape, box) -> Bounds:
    def min_max(shape_dim, box_index):
        return max(0, min(int(shape_dim * box_index), shape_dim - 1))

    left = min_max(shape[1], box[2])
    top = min_max(shape[0], box[3])
    right = min_max(shape[1], box[4])
    bottom = min_max(shape[0], box[5])

    return Bounds(left, top, right, bottom)