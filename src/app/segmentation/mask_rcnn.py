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


class MaskRCNN:
    net: cv2.dnn_Net
    classnames: List[str]

    def __init__(self: "MaskRCNN"):
        self.__load_net()
        self.__load_classnames()
        # model = cv2.dnn_SegmentationModel(net)

    def run(self: "MaskRCNN", imgs: List[ImageType]) -> List[AnalysedImage]:
        return [
            AnalysedImage(
                img=img,
                mask=self.__process_mask(mb),
                label=self.classnames[mb.classId],
            )
            for img in tqdm(imgs)
            for mb in self.__get_maskboxes(img)
        ]

    def __get_maskboxes(self: "MaskRCNN", img: ImageType) -> List[MaskBox]:
        blob = cv2.dnn.blobFromImage(img.np)
        self.net.setInput(blob)
        ([[boxes]], masks) = self.net.forward(
            ["detection_out_final", "detection_masks"]
        )
        return self.__match_masks_to_boxes(
            img.np.shape[:2], list(zip(boxes[:, 1:], masks))
        )

    def __match_masks_to_boxes(
        self: "MaskRCNN",
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
                bounds=self.__calc_bounds(frame, box),
            )
            for box, masks in detection_masks
            if box[1] > thresh
        ]

    def __process_mask(self: "MaskRCNN", maskbox: MaskBox) -> np.ndarray:
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

    def __calc_bounds(
        self: "MaskRCNN",
        frame: Tuple[int, ...],
        box: np.ndarray,
    ) -> Bounds:
        def min_max(shape_dim: int, box_dim: float) -> int:
            scaled_dim = int(shape_dim * box_dim)
            return int(max(0, min(scaled_dim, shape_dim - 1)))

        return Bounds(
            left=min_max(frame[1], box[2]),
            top=min_max(frame[0], box[3]),
            right=min_max(frame[1], box[4]),
            bottom=min_max(frame[0], box[5]),
        )

    def __load_net(self: "MaskRCNN", m: int = 0) -> None:
        weights = os.path.join(WEIGHTS_DIR, WEIGHTS[m])
        conf = os.path.join(WEIGHTS_DIR, CONFIG)
        self.net = cv2.dnn.readNetFromTensorflow(weights, conf)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def __load_classnames(self: "MaskRCNN") -> None:
        with open(os.path.join(WEIGHTS_DIR, CLASS_NAMES), "r") as file:
            self.classnames = file.read().split("\n")
