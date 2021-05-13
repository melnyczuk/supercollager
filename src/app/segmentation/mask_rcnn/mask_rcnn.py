import os
from typing import Iterable

import cv2  # type: ignore
import numpy as np

from .mask_box import MaskBox

WEIGHTS_DIR = os.path.abspath("weights/mask_rcnn")
CONFIG = "conf.pbtxt"
WEIGHTS = [
    "mask_rcnn_inception_v2_coco_2018_01_28.pb",
    "mask_rcnn_pyimagesearch_coco.pb",
    "mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.pb",
    "mask_rcnn_resnet50_atrous_coco_2018_01_28.pb",
]


class MaskRCNN:
    net: cv2.dnn_Net

    def __init__(self: "MaskRCNN"):
        self.__load_net()
        # self.model = cv2.dnn_SegmentationModel(net)

    def mask(
        self: "MaskRCNN",
        frame: np.ndarray,
        confidence_threshold: float = 0.0,
    ) -> Iterable[np.ndarray]:
        blob = cv2.dnn.blobFromImage(frame)
        self.net.setInput(blob)
        ([[boxes]], masks) = self.net.forward(
            ["detection_out_final", "detection_masks"]
        )
        shape = (frame.shape[0], frame.shape[1])
        for box, masks in zip(boxes[:, 1:], masks):
            if box[1] > confidence_threshold:
                maskbox = MaskBox(shape=shape, box=box, masks=masks)
                yield maskbox.mask.astype(np.uint8)
        return

    def __load_net(self: "MaskRCNN", m: int = 0) -> None:
        weights = os.path.join(WEIGHTS_DIR, WEIGHTS[m])
        conf = os.path.join(WEIGHTS_DIR, CONFIG)
        self.net = cv2.dnn.readNetFromTensorflow(weights, conf)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
