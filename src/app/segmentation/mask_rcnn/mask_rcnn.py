import os
from typing import Iterable, Union

import cv2  # type: ignore
import numpy as np

from src.app.masking import Masking
from src.app.roi import ROI

from .mask_box import MaskBox

WEIGHTS_DIR = os.path.abspath("weights")
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

    def image(
        self: "MaskRCNN",
        img: np.ndarray,
        rotate: Union[float, bool] = False,
    ) -> Iterable[np.ndarray]:
        for mask in self.__analyse(img):
            rgba = Masking.apply_mask(img=img, mask=mask, rotate=rotate)
            yield ROI.crop(rgba)

    def mask_frame(
        self: "MaskRCNN",
        frame: np.ndarray,
        confidence_threshold: float = 0.0,
    ) -> np.ndarray:
        try:
            masks = list(self.__analyse(frame, confidence_threshold))
            if not len(masks):
                raise ValueError()
            return np.array(np.mean(np.array(masks), axis=0), dtype=np.uint8)
        except Exception:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

    def __analyse(
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
