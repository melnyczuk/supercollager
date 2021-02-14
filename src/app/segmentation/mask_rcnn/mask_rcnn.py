import os
from typing import List, Union

import cv2  # type: ignore
import numpy as np
from tqdm.std import tqdm  # type:ignore

from src.app.image_type import ImageType
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

    def run(
        self: "MaskRCNN",
        imgs: List[ImageType],
        rotate: Union[float, bool] = False,
    ) -> List[ImageType]:
        return [
            ROI.crop(Masking.apply_mask(img=img, mask=mask, rotate=rotate))
            for img in tqdm(imgs)
            for mask in self.__analyse(img)
        ]

    def __analyse(
        self: "MaskRCNN",
        img: ImageType,
        thresh: float = 0.0,
    ) -> List[np.ndarray]:
        blob = cv2.dnn.blobFromImage(img.np)
        self.net.setInput(blob)
        ([[boxes]], masks) = self.net.forward(
            ["detection_out_final", "detection_masks"]
        )
        shape = (img.np.shape[0], img.np.shape[1])
        return [
            MaskBox(shape=shape, box=box, masks=masks).mask
            for box, masks in zip(boxes[:, 1:], masks)
            if box[1] > thresh
        ]

    def __load_net(self: "MaskRCNN", m: int = 0) -> None:
        weights = os.path.join(WEIGHTS_DIR, WEIGHTS[m])
        conf = os.path.join(WEIGHTS_DIR, CONFIG)
        self.net = cv2.dnn.readNetFromTensorflow(weights, conf)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
