from dataclasses import dataclass

from .gluoncv import GluonCVSegmentation
from .mask_rcnn import MaskRCNNSegmentation


@dataclass(frozen=True)
class Segmentation:
    gluoncv = GluonCVSegmentation.run
    mask_rcnn = MaskRCNNSegmentation.run
