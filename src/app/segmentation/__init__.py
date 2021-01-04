from dataclasses import dataclass

from src.app.segmentation.gluoncv import GluonCVSegmentation
from src.app.segmentation.mask_rcnn import MaskRCNNSegmentation


@dataclass(frozen=True)
class Segmentation:
    gluoncv = GluonCVSegmentation.run
    mask_rcnn = MaskRCNNSegmentation.run
