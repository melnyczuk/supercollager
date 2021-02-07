from dataclasses import dataclass

from src.app.segmentation.mask_rcnn import MaskRCNNSegmentation


@dataclass(frozen=True)
class Segmentation:
    mask_rcnn = MaskRCNNSegmentation.run
