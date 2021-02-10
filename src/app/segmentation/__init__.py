class Segmentation:
    def __init__(self: "Segmentation"):
        from src.app.segmentation.mask_rcnn import MaskRCNNSegmentation

        self.mask_rcnn = MaskRCNNSegmentation.run
