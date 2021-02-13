class Segmentation:
    def __init__(self: "Segmentation"):
        from src.app.segmentation.mask_rcnn import MaskRCNN

        self.mask_rcnn = MaskRCNN().run
