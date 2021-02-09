from typing import List

from PIL import Image  # type: ignore

from src.app.roi import ROI
from src.app.types import ImageType


class Composition:
    @staticmethod
    def layer_images(imgs: List[ImageType], background: int = 0) -> ImageType:
        edge = sorted(
            [side for img in imgs for side in img.dimensions], reverse=True
        )[0]
        canvas = Image.new("RGBA", (edge, edge), background)

        for img in imgs:
            canvas.alpha_composite(img.pil)

        return ROI.crop(ImageType(canvas.convert("RGB")))
