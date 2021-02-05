from typing import List

from PIL import Image  # type: ignore
from PIL.Image import Image as ImageType  # type: ignore

from src.app.roi import ROI


class Composition:
    @staticmethod
    def layer_images(imgs: List[ImageType], background: int = 0) -> ImageType:
        edge = sorted([n for a in imgs for n in a.size], reverse=True)[0]
        canvas = Image.new("RGBA", (edge, edge), background)

        for img in imgs:
            canvas.alpha_composite(img)

        return ROI.crop(canvas)
