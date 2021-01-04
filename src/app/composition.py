from typing import List

from PIL import Image  # type: ignore

from src.app.roi import ROI


class Composition:
    @staticmethod
    def layer_images(imgs: List[Image.Image], background: int = 0) -> Image:
        edge = sorted([n for a in imgs for n in a.size], reverse=True)[0]
        canvas = Image.new("RGBA", (edge, edge), background)

        for img in imgs:
            canvas.alpha_composite(img)

        return ROI.crop_pil(canvas)
