from typing import List, Tuple

from PIL import Image  # type: ignore

from src.app.roi import ROI


class Composition:
    @staticmethod
    def layer_images(imgs: List[Image.Image], background: int = 0) -> Image:
        edge = sorted([n for a in imgs for n in a.size], reverse=True)[0]
        canvas = Image.new("RGBA", (edge, edge), background)

        for img in imgs:
            offset = tuple(
                (canvas.size[i] - img.size[i]) // 2 for i in range(2)
            )
            canvas.alpha_composite(img, offset)

        return ROI.crop_roi(canvas)
