from typing import List

import numpy as np
from PIL import Image  # type: ignore

from src.app.roi import ROI


class Composition:
    @staticmethod
    def layer_images(
        imgs: List[np.ndarray],
        background: int = 0,
    ) -> np.ndarray:
        edge = Composition.__get_longest_edge(imgs)
        canvas = Image.new("RGBA", (edge, edge), background)
        for img in imgs:
            pil = Image.fromarray(img)
            canvas.alpha_composite(pil)
        return ROI.crop(np.array(canvas.convert("RGB"), dtype=np.uint8))

    @staticmethod
    def __get_longest_edge(imgs: List[np.ndarray]) -> int:
        return sorted(
            set(edge for img in imgs for edge in img.shape[:2]),
            reverse=True,
        )[0]
