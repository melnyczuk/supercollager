from typing import List, Tuple

import cv2  # type: ignore
import numpy as np
from PIL import Image  # type: ignore


class Composition:
    @staticmethod
    def layer_images(imgs: List[Image.Image], background: int = 0) -> Image:
        edge = _get_max_edge(imgs)
        canvas = Image.new("RGBA", (edge, edge), background)

        for img in imgs:
            canvas.alpha_composite(img)

        return Composition.crop_roi(canvas, background + 1)

    @staticmethod
    def crop_roi(pil_img: Image, threshold: int) -> Image:
        (x, y, w, h) = Composition.get_roi(np.array(pil_img), threshold)
        return pil_img.crop([x, y, x + w, y + h])

    @staticmethod
    def get_roi(
        np_img: np.ndarray, threshold: int
    ) -> Tuple[int, int, int, int]:
        grey = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        (_, binary) = cv2.threshold(grey, threshold, 255, cv2.THRESH_BINARY)
        (ctrs, _) = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        sorted_contours = sorted(ctrs, key=lambda ctr: ctr.size, reverse=True)
        return (
            cv2.boundingRect(sorted_contours[0])
            if (len(sorted_contours))
            else [0, 0, 0, 0]
        )


def _get_max_edge(imgs: List[Image.Image]) -> int:
    return sorted([n for a in imgs for n in a.size], reverse=True)[0]
