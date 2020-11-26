from typing import List

import cv2  # type: ignore
import numpy as np
from PIL import Image  # type: ignore


class Composition:
    @staticmethod
    def layer_images(
        imgs: List[Image.Image],
        background: int = 0,
    ) -> Image:
        edge = _get_max_edge(imgs)
        canvas = Image.new("RGBA", (edge, edge), background)

        for img in imgs:
            canvas.alpha_composite(img)

        return _crop_roi(canvas, background + 1)


def _crop_roi(img: Image.Image, threshold: int) -> Image.Image:
    greyscale = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    (_, binary) = cv2.threshold(greyscale, threshold, 255, cv2.THRESH_BINARY)
    (ctrs, _) = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    sorted_contours = sorted(ctrs, key=lambda ctr: ctr.size, reverse=True)
    (x, y, w, h) = cv2.boundingRect(sorted_contours[0])
    return img.crop([x, y, x + w, y + h])


def _get_max_edge(imgs: List[Image.Image]) -> int:
    return sorted([n for a in imgs for n in a.size], reverse=True)[0]
