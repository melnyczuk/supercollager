import cv2  # type: ignore
import numpy as np
from PIL import ImageEnhance, ImageOps  # type: ignore
from PIL.Image import Image as PilImage  # type: ignore


class PilPostProcess:
    img: PilImage

    def __init__(self: "PilPostProcess", img: PilImage) -> None:
        self.img = img

    def contrast(
        self: "PilPostProcess", factor: float = 0.0
    ) -> "PilPostProcess":
        auto = ImageOps.autocontrast(self.img)
        self.img = ImageEnhance.Contrast(auto).enhance(factor)
        return self

    def color(self: "PilPostProcess", factor: float = 0.0) -> "PilPostProcess":
        self.img = ImageEnhance.Color(self.img).enhance(factor)
        return self


class NpPostProcess:
    img: np.ndarray

    def __init__(self: "NpPostProcess", img: np.ndarray) -> None:
        self.img = img

    def gain(self: "NpPostProcess", gain: int) -> "NpPostProcess":
        self.img = np.minimum(self.img.astype(np.float64) * gain, 254).astype(
            np.uint8
        )
        return self

    def blur(self: "NpPostProcess", size: int) -> "NpPostProcess":
        ensure_odd = size * 2 + 1
        self.img = cv2.GaussianBlur(self.img, (ensure_odd, ensure_odd), 0)
        return self
