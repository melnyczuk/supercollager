import cv2  # type: ignore
import numpy as np
from PIL import Image, ImageEnhance, ImageOps  # type: ignore


class PostProcess:
    __img: np.ndarray

    def __init__(self: "PostProcess", img: np.ndarray) -> None:
        self.__img = img

    def done(self: "PostProcess") -> np.ndarray:
        return self.__img

    def contrast(self: "PostProcess", factor: float) -> "PostProcess":
        img = Image.fromarray(self.__img)
        auto = ImageOps.autocontrast(img)
        enhanced = ImageEnhance.Contrast(auto).enhance(factor)
        self.__img = np.array(enhanced, dtype=np.uint8)
        return self

    def color(self: "PostProcess", factor: float) -> "PostProcess":
        img = Image.fromarray(self.__img)
        enhanced = ImageEnhance.Color(img).enhance(factor)
        self.__img = np.array(enhanced, dtype=np.uint8)
        return self

    def gain(self: "PostProcess", factor: float) -> "PostProcess":
        img = self.__img.astype(np.float64) * factor
        self.__img = np.minimum(img, 254).astype(np.uint8)
        return self

    def blur(self: "PostProcess", factor: int) -> "PostProcess":
        if factor == 0:
            return self
        ensure_odd = factor * 2 - 1
        self.__img = cv2.GaussianBlur(
            self.__img, (ensure_odd, ensure_odd), 0
        ).astype(np.uint8)
        return self
