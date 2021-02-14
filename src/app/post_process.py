from PIL import ImageEnhance, ImageOps  # type: ignore
from PIL.Image import Image as PilImage  # type: ignore

from src.app.image_type import ImageType


class PostProcess:
    img: PilImage

    def __init__(self: "PostProcess", img: ImageType) -> None:
        self.img = img.pil

    def contrast(self: "PostProcess", factor: float = 0.0) -> "PostProcess":
        auto = ImageOps.autocontrast(self.img)
        self.img = ImageEnhance.Contrast(auto).enhance(factor)
        return self

    def color(self: "PostProcess", factor: float = 0.0) -> "PostProcess":
        self.img = ImageEnhance.Color(self.img).enhance(factor)
        return self
