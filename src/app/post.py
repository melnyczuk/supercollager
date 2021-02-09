from PIL import ImageEnhance, ImageOps  # type: ignore
from PIL.Image import Image as PilImage  # type: ignore

from src.app.types import ImageType


class Post:
    @staticmethod
    def process(img: ImageType) -> ImageType:
        contrasted = _contrast(img.pil, 1.2)
        enhanced_colour = _enhance_colour(contrasted, 1.2)
        return ImageType(enhanced_colour)


def _contrast(img: PilImage, factor: float = 0.0) -> PilImage:
    auto = ImageOps.autocontrast(img)
    return ImageEnhance.Contrast(auto).enhance(factor)


def _enhance_colour(img: PilImage, factor: float = 0.0) -> PilImage:
    return ImageEnhance.Color(img).enhance(factor)
