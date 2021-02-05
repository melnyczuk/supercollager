from PIL import ImageEnhance, ImageOps  # type: ignore
from PIL.Image import Image as ImageType  # type: ignore


class Post:
    @staticmethod
    def process(pil_img: ImageType) -> ImageType:
        contrasted = _contrast(pil_img, 1.2)
        return ImageEnhance.Color(contrasted).enhance(1.2)


def _contrast(pil_img: ImageType, factor: float = 0.0) -> ImageType:
    return ImageEnhance.Contrast(ImageOps.autocontrast(pil_img)).enhance(factor)
