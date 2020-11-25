from PIL import Image, ImageEnhance, ImageOps  # type: ignore


class Post:
    @staticmethod
    def run(img: Image.Image) -> Image.Image:
        contrasted = _contrast(img, 1.2)
        return ImageEnhance.Color(contrasted).enhance(1.2)


def _contrast(img: Image.Image, factor: float = 0.0) -> Image.Image:
    return ImageEnhance.Contrast(ImageOps.autocontrast(img)).enhance(factor)
