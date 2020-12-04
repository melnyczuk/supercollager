from PIL import Image, ImageEnhance, ImageOps  # type: ignore


class Post:
    @staticmethod
    def run(pil_img: Image) -> Image:
        contrasted = _contrast(pil_img, 1.2)
        return ImageEnhance.Color(contrasted).enhance(1.2)


def _contrast(pil_img: Image, factor: float = 0.0) -> Image:
    return ImageEnhance.Contrast(ImageOps.autocontrast(pil_img)).enhance(factor)
