from typing import Iterable

from PIL import Image  # type: ignore

from src.app.image_type import ImageType
from src.app.roi import ROI


class Composition:
    @staticmethod
    def layer_images(
        imgs: Iterable[ImageType], background: int = 0
    ) -> ImageType:
        edge = Composition.__get_longest_edge(imgs)
        canvas = Image.new("RGBA", (edge, edge), background)
        for img in imgs:
            canvas.alpha_composite(img.pil)
        return ROI.crop(ImageType(canvas.convert("RGB")))

    @staticmethod
    def __get_longest_edge(imgs: Iterable[ImageType]) -> int:
        return sorted(
            set(edge for img in imgs for edge in img.dimensions),
            reverse=True,
        )[0]
