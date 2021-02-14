from typing import Union

from src.app.image_type import ImageType


class Transform:
    @staticmethod
    def rotate(
        img: ImageType,
        rotate: Union[float, bool] = False,
    ) -> ImageType:
        if not rotate:
            return img

        rotation = 90.0 if type(rotate) == bool else rotate
        return ImageType(img.pil.rotate(rotation).resize(img.dimensions))
