from typing import Optional

import cv2  # type: ignore

from src.app.image_type import ImageType
from src.app.region import Region


class ROI:
    @staticmethod
    def crop(img: ImageType, threshold: int = None) -> ImageType:
        if not (region := ROI.__get_roi(img, threshold)):
            return img
        return ImageType(region.crop(img.np))

    @staticmethod
    def __get_roi(img: ImageType, threshold: int = None) -> Optional[Region]:
        grey = ROI.__get_grey(img)
        thresh = threshold if threshold else ROI.__get_threshold(grey)
        (_, bin) = cv2.threshold(grey.np, thresh, 255, cv2.THRESH_BINARY)
        (ctrs, _) = cv2.findContours(
            bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not ctrs:
            return None
        (left, top, w, h) = cv2.boundingRect(
            sorted(ctrs, key=lambda ctr: ctr.size, reverse=True)[0]
        )
        return Region(left=left, right=left + w, top=top, bottom=top + h)

    @staticmethod
    def __get_grey(img: ImageType) -> ImageType:
        if img.channels == 4:
            return ImageType(img.np[:, :, 3])
        if img.channels == 3:
            return ImageType(cv2.cvtColor(img.np, cv2.COLOR_RGB2GRAY))
        return img

    @staticmethod
    def __get_threshold(img: ImageType) -> int:
        histogram = img.pil.histogram()
        mode = max(*histogram)
        return histogram.index(mode)
