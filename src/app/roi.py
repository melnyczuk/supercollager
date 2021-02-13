from typing import Optional, Tuple

import cv2  # type: ignore

from src.app.types import ImageType

Region = Tuple[int, int, int, int]


class ROI:
    @staticmethod
    def crop(img: ImageType, threshold: int = None) -> ImageType:
        if not (region := ROI.__get_bounding_box(img, threshold)):
            return img
        (x0, y0, x1, y1) = region
        return ImageType(
            img.np[y0:y1, x0:x1, :]
            if img.channels > 1
            else img.np[y0:y1, x0:x1]
        )

    @staticmethod
    def __get_bounding_box(
        img: ImageType,
        threshold: int = None,
    ) -> Optional[Region]:
        (x0, y0, w, h) = ROI.__get_roi(img, threshold)
        if w == 0 or h == 0:
            return None
        (x1, y1) = (x0 + w, y0 + h)
        return (x0, y0, x1, y1)

    @staticmethod
    def __get_roi(img: ImageType, threshold: int = None) -> Region:
        grey = ROI.__get_grey(img)
        thresh = threshold if threshold else ROI.__get_threshold(grey)
        (_, bin) = cv2.threshold(grey.np, thresh, 255, cv2.THRESH_BINARY)
        (ctrs, _) = cv2.findContours(
            bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return (
            (0, 0, 0, 0)
            if not ctrs
            else cv2.boundingRect(
                sorted(ctrs, key=lambda ctr: ctr.size, reverse=True)[0]
            )
        )

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
