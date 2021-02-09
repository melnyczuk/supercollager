from typing import List, Union

from numpy.random import randint
from tqdm.std import tqdm  # type: ignore

from src.app.composition import Composition
from src.app.load import Load
from src.app.masking import Masking
from src.app.post import Post
from src.app.roi import ROI
from src.app.segmentation import Segmentation
from src.app.types import ImageType, LabelImage
from src.logger import logger


class App:
    @staticmethod
    def segment(
        uris: List[str] = [],
        rotate: Union[float, bool] = False,
    ) -> List[LabelImage]:
        imgs = Load.uris(uris)
        logger.log(f"loaded {len(imgs)} images")

        logger.log("analysing images:")
        analysed_images = Segmentation.mask_rcnn(imgs)
        logger.log(
            f"found {len(analysed_images)} segments in {len(imgs)} URIs finding {len(set(ai.label for ai in analysed_images))} different types of object"  # noqa: E501
        )

        logger.log("segmenting images:")
        return [
            LabelImage(img=_mask_crop(ai.img, ai.mask, rotate), label=ai.label)
            for ai in tqdm(analysed_images)
        ]

    @staticmethod
    def collage(
        uris: List[str],
        rotate: Union[float, bool] = False,
    ) -> ImageType:
        label_imgs = App.segment(uris=uris, rotate=rotate)
        imgs = [li.img for li in label_imgs]
        bg = randint(5, 15)
        comp = Composition.layer_images(imgs=imgs, background=bg)
        return Post.process(comp)


def _mask_crop(img, mask, rotate):
    masked_img = Masking.apply_mask(img, mask, rotate=rotate)
    return ROI.crop(masked_img)
