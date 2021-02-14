from typing import List, Union

from numpy.random import randint
from tqdm.std import tqdm  # type: ignore

from src.app.composition import Composition
from src.app.load import Load
from src.app.image_type import ImageType
from src.app.masking import Masking
from src.app.post_process import PostProcess
from src.app.roi import ROI
from src.app.segmentation import Segmentation
from src.logger import logger


class App:
    @staticmethod
    def segment(
        uris: List[str],
        rotate: Union[float, bool] = False,
    ) -> List[ImageType]:
        imgs = Load.uris(uris)
        logger.log(f"loaded {len(imgs)} images")

        logger.log("analysing images:")

        segments = Segmentation().mask_rcnn(imgs, rotate=rotate)
        logger.log(f"found {len(segments)} segments in {len(imgs)} URIs")
        return segments

    @staticmethod
    def collage(
        uris: List[str],
        rotate: Union[float, bool] = False,
    ) -> ImageType:
        label_imgs = App.segment(uris, rotate=rotate)
        imgs = [li.img for li in label_imgs]
        bg = int(randint(5, 15))
        comp = Composition.layer_images(imgs=imgs, background=bg)
        post = PostProcess(comp).contrast(1.2).color(1.2)
        return ImageType(post.img)
