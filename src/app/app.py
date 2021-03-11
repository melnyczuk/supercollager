from typing import List, Union

from numpy.random import randint

from src.app.composition import Composition
from src.app.image_type import ImageType
from src.app.post_process import PostProcess
from src.app.segmentation import Segmentation


class App:
    @staticmethod
    def segment(
        imgs: List[ImageType],
        rotate: Union[float, bool] = False,
    ) -> List[ImageType]:
        return Segmentation().mask_rcnn(imgs, rotate=rotate)

    @staticmethod
    def collage(
        imgs: List[ImageType],
        rotate: Union[float, bool] = False,
    ) -> ImageType:
        imgs = App.segment(imgs, rotate=rotate)
        bg = int(randint(5, 15))
        comp = Composition.layer_images(imgs=imgs, background=bg)
        post = PostProcess(comp).contrast(1.2).color(1.2)
        return ImageType(post.img)
