from typing import Iterable, List, Union

import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip  # type: ignore
from moviepy.video.io.VideoFileClip import VideoFileClip  # type: ignore
from tqdm.std import tqdm  # type:ignore

from src.app.composition import Composition
from src.app.image_type import ImageType
from src.app.keyframing import Keyframing
from src.app.post_process import NpPostProcess, PilPostProcess
from src.app.segmentation import Segmentation


class App:
    @staticmethod
    def segment(
        imgs: List[ImageType],
        rotate: Union[float, bool] = False,
    ) -> Iterable[ImageType]:
        segmentation = Segmentation()
        return segmentation.mask_rcnn.images(imgs, rotate=rotate)

    @staticmethod
    def collage(
        imgs: List[ImageType],
        rotate: Union[float, bool] = False,
    ) -> ImageType:
        segmentation = Segmentation()
        imgs = list(segmentation.mask_rcnn.images(imgs, rotate=rotate))
        bg = int(np.random.randint(5, 15))
        comp = Composition.layer_images(imgs=imgs, background=bg)
        post = PilPostProcess(comp.pil).contrast(1.2).color(1.2)
        return ImageType(post.img)

    @staticmethod
    def alpha_matte(
        video: VideoFileClip,
        keyframe_interval: int = 1,
        gain: int = 1,
        blur: int = 0,
        confidence_threshold: float = 0.0,
    ) -> Union[VideoFileClip, ImageSequenceClip]:
        segmentation = Segmentation()

        keyframes = tqdm(
            NpPostProcess(
                segmentation.mask_rcnn.mask_frame(
                    frame,
                    confidence_threshold=confidence_threshold,
                )
            )
            .gain(gain)
            .blur(blur)
            .img
            for f, frame in enumerate(video.iter_frames())
            if f % keyframe_interval == 0
        )

        mask = Keyframing.interpolate_frames(
            keyframes,
            duration=int(video.fps * video.duration),
            k_interval=keyframe_interval,
        )

        return ImageSequenceClip(
            [np.dstack((m, m, m)) for m in mask],
            with_mask=False,
            fps=video.fps,
        )
