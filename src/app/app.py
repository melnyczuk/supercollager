from typing import Iterable, List, Union

import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip  # type: ignore
from tqdm.std import tqdm  # type:ignore

from src.app.composition import Composition
from src.app.keyframing import Keyframing
from src.app.post_process import PostProcess
from src.app.segmentation import Segmentation


class App:
    @staticmethod
    def segment(
        imgs: List[np.ndarray],
        rotate: Union[float, bool] = False,
    ) -> Iterable[np.ndarray]:
        segmentation = Segmentation()
        return (
            segment
            for img in imgs
            for segment in segmentation.mask_rcnn.image(img, rotate=rotate)
        )

    @staticmethod
    def collage(
        imgs: List[np.ndarray],
        rotate: Union[float, bool] = False,
        contrast: float = 1.2,
        color: float = 1.2,
    ) -> np.ndarray:
        segments = App.segment(imgs, rotate=rotate)
        bg = int(np.random.randint(5, 15))
        comp = Composition.layer_images(imgs=list(segments), background=bg)
        return PostProcess(comp).contrast(contrast).color(color).done()

    @staticmethod
    def alpha_matte(
        video: VideoFileClip,
        keyframe_interval: int = 1,
        gain: int = 1,
        blur: int = 0,
        confidence_threshold: float = 0.0,
    ) -> Iterable[np.ndarray]:
        segmentation = Segmentation()

        keyframes = tqdm(
            PostProcess(
                segmentation.mask_rcnn.mask_frame(
                    frame,
                    confidence_threshold=confidence_threshold,
                )
            )
            .gain(gain)
            .blur(blur)
            .done()
            for f, frame in enumerate(video.iter_frames())
            if f % keyframe_interval == 0
        )

        return Keyframing.interpolate_frames(
            keyframes,
            duration=int(video.fps * video.duration),
            k_interval=keyframe_interval,
        )
