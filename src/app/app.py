from random import random
from typing import Any, Iterable, List, Tuple, Union

import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip  # type:ignore
from tqdm.std import tqdm  # type:ignore

from src.app.composition import Composition
from src.app.keyframing import Keyframing
from src.app.masking import Masking
from src.app.post_process import PostProcess
from src.app.roi import ROI
from src.app.segmentation import Segmentation
from src.app.super_resolution import SuperResolution
from src.app.transform import Transform


def shuffl(arr: Iterable[Any]) -> List[Any]:
    return sorted(arr, key=lambda _: random())


class App:
    @staticmethod
    def masks(imgs: Iterable[np.ndarray]) -> Iterable[np.ndarray]:
        segmentation = Segmentation()
        return (
            ROI.crop(mask)
            for img in imgs
            for mask in segmentation.mask_rcnn.mask(img)
        )

    @staticmethod
    def segment(
        imgs: Iterable[np.ndarray],
        rotate: Union[float, bool] = False,
        shuffle: bool = False,
    ) -> Iterable[np.ndarray]:
        segmentation = Segmentation()
        return tqdm(
            ROI.crop(Masking.apply_mask(img=img, mask=mask, rotate=rotate))
            for img in (shuffl(imgs) if shuffle else imgs)
            for mask in segmentation.mask_rcnn.mask(img)
        )

    @staticmethod
    def collage(
        imgs: Iterable[np.ndarray],
        background: Union[None, int, Tuple[int, int, int]] = None,
        color: float = 1.2,
        contrast: float = 1.2,
        rotate: Union[float, bool] = False,
        shuffle: bool = False,
    ) -> Iterable[np.ndarray]:
        segments = App.segment(imgs, rotate=rotate, shuffle=shuffle)
        comp = Composition.layer_images(segments, background=background)
        return (PostProcess(comp).contrast(contrast).color(color).done(),)

    @staticmethod
    def super_resolution(
        imgs: Iterable[np.ndarray],
        device: str = "cuda",
    ) -> Iterable[np.ndarray]:
        esrgan = SuperResolution.esrgan(device=device)
        return (esrgan.run(img) for img in imgs)

    @staticmethod
    def abstracts(
        imgs: Iterable[np.ndarray],
        color: float = 1.2,
        contrast: float = 1.2,
        device: str = "cuda",
        dsize: Tuple[int, int] = (80, 64),
        limit: int = 100,
        n_segments: int = 10,
        rotate: Union[float, bool] = False,
        shuffle: bool = True,
        sr_cycles: int = 0,
    ) -> Iterable[np.ndarray]:
        rng = np.random.default_rng()
        segments = App.segment(imgs, rotate=rotate, shuffle=shuffle)
        limited = [x for _, x in zip(range(limit), segments)]
        chosen = rng.choice(np.array(limited, dtype=object), n_segments)
        resized = [Transform.resize(seg, dsize=dsize) for seg in chosen]
        comp = Composition.layer_images(shuffl(resized))
        for _ in range(sr_cycles):
            (comp,) = App.super_resolution((comp,), device=device)
        return (PostProcess(comp).contrast(contrast).color(color).done(),)

    @staticmethod
    def alpha_matte(
        video: VideoFileClip,
        blur: int = 0,
        confidence_threshold: float = 0.0,
        gain: int = 1,
        keyframe_interval: int = 1,
    ) -> Iterable[np.ndarray]:
        segmentation = Segmentation()

        keyframes = tqdm(
            np.array(np.mean(np.array(masks), axis=0), dtype=np.uint8)
            if len(
                masks := list(
                    segmentation.mask_rcnn.mask(
                        frame,
                        confidence_threshold=confidence_threshold,
                    )
                )
            )
            else np.zeros(frame.shape[:2], dtype=np.uint8)
            for f, frame in enumerate(video.iter_frames())
            if f % keyframe_interval == 0
        )

        return Keyframing.interpolate_frames(
            (PostProcess(kf).gain(gain).blur(blur).done() for kf in keyframes),
            duration=int(video.fps * video.duration),
            k_interval=keyframe_interval,
        )
