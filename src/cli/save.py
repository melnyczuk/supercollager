import os
from typing import Iterable

import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image

from src.constants import VALID_EXTS


class Save:
    fname: str
    dir: str
    count: int

    def jpg(self: "Save", *imgs: np.ndarray, **kwargs) -> None:
        self.__image(*imgs, ext="jpg", **kwargs)
        return

    def png(self: "Save", *imgs: np.ndarray, **kwargs) -> None:
        self.__image(*imgs, ext="png", **kwargs)
        return

    def mp4(self: "Save", *clips: Iterable[np.ndarray], **kwargs) -> None:
        self.__video(*clips, ext="mp4", **kwargs)
        return

    def avi(self: "Save", *clips: Iterable[np.ndarray], **kwargs) -> None:
        self.__video(*clips, ext="avi", **kwargs)
        return

    def webm(self: "Save", *clips: Iterable[np.ndarray], **kwargs) -> None:
        self.__video(*clips, ext="webm", **kwargs)
        return

    def __image(
        self: "Save",
        *imgs: np.ndarray,
        ext: str,
        **kwargs,
    ) -> None:
        for img in imgs:
            Image.fromarray(img.astype(np.uint8)).save(
                self.__get_path(ext),
                **kwargs,
            )
            self.__increment()
        return

    def __video(
        self: "Save",
        *clips: Iterable[np.ndarray],
        ext: str,
        fps: int,
        **kwargs,
    ) -> None:
        for clip in clips:
            ImageSequenceClip(
                [np.dstack((m, m, m)) for m in clip],  # type: ignore
                with_mask=False,
                fps=fps,
            ).write_videofile(self.__get_path(ext), **kwargs)
            self.__increment()
        return

    def __get_path(
        self: "Save",
        ext: str,
    ) -> str:
        fname = f"{Save.__remove_ext(self.fname)}-{self.count}"
        return f"{os.path.join(self.dir, fname)}.{ext}"

    def __increment(self: "Save") -> None:
        self.count += 1

    def __init__(self: "Save", fname=None, dir=None) -> None:
        if not fname:
            raise ValueError("Please provide file name")
        if not dir:
            raise ValueError("Please provide directory")
        if not os.path.isdir(dir):
            os.mkdir(dir)
        self.fname = fname
        self.dir = dir
        self.count = 0
        return

    @staticmethod
    def __remove_ext(fname: str) -> str:
        for ext in VALID_EXTS:
            if ext in fname:
                return fname.split(f".{ext}")[0]
        return fname
