import os
from typing import Iterable, Optional, Union

import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip  # type: ignore
from PIL import Image  # type: ignore

from src.constants import VALID_EXTS


class Save:
    fname: str
    dir: str

    def jpg(self: "Save", img: np.ndarray, index: Optional[int] = None) -> None:
        self.__image(img, ext="jpg", index=index)
        return

    def png(self: "Save", img: np.ndarray, index: Optional[int] = None) -> None:
        self.__image(img, ext="png", index=index)
        return

    def mp4(
        self: "Save",
        clip: Iterable[np.ndarray],
        **kwargs,
    ) -> None:
        self.__video(clip, ext="mp4", **kwargs)
        return

    def avi(
        self: "Save",
        clip: Iterable[np.ndarray],
        **kwargs,
    ) -> None:
        self.__video(clip, ext="avi", **kwargs)
        return

    def webm(
        self: "Save",
        clip: Iterable[np.ndarray],
        **kwargs,
    ) -> None:
        self.__video(clip, ext="webm", **kwargs)
        return

    def __video(
        self: "Save",
        clip: Iterable[np.ndarray],
        ext: str,
        fps: int,
        **kwargs,
    ):
        fpath = f"{os.path.join(self.dir, Save.__remove_ext(self.fname))}.{ext}"
        ImageSequenceClip(
            [np.dstack((m, m, m)) for m in clip], with_mask=False, fps=fps
        ).write_videofile(fpath, **kwargs)
        return

    def __image(
        self: "Save",
        img: np.ndarray,
        ext: str,
        index: Optional[int] = None,
    ):
        name = Save.__remove_ext(self.fname) + Save.__maybe_append(index)
        Image.fromarray(img).save(f"{os.path.join(self.dir, name)}.{ext}")
        return

    def __init__(self: "Save", fname=None, dir=None):
        if not fname:
            raise ValueError("Please provide file name")
        if not dir:
            raise ValueError("Please provide directory")
        if not os.path.isdir(dir):
            os.mkdir(dir)
        self.fname = fname
        self.dir = dir

    @staticmethod
    def __remove_ext(fname: str) -> str:
        for ext in VALID_EXTS:
            if ext in fname:
                return fname.split(f".{ext}")[0]
        return fname

    @staticmethod
    def __maybe_append(something: Union[str, int, None], sep: str = "-") -> str:
        return f"{sep}{something}" if something else ""
