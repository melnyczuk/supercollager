import os
from typing import Iterable, Tuple, Union

from cv2 import VideoWriter, VideoWriter_fourcc  # type:ignore
from numpy import ndarray  # type:ignore
from tqdm.std import tqdm  # type:ignore

from src.app.image_type import ImageType
from src.constants import VALID_EXTS


class Save:
    fname: str
    dir: str

    def one(
        self: "Save",
        img: ImageType,
        index: int = None,
    ) -> None:
        ext = "png" if img.channels == 4 else "jpg"
        name = Save.__remove_ext(self.fname) + Save.__maybe_append(index)
        img.pil.save(f"{os.path.join(self.dir, name)}.{ext}")
        return

    def many(self: "Save", images: Iterable[ImageType]) -> None:
        for index, img in tqdm(enumerate(images)):
            self.one(img, index=index)
        return

    def video(
        self: "Save",
        video: Iterable[ndarray],
        fps: int,
        shape: Tuple[int, int],
    ) -> None:
        fpath = f"{os.path.join(self.dir, Save.__remove_ext(self.fname))}.avi"
        writer = VideoWriter(
            fpath,
            VideoWriter_fourcc("D", "I", "V", "X"),
            fps,
            shape,
            isColor=False,
        )
        for frame in tqdm(video):
            writer.write(frame)
        writer.release()
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
