import os
from typing import List, Union

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

    def many(self: "Save", images: List[ImageType]) -> None:
        for index, img in enumerate(images):
            self.one(img, index=index)

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
