import os
from typing import List, Union

from src.app.types import ImageType, LabelImage
from src.constants import VALID_EXTS


class Save:
    def __init__(self: "Save", fname=None, dir=None):
        if not fname:
            raise ValueError("Please provide file name")
        if not dir:
            raise ValueError("Please provide directory")
        if not os.path.isdir(dir):
            os.mkdir(dir)
        self.fname = fname
        self.dir = dir

    def one(
        self: "Save",
        img: ImageType,
        index: int = None,
        label: str = None,
    ) -> None:
        ext = "png" if img.channels == 4 else "jpg"
        name = (
            _remove_ext(self.fname)
            + _maybe_append(index)
            + _maybe_append(label)
        )
        img.pil.save(f"{os.path.join(self.dir, name)}.{ext}")
        return

    def many(self: "Save", label_images: List[LabelImage]) -> None:
        for index, li in enumerate(label_images):
            self.one(li.img, label=li.label, index=index)


def _remove_ext(fname: str) -> str:
    for ext in VALID_EXTS:
        if ext in fname:
            return fname.split(f".{ext}")[0]
    return fname


def _maybe_append(something: Union[str, int, None], sep: str = "-") -> str:
    return f"{sep}{something}" if something else ""
