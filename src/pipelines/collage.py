from datetime import datetime
from typing import List

import numpy as np

from ..app import IO, Composition
from .segment import segment


def collage(
    uris: List[str] = [],
    dir: str = "./dump/output",
    n: int = -1,
    smooth: bool = False,
) -> None:
    transparencies = [np.array(img) for img in segment(uris, smooth)]
    comp = Composition.layer_to_image(transparencies, [47, 47, 47])
    now = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    IO.save.image(comp, fname=now, dir=dir, ext="jpg")
