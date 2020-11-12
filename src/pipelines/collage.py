from datetime import datetime

from src.logger import logger
from src.app import (
    Colors,
    Composition,
    IO,
    Noise,
    Transformation,
)
from .segment import segment

from typing import List


def collage(
    uris: List[str] = [],
    dir: str = "./dump/output",
    n: int = -1,
    smooth: bool = False,
) -> None:
    transparencies = segment(uris, int(smooth))
    background = Colors.pick()
    comp = Composition.layer_to_image(transparencies, background)
    big = Transformation.scale_up_nparray(comp, 2)
    jpg = Noise.jpg_artifact(big, 10)
    output = Noise.salt_pepper(jpg, 0.1)
    now = datetime.now().strftime("%Y%d%m-%H:%M:%S")
    logger.log(f"saving image to {dir}/{now}.jpg")
    IO.save.image(output, fname=now, dir=dir, ext=".jpg")
