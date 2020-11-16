import os
from datetime import datetime
from typing import List

from tqdm import tqdm  # type:ignore

from . import pipelines
from .app import IO
from .logger import logger

VALID_FORMATS = ["jpg", "png", "jpeg"]


def _parse_uris(uris: List[str]) -> List[str]:
    return [
        uri
        for nested in (
            [uri]
            if not os.path.isdir(uri)
            else [
                os.path.join(uri, file)
                for file in os.listdir(uri)
                if file.lower().split(".")[-1] in VALID_FORMATS
            ]
            for uri in uris
        )
        for uri in nested
    ]


def save(fn):
    @staticmethod
    def f(**kwargs):
        now = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        dir = kwargs.pop("dir", "./dump/output")
        fname = kwargs.pop("fname", now)
        ext = kwargs.pop("ext", "png")

        uris = _parse_uris(kwargs.pop("uris", []))

        for i, img in enumerate(tqdm(fn(uris=uris, **kwargs))):
            IO.save.image(
                img=img,
                fname=f"{fname}-{i}",
                dir=dir,
                ext=ext,
            )

        logger.log(f"saved images to {dir}")

    return f


class CLI:
    blocks = save(pipelines.blocks)
    collage = save(pipelines.collage)
    segment = save(pipelines.segment)


if __name__ == "__main__":
    from fire import Fire  # type:ignore

    Fire(CLI)
