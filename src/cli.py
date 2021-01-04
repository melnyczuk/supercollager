import os
from datetime import datetime
from typing import List

from src.app import IO
from src.logger import logger
from src.pipelines import LabelImage, collage, segment

VALID_FORMATS = ["jpg", "png", "jpeg", "tif", "tiff"]


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
    def f(**kwargs) -> None:
        now = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        dir = kwargs.pop("dir", "./dump/output")
        fname = kwargs.pop("fname", now)

        uris = _parse_uris(kwargs.pop("uris", []))

        result: List[LabelImage] = fn(uris=uris, **kwargs)

        for i, label_image in enumerate(result):
            img = label_image.pil_img
            f = f"{fname}-{i}{label_image.label}"
            IO.save.pil_img(pil_img=img, fname=f, dir=dir)

        logger.log(f"saved {len(result)} images to {dir}")

    return f


class CLI:
    collage = save(collage)
    segment = save(segment)


if __name__ == "__main__":
    from fire import Fire  # type:ignore

    Fire(CLI)
