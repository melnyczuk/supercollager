import os
from dataclasses import dataclass
from io import BytesIO
from typing import List

import requests
from PIL import Image  # type: ignore

from src.app.types import ImageType
from src.constants import VALID_EXTS


@dataclass(frozen=True)
class Load:
    @staticmethod
    def uri(uri: str) -> ImageType:
        resource = (
            uri
            if not uri.startswith("http")
            else BytesIO(requests.get(uri).content)
        )
        return ImageType(Image.open(resource).convert("RGB"))

    @staticmethod
    def uris(uris: List[str]) -> List[ImageType]:
        return [Load.uri(uri) for uri in _parse_uris(uris)]


def _parse_uris(input: List[str]) -> List[str]:
    return [
        uri
        for nested in (
            [uri]
            if not os.path.isdir(uri)
            else [
                os.path.join(uri, file)
                for file in os.listdir(uri)
                if file.lower().split(".")[-1] in VALID_EXTS
            ]
            for uri in input
        )
        for uri in nested
    ]
