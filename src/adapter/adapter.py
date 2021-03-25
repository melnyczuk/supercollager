import os
from io import BytesIO
from typing import Any, Iterable, List

from moviepy.video.io.VideoFileClip import VideoFileClip  # type: ignore
from PIL import Image  # type: ignore

from src.adapter.directory import DirectoryAdapter
from src.adapter.local_file import LocalFileAdapter
from src.adapter.url import UrlAdapter
from src.app.image_type import ImageType


class Adapter:
    @staticmethod
    def load(*inputs: str) -> List[ImageType]:
        return [
            ImageType(Image.open(file).convert("RGB"))
            for input in inputs
            for file in Adapter.__match(input)
        ]

    @staticmethod
    def video(input: str) -> VideoFileClip:
        return VideoFileClip(input)

    @staticmethod
    def __match(
        input: Any,
    ) -> Iterable[BytesIO]:
        if os.path.isfile(input):
            if data := LocalFileAdapter.load(input):
                yield data
            pass
        if os.path.isdir(input):
            for data in DirectoryAdapter.load(input):
                if data:
                    yield data
            pass
        if input.startswith("http"):
            for data in UrlAdapter.scrape_site(input):
                yield data
            pass
        pass
