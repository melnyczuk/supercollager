from io import BytesIO
from typing import List

import requests
from bs4 import BeautifulSoup  # type:ignore


class UrlAdapter:
    @staticmethod
    def load(input: str) -> BytesIO:
        return BytesIO(requests.get(input).content)

    @staticmethod
    def scrape_site(input: str) -> List[BytesIO]:
        resp = requests.get(input)
        soup = BeautifulSoup(resp.text, "html.parser")
        tags = soup.find_all("img")
        imgs = (img.get("src", "") for img in tags)
        return [UrlAdapter.load(src) for src in imgs if src.startswith("http")]
