from datetime import datetime

from src.logger import logger
from src.app import Colors, Composition, IO, Masking, Segmentation

from typing import List


class CLI:
    @staticmethod
    def blocks(
        urls: List[str],
        dir: str = "./dump",
        n: int = 5,
        ext: str = ".png",
    ) -> None:
        masks = [
            mask for url in urls for mask in Segmentation.masks_from_url(url)
        ]
        masks.sort(key=lambda x: x.size)
        [background, *color_list] = Colors.as_list(n + 1)

        logger.log(f"segmented f{len(masks)} masks from {len(urls)} urls")

        blocks = [
            Masking.to_rgba(mask, color_list[i])
            for i, mask in enumerate(masks[:n])
        ]

        comp = Composition.layer_to_image(blocks, background)

        now = datetime.now().strftime("%Y%d%m-%H:%M:%S")

        logger.log(f"saving image to {dir}/{now}.png")

        IO.save.image(comp, fname=now, dir=dir)

    @staticmethod
    def segments(url: str, root: str = "./dump") -> None:
        dir = f"{root}/{url.split('/')[-1].split('.')[0]}"
        data = Segmentation.from_url(url)
        logger.log(f"saving {len(data)} imgs")
        for d in data:
            IO.save.np_array(d["img"], fname=d["label"], dir=dir)


if __name__ == "__main__":
    from fire import Fire  # type:ignore

    Fire(CLI)
