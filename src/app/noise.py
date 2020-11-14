import io
from typing import List

import numpy as np
from PIL import Image  # type:ignore

from ..logger import logger


class Noise:
    @staticmethod
    def jpg_artifact(img: Image, n: int) -> Image:
        if n == 0:
            return img
        else:
            for _ in range(n):
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG")
                jpg = Image.frombuffer("RGB", img.size, buf.getvalue())
                jpg.save(buf, format="PNG")
                img = Image.frombuffer("RGB", jpg.size, buf.getvalue())
            return img

    @staticmethod
    def add_random_noise(img: Image, grain: float) -> Image:
        random_color = (
            np.random.random(  # type:ignore
                (img.size[1], img.size[0], 1),
            )
            * 255
        )
        rgb = np.repeat(random_color, 3)
        alpha = (
            np.random.random(  # type:ignore
                (img.size[1], img.size[0], 1),
            )
            * grain
        )
        noise = Image.fromarray(
            np.dstack(  # type:ignore
                (rgb, alpha),
            ).astype(np.uint8)
        )
        return Image.alpha_composite(img, noise)

    @staticmethod
    def salt_pepper(img: np.ndarray, intensity: float) -> Image:
        logger.log("Adding salt and pepper noise")

        def coords() -> List[np.ndarray]:
            return [
                np.random.randint(0, i - 1, int(intensity * img[0].size))
                for i in img.shape
            ]

        salt = coords()
        pepper = coords()
        img[salt[0], salt[1], :3] = 255
        img[pepper[0], pepper[1], :3] = 0

        return Image.fromarray(img)
