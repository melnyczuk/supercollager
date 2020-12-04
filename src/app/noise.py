import io
from typing import List

import numpy as np
from PIL import Image  # type:ignore

from ..logger import logger


class Noise:
    @staticmethod
    def jpg_artifact(pil_img: Image, n: int) -> Image:
        if n == 0:
            return pil_img
        else:
            for _ in range(n):
                buf = io.BytesIO()
                pil_img.convert("RGB").save(buf, format="JPEG")
                jpg = Image.frombuffer("RGB", pil_img.size, buf.getvalue())
                jpg.save(buf, format="PNG")
                pil_img = Image.frombuffer("RGB", jpg.size, buf.getvalue())
            return pil_img

    @staticmethod
    def add_random_noise(pil_img: Image, grain: float) -> Image:
        random_color = (
            np.random.random(  # type:ignore
                (pil_img.size[1], pil_img.size[0], 1),
            )
            * 255
        )
        rgb = np.repeat(random_color, 3)
        alpha = (
            np.random.random(  # type:ignore
                (pil_img.size[1], pil_img.size[0], 1),
            )
            * grain
        )
        noise = Image.fromarray(
            np.dstack(  # type:ignore
                (rgb, alpha),
            ).astype(np.uint8)
        )
        return Image.alpha_composite(pil_img, noise)

    @staticmethod
    def salt_pepper(np_img: np.ndarray, intensity: float) -> Image:
        logger.log("Adding salt and pepper noise")

        def coords() -> List[np.ndarray]:
            return [
                np.random.randint(0, i - 1, int(intensity * np_img[0].size))
                for i in np_img.shape
            ]

        salt = coords()
        pepper = coords()
        np_img[salt[0], salt[1], :3] = 255
        np_img[pepper[0], pepper[1], :3] = 0

        return Image.fromarray(np_img)
