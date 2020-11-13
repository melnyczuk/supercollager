from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm.std import tqdm  # type: ignore


class Composition:
    @staticmethod
    def layer_to_image(
        transparencies: List[np.ndarray],
        background: List[int] = [0, 0, 0],
        dtype=np.uint8,
    ) -> Image:
        dimensions = _get_canvas_dimensions(transparencies)

        canvas = Image.fromarray(
            np.full(
                (*dimensions, 4),
                np.array([*background, 255]),
                dtype=dtype,
            )
        )

        crops = [_crop_center(layer, dimensions) for layer in transparencies]
        imgs = [Image.fromarray(crop.astype(dtype)) for crop in crops]
        [canvas.alpha_composite(img) for img in imgs]
        return canvas

    @staticmethod
    def layer_to_np(
        transparencies: List[np.ndarray],
        background: List[int] = [0, 0, 0],
        dtype=np.uint8,
    ) -> np.ndarray:
        dimensions = _get_canvas_dimensions(transparencies)

        canvas = np.full(
            (*dimensions, 4),
            np.array([*background, 255]),
            dtype=dtype,
        )

        for layer in tqdm(transparencies):
            for x in range(canvas.shape[0]):
                for y in range(canvas.shape[1]):
                    crop = _crop_center(layer, dimensions)
                    canvas[x][y] = (
                        crop[x][y] if crop[x][y][3] > 50 else canvas[x][y]
                    )

        return canvas.astype(dtype)


def _get_canvas_dimensions(
    imgs: List[np.ndarray], reverse=False
) -> Tuple[int, int]:
    shapes = [a.shape for a in imgs]

    shapes.sort(reverse=reverse, key=(lambda a: a[0]))
    width = shapes[0][0]

    shapes.sort(reverse=reverse, key=(lambda a: a[1]))
    height = shapes[0][1]

    return (width, height)


def _crop_center(img: np.ndarray, shape: tuple) -> np.ndarray:
    diff = (img.shape[0] - shape[0], img.shape[1] - shape[1])
    x = (_ := int(diff[0] / 2), img.shape[0] - _)
    y = (_ := int(diff[1] / 2), img.shape[1] - _)
    return img[x[0] : x[1], y[0] : y[1]]  # noqa: E203
