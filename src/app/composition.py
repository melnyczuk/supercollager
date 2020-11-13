from typing import Iterable, Tuple

import numpy as np
from PIL import Image  # type: ignore
from tqdm.std import tqdm  # type: ignore


class Composition:
    @staticmethod
    def layer_to_image(
        layers: Iterable[np.ndarray],
        background: Tuple[int, int, int] = (0, 0, 0),
        aspect: float = 1.0,
    ) -> Image:
        canvas = _get_canvas(layers, background=background, aspect=aspect)
        for lay in tqdm(layers):
            img = Image.fromarray(lay.astype(np.uint8))
            canvas.alpha_composite(img)
        return canvas

    @staticmethod
    def layer_to_np(
        layers: Iterable[np.ndarray],
        background: Tuple[int, int, int] = (0, 0, 0),
        aspect: float = 1.0,
    ) -> np.ndarray:
        canvas = _get_canvas(layers, background=background, aspect=aspect)

        for layer in tqdm(layers):
            for x in range(canvas.shape[0]):
                for y in range(canvas.shape[1]):
                    canvas[x][y] = (
                        layer[x][y] if layer[x][y][3] > 50 else canvas[x][y]
                    )

        return canvas.astype(
            np.uint8,
        )


def _get_canvas(
    layers: Iterable[np.ndarray],
    background: Tuple[int, int, int] = (0, 0, 0),
    aspect: float = 1.0,
) -> np.ndarray:
    return Image.fromarray(
        np.full(
            (*_get_canvas_shape(layers, aspect), 4),
            np.array([*background, 255]),
            dtype=np.uint8,
        )
    )


def _get_canvas_shape(
    imgs: Iterable[np.ndarray], aspect: float = 1.0
) -> Tuple[int, int]:
    edge_lengths = [n for a in imgs for n in a.shape]
    edge_lengths.sort(reverse=True)
    height = edge_lengths[0]
    width = int(height * aspect)
    return (height, width)
