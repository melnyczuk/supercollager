from typing import List, Tuple

import numpy as np
from PIL import Image  # type: ignore
from tqdm.std import tqdm  # type: ignore


class Composition:
    @staticmethod
    def layer_to_image(
        layers: List[np.ndarray],
        background: Tuple[int, int, int] = (0, 0, 0),
        aspect: float = 1.0,
    ) -> Image:
        canvas = _get_canvas(layers, background=background, aspect=aspect)

        for layer in tqdm(layers):
            img = Image.fromarray(layer.astype(np.uint8))
            canvas.alpha_composite(img)

        return canvas

    @staticmethod
    def layer_to_np(
        layers: List[np.ndarray],
        background: Tuple[int, int, int] = (0, 0, 0),
        aspect: float = 1.0,
    ) -> np.ndarray:
        canvas = Composition.layer_to_image(
            layers=layers,
            background=background,
            aspect=aspect,
        )
        return np.array(canvas).astype(np.uint8)


def _get_canvas(
    layers: List[np.ndarray],
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
    imgs: List[np.ndarray],
    aspect: float = 1.0,
) -> Tuple[int, int]:
    edge_lengths = [n for a in imgs for n in a.shape]
    edge_lengths.sort(reverse=True)
    height = edge_lengths[0]
    width = int(height * aspect)
    return (height, width)
