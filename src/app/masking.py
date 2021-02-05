import random
from typing import Tuple, Union

import numpy as np
from PIL import Image  # type: ignore
from PIL.Image import Image as ImageType  # type: ignore


class Masking:
    @staticmethod
    def to_block_mat(mask: np.ndarray, blocky: bool = False) -> np.ndarray:
        if not blocky:
            return mask

        scalar = random.randint(3, 12)
        grain_size_matrix = np.ones([scalar, scalar])
        block_matrix_mask = np.kron(  # type: ignore
            mask[::scalar, ::scalar],
            grain_size_matrix,
        )

        return block_matrix_mask[: mask.shape[0], : mask.shape[1]]

    @staticmethod
    def to_rgba(mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
        rgb = np.full((*mask.shape, 3), np.array(color))
        return np.dstack((rgb, mask))  # type: ignore

    @staticmethod
    def apply_mask(np_img: np.ndarray, mask: np.ndarray, **kwargs) -> ImageType:
        rotated_mask = _rotate(mask, **kwargs)
        rgba = np.dstack((np_img, rotated_mask))  # type: ignore
        return Image.fromarray(rgba)


def _rotate(
    np_img: np.ndarray,
    rotate: Union[float, bool] = False,
) -> np.ndarray:
    if not rotate:
        return np_img

    rotation = 90.0 if type(rotate) == bool else rotate
    size = np_img.shape[:2][::-1]

    return np.array(Image.fromarray(np_img).rotate(rotation).resize(size))
