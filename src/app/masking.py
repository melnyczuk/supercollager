import random
from typing import Tuple, Union

import cv2  # type: ignore
import numpy as np

from src.app.image_type import ImageType
from src.app.transform import Transform


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
    def apply_mask(
        img: ImageType,
        mask: np.ndarray,
        rotate: Union[float, bool] = False,
    ) -> ImageType:
        alpha = Transform.rotate(ImageType(mask), rotate)
        rgba = np.dstack((img.np, alpha.np))  # type: ignore
        return ImageType(rgba)

    @staticmethod
    def upscale(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        return cv2.resize(mask, dsize=size, interpolation=cv2.INTER_LINEAR)
