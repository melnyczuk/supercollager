import random
from typing import List

import numpy as np


class Masking:
    @staticmethod
    def to_block_mat(mask: np.ndarray, smooth: bool = False) -> np.ndarray:
        if smooth:
            return mask

        scalar = random.randint(3, 12)
        grain_size_matrix = np.ones([scalar, scalar])
        block_matrix_mask = np.kron(  # type: ignore
            mask[::scalar, ::scalar],
            grain_size_matrix,
        )

        return block_matrix_mask[: mask.shape[0], : mask.shape[1]]

    @staticmethod
    def to_rgba(mask: np.ndarray, color_list: List[int]) -> np.ndarray:
        color = np.array(color_list, dtype=mask.dtype)  # type: ignore
        rgb = np.full(
            (*mask.shape, color.size),
            color,
            dtype=mask.dtype,  # type: ignore
        )
        return Masking.stack_alpha(rgb, mask)

    @staticmethod
    def stack_alpha(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        alpha = alpha * np.full(
            alpha.shape,
            255.0,
            dtype=rgb.dtype,  # type: ignore
        )
        return np.dstack((rgb, alpha))  # type: ignore
