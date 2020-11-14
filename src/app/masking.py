import random
from typing import Tuple

import numpy as np


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
        return Masking.stack_alpha(rgb=rgb, alpha=mask)

    @staticmethod
    def stack_alpha(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        return np.dstack((rgb, alpha * 255))  # type:ignore
