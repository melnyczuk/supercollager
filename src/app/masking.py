import numpy as np
from typing import List


class Masking:
    @staticmethod
    def to_block_mat(mask: np.ndarray, scalar: int) -> np.ndarray:
        if scalar < 1:
            return mask
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
        return Masking.draw_transparency(rgb, mask)

    @staticmethod
    def draw_transparency(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        alpha = mask * np.full(
            mask.shape,
            255.0,
            dtype=rgb.dtype,  # type: ignore
        )
        return np.dstack((rgb, alpha))  # type: ignore
