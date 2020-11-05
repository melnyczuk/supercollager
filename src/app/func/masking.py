import numpy as np
from numpy import kron as np_kron  # type: ignore


def to_block_mat(mask: np.ndarray, scalar: int) -> np.ndarray:
    if scalar < 1:
        return mask
    grain_size_matrix = np.ones([scalar, scalar], dtype=np.uint8)
    block_matrix_mask = np_kron(mask[::scalar, ::scalar], grain_size_matrix)
    return block_matrix_mask[: mask.shape[0], : mask.shape[1]]
