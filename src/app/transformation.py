import numpy as np

from src.logger import logger


class Transformation:
    @staticmethod
    def scale_up_nparray(arr: np.ndarray, scalar: int):
        logger.log(f"resizing image by factor {scalar}")
        big = np.zeros(
            (arr.shape[0] * scalar, arr.shape[1] * scalar, arr.shape[2])
            # dtype=arr.dtype,
        )
        big[0::scalar, 0::scalar] = arr
        big[1::scalar, 1::scalar] = arr
        big[0::scalar, 1::scalar] = arr
        big[1::scalar, 0::scalar] = arr
        return big