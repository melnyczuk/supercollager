import logging
from unittest import TestCase
import numpy as np

from src.app.func import masking

logging.disable(logging.INFO)


class MaskingTestCase(TestCase):
    def test__block_mat_mask(self):
        mask = np.array(
            [
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
                [4, 5, 6, 7, 8],
            ]
        )
        expected = np.array(
            [
                [1, 1, 3, 3, 5],
                [1, 1, 3, 3, 5],
                [3, 3, 5, 5, 7],
                [3, 3, 5, 5, 7],
            ]
        )
        output = masking.to_block_mat(mask, 2)
        np.testing.assert_array_equal(expected, output)
