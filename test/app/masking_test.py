from test.utils import describe, each, it
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from src.app import Masking


class MaskingTestCase(TestCase):
    @patch("random.randint", return_value=2)
    @describe
    def test_to_block_mat(self, mock_randint):
        @each([True, False])
        def produces_a_block_matrix_if_blocky_is_True(blocky):
            input = np.array(
                [
                    [1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 6],
                    [3, 4, 5, 6, 7],
                    [4, 5, 6, 7, 8],
                ]
            )
            block = np.array(
                [
                    [1, 1, 3, 3, 5],
                    [1, 1, 3, 3, 5],
                    [3, 3, 5, 5, 7],
                    [3, 3, 5, 5, 7],
                ]
            )
            output = Masking.to_block_mat(input, blocky)
            expected = block if blocky else input
            np.testing.assert_array_equal(expected, output)

    @describe
    def test_to_rgba(self):
        @it
        def produces_a_mask_of_a_single_colour():
            mask = np.array(
                [
                    [0, 1, 0, 1, 0],
                    [1, 1, 0, 1, 1],
                    [0, 0, 1, 0, 0],
                ]
            )
            color = [123, 234, 45]
            expected = np.array(
                [
                    [
                        [123, 234, 45, 0],
                        [123, 234, 45, 255],
                        [123, 234, 45, 0],
                        [123, 234, 45, 255],
                        [123, 234, 45, 0],
                    ],
                    [
                        [123, 234, 45, 255],
                        [123, 234, 45, 255],
                        [123, 234, 45, 0],
                        [123, 234, 45, 255],
                        [123, 234, 45, 255],
                    ],
                    [
                        [123, 234, 45, 0],
                        [123, 234, 45, 0],
                        [123, 234, 45, 255],
                        [123, 234, 45, 0],
                        [123, 234, 45, 0],
                    ],
                ]
            )
            output = Masking.to_rgba(mask, color)
            np.testing.assert_array_equal(expected, output)
