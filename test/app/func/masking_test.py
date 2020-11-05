from unittest import TestCase
import numpy as np

from test.utils import describe, it

from src.app.func import masking


class MaskingTestCase(TestCase):
    @describe
    def test_to_block_mat(self):
        @it
        def produces_a_block_matrix():
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

        @it
        def returns_the_mask_as_is_if_scale_is_1_or_less():
            mask = np.array(
                [
                    [1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 6],
                    [3, 4, 5, 6, 7],
                    [4, 5, 6, 7, 8],
                ]
            )
            output = masking.to_block_mat(mask, 1)
            np.testing.assert_array_equal(mask, output)

    @describe
    def test_to_color(self):
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
                        [0, 0, 0],
                        [123, 234, 45],
                        [0, 0, 0],
                        [123, 234, 45],
                        [0, 0, 0],
                    ],
                    [
                        [123, 234, 45],
                        [123, 234, 45],
                        [0, 0, 0],
                        [123, 234, 45],
                        [123, 234, 45],
                    ],
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [123, 234, 45],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                ]
            )
            output = masking.to_color(mask, color)
            np.testing.assert_array_equal(expected, output)
