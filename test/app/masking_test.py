from unittest import TestCase
import numpy as np

from test.utils import describe, it

from src.app import Masking


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
            output = Masking.to_block_mat(mask, 2)
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
            output = Masking.to_block_mat(mask, 1)
            np.testing.assert_array_equal(mask, output)

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

    @describe
    def test_draw_transparency(self):
        @it
        def draws_the_correct_transparency():
            img = np.array(
                [
                    [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
                    [[2, 3, 4], [3, 4, 5], [4, 5, 6]],
                    [[3, 4, 5], [4, 5, 6], [5, 6, 7]],
                    [[4, 5, 6], [5, 6, 7], [6, 7, 8]],
                ]
            )
            mask = np.array(
                [
                    [0, 1, 0],
                    [1, 0, 1],
                    [1, 0, 1],
                    [0, 1, 0],
                ]
            )
            expected = np.array(
                [
                    [[1, 2, 3, 0], [2, 3, 4, 255], [3, 4, 5, 0]],
                    [[2, 3, 4, 255], [3, 4, 5, 0], [4, 5, 6, 255]],
                    [[3, 4, 5, 255], [4, 5, 6, 0], [5, 6, 7, 255]],
                    [[4, 5, 6, 0], [5, 6, 7, 255], [6, 7, 8, 0]],
                ]
            )
            output = Masking.draw_transparency(img, mask)
            np.testing.assert_array_equal(expected, output)
