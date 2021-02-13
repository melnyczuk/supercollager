from test.utils import describe, each, it
from unittest import TestCase, mock

import numpy as np

from src.app.masking import Masking
from src.app.types import ImageType


class MaskingTestCase(TestCase):
    @mock.patch("random.randint", return_value=2)
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
                    [0, 255, 0, 255, 0],
                    [255, 255, 0, 255, 255],
                    [0, 0, 255, 0, 0],
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
    def test_apply_mask(self):
        @it
        def applies_a_mask_without_rotating():
            mask = np.array(
                [
                    [0, 255, 0, 255, 0],
                    [255, 255, 0, 255, 255],
                    [0, 0, 255, 0, 0],
                ]
            )

            arr = np.array(
                [
                    [
                        [125, 255, 0],
                        [0, 255, 125],
                        [125, 255, 0],
                        [0, 255, 125],
                        [125, 255, 0],
                    ],
                    [
                        [255, 125, 0],
                        [255, 0, 125],
                        [255, 0, 125],
                        [255, 125, 0],
                        [255, 0, 125],
                    ],
                    [
                        [0, 125, 255],
                        [125, 0, 255],
                        [0, 125, 255],
                        [125, 0, 255],
                        [0, 125, 255],
                    ],
                ]
            )

            expected = np.array(
                [
                    [
                        [125, 255, 0, 0],
                        [0, 255, 125, 255],
                        [125, 255, 0, 0],
                        [0, 255, 125, 255],
                        [125, 255, 0, 0],
                    ],
                    [
                        [255, 125, 0, 255],
                        [255, 0, 125, 255],
                        [255, 0, 125, 0],
                        [255, 125, 0, 255],
                        [255, 0, 125, 255],
                    ],
                    [
                        [0, 125, 255, 0],
                        [125, 0, 255, 0],
                        [0, 125, 255, 255],
                        [125, 0, 255, 0],
                        [0, 125, 255, 0],
                    ],
                ]
            )

            img = ImageType(arr)
            out = Masking.apply_mask(img, mask)
            np.testing.assert_array_equal(expected, out.np)

        @it
        def applies_a_mask_with_rotating():
            mask = np.array(
                [
                    [0, 255, 0, 255, 0],
                    [255, 255, 0, 255, 255],
                    [0, 0, 255, 0, 0],
                ]
            )

            arr = np.array(
                [
                    [
                        [125, 255, 0],
                        [0, 255, 125],
                        [125, 255, 0],
                        [0, 255, 125],
                        [125, 255, 0],
                    ],
                    [
                        [255, 125, 0],
                        [255, 0, 125],
                        [255, 0, 125],
                        [255, 125, 0],
                        [255, 0, 125],
                    ],
                    [
                        [0, 125, 255],
                        [125, 0, 255],
                        [0, 125, 255],
                        [125, 0, 255],
                        [0, 125, 255],
                    ],
                ]
            )

            expected = np.array(
                [
                    [
                        [125, 255, 0, 0],
                        [0, 255, 125, 255],
                        [125, 255, 0, 255],
                        [0, 255, 125, 0],
                        [125, 255, 0, 0],
                    ],
                    [
                        [255, 125, 0, 0],
                        [255, 0, 125, 0],
                        [255, 0, 125, 0],
                        [255, 125, 0, 255],
                        [255, 0, 125, 0],
                    ],
                    [
                        [0, 125, 255, 0],
                        [125, 0, 255, 255],
                        [0, 125, 255, 255],
                        [125, 0, 255, 0],
                        [0, 125, 255, 0],
                    ],
                ]
            )

            img = ImageType(arr)
            out = Masking.apply_mask(img, mask, True)
            np.testing.assert_array_equal(expected, out.np)

    @describe
    def test_upscale(self):
        @it
        def produces_an_upscaled_mask():
            input = np.array(
                [
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                ],
                dtype=np.uint8,
            )
            expected = np.array(
                [
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                ],
                dtype=np.uint8,
            )
            output = Masking.upscale(input, (9, 9))
            np.testing.assert_array_equal(expected, output)
