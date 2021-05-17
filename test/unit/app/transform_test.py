from test.utils import describe, it
from unittest import TestCase

import numpy as np

from src.app.transform import Transform


class TransformTest(TestCase):
    @describe
    def test_rotate(self):
        @it
        def does_not_rotate_if_rotate_is_False():
            input = np.array(
                [
                    [0, 1, 0],
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 0],
                ]
            )
            output = Transform.rotate(input, rotate=False)
            np.testing.assert_almost_equal(input, output)

        @it
        def rotates_90_deg_if_rotate_is_True():
            input = np.array(
                [
                    [0, 1, 0],
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 0],
                ]
            )
            expected = np.array(
                [
                    [0, 0, 0],
                    [1, 1, 0],
                    [1, 1, 1],
                    [0, 0, 0],
                ]
            )
            output = Transform.rotate(input, rotate=True)
            np.testing.assert_almost_equal(expected, output)

        @it
        def rotates_provided_deg_if_rotate_is_float():
            input = np.array(
                [
                    [0, 1, 0],
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 0],
                ]
            )
            expected = np.array(
                [
                    [1, 0, 0],
                    [1, 1, 1],
                    [0, 1, 1],
                    [0, 0, 1],
                ]
            )
            output = Transform.rotate(input, rotate=45)
            np.testing.assert_almost_equal(expected, output)

    @describe
    def test_resize(self):
        @it
        def upscales():
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
            output = Transform.resize(input, (9, 9))
            np.testing.assert_array_equal(expected, output)

        @it
        def downscales():
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
                    [0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0],
                ],
                dtype=np.uint8,
            )
            output = Transform.resize(input, (3, 3))
            np.testing.assert_array_equal(expected, output)
