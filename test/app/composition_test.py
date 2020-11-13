from test.utils import describe, it
from unittest import TestCase

import numpy as np

from src.app.composition import _get_canvas_shape


class CompositionTestCase(TestCase):
    @describe
    def test_get_canvas_dimensions(self):
        @it
        def gets_the_largest_possible_dimensions():
            imgs = [
                np.zeros((30, 40)),
                np.zeros((20, 80)),
                np.zeros((92, 34)),
                np.zeros((16, 45)),
                np.zeros((31, 90)),
            ]
            output = _get_canvas_shape(imgs, reverse=True)
            np.testing.assert_array_equal((92, 90), output)

        @it
        def gets_the_smallest_possible_dimensions():
            imgs = [
                np.zeros((30, 40)),
                np.zeros((20, 80)),
                np.zeros((92, 34)),
                np.zeros((16, 45)),
                np.zeros((31, 90)),
            ]
            output = _get_canvas_shape(imgs, reverse=False)
            np.testing.assert_array_equal((16, 34), output)
