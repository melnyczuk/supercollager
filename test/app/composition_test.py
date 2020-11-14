from test.utils import describe, each, it
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
            output = _get_canvas_shape(imgs)
            np.testing.assert_array_equal((92, 92), output)

        @each([(0.7, 64), (1.2, 110), (0.9, 82)])
        def returns_the_correct_aspect_ratio(args):
            (aspect, width) = args
            imgs = [
                np.zeros((30, 40)),
                np.zeros((20, 80)),
                np.zeros((92, 34)),
                np.zeros((16, 45)),
                np.zeros((31, 90)),
            ]
            output = _get_canvas_shape(imgs, aspect=aspect)
            np.testing.assert_array_equal((92, width), output)
