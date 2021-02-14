from test.utils import describe, each
from unittest import TestCase

import numpy as np
from PIL import Image

from src.app.image_type import ImageType


class ImageTypeTestCase(TestCase):
    @describe
    def test_inits_from_pil(self):
        @each(["L", "F", "RGB", "RGBA"])
        def with_pil_attr(mode):
            pil = Image.new(mode, (30, 40))
            out = ImageType(pil)
            self.assertEqual(out.pil, pil)

        @each(["L", "F", "RGB", "RGBA"])
        def with_np_attr(mode):
            pil = Image.new(mode, (30, 40))
            out = ImageType(pil)
            self.assertEqual(out.pil, pil)
            np.testing.assert_array_equal(out.np, np.array(pil))

        @each(["L", "F", "RGB", "RGBA"])
        def with_correct_dimensions(mode):
            pil = Image.new(mode, (30, 40))
            out = ImageType(pil)
            self.assertTupleEqual(out.dimensions, (30, 40))

        @each([("L", 1), ("F", 1), ("RGB", 3), ("RGBA", 4)])
        def with_correct_number_of_channels(args):
            mode, n_channels = args
            pil = Image.new(mode, (30, 40))
            out = ImageType(pil)
            self.assertEqual(out.channels, n_channels)

    @describe
    def test_inits_from_np(self):
        @each([(89, 91), (4, 5, 3), (26, 42, 4)])
        def with_pil_attr(shape):
            arr = np.ones(shape)
            out = ImageType(arr)
            self.assertEqual(out.pil, Image.fromarray(arr.astype(np.uint8)))

        @each([(89, 91), (4, 5, 3), (26, 42, 4)])
        def with_pil_attr(shape):
            arr = np.ones(shape)
            out = ImageType(arr)
            np.testing.assert_array_equal(out.np, arr)

        @each(
            [((89, 91), (91, 89)), ((4, 5, 3), (5, 4)), ((26, 42, 4), (42, 26))]
        )
        def gets_the_correct_dimensions(args):
            shape, size = args
            arr = np.ones(shape)
            out = ImageType(arr)
            self.assertTupleEqual(out.dimensions, size)

        @each([((89, 91), 1), ((4, 5, 3), 3), ((26, 42, 4), 4)])
        def with_correct_number_of_channels(args):
            shape, n_channels = args
            arr = np.ones(shape)
            out = ImageType(arr)
            self.assertEqual(out.channels, n_channels)
