from test.utils import describe, it
from unittest import TestCase, mock

import numpy as np
from PIL import Image

from src.app.composition import Composition


class CompositionTestCase(TestCase):
    @mock.patch("src.app.composition.ROI.crop", side_effect=lambda x: x)
    @describe
    def test_layer_images(self, roi_crop):
        @it
        def makes_a_canvas_image_with_the_largest_possible_dimensions():
            imgs = [
                np.zeros((30, 40, 4), dtype=np.uint8),
                np.zeros((20, 80, 4), dtype=np.uint8),
                np.zeros((92, 34, 4), dtype=np.uint8),
                np.zeros((16, 45, 4), dtype=np.uint8),
                np.zeros((31, 90, 4), dtype=np.uint8),
            ]

            output = Composition.layer_images(imgs)

            roi_crop.assert_called()
            self.assertEqual(92, output.shape[1])
            self.assertEqual(92, output.shape[0])
