from test.utils import describe, it
from unittest import TestCase, mock

from PIL import Image
import numpy as np

from src.app import App

# from src.app.types import ImageType


class End2EndTestCase(TestCase):
    @mock.patch("src.app.load.PIL.Image.open")
    @describe
    def test_collage(self, mock_Image_open):
        arr = (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.uint8,
            )
            * 255,
        )
        img = Image.fromarray(arr)
        mock_Image_open.return_value = img

        @it
        def runs():
            output = App.collage(["a-file.jpg"])

            self.assertEqual(output.pil, img)
            np.testing.assert_array_equal(output.np, arr)
