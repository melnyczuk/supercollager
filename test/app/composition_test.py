from test.utils import describe, it
from unittest import TestCase, mock

from PIL import Image

from src.app.composition import Composition
from src.app.types import ImageType


class CompositionTestCase(TestCase):
    @mock.patch("src.app.ROI.crop", side_effect=lambda x: x)
    @describe
    def test_layer_images(self, roi_crop):
        @it
        def makes_a_canvas_image_with_the_largest_possible_dimensions():
            imgs = [
                ImageType(Image.new("RGBA", (30, 40))),
                ImageType(Image.new("RGBA", (20, 80))),
                ImageType(Image.new("RGBA", (92, 34))),
                ImageType(Image.new("RGBA", (16, 45))),
                ImageType(Image.new("RGBA", (31, 90))),
            ]

            output = Composition.layer_images(imgs)

            roi_crop.assert_called()
            self.assertEqual(92, output.dimensions[0])
            self.assertEqual(92, output.dimensions[1])
