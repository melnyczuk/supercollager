from test.utils import describe, it
from unittest import TestCase, mock

from PIL import Image

from src.app import Composition


class CompositionTestCase(TestCase):
    @mock.patch("src.app.ROI.crop_pil", side_effect=lambda x: x)
    @describe
    def test_layer_images(self, roi_crop_pil):
        @it
        def makes_a_canvas_image_with_the_largest_possible_dimensions():
            imgs = [
                Image.new("RGBA", (30, 40)),
                Image.new("RGBA", (20, 80)),
                Image.new("RGBA", (92, 34)),
                Image.new("RGBA", (16, 45)),
                Image.new("RGBA", (31, 90)),
            ]

            output = Composition.layer_images(imgs)

            roi_crop_pil.assert_called()
            self.assertEqual(92, output.size[0])
            self.assertEqual(92, output.size[1])
