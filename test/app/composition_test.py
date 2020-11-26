from test.utils import describe, each, it
from unittest import TestCase

from PIL import Image

from src.app.composition import _get_max_edge


class CompositionTestCase(TestCase):
    @describe
    def test_get_canvas_dimensions(self):
        @it
        def gets_the_largest_possible_dimensions():
            imgs = [
                Image.new("L", (30, 40)),
                Image.new("L", (20, 80)),
                Image.new("L", (92, 34)),
                Image.new("L", (16, 45)),
                Image.new("L", (31, 90)),
            ]
            output = _get_max_edge(imgs)
            self.assertEqual(92, output)
