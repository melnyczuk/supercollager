import logging
from unittest import TestCase
import numpy as np

from src.app.func import segmentation

logging.disable(logging.INFO)


class SegmentationTestCase(TestCase):
    def test__block_mat_mask(self):
        mask = np.array(
            [
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
                [4, 5, 6, 7, 8],
            ]
        )
        expected = np.array(
            [
                [1, 1, 3, 3, 5],
                [1, 1, 3, 3, 5],
                [3, 3, 5, 5, 7],
                [3, 3, 5, 5, 7],
            ]
        )
        output = segmentation._block_mat_mask(mask, 2)
        np.testing.assert_array_equal(expected, output)

    def test__draw_transparency(self):
        img = np.array(
            [
                [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
                [[2, 3, 4], [3, 4, 5], [4, 5, 6]],
                [[3, 4, 5], [4, 5, 6], [5, 6, 7]],
                [[4, 5, 6], [5, 6, 7], [6, 7, 8]],
            ]
        )
        mask = np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [1, 0, 1],
                [0, 1, 0],
            ]
        )
        expected = np.array(
            [
                [[1, 2, 3, 0], [2, 3, 4, 255], [3, 4, 5, 0]],
                [[2, 3, 4, 255], [3, 4, 5, 0], [4, 5, 6, 255]],
                [[3, 4, 5, 255], [4, 5, 6, 0], [5, 6, 7, 255]],
                [[4, 5, 6, 0], [5, 6, 7, 255], [6, 7, 8, 0]],
            ]
        )
        output = segmentation._draw_transparency(img, mask)
        np.testing.assert_array_equal(expected, output)
