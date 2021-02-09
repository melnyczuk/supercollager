from test.utils import describe, it
from unittest import TestCase, mock

import cv2
import numpy as np
from PIL import Image

from src.app.roi import (
    ROI,
    _get_bounding_box,
    _get_grey,
    _get_roi,
    _get_threshold,
)
from src.app.types import ImageType


class ROITestCase(TestCase):
    @mock.patch("src.app.roi._get_bounding_box")
    @describe
    def test_crop(self, mock_get_bounding_box):
        mock_get_bounding_box.return_value = (3, 4, 8, 7)

        @it
        def crops_an_np_array_using_a_bounding_box():
            img = ImageType(Image.new("RGB", (16, 16)))
            out = ROI.crop(img, 100)
            mock_get_bounding_box.assert_called_with(img, 100)
            self.assertEqual(5, out.dimensions[0])
            self.assertEqual(3, out.dimensions[1])

    @mock.patch("src.app.roi._get_roi")
    @describe
    def test__get_bounding_box(self, mock_get_roi):
        mock_get_roi.return_value = (5, 3, 9, 9)

        @it
        def returns_a_bounding_box():
            img = ImageType(np.ones((16, 16)))
            out = _get_bounding_box(img, 100)
            mock_get_roi.assert_called_with(img, 100)
            self.assertTupleEqual(out, (5, 3, 14, 12))

    @describe
    def test__get_grey(self):
        @it
        def returns_the_alpha_channel_if_4_channels():
            alpha = np.random.rand(16, 16).astype(np.uint8)
            rgb = np.ones((16, 16, 3))
            img = ImageType(np.dstack((rgb, alpha)))
            out = _get_grey(img)
            np.testing.assert_array_equal(out.np, alpha)

        @it
        def converts_to_COLOR_RGB2GRAY_if_3_channels():
            img = ImageType(np.ones((16, 16, 3)))
            out = _get_grey(img)
            np.testing.assert_array_equal(out.np, np.ones((16, 16)))

        @it
        def returns_array_as_is_if_1_channel():
            img = ImageType(np.ones((16, 16)))
            out = _get_grey(img)
            np.testing.assert_array_equal(out.np, img.np)

    @mock.patch("src.app.roi._get_threshold")
    @describe
    def test__get_roi(
        self,
        mock_get_threshold,
    ):
        mock_get_threshold.return_value = 10

        @it
        def returns_a_bounding_box_if_contours_found():
            img = ImageType(
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
            roi = _get_roi(img)
            self.assertTupleEqual(roi, (1, 1, 7, 7))

        @it
        def returns_a_tuple_of_length_4_filled_with_0_if_no_contours():
            with mock.patch(
                "src.app.roi.cv2.findContours", return_value=([], mock.ANY)
            ):
                with mock.patch(
                    "src.app.roi.cv2.boundingRect"
                ) as mock_boundingRect:
                    img = ImageType(np.zeros((9, 9), dtype=np.uint8))
                    roi = _get_roi(img, 34)
                    self.assertTupleEqual(roi, (0, 0, 0, 0))
                    mock_boundingRect.assert_not_called()

        @it
        def calls__get_threshold_if_threshold_argument_is_None():
            mock_get_threshold.reset_mock()
            img = ImageType(np.zeros((9, 9), dtype=np.uint8))
            _get_roi(img)
            mock_get_threshold.assert_called_with(img)

    @describe
    def test__get_threshold(self):
        @it
        def returns_the_mode_of_the_image_histogram():
            img = ImageType(
                np.array(
                    [
                        [0, 160, 255],
                        [1, 160, 254],
                        [2, 160, 253],
                        [3, 160, 252],
                    ],
                    dtype=np.uint8,
                )
            )
            threshold = _get_threshold(img)
            self.assertEqual(threshold, 160)
