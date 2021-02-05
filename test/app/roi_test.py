from test.utils import describe, it
from unittest import TestCase, mock

import cv2
import numpy as np
from PIL import Image

from src.app import ROI
from src.app.roi import _get_bounding_box, _get_grey, _get_roi, _get_threshold


class ROITestCase(TestCase):
    @mock.patch("src.app.roi._get_bounding_box")
    @describe
    def test_crop(self, mock_get_bounding_box):
        mock_get_bounding_box.return_value = (3, 4, 8, 7)

        @it
        def crops_an_np_array_using_a_bounding_box():
            img = Image.new("RGB", (16, 16))
            arr = np.array(img)
            out = ROI.crop(arr, 100)
            mock_get_bounding_box.assert_called_with(arr, 100)
            self.assertEqual(5, out.shape[1])
            self.assertEqual(3, out.shape[0])

        @it
        def crops_a_pil_image_using_a_bounding_box():
            img = Image.new("RGB", (16, 16))
            out = ROI.crop(img, 100)
            mock_get_bounding_box.assert_called_with(mock.ANY, 100)
            self.assertEqual(5, out.size[0])
            self.assertEqual(3, out.size[1])

    @mock.patch("src.app.roi._get_roi")
    @mock.patch("src.app.roi._get_grey")
    @describe
    def test__get_bounding_box(self, mock_get_grey, mock_get_roi):
        grey = np.ones((16, 16))
        mock_get_grey.return_value = grey
        mock_get_roi.return_value = (5, 3, 9, 9)

        @it
        def returns_a_bounding_box():
            arr = np.ones((16, 16))
            out = _get_bounding_box(arr, 100)
            mock_get_roi.assert_called_with(grey, 100)
            self.assertTupleEqual(out, (5, 3, 14, 12))

    @mock.patch("src.app.roi.cv2.cvtColor")
    @describe
    def test__get_grey(self, mock_cvtColor):
        cvtd = np.ones((16, 16, 1))
        mock_cvtColor.return_value = cvtd

        @it
        def returns_the_alpha_channel_if_4_channels():
            alpha = np.random.rand(16, 16)
            rgb = np.ones((16, 16, 3))
            arr = np.dstack((rgb, alpha))
            out = _get_grey(arr)
            np.testing.assert_array_equal(out, alpha)

        @it
        def converts_to_COLOR_RGB2GRAY_if_3_channels():
            mock_cvtColor.reset_mock()
            arr = np.ones((16, 16, 3))
            out = _get_grey(arr)
            mock_cvtColor.assert_called_with(arr, cv2.COLOR_RGB2GRAY)
            np.testing.assert_array_equal(out, cvtd)

        @it
        def converts_to_COLOR_BAYER_GR2GRAY_if_2_channels():
            mock_cvtColor.reset_mock()
            arr = np.ones((16, 16, 2))
            out = _get_grey(arr)
            mock_cvtColor.assert_called_with(arr, cv2.COLOR_BAYER_GR2GRAY)
            np.testing.assert_array_equal(out, cvtd)

        @it
        def returns_array_as_is_if_1_channel():
            mock_cvtColor.reset_mock()
            arr = np.ones((16, 16, 1))
            out = _get_grey(arr)
            mock_cvtColor.assert_not_called()
            np.testing.assert_array_equal(out, arr)

        @it
        def raises_a_ValueError_if_0_channels():
            arr = np.ones((16, 16, 0))
            self.assertRaises(ValueError, _get_grey, arr)

    @mock.patch("src.app.roi._get_threshold")
    @describe
    def test__get_roi(
        self,
        mock_get_threshold,
    ):
        mock_get_threshold.return_value = 10

        @it
        def returns_a_bounding_box_if_contours_found():
            arr = np.array(
                np.array(
                    [
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [1], [0], [0], [0], [0]],
                        [[0], [0], [0], [1], [1], [1], [0], [0], [0]],
                        [[0], [0], [1], [1], [1], [1], [1], [0], [0]],
                        [[0], [1], [1], [1], [1], [1], [1], [1], [0]],
                        [[0], [0], [1], [1], [1], [1], [1], [0], [0]],
                        [[0], [0], [0], [1], [1], [1], [0], [0], [0]],
                        [[0], [0], [0], [0], [1], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    ],
                )
                * 255,
                dtype=np.uint8,
            )
            roi = _get_roi(arr)
            self.assertTupleEqual(roi, (1, 1, 7, 7))

        @it
        def returns_a_tuple_of_length_4_filled_with_0_if_no_contours():
            with mock.patch(
                "src.app.roi.cv2.findContours", return_value=([], mock.ANY)
            ):
                with mock.patch(
                    "src.app.roi.cv2.boundingRect"
                ) as mock_boundingRect:
                    arr = np.array(np.zeros((9, 9, 1)) * 255, dtype=np.uint8)
                    roi = _get_roi(arr, 34)
                    self.assertTupleEqual(roi, (0, 0, 0, 0))
                    mock_boundingRect.assert_not_called()

        @it
        def calls__get_threshold_if_threshold_argument_is_None():
            mock_get_threshold.reset_mock()
            arr = np.array(np.zeros((9, 9, 1)) * 255, dtype=np.uint8)
            _get_roi(arr)
            mock_get_threshold.assert_called_with(arr)

    @describe
    def test__get_threshold(self):
        @it
        def returns_the_mode_of_the_image_histogram():
            arr = np.array(
                [
                    [0, 160, 255],
                    [1, 160, 254],
                    [2, 160, 253],
                    [3, 160, 252],
                ],
                dtype=np.uint8,
            )
            threshold = _get_threshold(arr)
            self.assertEqual(threshold, 160)
