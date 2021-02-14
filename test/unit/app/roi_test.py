from test.utils import describe, it
from unittest import TestCase, mock

import numpy as np
from PIL import Image

from src.app.roi import ROI
from src.app.image_type import ImageType


class ROITestCase(TestCase):
    @describe
    def test_crop(self):
        @it
        def crops_an_np_array_using_a_bounding_box():
            arr = np.zeros((16, 16, 3)).astype(np.uint8)
            arr[3:13, 5:11, :] = np.ones((10, 6, 3)).astype(np.uint8) * 255
            img = ImageType(arr)
            out = ROI.crop(img, 100)
            self.assertEqual(6, out.dimensions[0])
            self.assertEqual(10, out.dimensions[1])

        @it
        def returns_the_original_image_if_a_roi_is_not_found():
            arr = np.zeros((16, 16, 3)).astype(np.uint8)
            arr[3:13, 5:11, :] = np.random.rand(10, 6, 3).astype(np.uint8) * 255
            img = ImageType(arr)
            out = ROI.crop(img, 100)
            np.testing.assert_array_equal(out.np, arr)

        @it
        def uses_the_alpha_channel_to_find_roi_if_provided():
            arr = np.ones((16, 16, 3)).astype(np.uint8) * 255
            alpha = np.zeros((16, 16)).astype(np.uint8)
            alpha[4:8, 3:8] = np.ones((4, 5)).astype(np.uint8) * 255
            img = ImageType(np.dstack((arr, alpha)))
            out = ROI.crop(img, 100)
            self.assertEqual(5, out.dimensions[0])
            self.assertEqual(4, out.dimensions[1])

        @it
        def only_finds_the_roi_that_exceeds_the_threshold():
            roi = np.ones((5, 8)).astype(np.uint8) * 99
            roi[2:4, 3:6] = np.ones((2, 3)).astype(np.uint8) * 101
            arr = np.zeros((16, 16)).astype(np.uint8)
            arr[3:8, 6:14] = roi
            img = ImageType(arr)
            out = ROI.crop(img, 100)
            self.assertEqual(3, out.dimensions[0])
            self.assertEqual(2, out.dimensions[1])

        @it
        def finds_the_roi_by_infering_a_threshold_if_none_provided():
            arr = np.zeros((16, 16, 3)).astype(np.uint8)
            arr[3:13, 5:11, :] = np.ones((10, 6, 3)).astype(np.uint8) * 255
            img = ImageType(arr)
            out = ROI.crop(img)
            self.assertEqual(6, out.dimensions[0])
            self.assertEqual(10, out.dimensions[1])
