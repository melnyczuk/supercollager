from test.utils import describe, it
from unittest import TestCase, mock

import numpy as np
from PIL import Image

from src.app.post_process import PostProcess


class PostProcessTestCase(TestCase):
    @describe
    def test_init(self):
        @it
        def returns_the_img_when_done():
            img = np.random.rand(9, 9, 3).astype(np.uint8)
            post = PostProcess(img).done()
            np.testing.assert_array_equal(img, post)

    @mock.patch("src.app.post_process.ImageEnhance")
    @mock.patch("src.app.post_process.ImageOps")
    @describe
    def test_contrast(self, mock_ImageOps, mock_ImageEnhance):
        autocontrasted = Image.new("RGB", (9, 9))
        mock_ImageOps.autocontrast.return_value = autocontrasted

        contrast_enhanced = Image.new("RGB", (9, 9))
        mock_ImageEnhance.Contrast.return_value = mock_ImageEnhance
        mock_ImageEnhance.enhance.return_value = contrast_enhanced

        @it
        def applies_autocontrast_and_contrast_enhance_to_img_attribute():
            img = np.random.rand(9, 9, 3).astype(np.uint8)
            post = PostProcess(img).contrast(4.2).done()
            pil = Image.fromarray(img)
            mock_ImageOps.autocontrast.assert_called_with(pil)
            mock_ImageEnhance.Contrast.assert_called_with(pil)
            mock_ImageEnhance.enhance.assert_called_with(4.2)
            np.testing.assert_array_equal(
                np.array(contrast_enhanced, dtype=np.uint8), post
            )

        @it
        def returns_itself():
            img = np.random.rand(9, 9, 3).astype(np.uint8)
            post = PostProcess(img)
            out = post.contrast(4.2)
            self.assertIsInstance(out, PostProcess)
            self.assertEqual(out, post)

    @mock.patch("src.app.post_process.ImageEnhance")
    @describe
    def test_color(self, mock_ImageEnhance):
        colour_enhanced = Image.new("RGB", (9, 9))
        mock_ImageEnhance.Color.return_value = mock_ImageEnhance
        mock_ImageEnhance.enhance.return_value = colour_enhanced

        @it
        def applies_ImageOps_Color_enhance_to_img_attribute():
            img = np.random.rand(9, 9, 3).astype(np.uint8)
            post = PostProcess(img).color(4.2).done()
            pil = Image.fromarray(img)
            mock_ImageEnhance.Color.assert_called_with(pil)
            mock_ImageEnhance.enhance.assert_called_with(4.2)
            np.testing.assert_array_equal(
                np.array(colour_enhanced, dtype=np.uint8), post
            )

        @it
        def returns_itself():
            img = np.random.rand(9, 9, 3).astype(np.uint8)
            post = PostProcess(img)
            out = post.color(4.2)
            self.assertIsInstance(out, PostProcess)
            self.assertEqual(out, post)
