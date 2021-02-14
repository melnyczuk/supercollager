from test.utils import describe, it
from unittest import TestCase, mock

from PIL import Image

from src.app.image_type import ImageType
from src.app.post_process import PostProcess


class PostProcessTestCase(TestCase):
    @describe
    def test_init(self):
        @it
        def stores_the_pil_image_of_an_ImageType_as_img_attribute():
            img = Image.new("RGB", (9, 9))
            post = PostProcess(ImageType(img))
            self.assertEqual(img, post.img)

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
        def calls_ImageOps_autocontrast():
            img = Image.new("RGB", (9, 9))
            post = PostProcess(ImageType(img))
            post.contrast(4.2)
            mock_ImageOps.autocontrast.assert_called_with(img)

        @it
        def applies_ImageOps_Contrast_enhance_to_img_attribute():
            img = Image.new("RGB", (9, 9))
            post = PostProcess(ImageType(img))
            post.contrast(4.2)
            mock_ImageEnhance.Contrast.assert_called_with(autocontrasted)
            mock_ImageEnhance.enhance.assert_called_with(4.2)
            self.assertEqual(contrast_enhanced, post.img)

        @it
        def returns_itself():
            img = Image.new("RGB", (9, 9))
            post = PostProcess(ImageType(img))
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
            img = Image.new("RGB", (9, 9))
            post = PostProcess(ImageType(img))
            post.color(4.2)
            mock_ImageEnhance.Color.assert_called_with(post.img)
            mock_ImageEnhance.enhance.assert_called_with(4.2)
            self.assertEqual(colour_enhanced, post.img)

        @it
        def returns_itself():
            img = Image.new("RGB", (9, 9))
            post = PostProcess(ImageType(img))
            out = post.color(4.2)
            self.assertIsInstance(out, PostProcess)
            self.assertEqual(out, post)
