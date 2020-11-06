from unittest import TestCase
from unittest.mock import patch
from PIL import Image
from numpy import int32, ndarray, testing as np_testing

from test.utils import describe, each, it

from src.app import IO


class LoadTestCase(TestCase):
    @patch("gluoncv.model_zoo.get_model")
    @describe
    def test_gluoncv_model(self, mock_model_zoo):
        @it
        def calls_gluoncv_get_model_with_the_correct_params():
            for pretrained in [None, True, False]:
                IO.load.gluoncv_model("test_name", pretrained=pretrained)
                mock_model_zoo.assert_called_with(
                    "test_name", pretrained=pretrained
                )

    @patch(
        "gluoncv.data.transforms.presets.rcnn.load_test", return_value=([], [])
    )
    @patch("gluoncv.utils.download", return_value="test_path.png")
    @describe
    def test_mxnet_array_from_url(self, mock_download, mock_load_test):
        @it
        def calls_download():
            url = "a/url/to/a/img.png"
            IO.load.mxnet_array_from_url(url)
            mock_download.assert_called_with(url, path="./dump/img.png")

        @it
        def calls_load_test():
            url = "a/url/to/a/img.png"
            IO.load.mxnet_array_from_url(url)
            mock_load_test.assert_called_with("test_path.png")


class SaveTestCase(TestCase):
    @patch("PIL.Image")
    @patch("PIL.Image.fromarray")
    @describe
    def test_np_array(self, mock_png_from_array, mock_Image):
        mock_png_from_array.return_value = mock_Image

        @each(
            [
                {"dir": None, "fname": "test.png"},
                {"dir": "./dir", "fname": None},
            ]
        )
        def throws_without_a_dir_and_a_fname(kwargs):
            self.assertRaises(ValueError, IO.save.np_array, [], **kwargs)

        @it
        def calls_png_package():
            arr = ndarray([2, 3], dtype=int32)
            kwargs = {"dir": "./dir", "fname": "test.png"}

            IO.save.np_array(arr, **kwargs)
            np_testing.assert_array_equal(
                mock_png_from_array.call_args[0][0], arr
            )

        @each(["test", "test.png"])
        def saves_a_file_regardless_of_ext(fname):
            IO.save.np_array([], fname=fname, dir="./dir")
            self.assertEqual(mock_Image.save.call_args[0][0], "./dir/test.png")

        @each([".jpg", ".png"])
        def saves_a_file_with_provided_ext(ext):
            IO.save.np_array([], fname="test", dir="./dir", ext=ext)
            self.assertEqual(
                mock_Image.save.call_args[0][0], f"./dir/test{ext}"
            )

    @patch("PIL.Image")
    @describe
    def test_image(self, mock_Image):
        @each(
            [
                {"dir": None, "fname": "test.png"},
                {"dir": "./dir", "fname": None},
            ]
        )
        def throws_without_a_dir_and_a_fname(kwargs):
            self.assertRaises(ValueError, IO.save.image, mock_Image, **kwargs)

        @each(["test", "test.png"])
        def saves_a_file_regardless_of_ext(fname):
            IO.save.image(mock_Image, fname=fname, dir="./dir")
            mock_Image.save.assert_called()
            self.assertEqual(mock_Image.save.call_args[0][0], "./dir/test.png")

        @each([".jpg", ".png"])
        def saves_a_file_with_provided_ext(ext):
            IO.save.image(mock_Image, fname="test", dir="./dir", ext=ext)
            mock_Image.save.assert_called()
            self.assertEqual(
                mock_Image.save.call_args[0][0], f"./dir/test{ext}"
            )
