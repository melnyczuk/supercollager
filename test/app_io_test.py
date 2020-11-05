import logging
from unittest import TestCase
from unittest.mock import patch
from numpy import int32, ndarray, testing as np_testing

from src.app.io import load, save

logging.disable(logging.INFO)


class LoadTestCase(TestCase):
    @patch("gluoncv.model_zoo.get_model")
    def test_gluoncv_model(self, mock_model_zoo):
        for pretrained in [None, True, False]:
            load.gluoncv_model("test_name", pretrained=pretrained)
            mock_model_zoo.assert_called_with(
                "test_name", pretrained=pretrained
            )

    @patch("gluoncv.data.transforms.presets.rcnn.load_test")
    @patch("gluoncv.utils.download")
    def test_mxnet_array_from_url(self, mock_download, mock_load_test):
        mock_download.return_value = "test_path.png"
        mock_load_test.return_value = ([], [])

        url = "a/url/to/a/img.png"
        load.mxnet_array_from_url(url)

        mock_download.assert_called_with(url, path="./dump/img.png")
        mock_load_test.assert_called_with("test_path.png")


class SaveTestCase(TestCase):
    def test_np_to_png_throws_without_dir_and_fname(self):
        for kwargs in [
            {"dir": None, "fname": "test.png"},
            {"dir": "./dir", "fname": None},
        ]:
            self.assertRaises(ValueError, save.np_to_png, [], **kwargs)

    @patch("PIL.Image.fromarray")
    def test_np_to_png_calls_png_package(self, mock_png_from_array):
        arr = ndarray([2, 3], dtype=int32)
        kwargs = {"dir": "./dir", "fname": "test.png"}

        save.np_to_png(arr, **kwargs)

        np_testing.assert_array_equal(mock_png_from_array.call_args[0][0], arr)

    @patch("PIL.Image")
    @patch("PIL.Image.fromarray")
    def test_np_to_png_saves_file(self, mock_png_from_array, mock_Image):
        mock_png_from_array.return_value = mock_Image

        for fname in ["test", "test.png"]:
            save.np_to_png([], fname=fname, dir="./dir")
            self.assertEqual(mock_Image.save.call_args[0][0], "./dir/test.png")
