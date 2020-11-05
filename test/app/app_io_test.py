import logging
from unittest import TestCase
from unittest.mock import patch
from numpy import int32, ndarray, testing as np_testing

from src.app.io import Load, Save

logging.disable(logging.INFO)


class LoadTestCase(TestCase):
    @patch("gluoncv.model_zoo.get_model")
    def test_gluoncv_model(self, mock_model_zoo):
        def it_calls_gluoncv_get_model_with_the_correct_params():
            for pretrained in [None, True, False]:
                Load.gluoncv_model("test_name", pretrained=pretrained)
                mock_model_zoo.assert_called_with(
                    "test_name", pretrained=pretrained
                )

        it_calls_gluoncv_get_model_with_the_correct_params()

    @patch("gluoncv.data.transforms.presets.rcnn.load_test")
    @patch("gluoncv.utils.download")
    def test_mxnet_array_from_url(self, mock_download, mock_load_test):
        mock_download.return_value = "test_path.png"
        mock_load_test.return_value = ([], [])

        def it_calls_download():
            url = "a/url/to/a/img.png"
            Load.mxnet_array_from_url(url)
            mock_download.assert_called_with(url, path="./dump/img.png")

        it_calls_download()

        def it_calls_load_test():
            url = "a/url/to/a/img.png"
            Load.mxnet_array_from_url(url)
            mock_load_test.assert_called_with("test_path.png")

        it_calls_load_test()


class SaveTestCase(TestCase):
    @patch("PIL.Image")
    @patch("PIL.Image.fromarray")
    def test_np_to_png(self, mock_png_from_array, mock_Image):
        mock_png_from_array.return_value = mock_Image

        def it_throws_without_dir_and_fname():
            for kwargs in [
                {"dir": None, "fname": "test.png"},
                {"dir": "./dir", "fname": None},
            ]:
                self.assertRaises(ValueError, Save.np_to_png, [], **kwargs)

        it_throws_without_dir_and_fname()

        def it_calls_png_package():
            arr = ndarray([2, 3], dtype=int32)
            kwargs = {"dir": "./dir", "fname": "test.png"}

            Save.np_to_png(arr, **kwargs)
            np_testing.assert_array_equal(
                mock_png_from_array.call_args[0][0], arr
            )

        it_calls_png_package()

        def it_saves_file():

            for fname in ["test", "test.png"]:
                Save.np_to_png([], fname=fname, dir="./dir")
                self.assertEqual(
                    mock_Image.save.call_args[0][0], "./dir/test.png"
                )

        it_saves_file()
