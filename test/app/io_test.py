from test.utils import describe, each, it
from unittest import TestCase
from unittest.mock import patch

import numpy as np

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
    @patch("gluoncv.utils.download", return_value="img.png")
    @describe
    def test_mxnet_array(self, mock_download, mock_load_test):
        @each(
            [["https://url/to/a/img.png", True], ["a/path/to/a/img.png", False]]
        )
        def calls_download(args):
            mock_download.reset_mock()
            [uri, calls_download] = args
            IO.load.mxnet_array(uri)
            if calls_download:
                mock_download.assert_called_with(
                    uri, path="./dump/input/img.png"
                )
            else:
                mock_download.assert_not_called()

        @each(
            [
                ["https://url/to/a/img.png", mock_download.return_value],
                ["a/path/to/a/img.png", "a/path/to/a/img.png"],
            ]
        )
        def calls_load_test(args):
            (uri, expected) = args
            IO.load.mxnet_array(uri)
            mock_load_test.assert_called_with(expected)


class SaveTestCase(TestCase):
    @patch("PIL.Image.fromarray")
    @patch("PIL.Image.Image")
    @describe
    def test_np_array(self, mock_Image, mock_Image_fromarray):
        mock_Image_fromarray.return_value = mock_Image

        @each(
            [
                {"dir": None, "fname": "test.png"},
                {"dir": "./dir", "fname": None},
            ]
        )
        def throws_without_a_dir_and_a_fname(kwargs):
            self.assertRaises(ValueError, IO.save.np_array, [], **kwargs)

        @it
        def calls_Image_fromarray():
            arr = np.ones((2, 2, 3), dtype=np.uint8) * 255
            kwargs = {"dir": "./dir", "fname": "test", "ext": "jpeg"}
            IO.save.np_array(arr, **kwargs)
            call = mock_Image_fromarray.call_args[0][0]
            np.testing.assert_array_equal(call, arr)

        @each(["test", "test.png"])
        def saves_a_file_regardless_of_ext(fname):
            arr = np.ones((2, 2, 3), dtype=np.uint8) * 255
            IO.save.np_array(arr, fname=fname, dir="./dir", ext="png")
            mock_Image.save.assert_called_with("./dir/test.png")

        @each(["jpeg", "png"])
        def saves_a_file_with_provided_ext(ext):
            arr = np.ones((2, 2, 3), dtype=np.uint8) * 255
            IO.save.np_array(arr, fname="test", dir="./dir", ext=ext)
            mock_Image.save.assert_called_with(f"./dir/test.{ext}")

    @patch("PIL.Image")
    @describe
    def test_image(self, mock_Image):
        mock_Image.convert.return_value = mock_Image

        @each(
            [
                {"dir": None, "fname": "test.png"},
                {"dir": "./dir", "fname": None},
            ]
        )
        def throws_without_a_dir_and_a_fname(kwargs):
            self.assertRaises(ValueError, IO.save.pil_img, mock_Image, **kwargs)

        @each([("jpg", "RGB"), ("png", "RGBA")])
        def calls_convert_with_the_correct_mode(vars):
            (ext, mode) = vars
            IO.save.pil_img(mock_Image, fname="test", dir="./dir", ext=ext)
            mock_Image.convert.assert_called_with(mode)

        @each(["test", "test.png"])
        def saves_a_file_regardless_of_ext(fname):
            IO.save.pil_img(mock_Image, fname=fname, dir="./dir", ext="png")
            mock_Image.convert.assert_called()
            mock_Image.save.assert_called_with("./dir/test.png")

        @each(["jpg", "png"])
        def saves_a_file_with_provided_ext(ext):
            IO.save.pil_img(mock_Image, fname="test", dir="./dir", ext=ext)
            mock_Image.save.assert_called_with(f"./dir/test.{ext}")
