from test.utils import describe, each, it
from unittest import TestCase, mock

import numpy as np

from src.app import IO


class IOTestCase(TestCase):
    @mock.patch("PIL.Image.open")
    @mock.patch("PIL.Image.Image")
    @describe
    def test_load(self, mock_Image, mock_Image_open):
        mock_Image.open.return_value = mock_Image
        mock_Image.convert.return_value = mock_Image

        @it
        def loads_a_local_image():
            IO.load("./file.jpg")
            mock_Image_open.assert_called_with("./file.jpg")

        @it
        def loads_a_remote_image():
            with mock.patch("src.app.io.BytesIO") as mock_BytesIO:
                with mock.patch("src.app.io.requests") as mock_request:
                    mock_BytesIO.return_value = "some bytes"
                    mock_request.content = "response content"
                    mock_request.get.return_value = mock_request

                    IO.load("https://lol.lol/file.jpg")

                    mock_request.get.assert_called_with(
                        "https://lol.lol/file.jpg"
                    )
                    mock_BytesIO.assert_called_with("response content")
                    mock_Image_open.assert_called_with("some bytes")

    @mock.patch("os.path.isdir", return_value=True)
    @mock.patch("PIL.Image.fromarray")
    @mock.patch("PIL.Image.Image")
    @describe
    def test_save(self, mock_Image, mock_Image_fromarray, mock_isdir):
        mock_Image_fromarray.return_value = mock_Image
        mock_Image.convert.return_value = mock_Image

        @each(
            [
                {"dir": None, "fname": "test.png"},
                {"dir": "./dir", "fname": None},
            ]
        )
        def throws_without_a_dir_and_a_fname(kwargs):
            self.assertRaises(ValueError, IO.save, [], **kwargs)
            self.assertRaises(ValueError, IO.save, mock_Image, **kwargs)

        @it
        def converts_np_array_to_pil_image():
            arr = np.ones((2, 2, 3), dtype=np.uint8) * 255
            kwargs = {"dir": "./dir", "fname": "test"}
            IO.save(arr, **kwargs)
            call = mock_Image_fromarray.call_args[0][0]
            np.testing.assert_array_equal(call, arr)

        @each([("RGB", "jpg"), ("RGBA", "png")])
        def calls_convert_with_the_correct_mode(vars):
            (mode, ext) = vars
            mock_Image.mode = mode
            IO.save(mock_Image, fname="test", dir="./dir")
            mock_Image.save.assert_called_with(f"./dir/test.{ext}")

        @each(["test", "test.png", "test.jpeg", "test.tiff"])
        def saves_a_file_regardless_of_ext(fname):
            mock_Image.mode = "RGBA"
            IO.save(mock_Image, fname=fname, dir="./dir")
            mock_Image.save.assert_called_with("./dir/test.png")

        @it
        def creates_a_dir_if_none_exists():
            with mock.patch("os.path.isdir", return_value=False):
                with mock.patch("src.app.io.mkdir") as mock_mkdir:
                    mock_Image.mode = "RGBA"
                    IO.save(mock_Image, fname="fnmae", dir="./dir")
                    mock_mkdir.assert_called_with("./dir")
