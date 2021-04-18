from test.utils import describe, each, it
from unittest import TestCase, mock

import numpy as np

from src.cli.save import Save


class SaveTestCase(TestCase):
    @mock.patch("os.path.isdir", return_value=True)
    @mock.patch("src.cli.save.Image.Image.save")
    @describe
    def test_save_one(self, mock_Image_save, mock_isdir):
        @each(
            [
                {"dir": None, "fname": "test.png"},
                {"dir": "./dir", "fname": None},
            ]
        )
        def throws_without_a_dir_and_a_fname(kwargs):
            self.assertRaises(ValueError, Save, *[], **kwargs)

        @it
        def saves_an_image():
            img = np.ones((2, 2, 3), dtype=np.uint8)
            Save(fname="test", dir="./dir").jpg(img)
            mock_Image_save.assert_called()
            # call = mock_Image_save.call_args[0][0]
            # call.assert_called()

        @each(["test", "test.png", "test.jpeg", "test.tiff"])
        def saves_a_file_regardless_of_ext(fname):
            img = np.ones((2, 2, 4), dtype=np.uint8)
            Save(fname=fname, dir="./dir").png(img)
            mock_Image_save.assert_called_with("./dir/test.png")

        @each(
            [
                (None, "./dir/fname.png"),
                (4, "./dir/fname-4.png"),
            ]
        )
        def maybe_appends_index(params):
            index, outpath = params
            Save(fname="fname", dir="./dir").png(
                np.ones((2, 2, 4), dtype=np.uint8),
                index=index,
            )
            mock_Image_save.assert_called_with(outpath)

        @it
        def create_dir_if_dir_does_not_exist():
            with mock.patch("os.path.isdir", return_value=False):
                with mock.patch("src.cli.save.os.mkdir") as mock_mkdir:
                    img = np.ones((2, 2, 3), dtype=np.uint8)
                    Save(fname="fnmae", dir="./dir").jpg(img)
                    mock_mkdir.assert_called_with("./dir")
