from test.utils import describe, each, it
from unittest import TestCase, mock

import numpy as np
from PIL import Image

from src.app.types import ImageType
from src.cli.save import Save


class SaveTestCase(TestCase):
    @mock.patch("os.path.isdir", return_value=True)
    @mock.patch("src.app.types.Image.Image.save")
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
        def saves_an_image_of_ImageType():
            img = ImageType(np.ones((2, 2, 3)))
            Save(fname="test", dir="./dir").one(img)
            mock_Image_save.assert_called()
            # call = mock_Image_save.call_args[0][0]
            # call.assert_called()

        @each([("RGB", "jpg"), ("RGBA", "png")])
        def calls_convert_with_the_correct_mode(vars):
            (mode, ext) = vars
            img = ImageType(Image.new(mode, (9, 9)))
            Save(fname="test", dir="./dir").one(img)
            mock_Image_save.assert_called_with(f"./dir/test.{ext}")

        @each(["test", "test.png", "test.jpeg", "test.tiff"])
        def saves_a_file_regardless_of_ext(fname):
            img = ImageType(np.ones((2, 2, 4)))
            Save(fname=fname, dir="./dir").one(img)
            mock_Image_save.assert_called_with("./dir/test.png")

        @each(
            [
                ({"label": None, "index": None}, "./dir/fname.png"),
                ({"label": None, "index": 4}, "./dir/fname-4.png"),
                ({"label": "test", "index": None}, "./dir/fname-test.png"),
                ({"label": "toast", "index": 6}, "./dir/fname-6-toast.png"),
            ]
        )
        def maybe_appends_label_and_index(params):
            kwargs, outpath = params
            Save(fname="fname", dir="./dir").one(
                ImageType(np.ones((2, 2, 4))),
                index=kwargs["index"],
                label=kwargs["label"],
            )
            mock_Image_save.assert_called_with(outpath)

        @it
        def create_dir_if_dir_does_not_exist():
            with mock.patch("os.path.isdir", return_value=False):
                with mock.patch("src.cli.save.os.mkdir") as mock_mkdir:
                    img = ImageType(np.ones((2, 2, 3)))
                    Save(fname="fnmae", dir="./dir").one(img)
                    mock_mkdir.assert_called_with("./dir")
