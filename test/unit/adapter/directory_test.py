from test.utils import describe, it
from unittest import TestCase, mock

from src.adapter.directory import DirectoryAdapter


class AdapterTestCase(TestCase):
    @mock.patch("src.adapter.directory.LocalFileAdapter")
    @mock.patch("src.adapter.directory.os")
    @describe
    def test_load(self, mock_os, mock_LocalFileAdapter):
        @it
        def raises_if_the_input_is_not_dir():
            mock_os.path.isdir.return_value = False
            with self.assertRaises(ValueError):
                list(DirectoryAdapter.load("an-image"))

        mock_LocalFileAdapter.load.side_effect = lambda x: x
        mock_os.path.isdir.side_effect = lambda x: "." not in x
        mock_os.path.join.side_effect = lambda x, y: x + "/" + y

        @it
        def returns_all_the_files_in_that_dir_if_input_is_dir():
            mock_os.listdir.return_value = [
                "file-1.jpg",
                "file-2.png",
                "file-4.tif",
            ]

            input = "a-dir"
            output = list(DirectoryAdapter.load(input))
            self.assertListEqual(
                output,
                [
                    "a-dir/file-1.jpg",
                    "a-dir/file-2.png",
                    "a-dir/file-4.tif",
                ],
            )

        @it
        def performs_recursively():
            mock_os.listdir.side_effect = (
                lambda x: [
                    "file-1.jpg",
                    "file-2.png",
                ]
                if x != "a-dir"
                else ["nested-dir"]
            )

            input = "a-dir"
            output = list(DirectoryAdapter.load(input))
            self.assertListEqual(
                output,
                [
                    "a-dir/nested-dir/file-1.jpg",
                    "a-dir/nested-dir/file-2.png",
                ],
            )
