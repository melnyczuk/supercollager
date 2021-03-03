from test.utils import describe, it
from unittest import TestCase, mock

from src.adapter.local_file import LocalFileAdapter


class LocalFileTestCase(TestCase):
    @mock.patch(
        "src.adapter.local_file.open",
        new=mock.mock_open(read_data="filecontent"),
    )
    @mock.patch("src.adapter.local_file.BytesIO")
    @mock.patch("src.adapter.local_file.os")
    @describe
    def test_uri(self, mock_os, mock_bytesio):
        @it
        def raises_if_the_input_is_not_dir():
            mock_os.path.isfile.return_value = False
            self.assertRaises(
                ValueError, LocalFileAdapter.load, *["./an-image"]
            )

        mock_os.path.isfile.return_value = True
        mock_bytesio.return_value = "some bytes"

        @it
        def loads_a_remote_image():
            result = LocalFileAdapter.load("file.jpg")
            mock_bytesio.assert_called_with("filecontent")
            self.assertEqual("some bytes", result)

        @it
        def does_not_load_a_file_with_invalid_extension():
            result = LocalFileAdapter.load("file.txt")
            self.assertEqual(None, result)
