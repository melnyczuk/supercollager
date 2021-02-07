from test.utils import describe, each, it
from unittest import TestCase, mock

from src.app.load import Load, _parse_uris


class LoadTestCase(TestCase):
    @mock.patch("src.app.load.ImageType", side_effect=lambda x: x)
    @mock.patch("src.app.load.Image.open")
    @mock.patch("src.app.load.Image.Image")
    @describe
    def test_uri(self, mock_Image, mock_Image_open, mock_ImageType):
        mock_Image.open.return_value = mock_Image
        mock_Image.convert.return_value = mock_Image

        @it
        def loads_a_local_image():
            Load.uri("./file.jpg")
            mock_Image_open.assert_called_with("./file.jpg")

        @it
        def loads_a_remote_image():
            with mock.patch("src.app.load.BytesIO") as mock_BytesIO:
                with mock.patch("src.app.load.requests") as mock_request:
                    mock_BytesIO.return_value = "some bytes"
                    mock_request.content = "response content"
                    mock_request.get.return_value = mock_request

                    Load.uri("https://lol.lol/file.jpg")

                    mock_request.get.assert_called_with(
                        "https://lol.lol/file.jpg"
                    )
                    mock_BytesIO.assert_called_with("response content")
                    mock_Image_open.assert_called_with("some bytes")

    @mock.patch("src.app.load.os")
    @describe
    def test__parse_uris(self, mock_os):
        @it
        def returns_a_list_with_a_single_uri_if_input_is_not_dir():
            mock_os.path.isdir.return_value = False
            input = ["./a-uri"]
            output = _parse_uris(input)
            self.assertListEqual(output, input)

        @it
        def returns_all_the_files_in_that_dir_if_input_is_dir():
            mock_os.listdir.return_value = [
                "file-1.jpg",
                "file-2.png",
                "file-3.txt",
                "file-4.tif",
            ]
            mock_os.path.isdir.return_value = True
            mock_os.path.join.side_effect = lambda x, y: x + "/" + y

            input = ["./a-dir"]
            output = _parse_uris(input)
            self.assertListEqual(
                output,
                [
                    "./a-dir/file-1.jpg",
                    "./a-dir/file-2.png",
                    "./a-dir/file-4.tif",
                ],
            )
