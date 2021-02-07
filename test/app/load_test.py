from test.utils import describe, each, it
from unittest import TestCase, mock

from src.app.load import Load


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
