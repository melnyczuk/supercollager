from test.utils import describe, it
from unittest import TestCase, mock

from src.adapter.url import UrlAdapter


class UrlAdapterTestCase(TestCase):
    @mock.patch("src.adapter.url.requests")
    @mock.patch("src.adapter.url.BytesIO")
    @describe
    def test_load(self, mock_bytesio, mock_request):
        mock_response = mock.MagicMock()
        mock_response.content = "response content"
        mock_request.get.return_value = mock_response
        mock_bytesio.return_value = "some bytes"

        @it
        def loads_a_remote_image():
            result = UrlAdapter.load("https://lol.lol/file.jpg")
            mock_request.get.assert_called_with("https://lol.lol/file.jpg")
            mock_bytesio.assert_called_with("response content")
            self.assertEqual("some bytes", result)

    @mock.patch("src.adapter.url.requests")
    @mock.patch("src.adapter.url.BytesIO")
    @describe
    def test_scape_site(self, mock_bytesio, mock_request):
        mock_response = mock.MagicMock()
        mock_response.content = "response content"
        mock_response.text = """
            <html>
                <body>
                    <img src="http://test.com" />
                    <img src="toast.com" />
                </body>
            </html>
        """
        mock_request.get.return_value = mock_response
        mock_bytesio.return_value = "some bytes"

        @it
        def loads_remote_image_from_image_tag_if_http():
            result = UrlAdapter.scrape_site("https://some.url")
            mock_request.get.assert_any_call("https://some.url")
            mock_request.get.assert_any_call("http://test.com")
            mock_bytesio.assert_called_with("response content")
            self.assertListEqual(["some bytes"], result)
