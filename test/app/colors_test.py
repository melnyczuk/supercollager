from unittest import TestCase
from unittest.mock import patch

from test.utils import describe, it

from src.app.colors.colors import _get_rand_indicies


class ColorsTestCase(TestCase):
    @patch("random.randint", side_effect=[3, 4, 5])
    @describe
    def test_to_list(self, mock_randint):
        def it_returns_a_list_of_unique_colors():

            return

    @patch("random.randint", side_effect=[3, 4, 5])
    @describe
    def test__get_rand_indicies(self, mock_randint):
        @it
        def returns_the_input_list_if_it_is_long_enough():
            existing = [2, 3, 4]
            output = _get_rand_indicies(3, existing)
            self.assertListEqual([2, 3, 4], output)
            mock_randint.assert_not_called()

        @it
        def returns_a_list_of_unique_ints():
            existing = [2, 3, 4]
            output = _get_rand_indicies(4, existing)
            mock_randint.assert_called()
            self.assertListEqual([2, 3, 4, 5], output)
