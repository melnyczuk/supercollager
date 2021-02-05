from random import randint
from test.utils import describe, each, it
from unittest import TestCase, mock

from src.app import Colors
from src.app.colors import color_lookup


class ColorsTestCase(TestCase):
    @mock.patch("random.randint")
    @describe
    def test_pick(self, mock_randint):
        @it
        def picks_a_colour_from_the_lookup():
            mock_randint.side_effect = [5]
            expected = list(color_lookup.values())[5]
            output = Colors.pick()
            self.assertTupleEqual(expected, output)

    @mock.patch("random.randint")
    @describe
    def test_as_list(self, mock_randint):
        @it
        def returns_a_list_of_unique_colors():
            mock_randint.side_effect = [1, 2, 5]
            lookup_vals = list(color_lookup.values())
            expected = [lookup_vals[1], lookup_vals[2], lookup_vals[5]]
            output = Colors.as_list(3)
            self.assertListEqual(expected, output)

        @each([len(color_lookup) + 1, len(color_lookup)])
        def returns_a_list_at_most_as_long_as_the_lookup(
            length,
        ):
            mock_randint.side_effect = [
                randint(0, len(color_lookup) - 1) for _ in range(10000)
            ]
            lookup_vals = list(color_lookup.values())
            output = Colors.as_list(length)
            self.assertEqual(len(lookup_vals), len(output))

    @mock.patch("random.randint", side_effect=[*range(100000)])
    @describe
    def test__get_rand_indicies(self, mock_randint):
        @it
        def returns_the_input_list_if_it_is_long_enough():
            existing = [4, 2, 3]
            output = Colors._get_rand_indicies(3, existing)
            self.assertListEqual([4, 2, 3], output)
            mock_randint.assert_not_called()

        @it
        def returns_a_list_of_unique_ints():
            output = Colors._get_rand_indicies(100000)
            mock_randint.assert_called()
            self.assertListEqual([*range(100000)], output)
