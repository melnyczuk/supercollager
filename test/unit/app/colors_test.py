from random import randint
from test.utils import describe, each, it
from unittest import TestCase, mock

from src.app.colors import Colors, color_lookup


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
    def test_generate(self, mock_randint):
        @it
        def generates_unique_colors():
            mock_randint.side_effect = [1, 2, 5]
            lookup_vals = list(color_lookup.values())
            expected = [lookup_vals[1], lookup_vals[2], lookup_vals[5]]
            output = list(Colors.generate(3))
            self.assertListEqual(expected, output)

        @each([len(color_lookup) + 1, len(color_lookup)])
        def does_not_generate_more_than_the_length_of_the_lookup(
            length,
        ):
            mock_randint.side_effect = [
                randint(0, len(color_lookup) - 1) for _ in range(10000)
            ]
            lookup_vals = list(color_lookup.values())
            output = list(Colors.generate(length))
            self.assertEqual(len(lookup_vals), len(output))

    n = 1001

    @mock.patch("random.randint", side_effect=[*range(n - 1), *range(n)])
    @describe
    def test__get_rand_indicies(self, mock_randint):
        @it
        def returns_the_input_list_if_it_is_long_enough():
            existing = 4, 2, 3
            output = list(Colors._get_rand_indicies(3, existing))
            self.assertListEqual([4, 2, 3], list(output))
            mock_randint.assert_not_called()

        @it
        def returns_a_list_of_unique_ints():
            output = list(Colors._get_rand_indicies(self.n))
            mock_randint.assert_called()
            self.assertListEqual(list(range(self.n)), output)
