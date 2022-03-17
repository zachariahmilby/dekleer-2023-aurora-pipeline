import pytest
from khan.pipeline.rectification import round_up_to_odd


class TestRoundUpToOdd:

    def test_if_even_integer_rounds_up_correctly(self):
        assert round_up_to_odd(12) == 13

    def test_if_just_below_even_integer_rounds_up_correctly(self):
        assert round_up_to_odd(11.99) == 13

    def test_if_just_above_even_integer_rounds_down_correctly(self):
        assert round_up_to_odd(12.01) == 13

    def test_if_odd_integer_returns_same(self):
        assert round_up_to_odd(11) == 11

    def test_if_just_below_odd_integer_returns_original_odd_integer(self):
        assert round_up_to_odd(10.99) == 11

    def test_if_just_above_odd_integer_returns_original_odd_integer(self):
        assert round_up_to_odd(11.01) == 11

    def test_if_zero_returns_one(self):
        assert round_up_to_odd(0) == 1
