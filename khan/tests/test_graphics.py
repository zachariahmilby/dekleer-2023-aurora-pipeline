import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from khan.graphics import color_dict, _keck_one_alt_az_axis, \
    _format_axis_date_labels


class TestColorDictionary:
    
    @pytest.fixture
    def red(self):
        yield '#D62728'

    @pytest.fixture
    def orange(self):
        yield '#FF7F0E'

    @pytest.fixture
    def yellow(self):
        yield '#FDB813'

    @pytest.fixture
    def green(self):
        yield '#2CA02C'

    @pytest.fixture
    def blue(self):
        yield '#0079C1'

    @pytest.fixture
    def violet(self):
        yield '#9467BD'

    @pytest.fixture
    def cyan(self):
        yield '#17BECF'

    @pytest.fixture
    def magenta(self):
        yield '#D64ECF'

    @pytest.fixture
    def brown(self):
        yield '#8C564B'

    @pytest.fixture
    def darkgrey(self):
        yield '#3F3F3F'

    @pytest.fixture
    def grey(self):
        yield '#7F7F7F'

    @pytest.fixture
    def lightgrey(self):
        yield '#BFBFBF'

    def test_red_hex_value(self, red):
        assert color_dict['red'] == red

    def test_orange_hex_value(self, orange):
        assert color_dict['orange'] == orange

    def test_yellow_hex_value(self, yellow):
        assert color_dict['yellow'] == yellow

    def test_green_hex_value(self, green):
        assert color_dict['green'] == green

    def test_blue_hex_value(self, blue):
        assert color_dict['blue'] == blue

    def test_violet_hex_value(self, violet):
        assert color_dict['violet'] == violet

    def test_cyan_hex_value(self, cyan):
        assert color_dict['cyan'] == cyan

    def test_magenta_hex_value(self, magenta):
        assert color_dict['magenta'] == magenta

    def test_brown_hex_value(self, brown):
        assert color_dict['brown'] == brown

    def test_darkgrey_hex_value(self, darkgrey):
        assert color_dict['darkgrey'] == darkgrey

    def test_grey_hex_value(self, grey):
        assert color_dict['grey'] == grey

    def test_lightgrey_hex_value(self, lightgrey):
        assert color_dict['lightgrey'] == lightgrey


class TestKeckOneAltAxAxis:

    def test_if_return_type_is_axis(self):
        fig, axis = plt.subplots(subplot_kw={'projection': 'polar'})
        assert isinstance(_keck_one_alt_az_axis(axis), plt.Axes) is True

    def test_failure_with_non_polar_axis(self):
        with pytest.raises(AttributeError):
            fig, axis = plt.subplots()
            _keck_one_alt_az_axis(axis)

    def test_theta_direction_value_is_negative(self):
        fig, axis = plt.subplots(subplot_kw={'projection': 'polar'})
        axis = _keck_one_alt_az_axis(axis)
        assert axis.get_theta_direction() == -1

    def test_theta_zero_location_is_north(self):
        fig, axis = plt.subplots(subplot_kw={'projection': 'polar'})
        axis = _keck_one_alt_az_axis(axis)
        assert axis.get_theta_offset() == np.pi/2

    def test_rmin_is_0(self):
        fig, axis = plt.subplots(subplot_kw={'projection': 'polar'})
        axis = _keck_one_alt_az_axis(axis)
        assert axis.get_rmin() == 0

    def test_rmax_is_90(self):
        fig, axis = plt.subplots(subplot_kw={'projection': 'polar'})
        axis = _keck_one_alt_az_axis(axis)
        assert axis.get_rmax() == 90


class TestFormatAxisDateLabels:

    @pytest.fixture
    def xticks(self):
        yield np.array([18786.0, 18786.041666666668, 18786.083333333332,
                        18786.125])

    @pytest.fixture
    def utc_times(self):
        yield np.array(['00:00', '01:00', '02:00', '03:00'])

    @pytest.fixture
    def california_times(self):
        yield np.array(['17:00', '18:00', '19:00', '20:00'])

    def test_utc_axis_positions(self, xticks):
        fig, axis = plt.subplots()
        time = pd.date_range(start='2021-06-08', periods=180, freq='min')
        axis.plot(time, np.ones_like(time))
        _format_axis_date_labels(axis)
        assert assert_array_equal(axis.get_xticks(), xticks) is None

    def test_utc_axis_labels(self, utc_times):
        fig, axis = plt.subplots()
        time = pd.date_range(start='2021-06-08', periods=180, freq='min')
        axis.plot(time, np.ones_like(time))
        _format_axis_date_labels(axis)
        fig.canvas.draw()
        labels = np.array([label.get_text()
                           for label in axis.get_xticklabels()])
        assert assert_array_equal(labels, utc_times) is None

    def test_california_axis_positions(self, xticks):
        fig, axis = plt.subplots()
        time = pd.date_range(start='2021-06-08', periods=180, freq='min')
        axis.plot(time, np.ones_like(time))
        axis = _format_axis_date_labels(axis)
        assert assert_array_equal(axis.get_xticks(), xticks) is None

    def test_california_axis_labels(self, california_times):
        fig, axis = plt.subplots()
        time = pd.date_range(start='2021-06-08', periods=180, freq='min')
        axis.plot(time, np.ones_like(time))
        axis = _format_axis_date_labels(axis)
        fig.canvas.draw()
        labels = np.array([label.get_text()
                           for label in axis.get_xticklabels()])
        assert assert_array_equal(labels, california_times) is None
    