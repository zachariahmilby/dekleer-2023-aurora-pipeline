import datetime

import pytest
import pytz

from khan.planner.time import _convert_string_to_datetime, \
    _convert_datetime_to_string, _convert_ephemeris_date_to_string, \
    _convert_to_california_time, _calculate_duration


class TestConvertStringToDatetime:

    @pytest.fixture
    def good_date_string(self):
        yield '2021-Jun-08 12:48'

    @pytest.fixture
    def bad_date_string_number_month(self):
        yield '2021-06-08 12:48'

    @pytest.fixture
    def bad_date_string_includes_seconds(self):
        yield '2021-06-08 12:48:00'

    @pytest.fixture
    def bad_date_string_includes_microseconds(self):
        yield '2021-06-08 12:48:00.00000'

    def test_good_date_string_returns_datetime_object(self, good_date_string):
        assert _convert_string_to_datetime(good_date_string) == \
               datetime.datetime(2021, 6, 8, 12, 48)

    def test_bad_date_string_raises_value_error(
            self, bad_date_string_number_month):
        with pytest.raises(ValueError):
            _convert_string_to_datetime(bad_date_string_number_month)

    def test_bad_date_string_includes_seconds_raises_value_error(
            self, bad_date_string_includes_seconds):
        with pytest.raises(ValueError):
            _convert_string_to_datetime(bad_date_string_includes_seconds)

    def test_bad_date_string_includes_microseconds_raises_value_error(
            self, bad_date_string_includes_microseconds):
        with pytest.raises(ValueError):
            _convert_string_to_datetime(bad_date_string_includes_microseconds)


class TestConvertDatetimeToString:

    @pytest.fixture
    def good_datetime_minutes(self):
        yield datetime.datetime(2021, 6, 8, 12, 48, tzinfo=pytz.utc)

    @pytest.fixture
    def good_datetime_without_timezone(self):
        yield datetime.datetime(2021, 6, 8, 12, 48)

    @pytest.fixture
    def bad_datetime_as_string(self):
        yield '2021-Jun-08 12:48'

    def test_good_datetime_minutes_returns_string(self, good_datetime_minutes):
        assert _convert_datetime_to_string(good_datetime_minutes) == \
               '2021-Jun-08 12:48 UTC'

    def test_bad_datetime_as_string_raises_type_error(
            self, bad_datetime_as_string):
        with pytest.raises(TypeError):
            _convert_datetime_to_string(bad_datetime_as_string)


class TestConvertEphemerisDateToString:

    @pytest.fixture
    def good_date_string(self):
        yield '2021-Jun-08 12:48'

    @pytest.fixture
    def bad_date_string_number_month(self):
        yield '2021-06-08 12:48'

    @pytest.fixture
    def bad_date_string_includes_seconds(self):
        yield '2021-06-08 12:48:00'

    @pytest.fixture
    def bad_date_string_includes_microseconds(self):
        yield '2021-06-08 12:48:00.00000'

    def test_good_date_string_returns_same(self, good_date_string):
        assert _convert_ephemeris_date_to_string(good_date_string) == \
               good_date_string

    def test_bad_date_string_raises_value_error(
            self, bad_date_string_number_month):
        with pytest.raises(ValueError):
            _convert_ephemeris_date_to_string(bad_date_string_number_month)

    def test_bad_date_string_includes_seconds_raises_value_error(
            self, bad_date_string_includes_seconds):
        with pytest.raises(ValueError):
            _convert_ephemeris_date_to_string(bad_date_string_includes_seconds)

    def test_bad_date_string_includes_microseconds_raises_value_error(
            self, bad_date_string_includes_microseconds):
        with pytest.raises(ValueError):
            _convert_ephemeris_date_to_string(
                bad_date_string_includes_microseconds)


class TestConvertToCaliforniaTime:

    @pytest.fixture
    def summer_utc(self):
        yield '2021-Jun-08 12:00'

    @pytest.fixture
    def winter_utc(self):
        yield '2021-Dec-08 12:00'

    @pytest.fixture
    def summer_pdt(self):
        yield '2021-Jun-08 05:00 PDT'

    @pytest.fixture
    def winter_pst(self):
        yield '2021-Dec-08 04:00 PST'

    def test_if_pdt_conversion_gives_correct_datetime(self, summer_utc,
                                                      summer_pdt):
        assert _convert_datetime_to_string(
            _convert_to_california_time(summer_utc)) == summer_pdt

    def test_if_pst_conversion_gives_correct_datetime(self, winter_utc,
                                                      winter_pst):
        assert _convert_datetime_to_string(
            _convert_to_california_time(winter_utc)) == winter_pst


class TestCalcualteDuration:

    @pytest.fixture
    def start_time(self):
        yield '2021-Jun-08 23:49'

    @pytest.fixture
    def end_time(self):
        yield '2021-Jun-09 02:03'

    def test_if_duration_correct(self, start_time, end_time):
        assert _calculate_duration(start_time, end_time) == '2:14'
