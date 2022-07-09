import datetime

import pytz


def _convert_string_to_datetime(time_string: str) -> datetime.datetime:
    """
    Convert an ephemeris table datetime string to a Python datetime object.
    """
    return datetime.datetime.strptime(time_string, '%Y-%b-%d %H:%M')


def _convert_datetime_to_string(datetime_object: datetime.datetime) -> str:
    """
    Convert a Python datetime object to a string with the format
    YYYY-MM-DD HH:MM.
    """
    return datetime.datetime.strftime(datetime_object,
                                      '%Y-%b-%d %H:%M %Z').strip()


def _convert_ephemeris_date_to_string(ephemeris_datetime: str) -> str:
    """
    Ensures an ephemeris datetime is in the proper format.
    """
    return _convert_datetime_to_string(
        _convert_string_to_datetime(ephemeris_datetime))


def _convert_to_california_time(utc_time_string: str) -> datetime.datetime:
    """
    Convert a UTC datetime string to local time at Caltech.
    """
    datetime_object = _convert_string_to_datetime(utc_time_string)
    timezone = pytz.timezone('America/Los_Angeles')
    datetime_object = pytz.utc.localize(datetime_object)
    return datetime_object.astimezone(timezone)


def _calculate_duration(starting_time: str, ending_time: str) -> str:
    """
    Determine duration between two datetime strings to minute precision.
    """
    duration = _convert_string_to_datetime(
        ending_time) - _convert_string_to_datetime(starting_time)
    minutes, seconds = divmod(duration.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}:{minutes:0>2}'
