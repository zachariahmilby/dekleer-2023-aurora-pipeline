import pytest
from khan.planner.base import _AngularSeparation, EclipsePrediction
from khan.planner.ephemeris import _get_ephemeris
import astropy.units as u
import numpy as np
from numpy.testing import assert_almost_equal
from pathlib import Path
import os


class TestAngularSeparation:

    @pytest.fixture
    def callisto_ephemeris(self):
        yield _get_ephemeris('2021-Jun-08 12:48', '2021-Jun-08 13:00',
                             target='Callisto')

    @pytest.fixture
    def target(self):
        yield 'Ganymede'
        
    @pytest.fixture
    def angles(self):
        yield np.array([372.7058794623298, 372.37117406275223,
                        372.10200468710946, 371.78681602715983,
                        371.4388539656776, 371.1829407139316,
                        370.86775389099597, 370.51979285176765,
                        370.26387804005765, 369.94869306478955,
                        369.6007330547752, 369.2987888988952,
                        369.02963355959645])

    def test_angular_separation_value_is_arcsec(
            self, callisto_ephemeris, target):
        separation = _AngularSeparation(callisto_ephemeris, target)
        assert separation.values.unit == u.arcsec

    def test_angular_separation_values_correct(
            self, callisto_ephemeris, target, angles):
        separation = _AngularSeparation(callisto_ephemeris, target)
        assert assert_almost_equal(separation.values.value, angles) is None


class TestEclipsePrediction:

    @pytest.fixture
    def start_time(self):
        yield '2021-06-01'

    @pytest.fixture
    def good_end_time(self):
        yield '2021-08-01'

    @pytest.fixture
    def bad_end_time(self):
        yield '2021-06-07'

    @pytest.fixture
    def target(self):
        yield 'Ganymede'

    def test_valid_eclipse_prediction(self, capfd, start_time, good_end_time,
                                      target):
        EclipsePrediction(start_time, good_end_time, target)
        out, err = capfd.readouterr()
        assert out == '\n' \
                      '3 Ganymede eclipse(s) identified between 2021-06-01 ' \
                      'and 2021-08-01.\n' \
                      '\n' \
                      'Starting Time (Keck/UTC) Ending Time (Keck/UTC) ' \
                      'Starting Time (California) Ending Time (California) ' \
                      'Duration Airmass Range  Relative Velocity\n' \
                      '2021-Jun-08 12:48        2021-Jun-08 15:17      ' \
                      '2021-Jun-08 05:48 PDT      2021-Jun-08 08:17 PDT    ' \
                      '2:29     1.173 to 1.581 -24.034 km/s     \n' \
                      '2021-Jul-14 09:40        2021-Jul-14 12:06      ' \
                      '2021-Jul-14 02:40 PDT      2021-Jul-14 05:06 PDT    ' \
                      '2:26     1.222 to 1.994 -16.160 km/s     \n' \
                      '2021-Jul-21 12:47        2021-Jul-21 15:29      ' \
                      '2021-Jul-21 05:47 PDT      2021-Jul-21 08:29 PDT    ' \
                      '2:42     1.181 to 1.670 -12.841 km/s     \n'

    def test_no_eclipses_raises_exception(self, capfd, start_time,
                                          bad_end_time, target):
        with pytest.raises(Exception):
            EclipsePrediction(start_time, bad_end_time, target)

    def test_if_graphics_creation_successtul(self, start_time, good_end_time,
                                             target):
        eclipse_prediction = EclipsePrediction(start_time, good_end_time,
                                               target)
        eclipse_prediction.save_summary_graphics()
        files = sorted(Path('.').glob('ganymede_2021*.pdf'))
        [os.remove(file) for file in files]  # cleanup output graphics
        assert len(files) == 3
