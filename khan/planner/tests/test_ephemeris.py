import numpy as np
import pytest
from numpy.testing import assert_array_equal

from khan.planner.ephemeris import _get_ephemeris, _get_eclipse_indices


def test_get_ephemeris(capfd):
    eph = _get_ephemeris('2021-06-01', '2021-09-01', 'Ganymede', step='15m')
    print(eph)
    out, err = capfd.readouterr()
    assert out == '  targetname      datetime_str      datetime_jd    ' \
                  '...  PABLon   PABLat\n'\
                  '     ---              ---                d         ' \
                  '...   deg      deg  \n'\
                  '-------------- ----------------- ----------------- ' \
                  '... -------- -------\n'\
                  'Ganymede (503) 2021-Jun-01 12:30 2459367.020833333 ' \
                  '... 325.5914 -0.8547\n'\
                  'Ganymede (503) 2021-Jun-01 12:45     2459367.03125 ' \
                  '... 325.5929 -0.8547\n'\
                  'Ganymede (503) 2021-Jun-01 13:00 2459367.041666667 ' \
                  '... 325.5944 -0.8547\n'\
                  'Ganymede (503) 2021-Jun-01 13:15 2459367.052083333 ' \
                  '... 325.5959 -0.8547\n'\
                  'Ganymede (503) 2021-Jun-01 13:30      2459367.0625 ' \
                  '... 325.5974 -0.8547\n'\
                  'Ganymede (503) 2021-Jun-01 13:45 2459367.072916667 ' \
                  '...  325.599 -0.8547\n'\
                  'Ganymede (503) 2021-Jun-01 14:00 2459367.083333333 ' \
                  '... 325.6005 -0.8547\n'\
                  'Ganymede (503) 2021-Jun-01 14:15     2459367.09375 ' \
                  '...  325.602 -0.8547\n'\
                  'Ganymede (503) 2021-Jun-01 14:30 2459367.104166667 ' \
                  '... 325.6035 -0.8547\n'\
                  'Ganymede (503) 2021-Jun-01 14:45 2459367.114583333 ' \
                  '...  325.605 -0.8547\n'\
                  '           ...               ...               ... ' \
                  '...      ...     ...\n'\
                  'Ganymede (503) 2021-Aug-31 10:30      2459457.9375 ' \
                  '... 326.6078 -1.0821\n'\
                  'Ganymede (503) 2021-Aug-31 10:45 2459457.947916667 ' \
                  '... 326.6074 -1.0821\n'\
                  'Ganymede (503) 2021-Aug-31 11:00 2459457.958333333 ' \
                  '... 326.6069 -1.0821\n'\
                  'Ganymede (503) 2021-Aug-31 11:15     2459457.96875 ' \
                  '... 326.6064 -1.0821\n'\
                  'Ganymede (503) 2021-Aug-31 11:30 2459457.979166667 ' \
                  '...  326.606 -1.0821\n'\
                  'Ganymede (503) 2021-Aug-31 11:45 2459457.989583333 ' \
                  '... 326.6056 -1.0821\n'\
                  'Ganymede (503) 2021-Aug-31 12:00         2459458.0 ' \
                  '... 326.6052 -1.0821\n'\
                  'Ganymede (503) 2021-Aug-31 12:15 2459458.010416667 ' \
                  '... 326.6047 -1.0821\n'\
                  'Ganymede (503) 2021-Aug-31 12:30 2459458.020833333 ' \
                  '... 326.6043 -1.0821\n'\
                  'Ganymede (503) 2021-Aug-31 12:45     2459458.03125 ' \
                  '... 326.6039 -1.0821\n'\
                  'Length = 2525 rows\n'


class TestGetEclipseIndices:

    @pytest.fixture
    def start_time(self):
        yield '2021-06-01'

    @pytest.fixture
    def end_time(self):
        yield '2021-09-01'

    @pytest.fixture
    def target(self):
        yield 'Ganymede'
        
    @pytest.fixture
    def summer_indices(self):
        yield np.array([198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
                        1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198,
                        1199, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405,
                        1406, 1407, 2383, 2384, 2385])

    def test_summer_2021_ganymede_eclipses(self, start_time, end_time, target,
                                           summer_indices):
        eph = _get_ephemeris('2021-06-01', '2021-09-01', 'Ganymede',
                             step='15m')
        ind = _get_eclipse_indices(eph)
        assert assert_array_equal(ind, summer_indices) is None
