import numpy as np
from astroquery.jplhorizons import Horizons


def _get_ephemeris(starting_datetime: str, ending_datetime: str, target: str,
                   step: str = '1m', airmass_lessthan: int | float = 2,
                   skip_daylight: bool = False) -> dict:
    """
    Query the JPL Horizons System and get a dictionary of ephemeris
    information.
    """
    epochs = {'start': starting_datetime, 'stop': ending_datetime,
              'step': step}
    obj = Horizons(id=target, location='568', epochs=epochs)
    return obj.ephemerides(airmass_lessthan=airmass_lessthan,
                           skip_daylight=skip_daylight)


def _get_eclipse_indices(ephemeris: dict) -> np.ndarray:
    """
    Search through an ephemeris table and find when a satellite is eclipsed by
    Jupiter and it's either night, astronomical or nautical twilight on Mauna
    Kea.
    """
    return np.where((ephemeris['sat_vis'] == 'u') &
                    (ephemeris['solar_presence'] != 'C') &
                    (ephemeris['solar_presence'] != '*'))[0]
