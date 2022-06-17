from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.constants import astropyconst20 as c

# define airmass unit
airmass_unit = u.def_unit('airmass')


def get_package_directory():
    """
    Return the absolute path to the package directory.
    """
    return Path(__file__).resolve().parent


def doppler_shift_wavelengths(
        wavelengths: u.Quantity, velocity: u.Quantity) -> u.Quantity:
    """
    Calculate doppler-shifted wavelengths.

    Parameters
    ----------
    wavelengths : astropy.units.Quantity
        Rest wavelengths with units.
    velocity : astropy.units.Quantity
        Doppler-shift velocity.

    Returns
    -------
    Doppler-shifted wavelengths.
    """
    return wavelengths * (1 - velocity.si / c.c)


def load_anc_file(filename: str) -> np.ndarray:
    """
    Function to load data tables from the ancillary directory.

    Parameters
    ----------
    filename : str
        Name of the ancillary file you want to load. Probably has the extension
        ``.dat``.

    Returns
    -------
    The file as a single numpy ndarray. You'll have to unpack it yourself.
    """
    return np.genfromtxt(
        Path(get_package_directory(), 'anc', filename).resolve(),
        skip_header=True)


def get_meridian_reflectivity(value: bool = False) -> dict:
    """
    This function retrieves reflectivity (also called I/F) for Jupiter's
    meridian from 320 to 1000 nm at 0.1 nm resolution. I stole these data
    from figures 1 and 6 in Woodman et al. (1979) "Spatially Resolved
    Reflectivities of Jupiter during the 1976 Opposition"
    (doi: 10.1016/0019-1035(79)90116-7).

    Parameters
    ----------
    value : bool
        If true, return the data without astropy units.

    Returns
    -------
    A dictionary with two keys: "wavelength" returns the wavelengths in nm and
    "reflectivity" returns the reflectivity.

    Examples
    --------
    Load Jupiter's meridian reflectivity with units.

    >>> I_over_F = get_meridian_reflectivity()
    >>> I_over_F['wavelength']
    <Quantity [320. , 320.1, 320.2, ..., 999.7, 999.8, 999.9] nm>
    >>> I_over_F['reflectivity']
    array([0.2845, 0.2842, 0.284 , ..., 0.2143, 0.2151, 0.2151])

    Load Jupiter's meridian reflectivity without units.

    >>> I_over_F = get_meridian_reflectivity(value=True)
    >>> I_over_F['wavelength']
    array([320. , 320.1, 320.2, ..., 999.7, 999.8, 999.9])
    >>> I_over_F['reflectivity']
    array([0.2845, 0.2842, 0.284 , ..., 0.2143, 0.2151, 0.2151])
    """
    data = load_anc_file('jupiter_meridian_reflectivity.dat')
    if value:
        return {'wavelength': data[:, 0], 'reflectivity': data[:, 1]}
    else:
        return {'wavelength': data[:, 0] * u.nm, 'reflectivity': data[:, 1]}


def get_mauna_kea_summit_extinction(value: bool = False) -> dict:
    """
    This function retrieves the wavelength-dependent airmass extinction in
    magnitudes/airmass for the summit of Mauna Kea from 320 to 1000 nm at
    0.2 nm resolution. I got this data from the download linked in Buton et al.
    (2013) "Atmospheric extinction properties above Mauna Kea from the Nearby
    SuperNova Factory spectro-photometric data set"
    (doi: 0.1051/0004-6361/201219834).

    Parameters
    ----------
    value : bool
        If true, return the data without astropy units.

    Returns
    -------
    A dictionary with two keys: "wavelength" returns the wavelengths in nm and
    "reflectivity" returns the extinction in magnitudes/airmass.

    Examples
    --------
    Load Mauna Kea summit extinction with units.

    >>> extinction = get_mauna_kea_summit_extinction()
    >>> extinction['wavelength']
    <Quantity [320. , 320.2, 320.4, ..., 999.4, 999.6, 999.8] nm>
    >>> extinction['extinction']
    <Quantity [0.8564351 , 0.8401497 , 0.8122926 , ..., 0.01448886, 0.0144809 ,
               0.01447293] mag / airmass>

    Load Mauna Kea summit extinction without units.

    >>> extinction = get_mauna_kea_summit_extinction(value=True)
    >>> extinction['wavelength']
    array([320. , 320.2, 320.4, ..., 999.4, 999.6, 999.8])
    >>> extinction['extinction']
    array([0.8564351 , 0.8401497 , 0.8122926 , ..., 0.01448886, 0.0144809 ,
           0.01447293])
    """
    data = load_anc_file('mauna_kea_airmass_extinction.dat')
    if value:
        return {'wavelength': data[:, 0],
                'extinction': data[:, 1]}
    else:
        return {'wavelength': data[:, 0] * u.nm,
                'extinction': data[:, 1] * u.mag / airmass_unit}


def get_solar_spectral_radiance(value: bool = False) -> dict:
    """
    This function retrieves the theoretical solar spectral radiance above
    Earth's atmosphere at 1 au from the Sun from 320 to 1000 nm at 0.5 nm
    resolution until 400 nm then 1 nm resolution until 1000 nm. The data are
    actually spectral irradiance (W/m²/nm) which I convert to radiance by
    dividing by pi, giving units of W/m²/nm/sr. The spectral irradiance data
    come from the 2000 ASTM Standard Extraterrestrial Spectrum Reference
    E-490-00 downloaded from
    https://www.nrel.gov/grid/solar-resource/spectra-astm-e490.html.

    Parameters
    ----------
    value : bool
        If true, return the data without astropy units.

    Returns
    -------
    A dictionary with two keys: "wavelength" returns the wavelengths in nm and
    "reflectivity" returns the radiance in W/m²/nm/sr.

    Examples
    --------
    Load Mauna Kea summit extinction with units. The doctest printout was
    insanely long so I'm just printing the zeroth items instead of the full
    arrays.

    >>> radiance = get_solar_spectral_radiance()
    >>> radiance['wavelength'][0]
    <Quantity 320. nm>
    >>> radiance['radiance'][0]
    <Quantity 0.24669016 W / (m2 nm sr)>

    Load Mauna Kea summit extinction without units.

    >>> radiance = get_solar_spectral_radiance(value=True)
    >>> radiance['wavelength'][0]
    320.0
    >>> radiance['radiance'][0]
    0.2466901617924378
    """
    data = load_anc_file('solar_spectral_irradiance.dat')
    if value:
        return {'wavelength': data[:, 0], 'radiance': data[:, 1] / np.pi}
    else:
        return {'wavelength': data[:, 0] * u.nm,
                'radiance': data[:, 1] / np.pi * u.watt / (
                        u.m ** 2 * u.nm * u.sr)}


def aurora_line_wavelengths() -> [u.Quantity]:
    """
    Retrieve a list of all of the aurora wavelengths. Each is in a sublist to
    keep triplets together.

    Returns
    -------
    A list of aurora line wavelengths as Astropy quantities.
    """
    return [[557.7330] * u.nm,
            [630.0304] * u.nm,
            [636.3776] * u.nm,
            [777.1944, 777.4166, 777.5388] * u.nm,
            [844.625, 844.636, 844.676] * u.nm,
            [656.2852] * u.nm]


def jovian_naif_codes() -> dict:
    """
    Navigation and Ancillary Information Facility (NAIF) unique ID codes for
    major Jovian system bodies.

    Returns
    -------
    A dictionary with the codes accessible by the name of the object: Jupiter,
    Io, Europa, Ganymede or Callisto.
    """
    return {
        'Jupiter': '599',
        'Io': '501',
        'Europa': '502',
        'Ganymede': '503',
        'Callisto': '504',
    }
