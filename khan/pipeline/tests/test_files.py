import astropy.units as u
import numpy as np

from khan.common import airmass_unit, get_meridian_reflectivity, \
    get_mauna_kea_summit_extinction, get_solar_spectral_radiance


class TestGetMeridianReflectivity:

    def test_if_first_wavelength_value_correct(self):
        assert get_meridian_reflectivity()['wavelength'][0] == 320 * u.nm

    def test_if_last_wavelength_value_correct(self):
        assert get_meridian_reflectivity()['wavelength'][-1] == 999.9 * u.nm

    def test_if_length_of_wavelength_correct(self):
        assert len(get_meridian_reflectivity()['wavelength']) == 6800

    def test_if_first_reflectivity_value_correct(self):
        assert get_meridian_reflectivity()['reflectivity'][0] == 0.2845

    def test_if_last_reflectivity_value_correct(self):
        assert get_meridian_reflectivity()['reflectivity'][-1] == 0.2151

    def test_if_length_of_reflectivity_correct(self):
        assert len(get_meridian_reflectivity()['reflectivity']) == 6800


class TestGetMaunaKeaSummitExtinction:

    def test_if_first_wavelength_value_correct(self):
        assert get_mauna_kea_summit_extinction()['wavelength'][0] == 320 * u.nm

    def test_if_last_wavelength_value_correct(self):
        assert get_mauna_kea_summit_extinction()['wavelength'][
                   -1] == 999.8 * u.nm

    def test_if_length_of_wavelength_correct(self):
        assert len(get_mauna_kea_summit_extinction()['wavelength']) == 3400

    def test_if_first_reflectivity_value_correct(self):
        assert get_mauna_kea_summit_extinction()['extinction'][
                   0] == 0.8564351 * u.mag / airmass_unit

    def test_if_last_reflectivity_value_correct(self):
        assert get_mauna_kea_summit_extinction()['extinction'][
                   -1] == 0.01447293 * u.mag / airmass_unit

    def test_if_length_of_reflectivity_correct(self):
        assert len(get_mauna_kea_summit_extinction()['extinction']) == 3400


class TestGetSolarSpectralRadiance:

    def test_if_first_wavelength_value_correct(self):
        assert get_solar_spectral_radiance()['wavelength'][0] == 320 * u.nm

    def test_if_last_wavelength_value_correct(self):
        assert get_solar_spectral_radiance()['wavelength'][-1] == 999.0 * u.nm

    def test_if_length_of_wavelength_correct(self):
        assert len(get_solar_spectral_radiance()['wavelength']) == 760

    def test_if_first_radiance_value_correct(self):
        assert get_solar_spectral_radiance()['radiance'][
                   0] == 0.77500 * 1 / np.pi * u.watt / (
                           u.m ** 2 * u.nm * u.sr)

    def test_if_last_radiance_value_correct(self):
        assert get_solar_spectral_radiance()['radiance'][
                   -1] == 0.74200 * 1 / np.pi * u.watt / (
                           u.m ** 2 * u.nm * u.sr)

    def test_if_length_of_radiance_correct(self):
        assert len(get_solar_spectral_radiance()['radiance']) == 760
