import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.constants import astropyconst20 as c
from astropy.io import fits
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from khan.analysis.data_retrieval import DataSubsection
from khan.common import get_meridian_reflectivity, get_solar_spectral_radiance


class FluxCalibration:
    """
    This class calculates the flux calibration from a set of Jupiter meridian
    observations.
    """
    def __init__(self, data_subsection: DataSubsection,
                 reduced_data_path: str | Path):
        """
        Parameters
        ----------
        data_subsection : DataSubsection
            The portion of the order containing the wavelength of interest.
        reduced_data_path : str or Path
            The location of the reduced data FITS files.
        """
        self._data_subsection = data_subsection
        self._flux_calibration_file = \
            Path(reduced_data_path, 'flux_calibration.fits.gz')
        self._slit_length_in_bins, self._slit_width_in_bins = \
            self._get_slit_dimensions_in_bins()
        self._slit_length_in_arcsec, self._slit_width_in_arcsec = \
            self._get_slit_dimensions_in_arcsec()
        self._n_bins_per_slit = (self._slit_length_in_bins
                                 * self._slit_width_in_bins).value
        self._theoretical_flux = self._calculate_jupiter_theoretical_flux()
        self._wavelength_dispersion = self._calculate_wavelength_dispersion()

    def _get_slit_dimensions_in_bins(self) -> (u.Quantity, u.Quantity):
        """
        Get the slit length and width in bins.
        """
        with fits.open(self._flux_calibration_file) as hdul:
            hdr = hdul[0].header
            slit_length = hdr['SLITLEN'] / hdr['SPASCALE']
            slit_width = hdr['SLITWID'] / hdr['SPESCALE']
        return slit_length * u.bin, slit_width * u.bin

    def _get_slit_dimensions_in_arcsec(self) -> (u.Quantity, u.Quantity):
        """
        Get the slit length and width in arcsec.
        """
        with fits.open(self._flux_calibration_file) as hdul:
            hdr = hdul[0].header
            slit_length = hdr['SLITLEN']
            slit_width = hdr['SLITWID']
        return slit_length * u.arcsec, slit_width * u.arcsec

    def _calculate_jupiter_theoretical_flux(self) -> u.Quantity:
        """
        Calculate the theoretical photon flux for a given Jupiter meridian
        observation in [R/nm/(electrons/sec)].
        """
        # load Jupiter meridian I/F and solar spectral radiance data
        reflectivity_data = get_meridian_reflectivity(value=True)
        spectral_radiance_data = get_solar_spectral_radiance()

        # load observation information from FITS file
        with fits.open(self._flux_calibration_file) as hdul:
            flux_images = \
                (hdul[0].data[:, self._data_subsection.order_index,
                 :, self._data_subsection.horizontal_slice_centers])
            datetimes = hdul['OBSERVATION_INFORMATION'].data['OBSDATE']

        # query Horizons to get Jupiter distance at time of observations
        epochs = [Time(datetime, format='isot', scale='utc').jd
                  for datetime in datetimes]
        eph = Horizons(id='599', location='568', epochs=epochs).ephemerides()
        jupiter_distance = eph['r']

        # calculate average distance and average flux image
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            average_flux_image = np.nanmean(flux_images, axis=0)
        average_distance = np.mean(jupiter_distance) * eph['r'].unit

        # calculate photon energy
        photon_energy = (c.h * c.c /
                         (self._data_subsection.shifted_wavelength_centers
                          * u.photon))
        photon_energy = photon_energy.to(u.J/u.photon)

        # interpolate solar spectrum over wavelengths
        solar_spectrum = \
            np.interp(self._data_subsection.shifted_wavelength_centers.value,
                      spectral_radiance_data['wavelength'].value,
                      spectral_radiance_data['radiance'].value)
        solar_spectrum *= spectral_radiance_data['radiance'].unit

        # calculate photon flux at Earth
        photon_flux = (solar_spectrum / photon_energy).to(u.R/u.nm)

        # interpolate Jupiter meridian reflectivity over wavelengths
        reflectivity = \
            np.interp(self._data_subsection.shifted_wavelength_centers.value,
                      reflectivity_data['wavelength'],
                      reflectivity_data['reflectivity'])

        # calculate theoretical flux
        theoretical_flux = reflectivity * photon_flux * (
                u.au / average_distance) ** 2

        median_observed_flux = \
            np.nanmedian(average_flux_image, axis=0) * u.electron/u.second
        total_flux_per_slit = median_observed_flux * self._n_bins_per_slit
        flux_calibration = \
            (theoretical_flux/total_flux_per_slit).to(
                u.R / u.nm / (u.electron / u.s))

        return (np.tile(flux_calibration.value,
                        (average_flux_image.shape[0], 1))
                * flux_calibration.unit)

    def _calculate_wavelength_dispersion(self) -> u.Quantity:
        """
        Calculate the wavelength dispersion (nm/bin).
        """
        dispersion = np.diff(
            self._data_subsection.shifted_wavelength_edges.value)
        return np.tile(dispersion, (self._theoretical_flux.shape[0], 1)) * u.nm

    @property
    def slit_length_in_bins(self) -> u.Quantity:
        return self._slit_length_in_bins

    @property
    def slit_width_in_bins(self) -> u.Quantity:
        return self._slit_width_in_bins

    @property
    def slit_length_in_arcsec(self) -> u.Quantity:
        return self._slit_length_in_arcsec

    @property
    def slit_width_in_arcsec(self) -> u.Quantity:
        return self._slit_width_in_arcsec

    @property
    def n_bins_per_slit(self) -> float:
        return self._n_bins_per_slit

    @property
    def calibration(self) -> u.Quantity:
        return self._theoretical_flux

    @property
    def wavelength_dispersion(self) -> u.Quantity:
        return self._wavelength_dispersion
