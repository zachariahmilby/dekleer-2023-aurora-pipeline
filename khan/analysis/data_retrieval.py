from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from khan.common import doppler_shift_wavelengths


class DataSubsection:
    """
    Retrieve a subsection of the data containing the set of wavelengths.
    """
    def __init__(self, wavelengths: list[u.Quantity],
                 reduced_data_path: str | Path):
        """
        Parameters
        ----------
        wavelengths : list[u.Quantity]
            The set of wavelengths you want to examine. Either a single
            wavelength or a doublet/triplet, but either way it should be a list
            even if just one value.
        reduced_data_path : str or Path
            The location of the reduced data FITS files.
        """
        self._wavelengths = wavelengths
        self._average_wavelength = np.mean([wavelength.value for wavelength
                                            in wavelengths])
        self._science_observations_file = Path(reduced_data_path,
                                               'science_observations.fits.gz')
        self._observation_datetimes = self._get_observation_datetimes()
        self._order_index = self._find_order_with_wavelengths()
        self._velocity, self._radius = self._get_target_velocity_and_radius()
        self._rest_wavelength_centers = self._get_wavelength_centers()
        self._shifted_wavelength_centers = \
            doppler_shift_wavelengths(self._rest_wavelength_centers,
                                      self._velocity)
        self._rest_wavelength_edges = self._get_wavelength_edges()
        self._shifted_wavelength_edges = \
            doppler_shift_wavelengths(self._rest_wavelength_edges,
                                      self._velocity)
        self._center_indices, self._edge_indices = \
            self._get_horizontal_index_limits()
        self._science_data = self._get_science_data()
        self._guide_satellite_data = self._get_guide_satellite_data()
        self._meshgrids = self._make_angular_meshgrids()
        self._center_meshgrids = self._make_center_meshgrids()
        self._spa_scale, self._spe_scale = self._get_bin_scales()

    def _find_order_with_wavelengths(self) -> int:
        """
        Get the index of the echelle order with the feature wavelengths.
        """
        found_order = None
        with fits.open(self._science_observations_file) as hdul:
            order_wavelengths = hdul['BIN_CENTER_WAVELENGTHS'].data
            for order in range(order_wavelengths.shape[0]):
                ind = np.abs(order_wavelengths[order]
                             - self._average_wavelength).argmin()
                if (ind > 0) & (ind < order_wavelengths[order].shape[0] - 1):
                    found_order = order
                    break
        if found_order is None:
            raise ValueError('Wavelengths not found!')
        return found_order

    def _get_wavelength_centers(self) -> u.Quantity:
        """
        Get the pixel center wavelengths for the found order.
        """
        with fits.open(self._science_observations_file) as hdul:
            return \
                hdul['BIN_CENTER_WAVELENGTHS'].data[self._order_index] * u.nm

    def _get_wavelength_edges(self) -> u.Quantity:
        """
        Get the pixel edge wavelengths for the found order.
        """
        with fits.open(self._science_observations_file) as hdul:
            return \
                hdul['BIN_EDGE_WAVELENGTHS'].data[self._order_index] * u.nm

    def _get_observation_datetimes(self) -> u.Quantity:
        """
        Get the dates and times of the observations.
        """
        with fits.open(self._science_observations_file) as hdul:
            return hdul['SCIENCE_OBSERVATION_INFORMATION'].data['OBSDATE']

    def _get_target_name(self) -> str:
        """
        Get the name of the target satellite.
        """
        with fits.open(self._science_observations_file) as hdul:
            return hdul[0].header['TARGET']

    def _get_target_velocity_and_radius(self) -> (u.Quantity, u.Quantity):
        """
        Get the target satellite relative velocity and angular radius from
        Horizons.
        """
        epochs = [Time(datetime, format='isot', scale='utc').jd
                  for datetime in self._observation_datetimes]
        eph = Horizons(id=self._get_target_name(), location='568',
                       epochs=epochs).ephemerides()
        target_velocity = np.mean(eph['delta_rate']) * eph['delta_rate'].unit
        target_radius = np.mean(eph['ang_width'] / 2) * eph['ang_width'].unit
        return target_velocity, target_radius

    def _get_horizontal_index_limits(self, pad: int = 30) -> (int, int):
        """
        Pad the horizontal index limits when selecting a data subsection to
        have some bins available for background subtraction. Default is
        30 bins on either side of the minimum and maximum index.
        """
        left_index = np.abs(self._shifted_wavelength_centers
                            - self._wavelengths[0]).argmin()
        if left_index - pad < 0:
            left_index = 0
        else:
            left_index = left_index - pad
        right_index = np.abs(self._shifted_wavelength_centers
                             - self._wavelengths[-1]).argmin()
        if right_index + pad == self._shifted_wavelength_centers.shape[0] - 1:
            right_index = self._shifted_wavelength_centers.shape[0] - 1
        else:
            right_index = right_index + pad
        return (slice(left_index, right_index+1, 1),
                slice(left_index, right_index+2, 1))

    def _get_science_data(self) -> np.ndarray:
        """
        Retrieve the target satellite image data.
        """
        with fits.open(self._science_observations_file) as hdul:
            return hdul['PRIMARY'].data[:, self._order_index, :]

    def _get_guide_satellite_data(self) -> np.ndarray:
        """
        Retrieve the guide satellite image data.
        """
        with fits.open(self._science_observations_file) as hdul:
            return hdul['GUIDE_SATELLITE'].data[:, self._order_index]

    def _make_angular_meshgrids(self) -> (np.ndarray, np.ndarray):
        """
        Make angular meshgrids for plotting images with proper physical
        dimensions.
        """
        _, n_spa, n_spe = self.science_data.shape
        with fits.open(self._science_observations_file) as hdul:
            spa_scale = hdul[0].header['SPASCALE']
            spe_scale = hdul[0].header['SPESCALE']
        spatial_edges = np.arange(0, n_spa + 1, 1) * spa_scale
        spectral_edges = np.arange(0, n_spe + 1, 1) * spe_scale
        return np.meshgrid(spectral_edges, spatial_edges)

    def _make_center_meshgrids(self) -> (np.ndarray, np.ndarray):
        """
        Make meshgrids of the pixel center values in relative angular space.
        """
        _, n_spa, n_spe = self.science_data.shape
        with fits.open(self._science_observations_file) as hdul:
            spa_scale = hdul[0].header['SPASCALE']
            spe_scale = hdul[0].header['SPESCALE']
        spatial_edges = np.arange(0.5, n_spa + 0.5, 1) * spa_scale
        spectral_edges = np.arange(0.5, n_spe + 0.5, 1) * spe_scale
        return np.meshgrid(spectral_edges, spatial_edges)

    def _get_bin_scales(self) -> (float, float):
        """
        Get the spatial and spectral bin scales in arcsec/bin.
        """
        with fits.open(self._science_observations_file) as hdul:
            return hdul[0].header['SPASCALE'], hdul[0].header['SPESCALE']

    @property
    def order_index(self) -> int:
        return self._order_index

    @property
    def observation_datetimes(self) -> np.ndarray:
        return self._observation_datetimes

    @property
    def horizontal_slice_centers(self) -> slice:
        return self._center_indices

    @property
    def feature_wavelengths(self) -> list[u.Quantity]:
        return self._wavelengths

    @property
    def average_wavelength(self) -> u.Quantity:
        return self._average_wavelength.squeeze() * u.nm

    @property
    def rest_wavelength_centers(self) -> u.Quantity:
        return self._rest_wavelength_centers[self._center_indices]

    @property
    def shifted_wavelength_centers(self) -> u.Quantity:
        return self._shifted_wavelength_centers[self._center_indices]

    @property
    def rest_wavelength_edges(self) -> u.Quantity:
        return self._rest_wavelength_centers[self._edge_indices]

    @property
    def shifted_wavelength_edges(self) -> u.Quantity:
        return self._shifted_wavelength_centers[self._edge_indices]

    @property
    def science_data(self) -> np.ndarray:
        return self._science_data[:, :, self._center_indices]

    @property
    def guide_satellite_data(self) -> np.ndarray:
        return self._guide_satellite_data[:, :, self._center_indices]

    @property
    def angular_meshgrids(self) -> (np.ndarray, np.ndarray):
        return self._meshgrids

    @property
    def center_meshgrids(self) -> (np.ndarray, np.ndarray):
        return self._center_meshgrids

    @property
    def target_velocity(self) -> u.Quantity:
        return self._velocity

    @property
    def target_angular_radius(self) -> u.Quantity:
        return self._radius

    @property
    def spatial_bin_scale(self) -> float:
        return self._spa_scale

    @property
    def spectral_bin_scale(self) -> float:
        return self._spe_scale
