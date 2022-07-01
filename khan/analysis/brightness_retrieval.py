from pathlib import Path

import astropy.units as u
import lmfit.model
import numpy as np
import statsmodels.api as sm
from astropy.constants import astropyconst20 as c
from astropy.io import fits
from lmfit.models import ConstantModel, GaussianModel
from lmfit import Parameters

from astroquery.jplhorizons import Horizons
from astropy.time import Time
from astropy.coordinates import SkyCoord

from khan.common import doppler_shift_wavelengths, \
    get_solar_spectral_radiance, get_meridian_reflectivity, jovian_naif_codes


class OrderData:
    """
    This class gets order images and other relevant information for a given
    wavelength or set of wavelengths. It selects a sub-section from ± 1 nm from
    the minimum and maximum wavelength provided.
    """
    def __init__(self, reduced_data_path: str | Path,
                 wavelengths: u.Quantity, emission_line_strengths: list[float],
                 seeing: u.Quantity = 1 * u.arcsec,
                 exclude: dict = None, bottom_trim: int = 2,
                 top_trim: int = 2):
        """
        Parameters
        ----------
        reduced_data_path : str or Path
            Directory with pipeline-output FITS file "flux_calibration.fits.gz"
            and "science_observations.fits.gz".
        wavelengths : astropy.units.quantity.Quantity
            Aurora wavelength(s) as an astropy Quantity.
        emission_line_strengths : list of floats
            Relative strengths of emission lines for fitting.
        seeing : u.Quantity
            How much you want to add to the target radius in arcseconds to
            account for increase in the apparent target size due to atmospheric
            or other conditions. Default is 1 arcsecond.
        exclude: dict
            Frames to eliminate from the processing of a particular average
            wavelength. The key to the dictionary has to match the average
            wavelength to 1 decimal place, e.g., '777.4 nm'. The value should
            be a list of frame numbers to exclude, starting from zero.
        top_trim: int
            How many rows to remove from the top edge of the order to eliminate
            artifacts from rectification. Default is 2.
        bottom_trim: int
            How many rows to remove from the bottom edge of the order to
            eliminate artifacts from rectification. Default is 2.
        """
        self._science_observations_path = Path(
            reduced_data_path, 'science_observations.fits.gz')
        self._flux_calibration_path = Path(
            reduced_data_path, 'flux_calibration.fits.gz')
        self._wavelengths = wavelengths
        self._emission_line_strengths = emission_line_strengths
        self._seeing = seeing
        self._exclude = exclude
        self._top = top_trim
        self._bottom = bottom_trim
        self._order_index = self._find_order_with_wavelengths()
        self._data = self._retrieve_data_from_fits()

    def _find_order_with_wavelengths(self) -> int:
        """
        Get the index of the echelle order with the feature wavelengths.
        """
        average_wavelength = self._wavelengths.mean()
        found_order = None
        with fits.open(self._science_observations_path) as hdul:
            order_wavelengths = hdul['BIN_CENTER_WAVELENGTHS'].data * u.nm
            for order in range(order_wavelengths.shape[0]):
                ind = np.abs(order_wavelengths[order]
                             - average_wavelength).argmin()
                if (ind > 0) & (ind < order_wavelengths[order].shape[0] - 1):
                    found_order = order
                    break
        if found_order is None:
            raise ValueError('Wavelengths not found!')
        return found_order

    def _find_subsection_horizontal_bounds(self) -> (int, int):
        """
        Find and return the indices corresponding to ±1 nm from the minimum
        and maximum aurora wavelength(s).
        """
        with fits.open(self._science_observations_path) as hdul:
            wavelengths = \
                hdul['BIN_CENTER_WAVELENGTHS'].data[self._order_index] \
                * u.nm
            minimum_wavelength = np.min(self._wavelengths) - 1 * u.nm
            maximum_wavelength = np.max(self._wavelengths) + 1 * u.nm
        left_index = np.abs(wavelengths - minimum_wavelength).argmin()
        right_index = np.abs(wavelengths - maximum_wavelength).argmin()
        return left_index, right_index

    def _retrieve_data_from_fits(self):
        """
        Get all the needed data from the FITS files and store as a dictionary.
        """
        left, right = self._find_subsection_horizontal_bounds()
        with fits.open(self._science_observations_path) as hdul:
            key = f'{self.aurora_wavelengths.mean():.1f}'
            try:
                included_in_average = np.ones(hdul[0].shape[0], dtype=bool)
                included_in_average[self._exclude[key]] = False
                include_indices = np.array([index for index
                                            in np.arange(hdul[0].shape[0])
                                            if index not in self._exclude[key]]
                                           )
            except (KeyError, TypeError):
                include_indices = np.arange(hdul[0].shape[0])
                included_in_average = np.ones_like(include_indices, dtype=bool)
            target_relative_velocity = \
                np.mean(
                    hdul['SCIENCE_OBSERVATION_INFORMATION'].data['RELVLCTY']
                ) * u.m / u.s
            data = {
                'target_images':
                    hdul['PRIMARY'].data[:, self._order_index,
                                         self._bottom:-self._top, left:right]
                    * u.electron / u.s,
                'trace_images':
                    hdul['GUIDE_SATELLITE'].data[:, self._order_index,
                                                 self._bottom:-self._top,
                                                 left:right]
                    * u.electron / u.s,
                'rest_wavelength_centers':
                    hdul['BIN_CENTER_WAVELENGTHS'].data[self._order_index,
                                                        left:right] * u.nm,
                'rest_wavelength_edges':
                    hdul['BIN_EDGE_WAVELENGTHS'].data[self._order_index,
                                                      left:right + 1] * u.nm,
                'shifted_wavelength_centers': doppler_shift_wavelengths(
                    hdul['BIN_CENTER_WAVELENGTHS'].data[self._order_index,
                                                        left:right] * u.nm,
                    target_relative_velocity),
                'shifted_wavelength_edges': doppler_shift_wavelengths(
                    hdul['BIN_EDGE_WAVELENGTHS'].data[self._order_index,
                                                      left:right+1] * u.nm,
                    target_relative_velocity),
                'echelle_order':
                    hdul['ECHELLE_ORDERS'].data[self._order_index],
                'filenames':
                    hdul['SCIENCE_OBSERVATION_INFORMATION'].data['FILENAME'],
                'observation_dates':
                    hdul['SCIENCE_OBSERVATION_INFORMATION'].data['OBSDATE'],
                'target_radii':
                    hdul['SCIENCE_OBSERVATION_INFORMATION'].data['A_RADIUS']
                    * u.arcsec,
                'slit_length': hdul['PRIMARY'].header['SLITLEN'] * u.arcsec,
                'slit_width': hdul['PRIMARY'].header['SLITWID'] * u.arcsec,
                'spatial_bin_scale':
                    hdul['PRIMARY'].header['SPASCALE'] * u.arcsec / u.bin,
                'spectral_bin_scale':
                    hdul['PRIMARY'].header['SPESCALE'] * u.arcsec / u.bin,
                'target_name': hdul['PRIMARY'].header['TARGET'],
                'included_in_average': included_in_average,
                'include_indices': include_indices,
            }
        with fits.open(self._flux_calibration_path) as hdul:
            data['flux_calibration_images'] = \
                hdul['PRIMARY'].data[:, self._order_index,
                                     self._bottom:-self._top, left:right] \
                * u.electron / u.s
            data['jupiter_distance'] = \
                (np.mean(hdul['OBSERVATION_INFORMATION'].data['DISTANCE'])
                 * u.au)
        return data

    @property
    def target_images(self) -> u.Quantity:
        return self._data['target_images']

    @property
    def average_target_image(self) -> u.Quantity:
        include_indices = self._data['include_indices']
        return (np.nanmean(self._data['target_images'][include_indices].value,
                           axis=0) * self._data['target_images'].unit)

    @property
    def trace_images(self) -> u.Quantity:
        return self._data['trace_images']

    @property
    def flux_calibration_images(self) -> u.Quantity:
        return self._data['flux_calibration_images']

    @property
    def aurora_wavelengths(self) -> u.Quantity:
        return self._wavelengths

    @property
    def line_strengths(self) -> list[float]:
        return self._emission_line_strengths

    @property
    def rest_wavelength_centers(self) -> u.Quantity:
        return self._data['rest_wavelength_centers']

    @property
    def rest_wavelength_edges(self) -> u.Quantity:
        return self._data['rest_wavelength_edges']

    @property
    def shifted_wavelength_centers(self) -> u.Quantity:
        return self._data['shifted_wavelength_centers']

    @property
    def shifted_wavelength_edges(self) -> u.Quantity:
        return self._data['shifted_wavelength_edges']

    @property
    def echelle_order(self) -> int:
        return self._data['echelle_order']

    @property
    def filenames(self) -> np.chararray:
        return self._data['filenames']

    @property
    def observation_dates(self) -> np.chararray:
        return self._data['observation_dates']

    @property
    def target_radius(self) -> u.Quantity:
        return self._data['target_radii'].mean()

    @property
    def jupiter_distance(self) -> u.Quantity:
        return self._data['jupiter_distance']

    @property
    def seeing(self) -> u.Quantity:
        return self._seeing

    @property
    def slit_length(self) -> u.Quantity:
        return self._data['slit_length']

    @property
    def slit_width(self) -> u.Quantity:
        return self._data['slit_width']

    @property
    def spatial_bin_scale(self) -> u.Quantity:
        return self._data['spatial_bin_scale']

    @property
    def spectral_bin_scale(self) -> u.Quantity:
        return self._data['spectral_bin_scale']

    @property
    def target_name(self) -> str:
        return self._data['target_name']

    @property
    def included_in_average(self) -> list[bool]:
        return self._data['included_in_average']

    @property
    def include_indices(self) -> np.ndarray:
        return self._data['include_indices']


class Background:
    """
    This class generates a fitted background and provides background-subtracted
    data images.
    """
    def __init__(self, order_data: OrderData, y_offset: int = 0):
        self._order_data = order_data
        self._y_offset = y_offset
        self._target_mask, self._background_mask = self._make_masks()
        self._backgrounds, self._average_background = \
            self._construct_background()
        self._background_electron_flux = \
            self._calculate_background_electron_flux()

    def _make_angular_meshgrids(self, wavelength: u.Quantity) \
            -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Make sky-projected meshgrids in arcseconds for image display.
        pcolormesh can't seem to handle astropy quantities, so these can't
        have units, unfortunately...

        The first set of two are the horizontal and vertical edges, the second
        set of two are the horizontal and vertical centers.
        """
        n_obs, ny, nx = self._order_data.target_images.shape
        wavelength_ind = \
            np.abs(self._order_data.shifted_wavelength_centers
                   - wavelength).argmin()
        horizontal_bins_arcsec = ((np.linspace(0, nx, nx + 1) - wavelength_ind)
                                  * self._order_data.spectral_bin_scale.value)
        vertical_bins_arcsec = (
                (np.linspace(-ny / 2, ny / 2, ny + 1))
                * self._order_data.spatial_bin_scale.value)
        x, y = np.meshgrid(horizontal_bins_arcsec, vertical_bins_arcsec)
        xc = x[:-1, :-1] + self._order_data.spectral_bin_scale.value / 2
        yc = y[:-1, :-1] + self._order_data.spatial_bin_scale.value / 2
        return x, y, xc, yc

    def _make_masks(self) -> (np.ndarray, np.ndarray):
        """
        Make masks for the target and background bins. If there are multiple
        wavelengths in the set, the mask is a composite mask of all of them.
        """
        apparent_radius = (self._order_data.target_radius
                           + self._order_data.seeing)
        target_masks = []
        for wavelength in self._order_data.aurora_wavelengths:
            xe, ye, xc, yc = self._make_angular_meshgrids(wavelength)
            mask = np.sqrt(
                xc**2 + (yc - self._y_offset
                         * self._order_data.spatial_bin_scale.value)**2)
            mask *= u.arcsec
            target_bins = np.where(mask <= apparent_radius)
            background_bins = np.where(mask > apparent_radius)
            target_mask = np.ones(mask.shape, dtype=float)
            target_mask[target_bins] = np.nan
            target_mask[background_bins] = 1
            target_masks.append(target_mask)
        target_mask = np.mean(target_masks, axis=0)
        background_mask = np.ones(target_mask.shape, dtype=float)
        target_bins = np.where(np.isnan(target_mask))
        background_bins = np.where(~np.isnan(target_mask))
        background_mask[target_bins] = 1
        background_mask[background_bins] = np.nan
        return target_mask, background_mask

    def _make_normalized_background_profile(self) -> (np.ndarray, np.ndarray):
        """
        Construct a "typical" slit-varying background by averaging across each
        science image with the target masked. Also do the same for the average
        image.
        """
        n_obs, _, _ = self._order_data.target_images.shape
        profiles = []
        ind = np.unique(np.where(~np.isnan(self._target_mask))[1])
        for obs in range(n_obs):
            masked_image = (self._order_data.target_images[obs]
                            * self._target_mask)
            characteristic_profile = np.nanmedian(masked_image[:, ind], axis=1)
            profiles.append(characteristic_profile
                            / np.nanmax(characteristic_profile))
        average_masked_image = \
            self._order_data.average_target_image * self._target_mask
        characteristic_profile = np.nanmedian(average_masked_image[:, ind],
                                              axis=1)
        characteristic_profile /= np.nanmax(characteristic_profile)
        return np.array(profiles), characteristic_profile

    def _construct_background(self) -> (u.Quantity, u.Quantity):
        """
        Calculate the background by fitting the profile to each column along
        with a constant and linear component. Also do the same for the average
        image.
        """
        background_profiles, average_background_profile = \
            self._make_normalized_background_profile()
        backgrounds = []
        n_obs, ny, nx = self._order_data.target_images.shape
        for obs in range(n_obs):
            masked_image = \
                self._order_data.target_images[obs] * self._target_mask
            background = np.zeros(masked_image.shape)
            profile = background_profiles[obs]
            constant = np.ones(profile.shape[0])
            fit_profile = np.array([constant, profile]).T
            for col in range(nx):
                result = sm.OLS(masked_image[:, col].value,
                                fit_profile, missing='drop').fit()
                best_fit_constant = result.params[0]
                best_fit_profile = result.params[1] * profile
                background[:, col] = best_fit_constant + best_fit_profile
            backgrounds.append(background)
        average_masked_image = \
            self._order_data.average_target_image * self._target_mask
        average_background = np.zeros(average_masked_image.shape)
        constant = np.ones(average_background_profile.shape[0])
        fit_profile = np.array([constant, average_background_profile]).T
        for col in range(nx):
            result = sm.OLS(average_masked_image[:, col].value,
                            fit_profile, missing='drop').fit()
            best_fit_constant = result.params[0]
            best_fit_profile = result.params[1] * average_background_profile
            average_background[:, col] = best_fit_constant + best_fit_profile

        return (np.array(backgrounds) * u.electron / u.s,
                average_background * u.electron / u.s)

    def _calculate_background_electron_flux(self) -> u.Quantity:
        """
        Calculate the average electron flux above and below the target mask
        bins in electrons/s/arcsec². Useful for comparing relative backgrounds
        between nights.
        """
        cols = np.unique(np.where(np.isnan(self._target_mask))[1])
        n_obs = self.backgrounds.shape[0]
        average_flux = [np.nanmean(self.backgrounds[obs, :, cols].value)
                        for obs in range(n_obs)] * u.electron / u.s
        average_flux /= (self._order_data.spatial_bin_scale * u.bin
                         * self._order_data.spectral_bin_scale * u.bin)
        return average_flux

    @property
    def backgrounds(self) -> u.Quantity:
        return self._backgrounds

    @property
    def average_background(self) -> u.Quantity:
        return self._average_background

    @property
    def background_electron_flux(self) -> u.Quantity:
        return self._background_electron_flux

    @property
    def target_mask(self) -> np.ndarray:
        return self._target_mask

    @property
    def background_mask(self) -> np.ndarray:
        return self._background_mask


class AuroraBrightness:

    def __init__(self, order_data: OrderData, background: Background,
                 save_path: str | Path):
        self._order_data = order_data
        self._background = background
        self._save_path = save_path
        self._jupiter_brightness = self._get_jupiter_brightness()
        self._jupiter_electron_flux = self._get_jupiter_electron_flux()
        self._calibration_factor = (self._jupiter_brightness
                                    / self._jupiter_electron_flux)
        self._calibrated_images, self._average_calibrated_image = \
            self._calibrate_images()
        self._line_spectra, self._average_line_spectrum, self._fit_results, \
            self._average_fit_result = self._make_line_spectra()

    def _get_jupiter_brightness(self) -> u.Quantity:
        """
        Calculate Jupiter's photon flux at the average wavelength.
        """
        radiance = get_solar_spectral_radiance()
        eye_over_f = get_meridian_reflectivity()
        reflectivity = np.interp(radiance['wavelength'].value,
                                 eye_over_f['wavelength'].value,
                                 eye_over_f['reflectivity'])
        photon_energy = c.h * c.c / (radiance['wavelength'] * u.photon)
        photon_flux = (radiance['radiance'] / photon_energy).to(u.R / u.nm)
        photon_flux *= reflectivity
        photon_flux *= (u.au / self._order_data.jupiter_distance) ** 2
        ind = np.abs(radiance['wavelength']
                     - self._order_data.aurora_wavelengths.mean()).argmin()
        return photon_flux[ind]

    def _get_jupiter_electron_flux(self):
        """
        Get the average electron flux from Jupiter in the column corresponding
        to the wavelength, multiplied by the total number of bins in the slit
        to get the total electron flux from Jupiter at that wavelength.
        """
        flux_calibration_image = \
            np.mean(self._order_data.flux_calibration_images, axis=0)
        ny, nx = flux_calibration_image.shape
        middle_index = int(nx/2)
        slit_half_width_bins = np.round(
            self._order_data.slit_width
            / self._order_data.spectral_bin_scale / 2).astype(int)
        slit_length_bins = np.round(
            self._order_data.slit_length
            / self._order_data.spatial_bin_scale).astype(int)
        n_bins_slit = int(2 * slit_half_width_bins.value
                          * slit_length_bins.value) * u.bin
        slit_angular_area = (self._order_data.slit_length
                             * self._order_data.slit_width)
        left = middle_index-slit_half_width_bins.value
        right = middle_index+slit_length_bins.value
        average_counts = np.mean(flux_calibration_image[:, left:right])
        total_flux = (average_counts * n_bins_slit.value
                      / slit_angular_area.to(u.sr))
        return total_flux

    def _get_dwavelength(self):
        """
        Calculate wavelength dispersion-per-bin.
        """
        ind = np.abs(self._order_data.shifted_wavelength_centers
                     - self._order_data.aurora_wavelengths.mean()).argmin()
        return np.diff(self._order_data.shifted_wavelength_edges)[ind] / u.bin

    def _convert_target_to_flux_per_disk(self, target_flux):
        """
        Scale target flux to the angular size of the target disk.
        """
        return target_flux / (np.pi * self._order_data.target_radius**2)

    def _save_background_subtracted_images(self, image: np.ndarray,
                                           filename: str):
        """
        Save a background-subtracted 2D spectrum image to a text file.
        """
        average_wavelength = self._order_data.aurora_wavelengths.mean()
        savepath = Path(self._save_path, f'{average_wavelength:.1f}',
                        'spectra_2D', filename)
        if not savepath.parent.exists():
            savepath.parent.mkdir(parents=True)
        # noinspection PyTypeChecker
        np.savetxt(savepath, image)

    def _calibrate_images(self) -> (u.Quantity, u.Quantity):
        """
        Calibrate a background-subtracted image to R (per target disk). Also do
        the same for the average image.
        """
        n_obs, ny, nx = self._order_data.target_images.shape
        calibrated_images = []
        for obs in range(n_obs):
            background_subtracted_image = (self._order_data.target_images[obs]
                                           - self._background.backgrounds[obs])
            converted_target_flux = self._convert_target_to_flux_per_disk(
                    background_subtracted_image)
            calibrated_image = (converted_target_flux
                                * self._calibration_factor
                                * self._order_data.slit_width
                                / self._order_data.spectral_bin_scale
                                * self._get_dwavelength()).to(u.R)
            self._save_background_subtracted_images(
                calibrated_image.value,
                self._order_data.filenames[obs].replace('.fits.gz', '.txt'))
            calibrated_images.append(calibrated_image.value)
        background_subtracted_average_image = \
            (self._order_data.average_target_image
             - self._background.average_background)
        converted_average_target_flux = self._convert_target_to_flux_per_disk(
            background_subtracted_average_image)
        calibrated_average_image = (converted_average_target_flux
                                    * self._calibration_factor
                                    * self._order_data.slit_width
                                    / self._order_data.spectral_bin_scale
                                    * self._get_dwavelength()).to(u.R)
        self._save_background_subtracted_images(calibrated_average_image.value,
                                                'average.txt')
        return np.array(calibrated_images) * u.R, calibrated_average_image

    def _fit_gaussian(self, spectrum):
        """
        Make a (composite) Gaussian equal to the number of lines in the set and
        fit it. I've added a constant to account for any residual left over
        after background subtraction. I've also fixed the width of each of the
        Gaussians to radius/sqrt(2*ln(2)), under the assumption that the target
        radius is the HWHM of the Gaussian.

        Update June 22, 2022: I've added specific line ratios for 777.4 nm and
        844.6 nm.
        """
        center_indices = [np.abs(self._order_data.shifted_wavelength_centers
                                 - wavelength).argmin()
                          for wavelength in
                          self._order_data.aurora_wavelengths]
        spectrum = spectrum.value
        x = np.arange(len(spectrum))
        n_lines = len(center_indices)
        prefixes = [f'gaussian{i + 1}_' for i in range(n_lines)]
        model = ConstantModel(prefix='constant_')
        model += np.sum([GaussianModel(prefix=prefix) for prefix in prefixes],
                        dtype=object)
        params = Parameters()
        params.add('constant_c', value=0, min=-np.inf, max=np.inf)
        for i, prefix in enumerate(prefixes):
            if i != 0:
                params.add(f'{prefix}amplitude', value=np.nanmax(spectrum),
                           min=-np.inf, max=np.inf, vary=False,
                           expr=f'gaussian1_amplitude '
                                f'* {self._order_data.line_strengths[i]}')
            else:
                params.add(f'{prefix}amplitude', value=np.nanmax(spectrum),
                           min=-np.inf, max=np.inf)
            if i == 0:
                params.add(f'{prefix}center', value=center_indices[i],
                           min=center_indices[i]-10,
                           max=center_indices[i]+10)
            else:
                dx = center_indices[i] - center_indices[0]
                params.add(f'{prefix}center', vary=False,
                           expr=f'gaussian1_center + {dx}')
            params.add(f'{prefix}sigma',
                       value=((self._order_data.target_radius
                              / self._order_data.spectral_bin_scale).value
                              / np.sqrt(2*np.log(2))),
                       vary=False)
        return model.fit(spectrum, params=params, x=x)

    def _save_line_spectra(self, unshifted_wavelengths: u.Quantity,
                           shifted_wavelengths: u.Quantity,
                           spectrum: u.Quantity,
                           fit_result: lmfit.model.ModelResult, filename: str):
        """
        Save line spectra to a text file. Also save the result of the fit.
        """
        average_wavelength = self._order_data.aurora_wavelengths.mean()
        savepath = Path(self._save_path, f'{average_wavelength:.1f}',
                        'spectra_1D', filename)
        if not savepath.parent.exists():
            savepath.parent.mkdir(parents=True)
        with open(savepath, 'w') as file:
            file.write('rest_wavelengths_[nm] doppler_shifted_wavelengths[nm] '
                       'observed_spectrum_[R/nm] fitted_spectrum_[R/nm] '
                       'fitted_uncertainty_[R/nm]\n')
            best_fit = fit_result.best_fit
            fit_uncertainty = fit_result.eval_uncertainty()
            for i in range(len(spectrum)):
                file.write(f'{unshifted_wavelengths[i].value} '
                           f'{shifted_wavelengths[i].value} '
                           f'{spectrum[i].value} '
                           f'{best_fit[i]} '
                           f'{fit_uncertainty[i]}\n')

        fit_results_savepath = Path(self._save_path,
                                    f'{average_wavelength:.1f}',
                                    'spectra_1D',
                                    filename.replace('.txt',
                                                     '_fit_report.txt'))
        if not fit_results_savepath.parent.exists():
            fit_results_savepath.parent.mkdir(parents=True)
        with open(fit_results_savepath, 'w') as file:
            file.write(fit_result.fit_report())

        fit_params_savepath = Path(self._save_path,
                                   f'{average_wavelength:.1f}',
                                   'spectra_1D',
                                   filename.replace('.txt', '_fit_params.txt'))
        if not fit_params_savepath.parent.exists():
            fit_params_savepath.parent.mkdir(parents=True)
        with open(fit_params_savepath, 'w') as file:
            file.write(f'parameter initial best_fit_value uncertainty\n')
            for param in fit_result.params.keys():
                file.write(f'{param} '
                           f'{fit_result.init_params[param].value} '
                           f'{fit_result.params[param].value} '
                           f'{fit_result.params[param].stderr}\n')

    def _make_line_spectra(self) -> (list[np.ndarray], np.ndarray,
                                     list[lmfit.model.ModelResult],
                                     lmfit.model.ModelResult):
        """
        Sum the brightness over the vertical aperture bins to make a line
        spectrum.
        """
        n_obs, ny, nx = self._order_data.target_images.shape
        rows = np.unique(
            np.where(np.isnan(self._background.target_mask))[0])
        fit_results = []
        spectra = []
        for obs in range(n_obs):
            spectrum = np.nansum(self._calibrated_images[obs, rows]
                                 / self._get_dwavelength(), axis=0)
            spectra.append(spectrum)
            fit_result = self._fit_gaussian(spectrum=spectrum)
            fit_results.append(fit_result)
            filename = self._order_data.filenames[obs].replace('.fits.gz',
                                                               '.txt')
            self._save_line_spectra(
                self._order_data.rest_wavelength_centers,
                self._order_data.shifted_wavelength_centers,
                spectrum, fit_result, filename)
        average_spectrum = np.nansum(
            self._average_calibrated_image[rows]
            / self._get_dwavelength(), axis=0)
        fit_result = self._fit_gaussian(spectrum=average_spectrum)
        self._save_line_spectra(self._order_data.rest_wavelength_centers,
                                self._order_data.shifted_wavelength_centers,
                                average_spectrum, fit_result, 'average.txt')
        return spectra, average_spectrum, fit_results, fit_result

    def _find_integration_range(self) -> np.ndarray:
        """
        Find all indices ± 0.05 nm from the line center.
        """
        dwavelength = 0.05 * u.nm
        indices = []
        for wavelength in self._order_data.aurora_wavelengths:
            ind0 = np.abs(self._order_data.shifted_wavelength_centers
                          - (wavelength - dwavelength)).argmin()
            ind1 = np.abs(self._order_data.shifted_wavelength_centers
                          - (wavelength + dwavelength)).argmin()
            indices.extend(list(np.arange(ind0, ind1+1, 1)))
        return np.unique(indices)

    def _get_measured_brightnesses(self) -> (list[u.Quantity], u.Quantity):
        """
        Integrate measured line spectrum.
        """
        n_obs = self._order_data.target_images.shape[0]
        brightnesses = []
        ind = self._find_integration_range()
        dwavelength = np.diff(
            self._order_data.shifted_wavelength_edges)[ind].value
        for obs in range(n_obs):
            brightness = np.nansum(
                (self._line_spectra[obs][ind].value
                 - self._fit_results[obs].params['constant_c']) * dwavelength)
            brightnesses.append(brightness)
        average_brightness = np.nansum(
            (self._average_line_spectrum[ind].value
             - self._average_fit_result.params['constant_c']) * dwavelength)
        return brightnesses * u.R, average_brightness * u.R

    def _get_measured_uncertainties(self) -> (list[u.Quantity], u.Quantity):
        """
        Integrate best-fit model spectrum uncertainty for 0.1 nm.
        """
        n_obs, _, nx = self._order_data.target_images.shape
        uncertainties = []
        ind = self._find_integration_range()
        dwavelength = np.diff(
            self._order_data.shifted_wavelength_edges)[ind].value
        ind = np.array([i for i in np.arange(nx)
                        if i not in self._find_integration_range()])
        for obs in range(n_obs):
            std = np.nanstd(self._line_spectra[obs][ind].value)
            uncertainty = np.sqrt(np.sum((std * dwavelength)**2))
            uncertainties.append(uncertainty)
        std = np.nanstd(self._average_line_spectrum[ind].value)
        average_uncertainty = np.sqrt(np.sum((std * dwavelength)**2))
        return uncertainties * u.R, average_uncertainty * u.R

    def _get_fitted_brightnesses(self) -> (u.Quantity, u.Quantity):
        """
        Integrate best-fit model spectrum.
        """
        n_obs = self._order_data.target_images.shape[0]
        brightnesses = []
        ind = self._find_integration_range()
        for obs in range(n_obs):
            brightness = np.trapz(
                self._fit_results[obs].best_fit[ind]
                - self._fit_results[obs].params['constant_c'],
                x=self._order_data.shifted_wavelength_centers[ind].value)
            brightnesses.append(brightness)
        average_brightness = np.trapz(
            self._average_fit_result.best_fit[ind]
            - self._average_fit_result.params['constant_c'],
            x=self._order_data.shifted_wavelength_centers[ind].value)
        return np.array(brightnesses) * u.R, average_brightness * u.R

    def _get_fitted_uncertainties(self) -> (u.Quantity, u.Quantity):
        """
        Integrate best-fit model spectrum uncertainty.
        """
        n_obs = self._order_data.target_images.shape[0]
        uncertainties = []
        ind = self._find_integration_range()
        for obs in range(n_obs):
            uncertainty = np.trapz(
                self._fit_results[obs].eval_uncertainty()[ind],
                x=self._order_data.shifted_wavelength_centers[ind].value)
            uncertainties.append(uncertainty)
        average_uncertainty = np.trapz(
            self._average_fit_result.eval_uncertainty()[ind],
            x=self._order_data.shifted_wavelength_centers[ind].value)
        return np.array(uncertainties) * u.R, average_uncertainty * u.R

    @staticmethod
    def _get_ephemeris(date, target):
        epoch = Time(date, format='isot', scale='utc').jd
        eph = Horizons(id=target, location='568', epochs=epoch).ephemerides()
        return eph

    def _calculate_angular_separation(self) -> u.Quantity:
        """
        Calculate the angular separation between Jupiter and the target
        satellite in Jupiter radii.
        """
        dates = self._order_data.observation_dates
        codes = jovian_naif_codes()
        separations = []
        for date in dates:
            jupiter_ephemeris = self._get_ephemeris(date, '599')
            target_ephemeris = self._get_ephemeris(
                date, codes[self._order_data.target_name])
            jupiter_position = SkyCoord(ra=jupiter_ephemeris['RA'],
                                        dec=jupiter_ephemeris['DEC'])
            target_position = SkyCoord(ra=target_ephemeris['RA'],
                                       dec=target_ephemeris['DEC'])
            separation = target_position.separation(
                jupiter_position)[0].to(u.arcsec).value
            radius = jupiter_ephemeris['ang_width'][0] / 2
            separation -= radius
            separations.append(separation/radius)
        return separations

    def save_results(self):
        """
        Save the results to a text file.

        Columns:
            date: The start date/time of the observation.
            brightness_[R]: The integrated brightness in the aperture(s) in
                            rayleigh.

        Returns
        -------
        None.
        """

        n_obs = self._order_data.target_images.shape[0]
        dates = self._order_data.observation_dates
        average_wavelength = self._order_data.aurora_wavelengths.mean()
        brightnesses, average_brightness = \
            self._get_measured_brightnesses()
        background_unc, average_background_unc = \
            self._get_measured_uncertainties()
        fitted_brightnesses, average_fitted_brightness = \
            self._get_fitted_brightnesses()
        fitted_uncertainties, average_fitted_uncertainty = \
            self._get_fitted_uncertainties()
        bg_electron_flux = self._background.background_electron_flux
        angular_separations = self._calculate_angular_separation()
        included_in_average = self._order_data.included_in_average

        filename = 'results.txt'
        savepath = Path(self._save_path, f'{average_wavelength:.1f}', filename)
        if not savepath.parent.exists():
            savepath.parent.mkdir(parents=True)

        header = 'date measured_brightness_[R] measured_uncertainty_[R] ' \
                 'fitted_brightness_[R] fitted_uncertainty_[R] ' \
                 'bg_electron_flux_[e/s/arcsec2] limb_separation_[r/R_J] ' \
                 'included_in_avg'
        with open(savepath, 'w') as file:
            file.write(header + '\n')
            for obs in range(n_obs):
                data_str = f'{dates[obs]} ' \
                           f'{brightnesses[obs].value} ' \
                           f'{background_unc[obs].value} ' \
                           f'{fitted_brightnesses[obs].value} ' \
                           f'{fitted_uncertainties[obs].value} ' \
                           f'{int(bg_electron_flux[obs].value)} ' \
                           f'{angular_separations[obs]:.2f} ' \
                           f'{included_in_average[obs]}\n'
                file.write(data_str)
            bg_avg = (bg_electron_flux[
                          self._order_data.include_indices].value.mean())
            file.write(f'average '
                       f'{average_brightness.value} '
                       f'{average_background_unc.value} '
                       f'{average_fitted_brightness.value} '
                       f'{average_fitted_uncertainty.value} '
                       f'{np.round(bg_avg, 0)} '
                       f'--- '
                       f'---\n')
