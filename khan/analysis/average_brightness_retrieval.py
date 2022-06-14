from pathlib import Path

import astropy.units as u
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from astropy.constants import astropyconst20 as c
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import fits
from lmfit.models import GaussianModel, ConstantModel, PolynomialModel

from khan.common import aurora_feature_wavelengths, \
    doppler_shift_wavelengths, get_solar_spectral_radiance, \
    get_meridian_reflectivity
from khan.graphics import data_cmap


class DataSubsection:

    def __init__(self, reduced_data_path: str | Path,
                 analysis_save_path: str | Path,
                 feature_wavelengths: [u.Quantity], top_trim: int,
                 bottom_trim: int, seeing: float):
        self._analysis_save_path = analysis_save_path
        self._science_observations_path = Path(
            reduced_data_path, 'science_observations.fits.gz')
        self._flux_calibration_path = Path(
            reduced_data_path, 'flux_calibration.fits.gz')
        self._feature_wavelengths = feature_wavelengths.value
        self._average_wavelength = np.mean(feature_wavelengths.value)
        self._order_index = self._find_order_with_wavelengths()
        self._top_trim = top_trim
        self._bottom_trim = bottom_trim
        self._seeing = seeing
        self._selected_data = self._get_data_subsections()

    def _find_order_with_wavelengths(self) -> int:
        """
        Get the index of the echelle order with the feature wavelengths.
        """
        found_order = None
        with fits.open(self._science_observations_path) as hdul:
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

    def _get_data_subsections(self):

        # loop through wavelengths
        target_subsections = []
        meshgrids = []
        masks = []
        vertical_centers = []

        hdul = fits.open(self._science_observations_path)
        calibration_hdul = fits.open(self._flux_calibration_path)

        n_obs = hdul['PRIMARY'].header['NAXIS4']
        bin_width = hdul['PRIMARY'].header['SPESCALE']
        bin_height = hdul['PRIMARY'].header['SPASCALE']
        slit_width = hdul['PRIMARY'].header['SLITWID']
        slit_length = hdul['PRIMARY'].header['SLITLEN']
        ny = (np.ceil(slit_length / bin_height).astype(int)
              - int(self._bottom_trim + self._top_trim))

        flux_calibration_data = \
            np.mean(calibration_hdul['PRIMARY'].data[:, self._order_index],
                    axis=0)
        jupiter_distance = np.mean(
            calibration_hdul['OBSERVATION_INFORMATION'].data['DISTANCE'])

        order_wavelength_centers = \
            hdul['BIN_CENTER_WAVELENGTHS'].data[self._order_index]
        order_wavelength_edges = \
            hdul['BIN_EDGE_WAVELENGTHS'].data[self._order_index]
        echelle_order = hdul['ECHELLE_ORDERS'].data[self._order_index]

        wavelength_centers = None
        wavelength_edges = None
        min_ind = None
        max_ind = None
        nx = None

        for obs in range(n_obs):

            target_data = hdul['PRIMARY'].data[obs, self._order_index]
            guide_satellite_data = hdul['GUIDE_SATELLITE'].data[
                obs, self._order_index]

            relative_velocity = \
                (hdul['SCIENCE_OBSERVATION_INFORMATION'].data['RELVLCTY'][obs]
                 * u.m / u.s)
            shifted_wavelength_centers = doppler_shift_wavelengths(
                order_wavelength_centers * u.nm, relative_velocity).value
            shifted_wavelength_edges = doppler_shift_wavelengths(
                order_wavelength_edges * u.nm, relative_velocity).value

            wavelength_indices = [
                np.abs(shifted_wavelength_centers - wavelength).argmin()
                for wavelength in self._feature_wavelengths]

            if obs == 0:
                min_ind = (np.floor(np.min(wavelength_indices)
                                    - ny * bin_height / bin_width / 2).astype(
                    int))
                max_ind = (np.ceil(np.max(wavelength_indices)
                                   + ny * bin_height / bin_width / 2).astype(
                    int))
                nx = max_ind - min_ind
                flux_calibration_data = \
                    flux_calibration_data[self._bottom_trim:-self._top_trim,
                                          min_ind:max_ind]
                wavelength_centers = \
                    shifted_wavelength_centers[min_ind:max_ind]
                wavelength_edges = shifted_wavelength_edges[min_ind:max_ind]

            target_subsections.append(
                target_data[self._bottom_trim:-self._top_trim,
                            min_ind:max_ind])

            # find vertical offset from trace frame
            trace_slice = \
                guide_satellite_data[self._bottom_trim:-self._top_trim,
                                     int(nx / 2)]
            good = np.where(~np.isnan(trace_slice))
            trace_x = np.arange(ny)
            peak = GaussianModel()
            offset = ConstantModel()
            model = peak + offset
            pars = offset.make_params(c=np.nanmedian(trace_slice))
            pars += peak.guess(trace_slice[good], x=trace_x[good])
            result = model.fit(trace_slice[good], pars,
                               x=trace_x[good])
            vertical_center = np.round(
                result.params['center']) * bin_height
            vertical_centers.append(np.round(result.params['center']))
            horizontal_center = nx / 2 * bin_width

            # make mask
            x, y = np.meshgrid(
                (np.linspace(0, nx - 1, nx) + 0.5) * bin_width,
                (np.linspace(0, ny - 1, ny) + 0.5) * bin_height
                )
            meshgrids.append(((x - horizontal_center), (y - vertical_center)))
            apparent_radius = \
                (hdul['SCIENCE_OBSERVATION_INFORMATION'].data['A_RADIUS'][obs]
                 + self._seeing)
            mask_list = []
            for index in wavelength_indices:
                mask = np.sqrt(
                    (x - (index - min_ind) * bin_width) ** 2 + (
                                y - vertical_center) ** 2)
                mask[np.where(mask <= apparent_radius)] = np.nan
                mask[np.where(~np.isnan(mask))] = 1
                mask_list.append(mask)
            mask = np.mean(mask_list, axis=0)
            masks.append(mask)

        average_radius = \
            np.mean(hdul['SCIENCE_OBSERVATION_INFORMATION'].data['A_RADIUS'])

        hdul.close()
        calibration_hdul.close()

        data = {
            'average_wavelength': self._average_wavelength,
            'feature_wavelengths': self._feature_wavelengths,
            'echelle_order': np.array(echelle_order),
            'target_images': np.array(target_subsections),
            'vertical_centers': np.array(vertical_centers),
            'masks': np.array(masks),
            'meshgrids': np.array(meshgrids),
            'calibration_image': flux_calibration_data,
            'jupiter_distance': jupiter_distance,
            'wavelength_centers': wavelength_centers,
            'wavelength_edges': wavelength_edges,
            'bin_width': bin_width,
            'bin_height': bin_height,
            'slit_width': slit_width,
            'slit_length': slit_length,
            'target_angular_radius': average_radius,
            'aperture': average_radius + self._seeing,
        }

        return data

    def save_quality_assurance_graphic(self):

        n_obs = self.target_images.shape[0]
        if n_obs >= 6:
            n_cols = 6
        else:
            n_cols = n_obs
        n_rows = np.ceil(n_obs / n_cols).astype(int)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows),
                                 constrained_layout=True, sharex='all',
                                 sharey='all')
        axes = axes.ravel()
        [ax.set_facecolor('tab:grey') for ax in axes]
        [ax.set_xticks([]) for ax in axes]
        [ax.set_yticks([]) for ax in axes]
        [ax.remove() for ax in axes[n_obs:]]

        vmin = np.nanpercentile(self.target_images, 1)
        vmax = np.nanpercentile(self.target_images, 99)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        for obs in range(n_obs):
            axes[obs].pcolormesh(self.meshgrids[obs][0],
                                 self.meshgrids[obs][1],
                                 self.target_images[obs],
                                 cmap=data_cmap(), norm=norm)
            axes[obs].set_aspect('equal')
        ylims = np.array(axes[0].get_ylim()) / self.bin_height
        ny_total = (ylims[1] - ylims[0]).astype(int)
        savepath = Path(self._analysis_save_path, 'individual',
                        f'{self._average_wavelength:.1f}nm.png')
        if not savepath.parent.exists():
            savepath.parent.mkdir(parents=True)
        plt.suptitle(f'Wavelength: {self._average_wavelength:.1f} nm',
                     fontweight='bold')
        plt.savefig(savepath, facecolor='white', transparent=False)
        plt.close()
        return ny_total

    @property
    def average_wavelength(self) -> float:
        return self._selected_data['average_wavelength']

    @property
    def feature_wavelengths(self) -> [float]:
        return self._selected_data['feature_wavelengths']

    @property
    def target_images(self) -> np.ndarray:
        return self._selected_data['target_images']

    @property
    def echelle_order(self) -> int:
        return self._selected_data['echelle_order']

    @property
    def vertical_centers(self) -> np.ndarray:
        return self._selected_data['vertical_centers']

    @property
    def masks(self) -> np.ndarray:
        return self._selected_data['masks']

    @property
    def meshgrids(self) -> np.ndarray:
        return self._selected_data['meshgrids']

    @property
    def calibration_image(self) -> np.ndarray:
        return self._selected_data['calibration_image']

    @property
    def jupiter_distance(self) -> float:
        return self._selected_data['jupiter_distance']

    @property
    def wavelength_centers(self) -> np.ndarray:
        return self._selected_data['wavelength_centers']

    @property
    def wavelength_edges(self) -> np.ndarray:
        return self._selected_data['wavelength_edges']

    @property
    def bin_width(self) -> float:
        return self._selected_data['bin_width']

    @property
    def bin_height(self) -> float:
        return self._selected_data['bin_height']

    @property
    def slit_width(self) -> float:
        return self._selected_data['slit_width']

    @property
    def slit_length(self) -> float:
        return self._selected_data['slit_length']

    @property
    def target_radius(self) -> float:
        return self._selected_data['target_angular_radius']

    @property
    def aperture(self) -> float:
        return self._selected_data['aperture']


class AverageImage:

    def __init__(self, analysis_save_path: str | Path,
                 data: DataSubsection, ny_total: int, exclude: dict,
                 background_degree: int = 1):
        self._analysis_save_path = analysis_save_path
        self._data = data
        self._ny_total = ny_total
        self._exclude = exclude
        self._degree = background_degree
        self._average = self.offset_images_vertically()
        self._background = self.calculate_background()

    def offset_images_vertically(self) -> np.ndarray:
        n_obs, ny, nx = self._data.target_images.shape
        averages = []
        key = f'{self._data.average_wavelength:.1f}'
        exclude = []
        if self._exclude is not None:
            if key in self._exclude.keys():
                exclude = self._exclude[key]
        obs_to_include = [i for i in range(n_obs) if i not in exclude]
        for obs in obs_to_include:
            ind = (np.max(self._data.vertical_centers)
                   - self._data.vertical_centers[obs]).astype(int)
            img = np.full((self._ny_total, nx), fill_value=np.nan)
            img[ind:ind + ny] = self._data.target_images[obs]
            averages.append(img)
        average = np.mean(averages, axis=0)
        rows = np.unique(np.where(~np.isnan(average))[0])
        return average[rows]

    def make_angular_meshgrids(self) -> (np.ndarray, np.ndarray):
        ny, nx = self._average.shape
        x, y = np.meshgrid(np.arange(0, nx+1, 1) * self._data.bin_width,
                           np.arange(0, ny+1, 1) * self._data.bin_height)
        x -= nx * self._data.bin_width / 2
        y -= ny * self._data.bin_height / 2
        return x, y

    def make_mask(self):
        _, _, nx = self._data.target_images.shape
        x, y = self.make_angular_meshgrids()
        x = x[:-1, :-1] + np.median(np.diff(x[0]))
        y = y[:-1, :-1] + np.median(np.diff(y[:, 0]))
        mask = np.sqrt(x ** 2 + y ** 2)
        bg_pixels = np.where(mask > self._data.aperture)
        target_pixels = np.where(mask <= self._data.aperture)
        mask[bg_pixels] = 1
        mask[target_pixels] = np.nan
        return mask

    def calculate_background(self):
        _, _, nx = self._data.target_images.shape
        background = np.zeros_like(self._average)
        mask = self.make_mask()
        masked_image = self._average.copy() * mask
        for col in range(nx):
            data_slice = masked_image[:, col]
            good = np.where(~np.isnan(data_slice))
            data_x = np.arange(len(data_slice))
            model = PolynomialModel(degree=self._degree, nan_policy='omit')
            pars = model.guess(data_slice[good], x=data_x[good])
            result = model.fit(data_slice[good], pars, x=data_x[good])
            background[:, col] = result.eval(x=data_x)
        return background

    def plot_aperture(self, axis, center=0):
        theta = np.linspace(0, 2 * np.pi, 361)
        axis.plot((self._data.target_radius * np.cos(theta) - center),
                  self._data.target_radius * np.sin(theta), color='w')
        axis.plot((self._data.aperture * np.cos(theta) - center),
                  self._data.aperture * np.sin(theta), color='w',
                  linestyle='--')

    def save_quality_assurance_graphic(self):
        background = self.calculate_background()
        subtracted = self._average - background
        smoothed = convolve(subtracted, Gaussian2DKernel(x_stddev=1))
        norm = colors.Normalize(vmin=0, vmax=np.percentile(self._average, 99))
        norm_subtracted = colors.Normalize(vmin=0,
                                           vmax=np.percentile(subtracted, 99))

        # if it's a set of overlapping features
        ny, nx = self._average.shape
        xpos = self._data.bin_width * np.array(
            [nx/2-np.abs(self._data.wavelength_centers-wavelength).argmin()
             for wavelength in self._data.feature_wavelengths])

        fig, axes = plt.subplots(2, 3, figsize=(7, 3.5),
                                 constrained_layout=True)
        axes = axes.ravel()
        [axis.set_xticks([]) for axis in axes]
        [axis.set_yticks([]) for axis in axes]
        x, y = self.make_angular_meshgrids()
        axes[0].pcolormesh(x, y, self._average, norm=norm)
        axes[1].pcolormesh(x, y, self._average, norm=norm)
        [self.plot_aperture(axes[1], center) for center in xpos]
        axes[2].pcolormesh(x, y, background, norm=norm)
        axes[3].pcolormesh(x, y, subtracted, norm=norm_subtracted)
        axes[4].pcolormesh(x, y, subtracted, norm=norm_subtracted)
        [self.plot_aperture(axes[4], center) for center in xpos]
        [self.plot_aperture(axes[5], center) for center in xpos]
        axes[5].pcolormesh(x, y, smoothed, norm=norm_subtracted)
        axes[0].set_title('Aligned and Averaged')
        axes[1].set_title('...with Aperture')
        axes[2].set_title(f'Fitted Background, Degree: {self._degree}')
        axes[3].set_title('Background-Subtracted')
        axes[4].set_title('...with Aperture')
        axes[5].set_title(r'Smoothed, $\sigma=1\,\mathrm{bin}$')
        savepath = Path(self._analysis_save_path, 'averaged',
                        f'{self._data.average_wavelength:.1f}nm.png')
        if not savepath.parent.exists():
            savepath.parent.mkdir(parents=True)
        fig.suptitle(f'Wavelength: {self._data.average_wavelength:.1f} nm',
                     fontweight='bold')
        [axis.set_aspect('equal') for axis in axes]
        plt.savefig(savepath, facecolor='white', transparent=False)
        plt.close()

    @property
    def average_image(self) -> np.ndarray:
        return self._average

    @property
    def background_subtracted(self) -> np.ndarray:
        return self._average - self._background

    @property
    def smoothed(self) -> np.ndarray:
        return convolve(self._average - self._background,
                        Gaussian2DKernel(x_stddev=1))

    @property
    def meshgrids(self) -> (np.ndarray, np.ndarray):
        return self.make_angular_meshgrids()

    @property
    def target_mask(self) -> np.ndarray:
        return self.make_mask()


class Calibration:

    def __init__(self, data: DataSubsection):
        self._data = data
        self._jupiter_brightness = self.get_jupiter_brightness()
        self._jupiter_electron_flux = self.get_jupiter_electron_flux()

    def get_jupiter_brightness(self) -> u.Quantity:
        radiance = get_solar_spectral_radiance()
        eye_over_f = get_meridian_reflectivity()
        reflectivity = np.interp(radiance['wavelength'].value,
                                 eye_over_f['wavelength'].value,
                                 eye_over_f['reflectivity'])
        photon_energy = c.h * c.c / (radiance['wavelength'] * u.photon)
        photon_flux = (radiance['radiance'] / photon_energy).to(u.R / u.nm)
        photon_flux *= reflectivity
        photon_flux *= (1 / self._data.jupiter_distance) ** 2
        ind = np.abs(radiance['wavelength'].value
                     - self._data.average_wavelength).argmin()
        return photon_flux[ind]

    def get_jupiter_electron_flux(self):
        ny, nx = self._data.calibration_image.shape
        middle_index = int(nx/2)
        slit_half_width_bins = np.round(
            self._data.slit_width / self._data.bin_width / 2).astype(int)
        slit_length_bins = np.round(
            self._data.slit_length / self._data.bin_height).astype(int)
        n_bins_slit = int(2 * slit_half_width_bins * slit_length_bins)
        slit_angular_area = (self._data.slit_length * self._data.slit_width
                             * u.arcsec**2)
        left = middle_index-slit_half_width_bins
        right = middle_index+slit_length_bins
        average_counts = np.mean(self._data.calibration_image[:, left:right])
        total_flux = (average_counts * n_bins_slit * u.electron / u.s /
                      u.m**2 / u.nm / slit_angular_area.to(u.sr))
        return total_flux

    @property
    def jupiter_brightness(self) -> u.Quantity:
        return self._jupiter_brightness

    @property
    def jupiter_electron_flux(self) -> u.Quantity:
        return self._jupiter_electron_flux

    @property
    def calibration(self) -> u.Quantity:
        return (self._jupiter_brightness
                / self._jupiter_electron_flux).to(u.photon/u.electron)

    @property
    def slit_width_bins(self) -> float:
        return np.round(self._data.slit_width/self._data.bin_width).astype(int)


class CalibratedData:

    def __init__(self, data: DataSubsection, average: AverageImage,
                 calibration: Calibration, analysis_save_path):
        self._data = data
        self._average = average
        self._calibration = calibration
        self._analysis_save_path = analysis_save_path
        self._calibrated_image = self.calibrate_average_image()
        self._retrieved_brightness = self.calculate_integrated_brightness()

    def get_dwavelength(self):
        ind = np.abs(self._data.wavelength_centers
                     - self._data.average_wavelength).argmin()
        return np.diff(self._data.wavelength_edges)[ind]

    def calibrate_average_image(self):
        target_unit = (u.electron / u.s / u.m**2 /
                       (np.pi * (self._data.target_radius * u.arcsec)**2))
        brightness = (self._average.background_subtracted * target_unit
                      * self._calibration.calibration
                      * self._calibration.slit_width_bins
                      * self.get_dwavelength()).to(u.R)
        return brightness

    def make_inverted_target_mask(self):
        mask = self._average.target_mask
        target_ind = np.where(np.isnan(mask))
        background_ind = np.where(~np.isnan(mask))
        mask[target_ind] = 1
        mask[background_ind] = np.nan
        return mask

    def calculate_integrated_brightness(self):
        if self.wavelength.value == 656.2852:
            brightness = np.nansum(self._calibrated_image
                                   * self.make_inverted_target_mask())
        else:
            keep = np.where(self._calibrated_image > 0)
            brightness = np.nansum(self._calibrated_image[keep]
                                   * self.make_inverted_target_mask()[keep])
        background = np.nanstd(self._calibrated_image
                               * self._average.target_mask)
        n_pixels = len(np.where(np.isnan(self._average.target_mask))[0])
        background_noise = background / np.sqrt(n_pixels)
        signal_noise = np.sqrt(np.abs(brightness.value)) * brightness.unit
        uncertainty = np.sqrt(background_noise**2 + signal_noise**2)
        retrieved_brightness = {
            'brightness': brightness,
            'uncertainty': uncertainty,
        }
        return retrieved_brightness

    def save_quality_assurance_graphic(self):
        norm = colors.Normalize(
            vmin=0, vmax=np.percentile(self._calibrated_image.value, 99))
        smoothed = convolve(self._calibrated_image,
                            Gaussian2DKernel(x_stddev=1))
        fig, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
        x, y = self._average.meshgrids
        img = axes[0].pcolormesh(x, y, self._calibrated_image, norm=norm)
        axes[1].pcolormesh(x, y, smoothed, norm=norm)
        cbar = plt.colorbar(img, ax=axes[1],
                            label='Photon Flux from Target Disc [R]')
        cbar.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.1f}'))
        axes[0].set_title('Not Smoothed')
        axes[1].set_title('Smoothed')
        [axis.set_xticks([]) for axis in axes]
        [axis.set_yticks([]) for axis in axes]
        [axis.set_aspect('equal') for axis in axes]
        fig.suptitle(f'{self._data.average_wavelength:.1f} nm',
                     fontweight='bold')
        savepath = Path(self._analysis_save_path, 'calibrated',
                        f'{self._data.average_wavelength:.1f}nm.png')
        if not savepath.parent.exists():
            savepath.parent.mkdir(parents=True)
        plt.savefig(savepath, facecolor='white', transparent=False)
        plt.close()

    @property
    def wavelength(self) -> u.Quantity:
        return self._data.average_wavelength * u.nm

    @property
    def brightness(self) -> u.Quantity:
        return self._retrieved_brightness['brightness']

    @property
    def uncertainty(self) -> u.Quantity:
        return self._retrieved_brightness['uncertainty']


def get_average_brightness(reduced_data_path: str | Path,
                           analysis_save_path: str | Path,
                           top_trim: int = 2, bottom_trim: int = 2,
                           seeing: int = 1, background_degree: int = 1,
                           exclude=None):
    results_file = Path(analysis_save_path, 'results.txt')
    with open(results_file, 'w') as file:
        file.write('wavelength_nm brightness_rayleighs '
                   'uncertainty_rayleighs\n')

    for wavelengths in aurora_feature_wavelengths():

        try:  # accounts for absent wavelengths
            data_subsection = DataSubsection(
                reduced_data_path=reduced_data_path,
                analysis_save_path=analysis_save_path,
                feature_wavelengths=wavelengths,
                top_trim=top_trim, bottom_trim=bottom_trim, seeing=seeing)
            ny_total = data_subsection.save_quality_assurance_graphic()
            average_image = AverageImage(analysis_save_path=analysis_save_path,
                                         data=data_subsection,
                                         ny_total=ny_total,
                                         background_degree=background_degree,
                                         exclude=exclude)
            average_image.save_quality_assurance_graphic()
            calibration = Calibration(data=data_subsection)
            calibrated_data = CalibratedData(
                data=data_subsection, average=average_image,
                calibration=calibration, analysis_save_path=analysis_save_path)
            calibrated_data.save_quality_assurance_graphic()

            with open(results_file, 'a') as file:
                file.write(f'{calibrated_data.wavelength.value:.1f} '
                           f'{calibrated_data.brightness.value:.2f} '
                           f'{calibrated_data.uncertainty.value:.2f}\n')

        except ValueError:
            continue
