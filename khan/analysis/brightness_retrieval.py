from pathlib import Path
import pickle

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from khan.analysis.background_removal import Masks, Background
from khan.analysis.data_retrieval import DataSubsection
from khan.analysis.flux_calibration import FluxCalibration
from khan.graphics import data_cmap

# lists of wavelengths to examine
feature_wavelengths = [[557.7330 * u.nm],
                       [630.0304 * u.nm],
                       [636.3776 * u.nm],
                       [777.1944 * u.nm],
                       [777.4166 * u.nm],
                       [777.5388 * u.nm],
                       [844.625 * u.nm, 844.636 * u.nm, 844.676 * u.nm],
                       [656.2852 * u.nm]]


class SurfaceBrightness:
    """
    Calculate and hold surface brightnesses and uncertainties.
    """
    def __init__(self, data_subsection: DataSubsection, masks: Masks,
                 background: Background, flux_calibration: FluxCalibration,
                 save_path: str | Path, index: int):
        """
        Parameters
        ----------
        data_subsection : DataSubsection
            The portion of the order containing the wavelength of interest.
        masks
        background
        flux_calibration
        save_path
        index
        """
        self._data_subsection = data_subsection
        self._masks = masks
        self._background = background
        self._flux_calibration = flux_calibration
        self._save_path = save_path
        self._index = index
        self._calibrated_science_image = self._calibrate_science_image()
        self._factor = self._calculate_seeing_scaling_factor()
        self._brightness = self._calculate_surface_brightness()
        self._uncertainty = self._calcualte_uncertainty()

    def _calculate_seeing_scaling_factor(self) -> float:
        """
        The aperture is larger than the angular area of the target satellite,
        so this calcualtes the increase in the brightness from the seeing.
        """
        observed_radius = (self._data_subsection.target_angular_radius.value
                           + self._masks.seeing)
        actual_radius = self._data_subsection.target_angular_radius.value
        return (observed_radius / actual_radius) ** 2

    def _calibrate_science_image(self) -> np.ndarray:
        """
        Calibrate the science image to rayleigh.
        """
        background_subtracted_image = \
            (self._data_subsection.science_data[self._index]
             - self._background.backgrounds[self._index])
        calibrated_image = (self._flux_calibration.calibration[self._index]
                            * background_subtracted_image
                            * self._flux_calibration.slit_width_in_arcsec
                            * self._flux_calibration.slit_length_in_arcsec
                            / (np.pi *
                               self._data_subsection.target_angular_radius**2)
                            * self._flux_calibration.slit_width_in_bins
                            * self._flux_calibration.wavelength_dispersion)
        return calibrated_image.value

    def _calculate_surface_brightness(self) -> float:
        """
        Because I've calibrated every pixel by Ganymede's angular size, the
        surface brightness is the sum over the aperture.
        """
        brightness = \
            np.nansum(self._calibrated_science_image
                      * self._masks.inverted_target_masks[self._index]
                      * self._masks.slit_edge_mask).squeeze()
        return brightness

    def _calcualte_uncertainty(self) -> u.Quantity:
        """
        This one is a little complicated. The easy part is the Poisson noise
        from the target itself, for which I just use the square-root of the
        retrieved brightness. I didn't propagate or characterize any of the
        other noise sources (like read noise, dark current, etc.), but instead
        I estimate the total of this remaining noise by using the standard
        deviation of what's left over after background subtraction and the
        masking of the satellite target. I then propagate this to the mean of
        the bins in the aperture by dividing by the square-root of the number
        of bins. Finally, I account for the difference between the angular size
        of the chosen aperture and the angular size of the target source by
        multiplying by the ratio of their angular areas.
        """
        noise = np.nanstd(self._calibrated_science_image
                          * self._masks.target_masks[self._index]
                          * self._masks.slit_edge_mask)
        n_bins = len(np.where(np.isnan(
            self._masks.target_masks[self._index]))[0])
        return np.sqrt(n_bins * noise + np.abs(self._brightness))

    def save_quality_assurance_graphic(self) -> None:
        """
        Save a quality-assurance graphic of the result.
        """
        fig, axes = plt.subplots(4, 1, figsize=(2, 8), sharex='all')
        [axis.set_xticks([]) for axis in axes]
        [axis.set_yticks([]) for axis in axes]
        x, y = self._data_subsection.angular_meshgrids
        axes[0].pcolormesh(x, y,
                           self._data_subsection.science_data[self._index]
                           * self._masks.slit_edge_mask,
                           cmap=data_cmap(), vmin=0)
        axes[0].set_title('Reduced Science Image')
        axes[1].pcolormesh(x, y, self._background.backgrounds[self._index]
                           * self._masks.slit_edge_mask,
                           cmap=data_cmap(), vmin=0)
        axes[1].set_title('Fitted Background')
        axes[2].pcolormesh(x, y, self._calibrated_science_image
                           * self._masks.slit_edge_mask, cmap=data_cmap(),
                           vmin=0)
        axes[2].set_title('Background-Subtracted Science Image')
        axes[3].pcolormesh(x, y, self._calibrated_science_image
                           * self._masks.slit_edge_mask
                           * self._masks.inverted_target_masks[self._index],
                           cmap=data_cmap(), vmin=0)
        axes[3].set_title('Target Size and Aperture')
        [axis.set_aspect('equal') for axis in axes]
        plt.tight_layout()
        filename = self._data_subsection.observation_datetimes[
            self._index].replace(':', '')
        path = Path(self._save_path,
                    f'{self._data_subsection.average_wavelength:.1f}',
                    f'{filename}.png')
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        plt.savefig(path)
        plt.close(fig=fig)

    def save_final_data(self):
        observation_date = self._data_subsection.observation_datetimes[
            self._index]
        path = Path(self._save_path,
                    f'{self._data_subsection.average_wavelength:.1f}',
                    f"{observation_date.replace(':', '')}.pickle")
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        data_dictionary = {
            'science_image': self._calibrated_science_image,
            'angular_meshgrids': self._data_subsection.angular_meshgrids,
            'datetime': observation_date
        }
        with open(path, 'wb') as file:
            pickle.dump(data_dictionary, file)

    @property
    def value(self) -> float:
        return self._brightness

    @property
    def uncertainty(self) -> u.Quantity:
        return self._uncertainty


def get_aurora_brightnesses(reduced_data_path: str | Path,
                            save_path: str | Path, seeing: int | float = 1,
                            top_trim: int = 1, bottom_trim: int = 1,
                            background_degree: int = 1) -> None:
    """
    Retrieve brightnesses and uncertainties for the following wavelengths:
    - 557.7330 nm O(¹S) to O(¹D)
    - 630.0304 nm O(¹D) to O(³P₀)
    - 636.3776 nm O(¹D) to O(³P₂)
    - 656.2852 nm Hα
    - 777.1944 nm O(⁵P₃) to O(⁵S₂)
    - 777.4166 nm O(⁵P₂) to O(⁵S₂)
    - 777.5388 nm O(⁵P₁) to O(⁵S₂)
    - 844.625 nm O(³P₀) to O(³S₁)
    - 844.636 nm O(³P₂) to O(³S₁)
    - 844.676 nm O(³P₁) to O(³S₁)

    You may want to run this once through, examine the results, then decide
    whether you need to change the seeing and trip off some rows from the top
    and/or bottom of each order, then re-run the retrievals.


    Parameters
    ----------
    reduced_data_path : str
        Absolute path to the location of the reduced data generated by the
        `reduce_data` function.
    save_path : str
        Absolute path to directory where you want brightness-retrieval output
        files saved.
    seeing : int or float
        Seeing for the night of the observations in arcseconds. Defaults to
        1.5 arcsecond but you can change this if the aperture isn't capturing
        the whole aurora emission (or is extending too far!)
    top_trim : int
        How many rows to trim off of the top of the orders. Mis-aligned flat
        fields and other edge effects might cause weird effects near the top
        and bottom boundaries of the slit.
    bottom_trim : int
        Same as `top_trim`, but for the bottom of the slit.
    background_degree : int
        The polynomial degree you want to estimate the background in the
        science images. Default is 1 (linear).

    Returns
    -------
    I dunno yet. Probably nothing.
    """
    for wavelengths in feature_wavelengths:
        brightnesses = []
        uncertainties = []
        try:  # skip any wavelengths that aren't captured in the data
            data_subsection = \
                DataSubsection(wavelengths=wavelengths,
                               reduced_data_path=reduced_data_path)
        except ValueError:
            continue
        masks = Masks(data_subsection=data_subsection, top_trim=top_trim,
                      bottom_trim=bottom_trim, seeing=seeing)
        background = Background(data_subsection=data_subsection,
                                masks=masks,
                                background_degree=background_degree)
        flux_calibration = FluxCalibration(
            data_subsection=data_subsection,
            reduced_data_path=reduced_data_path)

        print(f'Retrieving brightnesses at '
              f'{data_subsection.average_wavelength:.1f}...')

        # open file to hold results
        text_file = Path(save_path,
                         f'{data_subsection.average_wavelength:.1f} '
                         f'Results.txt')
        if not text_file.parent.exists():
            text_file.parent.mkdir(parents=True)
        with open(text_file, 'w') as file:
            file.write('date brightness_R uncertainty_R '
                       'avg_background_counts_per_s\n')

        for index in range(data_subsection.science_data.shape[0]):
            surface_brightness = SurfaceBrightness(
                data_subsection=data_subsection, masks=masks,
                background=background, flux_calibration=flux_calibration,
                save_path=save_path, index=index)
            surface_brightness.save_quality_assurance_graphic()
            surface_brightness.save_final_data()
            brightnesses.append(surface_brightness.value)
            uncertainties.append(surface_brightness.uncertainty)

            with open(text_file, 'a') as file:
                file.write(f'{data_subsection.observation_datetimes[index]} '
                           f'{brightnesses[-1]:.2f} {uncertainties[-1]:.2f} '
                           f'{background.average_backgrounds[index]:.2f}\n')
