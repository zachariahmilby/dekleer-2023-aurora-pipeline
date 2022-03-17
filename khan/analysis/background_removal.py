import warnings

import numpy as np
from lmfit.models import GaussianModel, PolynomialModel

from khan.analysis.data_retrieval import DataSubsection


class Masks:
    """
    This class generates and holds all of the masks for removing or isolating
    the target satellite and removing the edges of the slit. The masks are
    arrays of NaNs and 1s, so that when multipling an image by the mask, the
    NaN values will effectively mask out those regions. There's probably a way
    to do this better with Numpy's masked arrays, but I realized that too late.
    Maybe in a future release...
    """

    def __init__(self, data_subsection: DataSubsection,
                 top_trim: int = 0, bottom_trim: int = 0, seeing: float = 1):
        """
        Parameters
        ----------
        data_subsection : DataSubsection
            The portion of the order containing the wavelength of interest.
        top_trim : int
            How many rows to trim off of the top.
        bottom_trim : int
            How many rows to trim off of the bottom.
        seeing : float
            The amount by which to increase the planet's radius in order to
            make a larger aperture and capture the emssion spread out by
            atmospheric seeing. If the target satellite's radius is R, then
            the aperture has a size of π(R + seeing)².
        """
        self._data_susbection = data_subsection
        self._top_trim = top_trim
        self._bottom_trim = bottom_trim
        self._seeing = seeing
        self._horizontal_indices = self._find_horizontal_indices()
        self._vertical_indices = self.find_vertical_indices()
        self._slit_edge_mask = self._make_slit_edge_mask()
        self._target_masks = self._make_target_masks()
        self._inverted_target_masks = self._make_inverted_target_masks()

    def _find_horizontal_indices(self) -> list[int]:
        """
        Find the horizontal indices closest to the chosen wavelength(s).
        Coupled with `_find_vertical_indices` this will give the index tuple
        corresponding to the center of the target satellite at a given
        wavelength.
        """
        return [np.abs(self._data_susbection.shifted_wavelength_centers
                       - wavelength).argmin()
                for wavelength in self._data_susbection.feature_wavelengths]

    def find_vertical_indices(self) -> list[list[int]]:
        """
        Use the trace frames to find the approximate vertical centroid. Coupled
        with `_find_horizontal_indices` this will give the index tuple
        corresponding to the center of the target satellite at a given
        wavelength.
        """
        n_obs, n_spa, n_spe = self._data_susbection.guide_satellite_data.shape
        x = np.arange(n_spa)
        vertical_indices = []
        for trace in self._data_susbection.guide_satellite_data:
            image_indices = []
            for wavelength_index in self._horizontal_indices:
                model = GaussianModel()
                vertical_slice = trace[:, wavelength_index]
                params = model.guess(vertical_slice, x=x)
                fit = model.fit(vertical_slice, params, x=x)
                ind = np.round(fit.params['center'].value).astype(int)
                image_indices.append(ind)
            vertical_indices.append(image_indices)
        return vertical_indices

    def _make_slit_edge_mask(self) -> np.ndarray:
        """
        Make a mask which removes user-selected rows from the top and bottom
        of the image (corresponding to the spatial edges of the slit).
        """
        mask = np.ones_like(self._data_susbection.science_data[0])
        if self._bottom_trim != 0:
            mask[:self._bottom_trim, :] = np.nan
        if self._top_trim != 0:
            mask[-self._top_trim:, :] = np.nan
        return mask

    def _make_target_masks(self) -> np.ndarray:
        """
        Make a mask which removes the aperture in order to isolate all of
        the background.
        """
        n_obs, n_spa, n_spe = self._data_susbection.guide_satellite_data.shape
        radius = self._data_susbection.target_angular_radius
        spa_scale = self._data_susbection.spatial_bin_scale
        spe_scale = self._data_susbection.spectral_bin_scale
        x, y = self._data_susbection.center_meshgrids
        masks = []
        for obs in range(n_obs):
            line_masks = []
            for i, h_ind in enumerate(self._horizontal_indices):
                mask = np.ones((n_spa, n_spe))
                x0 = h_ind * spe_scale
                y0 = self._vertical_indices[obs][i] * spa_scale
                dist = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
                mask[np.where(dist < radius.value + self._seeing/2)] = np.nan
                line_masks.append(mask)
            line_mask = np.mean(line_masks, axis=0)
            masks.append(line_mask)
        return np.array(masks)

    def _make_inverted_target_masks(self) -> np.ndarray:
        """
        Make a mask which isolates the aperture and removes all of the
        background.
        """
        n_obs, n_spa, n_spe = self._data_susbection.guide_satellite_data.shape
        radius = self._data_susbection.target_angular_radius
        spa_scale = self._data_susbection.spatial_bin_scale
        spe_scale = self._data_susbection.spectral_bin_scale
        x, y = self._data_susbection.center_meshgrids
        masks = []
        for obs in range(n_obs):
            line_masks = []
            for i, h_ind in enumerate(self._horizontal_indices):
                mask = np.ones((n_spa, n_spe))
                x0 = h_ind * spe_scale
                y0 = self._vertical_indices[obs][i] * spa_scale
                dist = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
                mask[np.where(dist >= radius.value + self._seeing/2)] = np.nan
                line_masks.append(mask)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                line_mask = np.nanmean(line_masks, axis=0)
            masks.append(line_mask)
        return np.array(masks)

    @property
    def seeing(self) -> float:
        return self._seeing

    @property
    def slit_edge_mask(self) -> np.ndarray:
        return self._slit_edge_mask

    @property
    def target_masks(self) -> np.ndarray:
        return self._target_masks

    @property
    def inverted_target_masks(self) -> np.ndarray:
        return self._inverted_target_masks


class Background:
    """
    This class calculates and holds the fitted background of the science image.
    For each row in the science image, it fits a polynomial and stores the
    result in a master background image. This image can then be subtracted from
    the corresponding science image to isolate any brightness in the aperture.
    The default polynomial degree is 1 (linear), but a user can choose other
    degrees if desired. When fitting the background in columns including the
    aperture, the fit ignores the aperture pixels so it doesn't explicitly try
    to capture some of the aurora brightness.
    """
    def __init__(self, data_subsection: DataSubsection, masks: Masks,
                 background_degree: int = 1):
        """
        Parameters
        ----------
        data_subsection : DataSubsection
            The portion of the order containing the wavelength of interest.
        masks : Masks
            Masks for the data_subsection.
        background_degree : int
            What degree of background to fit.
        """
        self._data_subsection = data_subsection
        self._masks = masks
        self._background_degree = background_degree
        self._backgrounds = self._calculate_backgrounds()

    def _fit_background_to_vertical_slice(
            self, vertical_slice: np.ndarray) -> np.ndarray:
        """
        Fit a polynomial to a vertical slice to estimate the background. This
        ignores any NaNs from masks but estimates the background in those
        regions by evaluating the final fit over all vertical slice pixels.
        """
        good = np.where(~np.isnan(vertical_slice))
        x = np.arange(len(vertical_slice))
        model = PolynomialModel(degree=self._background_degree,
                                nan_policy='omit')
        params = model.guess(vertical_slice[good], x=x[good])
        fit = model.fit(vertical_slice[good], params, x=x[good])
        return fit.eval(x=x)

    def _calculate_backgrounds(self) -> list[np.ndarray]:
        """
        Construct a background image for each of the science images.
        """
        masked_images = (self._data_subsection.science_data
                         * self._masks.target_masks
                         * self._masks.slit_edge_mask)
        _, n_spa, n_spe = masked_images.shape
        backgrounds = []
        for image in masked_images:
            background = np.zeros_like(image)
            for spe in range(n_spe):
                background[:, spe] = \
                    self._fit_background_to_vertical_slice(image[:, spe])
            backgrounds.append(background)
        return backgrounds

    @property
    def backgrounds(self) -> list[np.ndarray]:
        return self._backgrounds
