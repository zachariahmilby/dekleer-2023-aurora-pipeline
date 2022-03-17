import pickle
import warnings
from copy import deepcopy
from pathlib import Path

import astropy.units as u
import numpy as np
from lmfit.models import GaussianModel
from scipy.signal import correlate
from sklearn.preprocessing import minmax_scale

from khan.common import get_package_directory
from khan.pipeline.images import CCDImage
from khan.pipeline.rectification import RectifiedData, \
    calculate_slit_width_in_bins


class WavelengthSolution:

    def __init__(self, rectified_master_arc: RectifiedData):
        """
        Class to automatically generate and hold wavelength solutions for each
        order.

        Parameters
        ----------
        rectified_master_arc : RectifiedData
            Rectified master arc lamp spectra.
        """
        self._rectified_arc = rectified_master_arc
        self._pixel_centers = \
            np.arange(rectified_master_arc.images[0][0].data.shape[1])
        self._pixel_edges = \
            np.arange(rectified_master_arc.images[0][0].data.shape[1] + 1)-0.5
        self._half_slit_width = \
            calculate_slit_width_in_bins(
                self._rectified_arc.images[0][0]).value / 2
        (input_centers, input_wavelengths, fit_wavelength_centers,
         fit_wavelength_edges, fit_orders, fit_ind, fit_functions) = \
            self._calculate_wavelength_solutions()
        self._input_pixel_centers = input_centers
        self._input_wavelengths = input_wavelengths
        self._fitted_wavelength_centers = fit_wavelength_centers
        self._fitted_wavelength_edges = fit_wavelength_edges
        self._orders = fit_orders
        self._order_indices = fit_ind
        self._fit_functions = fit_functions

    def _get_solution_templates(self) -> dict:
        """
        Retrieve the closest template for the wavelength solution. These
        templates are categorized based on detector setup (legacy or mosaic),
        the cross-disperser (red or blue), and the cross-disperser and echelle
        angles. After selecting the right detector setup and cross disperser,
        it returns the template closest to both angles. These probably won't
        exactly match the data, but they are close enough to give a decent
        wavelength solution.
        """
        detector_layout = \
            self._rectified_arc.images[0][0].anc['detector_layout']
        echelle_angle = \
            self._rectified_arc.images[0][0].anc['echelle_angle'].value
        cross_disperser_angle = \
            self._rectified_arc.images[0][0].anc['cross_disperser_angle'].value
        cross_disperser = \
            self._rectified_arc.images[0][0].anc['cross_disperser']
        template_filepath = Path(
            get_package_directory(), 'anc',
            f'{detector_layout}_detector_templates.pickle')
        templates = pickle.load(open(template_filepath, 'rb'))
        cross_dispersers = np.array([template['cross_disperser']
                                     for template in templates])
        echelle_angles = np.array([template['echelle_angle']
                                   for template in templates])
        cross_disperser_angles = np.array([template['cross_disperser_angle']
                                           for template in templates])
        nearest = min(zip(echelle_angles, cross_disperser_angles),
                      key=lambda point:
                      np.sqrt(np.sum((point - np.array(
                          [echelle_angle, cross_disperser_angle]
                      )) ** 2)))
        ind = np.where((cross_dispersers == cross_disperser)
                       & (echelle_angles == nearest[0])
                       & (cross_disperser_angles == nearest[1]))[0][0]
        return templates[ind]

    def _make_gaussian_emission_line(self, center: int | float) -> np.array:
        """
        Make a normalized Gaussian emission line at a given pixel position with
        a sigma equal to half of the slit width. This evaluates the emission
        line over the entire range of detector spectral pixels, so you can make
        a spectrum of lines by adding them up.
        """
        model = GaussianModel()
        params = model.make_params(amplitude=1, center=center,
                                   sigma=self._half_slit_width)
        return model.eval(x=self._pixel_centers, params=params)

    def _cross_correlate_template(self) -> \
            (np.ndarray, np.ndarray, list, list, np.ndarray):
        """
        Cross-correlate stacks of one-dimensional arc lamp spectra with stacks
        of the identified lines. This identifies and assigns wavelengths to
        pixels along the spectral dimension.

        Returns
        -------
        It's a little convoluted, but since this is for internal use only I
        kept it messy. The first item "selected_data" are the orders for which
        I was able to automatically generate a wavelength solution. The
        "good_data" array holds the corresponding order indices for retrieving
        data from the RectifiedData object. The "centers" are the pixel
        coordintes of the identified wavelengths in "wavelengths." Finally, the
        last item is an array of the one-dimensional arc lamp spectra
        corresponding to the selected data.
        """
        n_observations, n_orders = np.shape(self._rectified_arc.images)
        template = self._get_solution_templates()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arc_images = np.array([self._rectified_arc.images[0][order].data
                                   for order in range(n_orders)])
            observed_arc = minmax_scale(np.nanmean(arc_images, axis=1))
            observed_arc[np.where(np.isnan(observed_arc))] = 0
            artificial_arc_spectrum = np.zeros((len(template['orders']),
                                                observed_arc.shape[1]))
            for i, p in enumerate(template['line_centers']):
                artificial_arc_spectrum[i] = \
                    minmax_scale(
                        np.sum([self._make_gaussian_emission_line(line)
                                for line in p], axis=0))
            cross_correlation_matrix = correlate(observed_arc,
                                                 artificial_arc_spectrum,
                                                 mode='same')
        ind = np.unravel_index(cross_correlation_matrix.argmax(),
                               cross_correlation_matrix.shape)
        spatial_offset = artificial_arc_spectrum.shape[0]/2 - ind[0]
        spectral_offset = artificial_arc_spectrum.shape[1]/2 - ind[1]
        xmesh_data, ymesh_data = \
            np.meshgrid(np.linspace(0, observed_arc.shape[1],
                                    observed_arc.shape[1] + 1),
                        np.linspace(0, observed_arc.shape[0],
                                    observed_arc.shape[0] + 1))
        xmesh_template, ymesh_template = \
            np.meshgrid(np.linspace(0, artificial_arc_spectrum.shape[1],
                                    artificial_arc_spectrum.shape[1] + 1),
                        np.linspace(0, artificial_arc_spectrum.shape[0],
                                    artificial_arc_spectrum.shape[0] + 1))

        _, good_data, good_arcs = \
            np.intersect1d(ymesh_data[:-1, 0],
                           ymesh_template[:-1, 0] - spatial_offset,
                           return_indices=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selected_data = np.nanmean([self._rectified_arc.images[0][i].data
                                        for i in good_data], axis=1)
            selected_data = np.array([minmax_scale(i) for i in selected_data])
        centers = [np.round(template['line_centers'][i]
                            - spectral_offset).astype(int) for i in good_arcs]
        wavelengths = [template['line_wavelengths'][i] for i in good_arcs]
        selected_arcs = np.array(template['orders'])[good_arcs]
        return selected_data, good_data, centers, wavelengths, selected_arcs

    def _calculate_wavelength_solutions(self) \
            -> (list, list, np.ndarray, np.ndarray, np.ndarray,
                np.ndarray, list):
        """
        This is where the magic happens. This goes through all of the orders,
        fits a Gaussian with a sigma of 5 pixels to the arc lamp spectrum at an
        identified wavelength pixel, finds an ideal center from the Gaussian
        fit, then stores that center and its corresponding wavelength. Finally,
        it calculates a third-degree polynomial fit to the results and
        calcualtes the wavelengths of the order pixel centers and pixel edges.

        Returns
        -------
        A couple of things. First and second are the calculated wavelengths for
        the pixel centers and edges. Then the order numbers and the indices in
        the RectifiedData these solutions correspond to.
        """
        selected_data, data_indices, peaks, wavelengths, orders = \
            self._cross_correlate_template()
        input_centers = []
        input_wavelengths = []
        wavelength_centers = []
        wavelength_edges = []
        fit_functions = []
        for i in range(len(selected_data)):
            better_centers = []
            better_wavelengths = []
            use_ind = np.where((peaks[i] > 5)
                               & (peaks[i] < np.max(self._pixel_centers) - 5)
                               )[0]
            for wave, line in zip(wavelengths[i][use_ind], peaks[i][use_ind]):
                x = np.arange(line - 5, line + 5 + 1)
                y = selected_data[i][line - 5:line + 5 + 1]
                model = GaussianModel()
                params = model.guess(y, x=x)
                try:
                    result = model.fit(y, params, x=x)
                except (ValueError, IndexError):
                    continue
                if np.abs(line - result.params['center'].value) > 5:
                    continue  # skip bad ones
                else:
                    better_centers.append(result.params['center'].value)
                    better_wavelengths.append(wave)
            better_centers = np.array(better_centers)
            better_wavelengths = np.array(better_wavelengths)
            input_centers.append(better_centers)
            input_wavelengths.append(better_wavelengths)
            fit = np.poly1d(np.polyfit(better_centers, better_wavelengths,
                                       deg=3))
            wavelength_centers.append(fit(self._pixel_centers))
            wavelength_edges.append(fit(self._pixel_edges))
            fit_functions.append(fit)

        return (input_centers, input_wavelengths,
                np.array(wavelength_centers), np.array(wavelength_edges),
                orders, data_indices, fit_functions)

    def select_orders_with_solutions(
            self, data: list[list[CCDImage]]) -> list[list[CCDImage]]:
        """
        Select the subset of a set of CCDImage data with wavelength solutions.
        This may trim away some of marginal orders on either the red or blue
        detector.

        Parameters
        ----------
        data : list[list[CCDImage]]
            A set of rectified data. Can be before or after cosmic ray removal.

        Returns
        -------
        The subset of the set of rectified data with wavelength solutions.
        """
        n_observations, n_orders = np.shape(data)
        selected_images = []
        for obs in range(n_observations):
            selected_order_images = []
            for ind, order in enumerate(self._order_indices):
                image = data[obs][order].data
                anc = deepcopy(data[obs][order].anc)
                anc['order'] = self._orders[ind]
                anc['pixel_center_wavelengths'] = \
                    self._fitted_wavelength_centers[ind] * u.nm
                anc['pixel_edge_wavelengths'] = \
                    self._fitted_wavelength_edges[ind] * u.nm
                selected_order_images.append(CCDImage(image, anc))
            selected_images.append(selected_order_images)
        return selected_images

    @property
    def input_pixel_centers(self) -> list:
        return self._input_pixel_centers

    @property
    def input_wavelengths(self) -> list:
        return self._input_wavelengths

    @property
    def pixel_centers(self) -> np.ndarray:
        return self._pixel_centers

    @property
    def pixel_center_wavelengths(self) -> np.ndarray:
        return self._fitted_wavelength_centers

    @property
    def pixel_edges(self) -> np.ndarray:
        return self._pixel_edges

    @property
    def pixel_edge_wavelengths(self) -> np.ndarray:
        return self._fitted_wavelength_edges

    @property
    def orders(self) -> np.ndarray:
        return self._orders

    @property
    def fit_functions(self) -> list[np.poly1d]:
        return self._fit_functions
