from copy import deepcopy

import astropy.units as u
import numpy as np
from astropy.convolution import convolve, Gaussian1DKernel
from lmfit.models import GaussianModel
from scipy.signal import find_peaks
from sklearn.metrics import r2_score

from khan.pipeline.images import MasterFlat, MasterTrace, CCDImage


def round_up_to_odd(
        number: int | float | list | np.ndarray) -> int | np.ndarray:
    """
    Rounds a number to the nearest odd integer.
    """
    return (np.floor(np.round(number) / 2) * 2 + 1).astype(int)


def calculate_slit_width_in_bins(ccd_image: CCDImage) -> u.Quantity:
    """
    Get the slit width in bins from a master trace image.
    """
    slit_width_arcsec = ccd_image.anc['slit_width']
    bin_scale = ccd_image.anc['spectral_bin_scale']
    return np.round(slit_width_arcsec / bin_scale).astype(int)


def calculate_slit_length_in_bins(ccd_image: CCDImage) -> u.Quantity:
    """
    Get the slit length in bins from a master trace image.
    """
    slit_length_arcsec = ccd_image.anc['slit_length']
    bin_scale = ccd_image.anc['spatial_bin_scale']
    return np.round(slit_length_arcsec / bin_scale).astype(int)


class OrderTraces:
    """
    This class calculates the order traces, then cross-correlates an artifical
    flat-field to find the order edges.
    """
    def __init__(self, master_trace: MasterTrace, master_flat: MasterFlat):
        """
        Parameters
        ----------
        master_trace : MasterTrace
            A master trace image. Can be anything really, as long as it's a
            point-source like object taken under the same detector setup as
            your science data.
        master_flat : MasterFlat
            The master flat field image.
        """
        self._master_trace = master_trace
        self._master_flat = master_flat
        self._edges = [self._cross_correlate_with_flat(i)
                       for i in range(self._master_trace.n_detectors)]

    def __getitem__(self, index):
        return [self._edges[index][0], self._edges[index][1]]

    def _smoothed_trace_image_slice(self, trace_image_slice: np.ndarray) \
            -> np.ndarray:
        """
        Smooth a vertical slice through the trace image using a Gaussian kernel
        with a width scaled to the detector spatial binning.
        """
        kernel_size = round_up_to_odd(
            7 / self._master_trace[0].anc['spatial_binning'].value)
        smoothing_kernel = Gaussian1DKernel(kernel_size)
        vertical_slice = convolve(trace_image_slice, smoothing_kernel)
        return vertical_slice / np.nanmax(vertical_slice)

    def _get_peaks(self, detector: int, index: int) -> np.ndarray:
        """
        Gets peaks along a vertical slice of the trace frame. Also remove any
        spurious peaks by seeing if the fit improves without them.
        """
        # Some of the light blocked by the filter leaks through in mosaic
        # detector 3; increasing the prominence eliminates false peaks from
        # this leaked light.
        if detector == 2:
            prominence = 0.1
        else:
            prominence = 0.025
        trace_image = self._master_trace[detector].data
        vertical_slice = self._smoothed_trace_image_slice(
            trace_image[:, index])
        peaks, _ = find_peaks(vertical_slice, prominence=prominence)
        for i in range(len(peaks) - 1):
            if vertical_slice[peaks[i + 1]] < 0.2 * vertical_slice[peaks[i]]:
                examination_slice = \
                    np.concatenate((np.arange(0, i + 1),
                                    np.arange(i + 2, len(peaks))))
                peak_number = np.arange(len(examination_slice))
                fit = np.poly1d(np.polyfit(peak_number,
                                           peaks[examination_slice], deg=3))
                if r2_score(peaks[examination_slice],
                            fit(peak_number)) > 0.99999:
                    peaks[i + 1] = -1
        return peaks[np.where(peaks != -1)]

    def _get_better_peak_locations(
            self, detector: int, index: int) -> np.ndarray:
        """
        Return better trace centers using a Gaussian fit.
        """
        trace_image = self._master_trace[detector].data
        half_slit_length = \
            int((calculate_slit_length_in_bins(
                self._master_trace[0]).value - 1) / 2)
        vertical_slice = self._smoothed_trace_image_slice(
            trace_image[:, index])
        peaks = self._get_peaks(detector=detector, index=index)
        fitted_peaks = np.full_like(peaks, np.nan, dtype=float)
        for i, peak in enumerate(peaks):
            # can't fit a Gaussian to a peak at the edge of the detector
            if (peak - half_slit_length > 0) \
                    & (peak + half_slit_length < len(vertical_slice)):
                model = GaussianModel()
                fit_x = np.arange(peak - half_slit_length,
                                  peak + half_slit_length)
                fit_y = vertical_slice[peak - half_slit_length:
                                       peak + half_slit_length]
                initial_params = model.guess(fit_y, fit_x)
                fit = model.fit(fit_y, params=initial_params, x=fit_x)
                fitted_peaks[i] = fit.params['center'].value
            else:
                continue
        return fitted_peaks[np.where(~np.isnan(fitted_peaks))]

    def _interpolate_additional_peaks(
            self, detector: int, index: int) -> np.ndarray:
        """
        Interpolate 2 extra peaks to make sure there aren't any orders near the
        top or bottom of the detector missing peaks.
        """
        peaks = self._get_better_peak_locations(detector=detector, index=index)
        interpolation_domain = np.arange(-2, len(peaks) + 1)
        fit = np.poly1d(np.polyfit(np.arange(len(peaks)), peaks, deg=3))
        interpolated_peaks = fit(interpolation_domain)
        return interpolated_peaks

    def _get_slices(self, detector: int) -> np.ndarray:
        """
        Indices for 11 vertical slices along the spectral dimension starting at
        index 100 and ending at index -100.
        """
        return np.linspace(100,
                           self._master_trace[detector].data.shape[1] - 100,
                           11, dtype=int)

    def _get_column_indices(self, detector: int) -> np.ndarray:
        """
        A range of integers covering the range of spectral bins for
        interpolation.
        """
        return np.arange(self._master_trace[detector].data.shape[1])

    def _find_trace_intercepts(self, detector: int) -> np.ndarray:
        """
        Wrapper method which finds the best centers at 11 points per order
        using the methods defined above.
        """
        slices = self._get_slices(detector)
        extended_peaks = []
        for index in slices:
            interpolated_peaks = self._interpolate_additional_peaks(
                detector=detector, index=index)
            extended_peaks.append(interpolated_peaks)
        slope = np.diff([i[0] for i in extended_peaks])
        discontinuities = np.flip(np.where(slope < 0)[0])
        for index in discontinuities:
            for item in range(index + 1, len(extended_peaks)):
                extended_peaks[item] = extended_peaks[item][1:]
        minimum_length = np.min([len(i) for i in extended_peaks])
        for index in range(len(extended_peaks)):
            extended_peaks[index] = extended_peaks[index][:minimum_length]
        extended_peaks = np.asarray(extended_peaks).T
        n_rows, n_columns = extended_peaks.shape
        for row in range(n_rows):
            if len(np.where(extended_peaks[row] < 0)[0]) == n_columns:
                extended_peaks[row, :] = np.nan
            elif len(np.where(extended_peaks[row]
                              >= self._master_trace[detector].data.shape[1])[0]
                     ) == n_columns:
                extended_peaks[row, :] = np.nan
        return extended_peaks

    def _trace_the_traces(self, detector: int) -> np.ndarray:
        """
        Calculate the trace vertical indices for every order in a detector.
        """
        columns = self._get_column_indices(detector)
        slices = self._get_slices(detector)
        peaks = self._find_trace_intercepts(detector)
        trace_indices = []
        for index, row in enumerate(peaks):
            if len(np.where(~np.isnan(row))[0]) < 4:
                continue
            else:
                good_points = np.where(~np.isnan(row))[0]
                fit = np.poly1d(np.polyfit(slices[good_points],
                                           row[good_points], deg=3))
                trace_indices.append(np.round(fit(columns)).astype(int))
        return np.array(trace_indices)

    def _make_artificial_flat(self, detector: int) -> np.ndarray:
        """
        Make an artificial binary flatfield using the traces. The result is an
        array of zeros and ones similar to the flatfield.
        """
        traces = self._trace_the_traces(detector=detector)
        n_rows, n_columns = self._master_trace[detector].data.shape
        slit_length = \
            calculate_slit_length_in_bins(self._master_trace[0]).value
        traces -= np.nanmin(traces)  # offset to zero
        binary_flat = np.zeros((np.nanmax(traces) + slit_length, n_columns))
        for row in range(traces.shape[0]):
            for column in range(traces.shape[1]):
                ind = traces[row, column]
                binary_flat[ind:ind + slit_length, column] = 1
        return binary_flat

    def _cross_correlate_along_spatial_axis(
            self, detector: int) -> (int, np.ndarray):
        """
        For some reason there's no way to cross-correlate along just one axis,
        so I had to write it myself. This returns the vertical offset for the
        traces from zero and the binary flatfield with the highest
        cross-correlation.
        """
        bf = self._make_artificial_flat(detector=detector)
        ind = slice(0, self._master_flat[detector].data.shape[0])
        master_flat = self._master_flat[detector].data
        slit_length = \
            calculate_slit_length_in_bins(self._master_trace[0]).value
        ccr = np.array([np.sum(master_flat * np.roll(bf, -i, axis=0)[ind])
                        for i in range(-4 * slit_length, 4 * slit_length)])
        offset = 4 * slit_length - ccr.argmax()
        return offset, np.roll(bf, offset, axis=0)[ind]

    def _cross_correlate_with_flat(
            self, detector: int) -> (np.ndarray, np.ndarray):
        """
        Cross-correlate the binary flat field with the real flat field to
        determine the edges of the orders. Returns the bottom and top edges of
        the orders for a given detector.
        """
        traces = self._trace_the_traces(detector=detector)
        traces -= np.nanmin(traces)  # offset to zero
        offset, bf = self._cross_correlate_along_spatial_axis(detector)
        traces += offset
        bf[np.where(bf == 0)] = np.nan
        slit_length = \
            calculate_slit_length_in_bins(self._master_trace[0]).value
        return traces, traces + slit_length, bf

    def get_bottom_edges(self, detector: int) -> np.ndarray:
        """
        Return an array of the bottom edges of identified orders for a given
        detector.

        Parameters
        ----------
        detector : int
            The detector number.

        Returns
        -------
        An array with shape (n_orders, n_spectral_pixels).
        """
        return self._edges[detector][0]

    def get_top_edges(self, detector: int) -> np.ndarray:
        """
        Return an array of the top edges of identified orders for a given
        detector.

        Parameters
        ----------
        detector : int
            The detector number.

        Returns
        -------
        An array with shape (n_orders, n_spectral_pixels).
        """
        return self._edges[detector][1]

    def order_mask(self, detector: int) -> np.ndarray:
        """
        Return a masking array of ones and NaNs to remove non-order pixels when
        displaying data.

        Parameters
        ----------
        detector : int
            The detector number.

        Returns
        -------
        An array with shape (n_orders, n_spectral_pixels).
        """
        return self._edges[detector][2]


def rectify_legacy_data(order_traces: OrderTraces,
                        image_data: list[CCDImage]) -> list[CCDImage]:
    """
    Rectify legacy data. Orders that extend off the detector are padded with
    NaNs.
    """
    lower_limit = np.nanmin(order_traces.get_bottom_edges(0))
    upper_limit = np.nanmax(order_traces.get_top_edges(0))
    n_rows = np.nanmax([upper_limit - lower_limit,
                        image_data[0].data.shape[1] - lower_limit])
    extended_image = np.full((n_rows, image_data[0].data.shape[1]),
                             fill_value=np.nan)
    extended_image[-lower_limit:-lower_limit + image_data[0].data.shape[0]] = \
        image_data[0].data
    bottom_edges = order_traces.get_bottom_edges(0) - lower_limit
    top_edges = order_traces.get_top_edges(0) - lower_limit
    slit_length = top_edges[0][0] - bottom_edges[0][0]
    n_orders = bottom_edges.shape[0]
    rectified_data = []
    for order in range(n_orders):
        rectified_image = np.full((slit_length, bottom_edges.shape[1]),
                                  fill_value=np.nan)
        for col in range(bottom_edges.shape[1]):
            order_slice = slice(bottom_edges[order, col],
                                top_edges[order, col])
            rectified_image[:, col] = extended_image[order_slice, col]
        anc = deepcopy(image_data[0].anc)
        anc['reductions_applied'].append('rectified')
        rectified_data.append(CCDImage(rectified_image, anc))
    return rectified_data


def rectify_mosaic_data(order_traces: OrderTraces,
                        image_data: list[CCDImage]) -> list[CCDImage]:
    """
    Rectify mosaic data. Orders that extend off the detector are padded with
    NaNs.
    """
    lower_limit = np.nanmin(order_traces.get_bottom_edges(0))
    upper_limit = (image_data[-1].anc['lower_left_corner'][0]
                   + image_data[-1].data.shape[0] - lower_limit
                   + np.max([0, np.nanmax(order_traces.get_top_edges(-1)
                                          - image_data[-1].data.shape[0])]))
    combined_image = np.full((upper_limit, image_data[-1].data.shape[1]),
                             fill_value=np.nan)
    for image in image_data:
        offset = image.anc['lower_left_corner'][0] - lower_limit
        combined_image[offset:offset + image.data.shape[0], :] = \
            image.data
    rectified_data = []
    for detector in range(3):
        offset = image_data[detector].anc['lower_left_corner'][0] - lower_limit
        bottom_edges = order_traces.get_bottom_edges(detector) + offset
        top_edges = order_traces.get_top_edges(detector) + offset
        slit_length = top_edges[0][0] - bottom_edges[0][0]
        n_orders = bottom_edges.shape[0]
        for order in range(n_orders):
            # check if order overlaps with last order from previous detector,
            # if it does then continue to the next order
            if (detector > 0) & (order == 0):
                old_offset = \
                    image_data[detector-1].anc['lower_left_corner'][0] \
                    - lower_limit
                old_bottom_edges = \
                    order_traces.get_bottom_edges(detector-1) + old_offset
                diff = np.median(bottom_edges[0] - old_bottom_edges[-1])
                if diff < slit_length:
                    continue
            rectified_image = np.full((slit_length, bottom_edges.shape[1]),
                                      fill_value=np.nan)
            for col in range(bottom_edges.shape[1]):
                order_slice = slice(bottom_edges[order, col],
                                    top_edges[order, col])
                rectified_image[:, col] = combined_image[order_slice, col]
            anc = deepcopy(image_data[detector].anc)
            anc['reductions_applied'].append('rectified')
            rectified_data.append(CCDImage(rectified_image, anc))
    return rectified_data


class RectifiedData:
    """
    This class rectifies a set of observations using order traces.
    """
    def __init__(self, order_traces: OrderTraces, data: list[list[CCDImage]]):
        """
        Parameters
        ----------
        order_traces : OrderTraces
            The top and bottom edges of orders identified from the flat field.
        data : list[list[CCDImage]]
            CCDImage objects in the nested-list shape (number of observations,
            number of  detectors).
        """
        self._order_traces = order_traces
        self._data = data
        self._rectified_data = self._get_rectified_data()

    def _get_rectified_data(self) -> list[list[CCDImage]]:
        """
        Get rectified data from a single data observation.
        """
        rectified_data = []
        n_observations, n_detectors = np.shape(self._data)
        for obs in range(n_observations):
            if n_detectors == 1:
                rectified_data.append(rectify_legacy_data(self._order_traces,
                                                          self._data[obs]))
            else:
                rectified_data.append(rectify_mosaic_data(self._order_traces,
                                                          self._data[obs]))
        return rectified_data

    @property
    def images(self) -> list[list[CCDImage]]:
        """
        Returns
        -------
        The rectified images as a nested list of CCDImage objects with the
        shape (number of observations, number of orders).
        """
        return self._rectified_data
