import warnings
from copy import deepcopy
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits

from khan.pipeline.files import InstrumentCalibrationFiles, \
    FluxCalibrationFiles, GuideSatelliteFiles, ScienceFiles


def parse_mosaic_detector_slice(slice_string: str) -> tuple[slice, slice]:
    """
    Extract the Python slice which trims detector edges in the spatial
    dimension for mosaic data.
    """
    indices = np.array(slice_string.replace(':', ',').replace('[', '')
                       .replace(']', '').split(',')).astype(int)
    indices[[0, 2]] -= 1
    return slice(indices[0], indices[1], 1), slice(indices[2], indices[3], 1)


def determine_detector_layout(hdul: fits.HDUList) -> str:
    """
    If the HDUList has only one item, it's legacy data using the single
    detector. If not, then it is the three-detector mosaic arrangement.
    """
    if len(hdul) == 1:
        return 'legacy'
    elif len(hdul) == 4:
        return 'mosaic'
    else:
        raise Exception('Unknown detector layout!')


def get_mosaic_detector_corner_coordinates(image_header: fits.header.Header,
                                           binning: np.ndarray) -> np.ndarray:
    """
    Determine the relative physical coordinate of a mosaic detector's lower
    left corner. These coordinates let you replicate the actual physical layout
    of the detectors including the physical separation between them.
    """
    n_rows, n_columns = image_header['CRVAL1G'], image_header['CRVAL2G']
    spatial_coordinate = \
        np.abs(np.ceil(2048 - n_rows - 1) / binning[0]).astype(int)
    spectral_coordinate = np.ceil(n_columns - 1).astype(int)
    return np.array([spatial_coordinate, spectral_coordinate])


def reformat_observers(observer_names: str):
    """
    Reformat observer names so they are separated by commas.
    """
    return ', '.join([name.strip()
                      for name in observer_names.replace('/', ',').split(',')])


class CCDImage:
    """
    This class holds a HIRES CCD image and ancillary information about it.
    """

    def __init__(self, image_data: np.ndarray, anc: dict):
        """
        Parameters
        ----------
        image_data : numpy.ndarray
            The actual image data.
        anc : dict
            A dictionary containing all relevant ancillary information, like
            original file name, exposure time, gain, etc.
        """
        self._data = image_data
        self._anc = anc

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def anc(self) -> dict:
        return self._anc


def get_legacy_data(file_path: Path) -> list[CCDImage]:
    """
    Retrieve the CCD image and ancillary metadata from a HIRES single-detector
    legacy observation.
    """
    with fits.open(file_path) as hdul:
        header = hdul[0].header
        binning = np.array(header['BINNING'].split(',')).astype(int)
        slice0 = header['PREPIX']
        slice1 = header['NAXIS1'] - header['POSTPIX']
        detector_image = hdul[0].data[:, slice0:slice1]
        anc = {'file_name': file_path.name,
               'date_time': header['DATE'],
               'observers': reformat_observers(header['OBSERVER']),
               'detector_layout': 'legacy',
               'unit': u.adu,
               'exposure_time': header['EXPTIME'] * u.second,
               'airmass': float(header['AIRMASS']) * u.def_unit('airmass'),
               'gain': header['CCDGN01'] * u.electron / u.adu,
               'read_noise': header['CCDRN01'] * u.electron,
               'slit_length': header['SLITLEN'] * u.arcsec,
               'slit_width': header['SLITWIDT'] * u.arcsec,
               'cross_disperser': header['XDISPERS'].lower(),
               'cross_disperser_angle':
                   np.round(header['XDANGL'], 5) * u.degree,
               'echelle_angle': np.round(header['ECHANGL'], 5) * u.degree,
               'spatial_binning': (binning[0] * u.pixel / u.bin).astype(int),
               'spatial_bin_scale': header['SPATSCAL'] * u.arcsec / u.bin,
               'spectral_binning': (binning[1] * u.pixel / u.bin).astype(int),
               'spectral_bin_scale': header['DISPSCAL'] * u.arcsec / u.bin,
               'pixel_size': 24 * u.micron,
               'reductions_applied': []}
    return [CCDImage(detector_image, anc)]


def get_mosaic_data(file_path: Path) -> list[CCDImage]:
    """
    Retrieve the CCD image and ancillary metadata from a HIRES mosaic-detector
    observation.
    """
    images = []
    with fits.open(file_path) as hdul:
        header = hdul[0].header
        binning = np.array(header['BINNING'].split(',')).astype(int)
        for detector_number in range(1, 4):
            image_header = hdul[detector_number].header
            detector_slice = parse_mosaic_detector_slice(
                image_header['DATASEC'])
            detector_image = np.flipud(
                hdul[detector_number].data.T[detector_slice])
            anc = {'file_name': file_path.name,
                   'date_time': header['DATE_BEG'],
                   'observers': reformat_observers(header['OBSERVER']),
                   'detector_layout': 'mosaic',
                   'detector_number': detector_number - 1,
                   'lower_left_corner':
                       get_mosaic_detector_corner_coordinates(image_header,
                                                              binning),
                   'unit': u.adu,
                   'exposure_time': header['EXPTIME'] * u.second,
                   'airmass': float(header['AIRMASS']) * u.def_unit('airmass'),
                   'gain':
                       header[f'CCDGN0{detector_number}'] * u.electron / u.adu,
                   'read_noise':
                       header[f'CCDRN0{detector_number}'] * u.electron,
                   'slit_length': header['SLITLEN'] * u.arcsec,
                   'slit_width': header['SLITWIDT'] * u.arcsec,
                   'cross_disperser': header['XDISPERS'].lower(),
                   'cross_disperser_angle':
                       np.round(header['XDANGL'], 5) * u.degree,
                   'echelle_angle': np.round(header['ECHANGL'], 5) * u.degree,
                   'spatial_binning':
                       (binning[0] * u.pixel / u.bin).astype(int),
                   'spatial_bin_scale': header['SPATSCAL'] * u.arcsec / u.bin,
                   'spectral_binning':
                       (binning[1] * u.pixel / u.bin).astype(int),
                   'spectral_bin_scale': header['DISPSCAL'] * u.arcsec / u.bin,
                   'pixel_size': 15 * u.micron,
                   'reductions_applied': []}
            images.append(CCDImage(detector_image, anc))
    return images


def get_instrument_calibration_images(directory: str,
                                      file_type: str) -> list[list[CCDImage]]:
    """
    Retrieve a nested list of instrument calibration images (bias, flat, arc or
    trace).
    """
    try:
        return [get_legacy_data(file_path) for file_path in
                InstrumentCalibrationFiles(directory, file_type).file_paths]
    except KeyError:
        return [get_mosaic_data(file_path) for file_path in
                InstrumentCalibrationFiles(directory, file_type).file_paths]


def get_flux_calibration_images(directory: str) -> list[list[CCDImage]]:
    """
    Retrieve a nested list of Jupiter meridian flux calibration files.
    """
    try:
        return [get_legacy_data(file_path) for file_path in
                FluxCalibrationFiles(directory).file_paths]
    except KeyError:
        return [get_mosaic_data(file_path) for file_path in
                FluxCalibrationFiles(directory).file_paths]


def get_guide_satellite_images(directory: str) -> list[list[CCDImage]]:
    """
    Retrieve a nested list of guide satellite observations.
    """
    try:
        return [get_legacy_data(file_path) for file_path in
                GuideSatelliteFiles(directory).file_paths]
    except KeyError:
        return [get_mosaic_data(file_path) for file_path in
                GuideSatelliteFiles(directory).file_paths]


def get_science_images(directory: str) -> list[list[CCDImage]]:
    """
    Retrieve a nested list of Galilean satellite science observations.
    """
    try:
        return [get_legacy_data(file_path) for file_path in
                ScienceFiles(directory).file_paths]
    except KeyError:
        return [get_mosaic_data(file_path) for file_path in
                ScienceFiles(directory).file_paths]


def median_image(images: list[list[CCDImage]]) -> list[CCDImage]:
    """
    Calculate the median of a set of CCD images along the 0th-axis. Also
    modifies the resulting ancillary dictionary to remove the original file
    names, dates/times, exposure times and airmasses. Stores a record in the
    ``reductions_applied`` list in the ancillary dictionary.
    """
    _, n_detectors = np.shape(images)
    image_data = np.array([[image[detector].data
                            for detector in range(n_detectors)]
                           for image in images])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        median_images = np.nanmedian(image_data, axis=0)
    new_median_images = []
    for detector in range(n_detectors):
        new_anc = deepcopy(images[0][detector].anc)
        new_anc['file_name'] = None
        new_anc['date_time'] = None
        new_anc['exposure_time'] = None
        new_anc['airmass'] = None
        new_anc['reductions_applied'].append('median')
        new_median_images.append(CCDImage(median_images[detector, :, :],
                                          new_anc))
    return new_median_images


def normalize_image(images: list[CCDImage]) -> list[CCDImage]:
    """
    Normalize a set of median instrument calibration images. Stores a record in
    the ``reductions_applied`` list in the ancillary dictionary.
    """
    n_detectors = len(images)
    normalized_images = []
    for detector in range(n_detectors):
        image = images[detector].data
        normalized_image = image / np.nanmax(image)
        normalized_image[np.where(normalized_image <= 0)] = 1
        new_anc = deepcopy(images[detector].anc)
        new_anc['unit'] = u.dimensionless_unscaled
        new_anc['reductions_applied'].append('normalize')
        normalized_images.append(CCDImage(normalized_image, new_anc))
    return normalized_images


def set_zeros_to_nan(images: list[CCDImage]) -> list[CCDImage]:
    """
    Set saturated pixels in a a set of images to NaN. For HIRES data, saturated
    pixels seem to become zeros (see saturated arc lamp exposures).
    """
    n_detectors = len(images)
    nan_images = []
    for detector in range(n_detectors):
        nan_index = np.where(images[detector].data == 0)
        new_image = images[detector].data.astype(float)
        new_image[nan_index] = np.nan
        new_anc = deepcopy(images[detector].anc)
        new_anc['reductions_applied'].append('remove_saturated_bins')
        nan_images.append(CCDImage(new_image, new_anc))
    return nan_images


def remove_outliers(images: list[CCDImage]) -> list[CCDImage]:
    """
    Calculates the 1st and 99th percentile values for a set of detector images,
    then sets any values below the 1st percentile to that value and any values
    above the 99th percentile to that value.
    """
    n_detectors = len(images)
    new_images = []
    for detector in range(n_detectors):
        new_image = images[detector].data
        minimum_value = np.percentile(new_image, 1)
        maximum_value = np.percentile(new_image, 99)
        new_image[np.where(new_image < minimum_value)] = minimum_value
        new_image[np.where(new_image > maximum_value)] = maximum_value
        new_anc = deepcopy(images[detector].anc)
        new_anc['reductions_applied'].append('remove_outliers')
        new_images.append(CCDImage(new_image, new_anc))
    return new_images


class MasterBias:
    """
    This class holds a master bias image (or set of images for mosaic detector
    data). The master bias images can either be accessed as a list through the
    "images" property, or by directly indexing the object, e.g., MasterBias[0].
    """

    def __init__(self, directory: str):
        """
        Parameters
        ----------
        directory : str
            Absolute path to the parent directory containing the ``bias``
            directory.
        """
        self._master_bias = self._make_master_bias(directory)

    def __getitem__(self, indices):
        """
        Allows the MasterBias object to be indexed.
        """
        return self._master_bias[indices]

    @staticmethod
    def _make_master_bias(directory: str) -> list[CCDImage]:
        """
        Retrieve the nested list of bias image data and get the median images
        for each detector.
        """
        images = get_instrument_calibration_images(directory, 'bias')
        return median_image(images)

    def subtract_bias(self, images: list[CCDImage]) -> list[CCDImage]:
        """
        Built-in method to subtract the master bias from a list of detector
        images, either 1 or 3 depending on the detector layout. This doesn't
        work if passing a nested list for multiple observations, so just pass
        one single observation at a time. Stores a record in the
        ``reductions_applied`` list in the ancillary dictionary.

        Parameters
        ----------
        images : list[CCDImage]
            A set of detector images from which you want the bias subtracted.

        Returns
        -------
        A list of detector images with the bias subtracted.
        """
        new_images = []
        for detector, image in enumerate(images):
            new_image = image.data - self._master_bias[detector].data
            new_anc = deepcopy(image.anc)
            new_anc['reductions_applied'].append('bias_subtracted')
            new_images.append(CCDImage(new_image, new_anc))
        return new_images

    @property
    def n_detectors(self) -> int:
        """
        Returns
        -------
        1 if legacy data or 3 if mosaic data.
        """
        return np.shape(self._master_bias)[0]

    @property
    def images(self):
        """
        Returns
        -------
        The master bias data as a list of CCDImage objects, one per detector.
        """
        return self._master_bias


class MasterFlat:
    """
    This class holds a master flat image (or set of images for mosaic detector
    data). The master flat images can either be accessed as a list through the
    "images" property, or by directly indexing the object, e.g., MasterFlat[0].
    """

    def __init__(self, directory: str, master_bias: MasterBias):
        """
        Parameters
        ----------
        directory : str
            Absolute path to the parent directory containing the ``flat``
            directory.
        master_bias : MasterBias
            A MasterBias object so the master flat images can have the bias
            subtracted from them.
        """
        self._master_flat = self._make_master_flat(directory, master_bias)

    def __getitem__(self, indices):
        """
        Allows the MasterFlat object to be indexed.
        """
        return self._master_flat[indices]

    @staticmethod
    def _make_master_flat(directory: str,
                          master_bias: MasterBias) -> list[CCDImage]:
        """
        Retrieve the nested list of flat image data, get the median images for
        each detector, subtract the master bias and normalize.
        """
        images = get_instrument_calibration_images(directory, 'flat')
        median_images = median_image(images)
        bias_subtracted_images = master_bias.subtract_bias(median_images)
        return normalize_image(bias_subtracted_images)

    def flat_field_correct(self, images: list[CCDImage]):
        """
        Built-in method to flat-field correct a list of detector images, either
        1 or 3  depending on the detector layout. This doesn't work if passing
        a nested list for multiple observations, so just pass one single
        observation at a time. Stores a record in the ``reductions_applied``
        list in the ancillary dictionary.

        Parameters
        ----------
        images : list[CCDImage]
            A set of detector images you want to flat-field correct.

        Returns
        -------
        A list of flat-field-corrected detector images.
        """
        new_images = []
        for detector, image in enumerate(images):
            new_image = image.data / self._master_flat[detector].data
            new_anc = deepcopy(image.anc)
            new_anc['reductions_applied'].append('flat_field_corrected')
            new_images.append(CCDImage(new_image, new_anc))
        return new_images

    @property
    def n_detectors(self) -> int:
        """
        Returns
        -------
        1 if legacy data or 3 if mosaic data.
        """
        return np.shape(self._master_flat)[0]

    @property
    def images(self):
        """
        Returns
        -------
        The master bias data as a list of CCDImage objects, one per detector.
        """
        return self._master_flat


class MasterArc:
    """
    This class holds a master arc image (or set of images for mosaic detector
    data). The master arc images can either be accessed as a list through the
    "images" property, or by directly indexing the object, e.g., MasterArc[0].
    """

    def __init__(self, directory: str, master_bias: MasterBias):
        """
        Note: the master arc image doesn't really need flat-fielding, so I'm
        skipping it.

        Parameters
        ----------
        directory : str
            Absolute path to the parent directory containing the ``arc``
            directory.
        master_bias : MasterBias
            A MasterBias object so the master arc images can have the bias
            subtracted from them.
        """
        self._master_arc = self._make_master_arc(directory, master_bias)

    def __getitem__(self, indices):
        """
        Allows the MasterFlat object to be indexed.
        """
        return self._master_arc[indices]

    @staticmethod
    def _make_master_arc(directory: str,
                         master_bias: MasterBias) -> list[CCDImage]:
        """
        Retrieve the nested list of arc image data, get the median images for
        each detector and subtract the master bias.
        """
        images = get_instrument_calibration_images(directory, 'arc')
        desaturated_images = [set_zeros_to_nan(images) for images in images]
        median_images = median_image(desaturated_images)
        return master_bias.subtract_bias(median_images)

    @property
    def n_detectors(self) -> int:
        """
        Returns
        -------
        1 if legacy data or 3 if mosaic data.
        """
        return np.shape(self._master_arc)[0]

    @property
    def images(self):
        """
        Returns
        -------
        The master bias data as a list of CCDImage objects, one per detector.
        """
        return self._master_arc


class MasterTrace:
    """
    This class holds a master trace image (or set of images for mosaic detector
    data). The master trace images can either be accessed as a list through the
    "images" property, or by directly indexing the object, e.g.,
    MasterTrace[0].
    """

    def __init__(self, directory: str, master_bias: MasterBias):
        """
        Note: the master arc image doesn't really need flat-fielding, so I'm
        skipping it.

        Parameters
        ----------
        directory : str
            Absolute path to the parent directory containing the ``arc``
            directory.
        master_bias : MasterBias
            A MasterBias object so the master arc images can have the bias
            subtracted from them.
        """
        self._master_trace = self._make_master_trace(directory, master_bias)

    def __getitem__(self, indices):
        """
        Allows the MasterFlat object to be indexed.
        """
        return self._master_trace[indices]

    @staticmethod
    def _make_master_trace(directory: str,
                           master_bias: MasterBias) -> list[CCDImage]:
        """
        Retrieve the nested list of trace image data, get the median images for
        each detector and subtract the master bias.
        """
        images = get_instrument_calibration_images(directory, 'trace')[0]
        return master_bias.subtract_bias(images)

    @property
    def n_detectors(self) -> int:
        """
        Returns
        -------
        1 if legacy data or 3 if mosaic data.
        """
        return np.shape(self._master_trace)[0]

    @property
    def images(self):
        """
        Returns
        -------
        The master bias data as a list of CCDImage objects, one per detector.
        """
        return self._master_trace
