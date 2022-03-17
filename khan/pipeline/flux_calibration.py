from copy import deepcopy

import numpy as np

from khan.common import get_mauna_kea_summit_extinction
from khan.pipeline.images import CCDImage


def correct_for_airmass_extinction(data: list[list[CCDImage]]) \
        -> list[list[CCDImage]]:
    """
    Apply the wavelength- and airmass-dependent flux correction. For a
    complete description of the data used, see the function
    `khan.pipeline.files.get_mauna_kea_summit_extinction`.

    Parameters
    ----------
    data : list[list[CCDImage]]
        A set of rectified data. Can be before or after cosmic ray removal.

    Returns
    -------
    The data with flux counts corrected for airmass extinction.
    """
    calibration_data = get_mauna_kea_summit_extinction(value=True)
    extinction_corrected_frames = []
    n_observations, n_orders = np.shape(data)
    for obs in range(n_observations):
        extinction_corrected_orders = []
        for order in range(n_orders):
            image = data[obs][order].data
            anc = deepcopy(data[obs][order].anc)
            if 'pixel_center_wavelengths' not in anc.keys():
                raise Exception('No wavelength solution in the ancillary data '
                                'dictionary. Make sure you run '
                                '`select_orders_with_solutions` on these data '
                                'first!')
            wavelengths = anc['pixel_center_wavelengths'].value
            interp_extinction = np.interp(wavelengths,
                                          calibration_data['wavelength'],
                                          calibration_data['extinction'])
            factor = anc['airmass'].value / 100 ** (1 / 5)
            extinction = np.tile(10 ** (interp_extinction * factor),
                                 (image.shape[0], 1))
            anc['reductions_applied'].append('airmass_ext_corrected')
            extinction_corrected_data = image * extinction
            extinction_corrected_orders.append(
                CCDImage(extinction_corrected_data, anc))
        extinction_corrected_frames.append(extinction_corrected_orders)
    return extinction_corrected_frames
