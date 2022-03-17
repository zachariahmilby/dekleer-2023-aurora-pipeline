from copy import deepcopy

import astropy.units as u
from ccdproc import cosmicray_lacosmic

from khan.pipeline.images import CCDImage


def clean_cosmic_rays(images: list[CCDImage]) -> list[CCDImage]:
    """
    Identify and remove cosmic rays using the L.A.Cosmic algorithm.
    """
    clean_images = []
    for image in images:
        clean_image, _ = cosmicray_lacosmic(image.data, gain=image.anc['gain'],
                                            readnoise=image.anc['read_noise'],
                                            gain_apply=False)
        anc = deepcopy(image.anc)
        anc['reductions_applied'].append('cosmic_rays_removed')
        clean_images.append(CCDImage(clean_image, anc))
    return clean_images


def gain_correct(images: list[CCDImage]) -> list[CCDImage]:
    """
    Apply gain correction to convert final science images from ADU to
    electrons.
    """
    corrected_images = []
    for image in images:
        gain_corrected_image = image.data * image.anc['gain'].value
        anc = deepcopy(image.anc)
        anc['unit'] = u.electron
        anc['reductions_applied'].append('gain_corrected')
        corrected_images.append(CCDImage(gain_corrected_image, anc))
    return corrected_images
