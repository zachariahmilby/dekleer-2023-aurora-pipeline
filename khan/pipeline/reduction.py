import time
from datetime import timedelta
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astroquery.jplhorizons import Horizons
import astropy.units as u

from khan.graphics import bias_cmap, flat_cmap, arc_cmap, data_cmap
from khan.pipeline.flux_calibration import correct_for_airmass_extinction
from khan.pipeline.images import get_flux_calibration_images, \
    get_guide_satellite_images, get_science_images, MasterBias, MasterFlat, \
    MasterArc, MasterTrace, CCDImage
from khan.pipeline.instrument_correction import clean_cosmic_rays, gain_correct
from khan.pipeline.quality_assurance import \
    make_master_calibration_image_quality_assurance_graphic, \
    make_order_trace_quality_assurance_graphic, \
    make_instrument_correction_quality_assurance_graphic, \
    make_wavelength_solution_quality_assurance_graphic
from khan.pipeline.rectification import OrderTraces, RectifiedData
from khan.pipeline.wavelength_calibration import WavelengthSolution


def instrument_artifact_correct(
        images: list[list[CCDImage]], master_bias: MasterBias,
        master_flat: MasterFlat, save_path: str,
        sub_directory: str, qa: bool = True) -> list[list[CCDImage]]:
    """
    Apply instrument-artifact corrections and save a quality assurance graphic.
    """
    corrected_images = []
    for detector_images in images:
        cosmic_ray_cleaned_images = clean_cosmic_rays(detector_images)
        bias_subtracted_images = \
            master_bias.subtract_bias(cosmic_ray_cleaned_images)
        flat_corrected_images = \
            master_flat.flat_field_correct(bias_subtracted_images)
        gain_corrected_images = gain_correct(flat_corrected_images)
        if qa:
            make_instrument_correction_quality_assurance_graphic(
                detector_images, cosmic_ray_cleaned_images,
                bias_subtracted_images, flat_corrected_images,
                gain_corrected_images, save_path, sub_directory)
        corrected_images.append(gain_corrected_images)
    return corrected_images


def save_flux_calibration_data(
        flux_calibration_images: list[list[CCDImage]], save_path: str) -> None:
    """
    Save reduced Jupiter flux calibration data to...ugh...a FITS file. I know,
    I'm sorry, too...
    """
    n_obs, n_orders = np.shape(flux_calibration_images)
    primary_data = []
    for obs in range(n_obs):
        order_data = []
        for order in range(n_orders):
            data = flux_calibration_images[obs][order].data
            exposure_time = \
                flux_calibration_images[obs][order].anc['exposure_time']
            order_data.append(data / exposure_time.value)
        primary_data.append(order_data)
    base_anc = flux_calibration_images[0][0].anc

    # calculate ephemeris quantities
    dates = [flux_calibration_images[obs][0].anc['date_time']
             for obs in range(n_obs)]
    epoch = Time(dates, format='isot', scale='utc').jd
    eph = Horizons(id='599', location='568',
                   epochs=epoch).ephemerides()
    distance = eph['r']

    primary = fits.PrimaryHDU(np.array(primary_data))
    header = primary.header
    header['NAXIS1'] = (header['NAXIS1'], 'number of spectral bins')
    header['NAXIS2'] = (header['NAXIS2'], 'number of spatial bins')
    header['NAXIS3'] = (header['NAXIS3'], 'number of echelle orders')
    header['NAXIS4'] = (header['NAXIS4'], 'number of observations')
    header.append(('TARGET', 'Jupiter', 'name of target body'))
    header.append(('BUNIT', 'electrons/second',
                   'physical units of primary extension'))
    header.append(('OBSERVER', base_anc['observers'],
                   'last names of observers'))
    header.append(('LAYOUT', base_anc['detector_layout'],
                   'detector layout (legacy or mosaic)'))
    header.append(('SLITLEN', base_anc['slit_length'].value,
                   'slit length [arcsec]'))
    header.append(('SLITWID', base_anc['slit_width'].value,
                   'slit width [arcsec]'))
    header.append(('XDNAME', base_anc['cross_disperser'],
                   'name of cross diserpser'))
    header.append(('XDANG', base_anc['cross_disperser_angle'].value,
                   'cross disperser angle [deg]'))
    header.append(('ECHANG', base_anc['echelle_angle'].value,
                   'echelle angle [deg]'))
    header.append(('SPABIN', base_anc['spatial_binning'].value,
                   'spatial binning [pix/bin]'))
    header.append(('SPEBIN', base_anc['spectral_binning'].value,
                   'spectral binning [pix/bin]'))
    header.append(('SPASCALE', base_anc['spatial_bin_scale'].value,
                   'spatial bin scale [arcsec/bin]'))
    header.append(('SPESCALE', base_anc['spectral_bin_scale'].value,
                   'spectral bin scale [arcsec/bin]'))
    header.append(('PIXWIDTH', base_anc['pixel_size'].value,
                   'pixel width [micron]'))
    for i, redux in enumerate(base_anc['reductions_applied']):
        header.append((f'REDUX{i:0>2}', redux,
                       'reduction applied to primary extension'))

    filenames = fits.Column(
        name='FILENAME', format='25A',
        array=[flux_calibration_images[obs][0].anc['file_name']
               for obs in range(n_obs)])
    exposure_times = fits.Column(
        name='EXPTIME', format='D', unit='seconds',
        array=[flux_calibration_images[obs][0].anc['exposure_time'].value
               for obs in range(n_obs)]
    )
    observation_dates = fits.Column(
        name='OBSDATE', format='19A',
        array=[flux_calibration_images[obs][0].anc['date_time']
               for obs in range(n_obs)]
    )
    airmasses = fits.Column(
        name='AIRMASS', format='E',
        array=[flux_calibration_images[obs][0].anc['airmass'].value
               for obs in range(n_obs)]
    )
    distances = fits.Column(
        name='DISTANCE', format='E', unit='au',
        array=distance
    )
    obs_table_hdu = fits.BinTableHDU.from_columns(
        [filenames, exposure_times, observation_dates, airmasses, distances],
        name='OBSERVATION_INFORMATION')
    header = obs_table_hdu.header
    header['TTYPE1'] = (header['TTYPE1'], 'original file name')
    header['TTYPE2'] = (header['TTYPE2'], 'exposure time')
    header['TTYPE3'] = (header['TTYPE3'], 'date at start of exposure')
    header['TTYPE4'] = (header['TTYPE4'], 'airmass')

    orders = np.array([flux_calibration_images[0][order].anc['order']
                       for order in range(n_orders)]).squeeze()
    ech_orders_hdu = fits.ImageHDU(orders, name='ECHELLE_ORDERS')
    wavelength_centers = np.array([flux_calibration_images[0][order].anc[
                                       'pixel_center_wavelengths'].value
                                   for order in range(n_orders)]).squeeze()
    wavelength_centers_hdu = fits.ImageHDU(wavelength_centers,
                                           name='BIN_CENTER_WAVELENGTHS')
    header = wavelength_centers_hdu.header
    header['NAXIS1'] = (header['NAXIS1'], 'number of spectral bin centers')
    header['NAXIS2'] = (header['NAXIS2'], 'number of echelle orders')
    header.append(('BUNIT', 'nm', 'wavelength physical unit'))
    wavelength_edges = np.array([flux_calibration_images[0][order].anc[
                                       'pixel_edge_wavelengths'].value
                                 for order in range(n_orders)]).squeeze()
    wavelength_edges_hdu = fits.ImageHDU(wavelength_edges,
                                         name='BIN_EDGE_WAVELENGTHS')
    header = wavelength_edges_hdu.header
    header['NAXIS1'] = (header['NAXIS1'], 'number of spectral bin edges')
    header['NAXIS2'] = (header['NAXIS2'], 'number of echelle orders')
    header.append(('BUNIT', 'nm', 'wavelength physical unit'))
    hdul = fits.HDUList([primary, wavelength_centers_hdu, wavelength_edges_hdu,
                         ech_orders_hdu, obs_table_hdu])
    file_name = Path(save_path, 'flux_calibration.fits.gz')
    if not file_name.parent.exists():
        file_name.parent.mkdir(parents=True)
    hdul.writeto(file_name, overwrite=True)


def save_science_target_data(
        science_target_images: list[list[CCDImage]], science_target_name: str,
        guide_satellite_images: list[list[CCDImage]],
        guide_satellite_name: str, save_path: str) -> None:
    """
    Save reduced science target and guide satellite calibration data to a FITS
    file.
    """

    # science target images
    n_obs, n_orders = np.shape(science_target_images)
    primary_data = []
    for obs in range(n_obs):
        order_data = []
        for order in range(n_orders):
            data = science_target_images[obs][order].data
            exposure_time = \
                science_target_images[obs][order].anc['exposure_time']
            order_data.append(data / exposure_time.value)
        primary_data.append(order_data)

    # calculate ephemeris quantities
    dates = [science_target_images[obs][0].anc['date_time']
             for obs in range(n_obs)]
    epoch = Time(dates, format='isot', scale='utc').jd
    ids = {'Jupiter': '599', 'Io': '501', 'Europa': '502',
           'Ganymede': '503', 'Callisto': '504'}
    eph = Horizons(id=ids[science_target_name], location='568',
                   epochs=epoch).ephemerides()
    velocities = eph['delta_rate'].to(u.m / u.s)
    radii = (eph['ang_width'] / 2) * u.arcsec

    base_anc = science_target_images[0][0].anc
    primary = fits.PrimaryHDU(np.array(primary_data))
    header = primary.header
    header['NAXIS1'] = (header['NAXIS1'], 'number of spectral bins')
    header['NAXIS2'] = (header['NAXIS2'], 'number of spatial bins')
    header['NAXIS3'] = (header['NAXIS3'], 'number of echelle orders')
    header['NAXIS4'] = (header['NAXIS4'], 'number of observations')
    header.append(('TARGET', science_target_name.capitalize(),
                   'name of science target body'))
    header.append(('BUNIT', 'electrons/second',
                   'physical units of primary extension'))
    header.append(('OBSERVER', base_anc['observers'],
                   'last names of observers'))
    header.append(('LAYOUT', base_anc['detector_layout'],
                   'detector layout (legacy or mosaic)'))
    header.append(('SLITLEN', base_anc['slit_length'].value,
                   'slit length [arcsec]'))
    header.append(('SLITWID', base_anc['slit_width'].value,
                   'slit width [arcsec]'))
    header.append(('XDNAME', base_anc['cross_disperser'],
                   'name of cross diserpser'))
    header.append(('XDANG', base_anc['cross_disperser_angle'].value,
                   'cross disperser angle [deg]'))
    header.append(('ECHANG', base_anc['echelle_angle'].value,
                   'echelle angle [deg]'))
    header.append(('SPABIN', base_anc['spatial_binning'].value,
                   'spatial binning [pix/bin]'))
    header.append(('SPEBIN', base_anc['spectral_binning'].value,
                   'spectral binning [pix/bin]'))
    header.append(('SPASCALE', base_anc['spatial_bin_scale'].value,
                   'spatial bin scale [arcsec/bin]'))
    header.append(('SPESCALE', base_anc['spectral_bin_scale'].value,
                   'spectral bin scale [arcsec/bin]'))
    header.append(('PIXWIDTH', base_anc['pixel_size'].value,
                   'pixel width [micron]'))
    for i, redux in enumerate(base_anc['reductions_applied']):
        header.append((f'REDUX{i:0>2}', redux,
                       'reduction applied to primary extension'))

    primary_filenames = fits.Column(
        name='FILENAME', format='25A',
        array=[science_target_images[obs][0].anc['file_name']
               for obs in range(n_obs)])
    primary_exposure_times = fits.Column(
        name='EXPTIME', format='D', unit='seconds',
        array=[science_target_images[obs][0].anc['exposure_time'].value
               for obs in range(n_obs)]
    )
    primary_observation_dates = fits.Column(
        name='OBSDATE', format='19A',
        array=[science_target_images[obs][0].anc['date_time']
               for obs in range(n_obs)]
    )
    primary_airmasses = fits.Column(
        name='AIRMASS', format='E',
        array=[science_target_images[obs][0].anc['airmass'].value
               for obs in range(n_obs)]
    )
    primary_angular_size = fits.Column(
        name='A_RADIUS', format='E', unit='arcseconds', array=radii
    )
    primary_relative_velocity = fits.Column(
        name='RELVLCTY', format='E', unit='meters/second', array=velocities
    )
    primary_obs_table_hdu = fits.BinTableHDU.from_columns(
        [primary_filenames, primary_exposure_times, primary_observation_dates,
         primary_airmasses, primary_angular_size, primary_relative_velocity],
        name='SCIENCE_OBSERVATION_INFORMATION')
    header = primary_obs_table_hdu.header
    header['TTYPE1'] = (header['TTYPE1'], 'original file name')
    header['TTYPE2'] = (header['TTYPE2'], 'exposure time')
    header['TTYPE3'] = (header['TTYPE3'], 'date at start of exposure')
    header['TTYPE4'] = (header['TTYPE4'], 'airmass')
    header['TTYPE5'] = (header['TTYPE5'], 'target apparent angular radius')
    header['TTYPE6'] = (header['TTYPE6'], 'target apparent relative velocity')

    # guide satellite image
    n_obs_guide, n_orders_guide = np.shape(guide_satellite_images)
    if n_obs_guide == 1:
        guide_satellite_images = [guide_satellite_images[0]] * n_obs
    elif n_obs_guide == n_obs:
        pass
    else:
        raise Exception('This algorithm can only handle the same number of '
                        'guide satellite observations as science target '
                        'observations or a single guide satellite observation.'
                        )
    guide_satellite_data = []
    for obs in range(n_obs):
        order_data = []
        for order in range(n_orders):
            data = guide_satellite_images[obs][order].data
            exposure_time = \
                guide_satellite_images[obs][order].anc['exposure_time']
            order_data.append(data / exposure_time.value)
        guide_satellite_data.append(order_data)
    guide_satellite_hdu = fits.ImageHDU(
        np.array(guide_satellite_data).squeeze(), name='GUIDE_SATELLITE')
    header = guide_satellite_hdu.header
    header['NAXIS1'] = (header['NAXIS1'], 'number of spectral bins')
    header['NAXIS2'] = (header['NAXIS2'], 'number of spatial bins')
    header['NAXIS3'] = (header['NAXIS3'], 'number of echelle orders')
    header['NAXIS4'] = (header['NAXIS4'], 'number of observations')
    header.append(('TARGET', guide_satellite_name.capitalize(),
                   'name of guide satellite body'))
    header.append(('BUNIT', 'electrons/second',
                   'physical units of primary extension'))
    guide_satellite_filenames = fits.Column(
        name='FILENAME', format='25A',
        array=[guide_satellite_images[obs][0].anc['file_name']
               for obs in range(n_obs)])
    guide_satellite_exposure_times = fits.Column(
        name='EXPTIME', format='D', unit='seconds',
        array=[guide_satellite_images[obs][0].anc['exposure_time'].value
               for obs in range(n_obs)]
    )
    guide_satellite_observation_dates = fits.Column(
        name='OBSDATE', format='19A',
        array=[guide_satellite_images[obs][0].anc['date_time']
               for obs in range(n_obs)]
    )
    guide_satellite_airmasses = fits.Column(
        name='AIRMASS', format='E',
        array=[guide_satellite_images[obs][0].anc['airmass'].value
               for obs in range(n_obs)]
    )
    guide_satellite_obs_table_hdu = fits.BinTableHDU.from_columns(
        [guide_satellite_filenames, guide_satellite_exposure_times,
         guide_satellite_observation_dates, guide_satellite_airmasses],
        name='GUIDE_SATELLITE_OBSERVATION_INFORMATION')
    header = guide_satellite_obs_table_hdu.header
    header['TTYPE1'] = (header['TTYPE1'], 'original file name')
    header['TTYPE2'] = (header['TTYPE2'], 'exposure time')
    header['TTYPE3'] = (header['TTYPE3'], 'date at start of exposure')
    header['TTYPE4'] = (header['TTYPE4'], 'airmass')

    orders = np.array([science_target_images[0][order].anc['order']
                       for order in range(n_orders)]).squeeze()
    ech_orders_hdu = fits.ImageHDU(orders, name='ECHELLE_ORDERS')
    wavelength_centers = np.array([science_target_images[0][order].anc[
                                       'pixel_center_wavelengths'].value
                                   for order in range(n_orders)]).squeeze()
    wavelength_centers_hdu = fits.ImageHDU(wavelength_centers,
                                           name='BIN_CENTER_WAVELENGTHS')
    header = wavelength_centers_hdu.header
    header['NAXIS1'] = (header['NAXIS1'], 'number of spectral bin centers')
    header['NAXIS2'] = (header['NAXIS2'], 'number of echelle orders')
    header.append(('BUNIT', 'nm', 'wavelength physical unit'))
    wavelength_edges = np.array([science_target_images[0][order].anc[
                                       'pixel_edge_wavelengths'].value
                                 for order in range(n_orders)]).squeeze()
    wavelength_edges_hdu = fits.ImageHDU(wavelength_edges,
                                         name='BIN_EDGE_WAVELENGTHS')
    header = wavelength_edges_hdu.header
    header['NAXIS1'] = (header['NAXIS1'], 'number of spectral bin edges')
    header['NAXIS2'] = (header['NAXIS2'], 'number of echelle orders')
    header.append(('BUNIT', 'nm', 'wavelength physical unit'))
    hdul = fits.HDUList([primary, guide_satellite_hdu, wavelength_centers_hdu,
                         wavelength_edges_hdu, ech_orders_hdu,
                         primary_obs_table_hdu, guide_satellite_obs_table_hdu])
    file_name = Path(save_path, 'science_observations.fits.gz')
    if not file_name.parent.exists():
        file_name.parent.mkdir(parents=True)
    hdul.writeto(file_name, overwrite=True)


def reduce_data(science_target_name: str, guide_satellite_name: str,
                source_data_path: str | Path, save_path: str | Path,
                quality_assurance: bool = True) -> None:
    """
    Wrapper function to reduce a set of Keck/HIRES Galilean satellite aurora
    observations.

    Parameters
    ----------
    science_target_name : str
        Name of science target, e.g., "Ganymede".
    guide_satellite_name : str
        Name of guide satellite, e.g., "Io".
    source_data_path : str
        The path to the directory containing the subdirectories ``bias``,
        ``flat``, ``arc``, ``trace``, ``flux_calibration``,
        ``guide_satellite`` and ``science``.
    save_path : str
        The location where you want the reduced data saved. The pipeline will
        save the reduced data products here and will create some
        sub-directories and fill them with quality-assurance graphics.
    quality_assurance : bool
        Whether or not to save quality-assurance graphics while running the
        pipeline. Matplotlib has memory leaks and sometimes this will cause a
        system exit if the memory pressure gets to be too high. I haven't found
        a way around this yet. I wouldn't try running multiple nights in a loop
        if I were you...
    """

    # store starting time
    print(f'Running data reduction pipeline on {source_data_path}:')
    starting_time = time.time()

    # make master bias frame
    print('   Making master bias...')
    master_bias = MasterBias(source_data_path)
    if quality_assurance:
        make_master_calibration_image_quality_assurance_graphic(
            master_bias, save_path, 'master_bias.pdf', bias_cmap())

    print('   Making master flat...')
    master_flat = MasterFlat(source_data_path, master_bias=master_bias)
    if quality_assurance:
        make_master_calibration_image_quality_assurance_graphic(
            master_flat, save_path, 'master_flat.pdf', flat_cmap(), cbar=False)

    print('   Making master arc...')
    master_arc = MasterArc(source_data_path, master_bias=master_bias)
    if quality_assurance:
        make_master_calibration_image_quality_assurance_graphic(
            master_arc, save_path, 'master_arc.pdf', arc_cmap())

    print('   Making master trace...')
    master_trace = MasterTrace(source_data_path, master_bias=master_bias)
    if quality_assurance:
        make_master_calibration_image_quality_assurance_graphic(
            master_trace, save_path, 'master_trace.pdf', data_cmap())

    # find order edges
    print('   Tracing order edges...')
    order_traces = OrderTraces(master_trace, master_flat)
    if quality_assurance:
        make_order_trace_quality_assurance_graphic(order_traces, master_flat,
                                                   save_path)

    # correct instrument artifacts
    print('   Removing instrument artifacts from flux calibration images...')
    flux_calibration_images = \
        instrument_artifact_correct(
            get_flux_calibration_images(source_data_path),
            master_bias=master_bias, master_flat=master_flat,
            save_path=save_path, sub_directory='flux_calibration',
            qa=quality_assurance)
    print('   Removing instrument artifacts from guide satellite images...')
    guide_satellite_images = \
        instrument_artifact_correct(
            get_guide_satellite_images(source_data_path),
            master_bias=master_bias, master_flat=master_flat,
            save_path=save_path, sub_directory='guide_satellite',
            qa=quality_assurance)
    print('   Removing instrument artifacts from science target images...')
    science_target_images = \
        instrument_artifact_correct(get_science_images(source_data_path),
                                    master_bias=master_bias,
                                    master_flat=master_flat,
                                    save_path=save_path,
                                    sub_directory='science',
                                    qa=quality_assurance)

    # rectify science data and arc lamp exposures
    print('   Rectifying data...')
    arc_lamp_images = RectifiedData(order_traces, [master_arc.images])
    flux_calibration_images = \
        RectifiedData(order_traces, flux_calibration_images)
    guide_satellite_images = \
        RectifiedData(order_traces, guide_satellite_images)
    science_target_images = \
        RectifiedData(order_traces, science_target_images)

    # wavelength solution calculation
    print('   Calculating wavelength solution...')
    wavelength_solution = WavelengthSolution(arc_lamp_images)
    if quality_assurance:
        make_wavelength_solution_quality_assurance_graphic(wavelength_solution,
                                                           save_path)

    # extract orders with wavelength solutions
    print('   Extracting orders with wavelength solutions...')
    flux_calibration_images = \
        wavelength_solution.select_orders_with_solutions(
            flux_calibration_images.images)
    guide_satellite_images = \
        wavelength_solution.select_orders_with_solutions(
            guide_satellite_images.images)
    science_target_images = \
        wavelength_solution.select_orders_with_solutions(
            science_target_images.images)

    # airmass extinction correction
    print('   Correcting for airmass extinction...')
    flux_calibration_images = \
        correct_for_airmass_extinction(flux_calibration_images)
    guide_satellite_images = \
        correct_for_airmass_extinction(guide_satellite_images)
    science_target_images = \
        correct_for_airmass_extinction(science_target_images)

    # save results
    print(f'   Saving reduced data to {save_path}...')
    save_flux_calibration_data(flux_calibration_images, save_path)
    save_science_target_data(science_target_images, science_target_name,
                             guide_satellite_images, guide_satellite_name,
                             save_path)

    # print completion message
    print(f'Processing complete, time elapsed '
          f'{timedelta(seconds=time.time() - starting_time)}.')
