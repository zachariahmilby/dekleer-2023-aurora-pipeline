from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from astroquery.jplhorizons import Horizons
import warnings
from tqdm import tqdm

from khan.common import jovian_naif_codes


class FilesDirectory:
    """
    This class makes a graphical quicklook product for each of the raw data
    files. It also makes a summary spreadsheet in CSV format for a brief
    overview of the data in each of the observation night directories.
    """
    def __init__(self, directory: str or Path, target: str):
        """
        Parameters
        ----------
        directory : str or Path
            Path to directory containing FITS files in '.fits.gz' format.
        target : str
            Satellite target name: Io, Europa, Ganymede or Callisto.
        """
        self._directory = Path(directory).resolve()
        self._target = target

    @staticmethod
    def _get_ephemerides(date: str, target: str):
        """
        Get ephemeris information for the target satellite.
        """
        target_ids = jovian_naif_codes()
        epoch = Time(date, format='isot', scale='utc').jd
        obj = Horizons(id=target_ids[target], location='568',
                       epochs=epoch)
        return obj.ephemerides()
                
    @staticmethod
    def _make_info_text_block(data):
        """
        Make strings of ancillary information for the graphic.
        """
        pixel_value_range = data["Pixel Value Range"].split(', ')
        left_column = fr'$\bf{{File\ Name\!:}}$ '
        left_column += fr'$\tt{{{data["Filename"]}}}$' + '\n'
        left_column += fr'$\bf{{Observation\ Date\!:}}$ '
        left_column += fr'{data["Observation Date"]} UTC' + '\n'
        left_column += fr'$\bf{{Object\ (Target)\!:}}$ ' \
                       + data["Object (Target)"].replace('_', r'\_') + '\n'
        left_column += fr'$\bf{{Exposure\ Time\!:}}$ ' \
                       fr'{data["Exposure Time [s]"]:.2f} seconds' + '\n'
        left_column += fr'$\bf{{Target\ Visibility\ Code\!:}}$ ' \
                       fr'{data["Target Visibility Code"]}' + '\n'
        left_column += fr'$\bf{{Airmass\!:}}$ ' \
                       fr'{data["Airmass"]}' + '\n'
        left_column += fr'$\bf{{Pixel\ Value\ Range\!:}}$ ' \
                       fr'{int(pixel_value_range[0]):,} to ' \
                       fr'{int(pixel_value_range[1]):,}' + '\n'
        left_column += fr'$\bf{{Observers\!:}}$ {data["Observers"]}'

        right_column = fr'$\bf{{Observation\ Type\!:}}$ ' \
                       fr'{data["Observation Type"]}' + '\n'
        right_column += fr'$\bf{{Decker\!:}}$ {data["Decker"]}' + '\n'
        right_column += fr'$\bf{{Lamp\!:}}$ {data["Lamp"]}' + '\n'
        right_column += fr'$\bf{{Filter\ 1\!:}}$ {data["Filter 1"]}' + '\n'
        right_column += fr'$\bf{{Filter\ 2\!:}}$ {data["Filter 2"]}' + '\n'
        right_column += fr'$\bf{{Binning\!:}}$ {data["Binning"]}' + '\n'
        right_column += fr'$\bf{{Echelle\ Angle\!:}}$ '
        right_column += fr'${data["Echelle Angle [deg]"]:.5f}\degree$' + '\n'
        right_column += fr'$\bf{{Cross\ Disperser\ Angle:\!}}$ '
        right_column += fr'${data["Cross Disperser Angle [deg]"]:.4f}\degree$'

        return left_column, right_column

    def _save_quicklook(self, image: np.ndarray, data: dict,
                        subdirectory: str = ''):
        """
        Save graphic quicklook to file.
        """
        left_column, right_column = self._make_info_text_block(data)
        fig, axes = plt.subplots(2, 1, figsize=(6, 4),
                                 gridspec_kw={'height_ratios': [1, 3]},
                                 constrained_layout=True)
        axes[1].pcolormesh(
            image, norm=colors.LogNorm(vmin=np.nanpercentile(image, 1),
                                       vmax=np.nanpercentile(image, 99)),
            rasterized=True)
        [ax.set_xticks([]) for ax in axes]
        [ax.set_yticks([]) for ax in axes]
        [ax.set_frame_on(False) for ax in axes]
        axes[0].text(0, 1, left_column, ha='left', va='top',
                     transform=axes[0].transAxes, linespacing=1.5)
        axes[0].text(0.5, 1, right_column, ha='left', va='top',
                     transform=axes[0].transAxes, linespacing=1.5)
        fig.canvas.draw()
        savepath = Path(self._directory, subdirectory,
                        data['Filename'].replace('fits.gz', 'png'))
        plt.savefig(savepath)
        plt.close(fig)


class RawFiles(FilesDirectory):

    def _get_files(self):
        """
        Get a list of the FITS files.
        """
        return sorted(self._directory.glob('*.fits.gz'))

    def _get_information(self, hdul: fits.HDUList):
        """
        Return a dictionary containing all of the necessary ancillary
        information and the detector images appropriately rotated and stacked.
        """
        eph = self._get_ephemerides(hdul[0].header['date'], self._target)
        header = hdul['PRIMARY'].header
        try:  # under a different name in the 1998 data
            obstype = header['obstype']
        except KeyError:
            obstype = header['imagetyp']
        try:
            image = np.flipud(np.vstack(
                (hdul[3].data.T, hdul[2].data.T, hdul[1].data.T)))
        except IndexError:  # for the 1998 single-detector images
            image = hdul[0].data
        minimum_scale_value = np.percentile(image, 1)
        maximum_scale_value = np.percentile(image, 99)
        file_name = Path(hdul.filename()).name
        data = {'Filename': file_name,
                'Observation Date': hdul[0].header['date'].replace('T', ' '),
                'Object (Target)': header['object'],
                'Exposure Time [s]': np.round(header['exptime'], 2),
                'Target Visibility Code': eph['sat_vis'][0],
                'Pixel Value Range': f'{int(minimum_scale_value)}, '
                                     f'{int(maximum_scale_value)}',
                'Observers': header['observer'],
                'Observation Type': obstype,
                'Decker': header['deckname'],
                'Lamp': header['lampname'],
                'Filter 1': header['fil1name'],
                'Filter 2': header['fil2name'],
                'Binning': header['binning'],
                'Airmass': header['airmass'],
                'Echelle Angle [deg]': np.round(header['echangl'], 5),
                'Cross Disperser Angle [deg]': np.round(header['xdangl'], 4),
                'Pixel Scale [arcsec/pixel]': header['dispscal'],
                'Target Angular Size [arcsec]': eph['ang_width'][0],
                'Target Relative Velocity [km/s]': eph['delta_rate'][0],
                }
        return data, image

    def process_files(self):
        """
        Wrapper function to make the quicklooks and save the summary CSV.
        """
        print('Processing raw data files...')
        df = pd.DataFrame()
        for file in tqdm(self._get_files()):
            with fits.open(file) as hdul:
                data, image = self._get_information(hdul)
                self._save_quicklook(image, data)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = df.append(data, ignore_index=True)
        savepath = Path(self._directory, 'file_information.csv')
        df.to_csv(savepath, index=False)


class SelectedFiles(FilesDirectory):
    """
    This class looks at each of the manually selected files (sorted into file
    types of bias, flat, arc, calibration, trace and science) and produces
    another summary spreadsheet with metadata relevant to each of the files.
    """

    def _get_files(self):
        """
        Get a list of the FITS files.
        """
        return sorted(self._directory.glob('*/*.fits.gz'))

    def _get_filetypes(self):
        """
        Get the names of the subdirectories.
        """
        return [filepath.parent.name for filepath in self._get_files()]

    def _get_information(self, hdul: fits.HDUList, filetype: str):
        """
        Return a dictionary containing all of the necessary ancillary
        information and the detector images appropriately rotated and stacked.
        """
        eph = self._get_ephemerides(hdul[0].header['date'], self._target)
        jupiter_eph = self._get_ephemerides(hdul[0].header['date'], 'Jupiter')
        header = hdul['PRIMARY'].header
        try:  # under a different name in the 1998 data
            obstype = header['obstype']
        except KeyError:
            obstype = header['imagetyp']
        try:
            image = np.flipud(np.vstack(
                (hdul[3].data.T, hdul[2].data.T, hdul[1].data.T)))
        except IndexError:  # for the 1998 single-detector images
            image = hdul[0].data
        minimum_scale_value = np.percentile(image, 1)
        maximum_scale_value = np.percentile(image, 99)
        file_name = Path(hdul.filename()).name
        date = hdul[0].header['date']
        data = {'Filename': file_name,
                'File Type': filetype,
                'Julian Date': Time(date, format='isot', scale='utc').jd,
                'Observation Date': date.replace('T', ' '),
                'Object (Target)': header['object'],
                'Exposure Time [s]': np.round(header['exptime'], 2),
                'Target Visibility Code': eph['sat_vis'][0],
                'Pixel Value Range': f'{int(minimum_scale_value)}, '
                                     f'{int(maximum_scale_value)}',
                'Observers': header['observer'],
                'Observation Type': obstype,
                'Decker': header['deckname'],
                'Lamp': header['lampname'],
                'Filter 1': header['fil1name'],
                'Filter 2': header['fil2name'],
                'Binning': header['binning'],
                'Airmass': header['airmass'],
                'Echelle Angle [deg]': np.round(header['echangl'], 5),
                'Cross Disperser Angle [deg]': np.round(header['xdangl'], 4),
                'Pixel Scale [arcsec/pixel]': header['dispscal'],
                'Target Angular Size [arcsec]': eph['ang_width'][0],
                'Target Relative Velocity [km/s]': eph['delta_rate'][0],
                'Target Sub-Observer Latitude [deg]': eph['PDObsLat'][0],
                'Target Sub-Observer Longitude [deg]': eph['PDObsLon'][0],
                'Target North Pole Angle from Disk Center [deg]':
                    eph['NPole_ang'][0],
                'Target North Pole Distance from Disk Center [arcsec]':
                    eph['NPole_dist'][0],
                'Jupiter Relative Velocity [km/s]':
                    jupiter_eph['delta_rate'][0],
                }
        return data, image

    def _check_if_quicklook_exists(self, filetype, hdul: fits.HDUList):
        file = Path(self._directory, filetype,
                    Path(hdul.filename().replace('.fits.gz', '.png')).name)
        return file.exists()

    def _make_observation_planning_graphic(self, date: Time):
        from obsplan import EclipsePrediction
        date = Time(date, format='isot', scale='utc')
        start_time = date - 1 * u.day
        end_time = date + 1 * u.day
        eclipse_prediction = EclipsePrediction(
            starting_datetime=start_time.to_value('iso', subfmt='date'),
            ending_datetime=end_time.to_value('iso', subfmt='date'),
            target=self._target)
        eclipse_prediction.save_summary_graphics(str(self._directory))

    def process_files(self, make_planning_graphic: bool = False):
        """
        Wrapper function to make the quicklooks and save the summary CSV.
        """
        print('Processing selected data files...')
        df = pd.DataFrame()
        files = self._get_files()
        filetypes = self._get_filetypes()
        for i in tqdm(range(len(files))):
            with fits.open(files[i]) as hdul:
                data, image = self._get_information(hdul, filetypes[i])
                if not self._check_if_quicklook_exists(filetypes[i], hdul):
                    self._save_quicklook(image, data,
                                         subdirectory=filetypes[i])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = df.append(data, ignore_index=True)
        savepath = Path(self._directory, 'file_information.csv')
        df.sort_values('Julian Date', inplace=True)
        df.to_csv(savepath, index=False)
        if make_planning_graphic:
            with fits.open(files[0]) as hdul:
                self._make_observation_planning_graphic(hdul[0].header['date'])


class InstrumentCalibrationFiles:
    """
    This object holds file paths for instrument calibration files: bias, flat,
    arc or trace.
    """

    def __init__(self, directory: str, file_type: str):
        """
        Parameters
        ----------
        directory : str
            Absolute path to the parent directory containing the ``bias``,
            ``flat``, ``arc`` and ``trace`` sub-directories.
        file_type : str
            The type of instrument calibration file: ``bias``, ``flat``,
            ``arc`` and ``trace``.
        """
        self._file_paths = Path(directory).glob(f'{file_type}/*.fits*')

    def __str__(self):
        files = sorted(self._file_paths)
        print_string = f'Directory:\n   {files[0].parent}\n'
        print_string += 'Files: \n'
        print_string += '\n'.join([f'   {file.name}' for file in files])
        return print_string

    @property
    def file_paths(self) -> list[Path]:
        return sorted(self._file_paths)


class FluxCalibrationFiles:
    """
    This object holds file paths for Jupiter meridian flux calibration files.
    """

    def __init__(self, directory: str):
        """
        Parameters
        ----------
        directory : str
            Absolute path to the parent directory containing the
            ``calibration`` sub-directory.
        """
        self._file_paths = Path(directory).glob('calibration/*.fits*')

    def __str__(self):
        files = sorted(self._file_paths)
        print_string = f'Directory:\n   {files[0].parent}\n'
        print_string += 'Files: \n'
        print_string += '\n'.join([f'   {file.name}' for file in files])
        return print_string

    @property
    def file_paths(self) -> list[Path]:
        return sorted(self._file_paths)


class GuideSatelliteFiles:
    """
    This object holds file paths for guide satellite files.
    """

    def __init__(self, directory: str):
        """
        Parameters
        ----------
        directory : str
            Absolute path to the parent directory containing the
            ``sub_directory`` sub-directory.
        """
        self._file_paths = Path(directory, 'guide_satellite').glob('*.fits*')

    def __str__(self):
        files = sorted(self._file_paths)
        print_string = f'Directory:\n   {files[0].parent}\n'
        print_string += 'Files: \n'
        print_string += '\n'.join([f'   {file.name}' for file in files])
        return print_string

    @property
    def file_paths(self) -> list[Path]:
        return sorted(self._file_paths)


class ScienceFiles:
    """
    This object holds file paths for science target files.
    """

    def __init__(self, directory: str):
        """
        Parameters
        ----------
        directory : str
            Absolute path to the parent directory containing the
            ``sub_directory`` sub-directory.
        """
        self._file_paths = Path(directory, 'science').glob('*.fits*')

    def __str__(self):
        files = sorted(self._file_paths)
        print_string = f'Directory:\n   {files[0].parent}\n'
        print_string += 'Files: \n'
        print_string += '\n'.join([f'   {file.name}' for file in files])
        return print_string

    @property
    def file_paths(self) -> list[Path]:
        return sorted(self._file_paths)
