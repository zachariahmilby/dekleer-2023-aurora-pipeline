from pathlib import Path


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
