from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

# set graphics style
plt.style.use(Path(Path(__file__).resolve().parent, 'anc/rcparams.mplstyle'))


# custom colors dictionary
color_dict = {'red': '#D62728', 'orange': '#FF7F0E', 'yellow': '#FDB813',
              'green': '#2CA02C', 'blue': '#0079C1', 'violet': '#9467BD',
              'cyan': '#17BECF', 'magenta': '#D64ECF', 'brown': '#8C564B',
              'darkgrey': '#3F3F3F', 'grey': '#7F7F7F', 'lightgrey': '#BFBFBF'}


def percentile_norm(data: np.ndarray) -> colors.Normalize:
    """
    Normalize an image array to a range from its 1st to 99th percentiles. Helps
    to remove outliers when displaying an image.
    """
    return colors.Normalize(vmin=np.nanpercentile(data, 1),
                            vmax=np.nanpercentile(data, 99))


def bias_cmap() -> colors.ListedColormap:
    """
    Return the "cividis" colormap with NaNs set to grey.
    """
    cmap = plt.get_cmap('cividis').copy()
    cmap.set_bad(color_dict['grey'])
    return cmap


def flat_cmap() -> colors.ListedColormap:
    """
    Return the "bone" colormap with NaNs set to red.
    """
    cmap = plt.get_cmap('bone').copy()
    cmap.set_bad(color_dict['red'])
    return cmap


def arc_cmap() -> colors.ListedColormap:
    """
    Return the "inferno" colormap with NaNs set to grey.
    """
    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad(color_dict['grey'])
    return cmap


def data_cmap() -> colors.ListedColormap:
    """
    Return the "viridis" colormap with NaNs set to grey.
    """
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color_dict['grey'])
    return cmap


def turn_off_ticks(axis: plt.Axes) -> None:
    """
    Turn off an axis' ticks.
    """
    axis.set_xticks([])
    axis.set_yticks([])


def convert_wavelength_to_rgb(
        wavelength: int | float, gamma: float = 0.8) -> str:
    """
    Convert a wavelength in nanometers to its equivalent visible light color
    (black if UV or IR).

    Parameters
    ----------
    wavelength : float
        Wavelength in nm.
    gamma : float
        No idea. It was part of the code when I stole it.

    Returns
    -------
    The RGB color equivalent (0-255) as a string. Plug into Matplotlib and...
    voila!
    """
    wavelength = float(wavelength)
    if 380 <= wavelength < 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        r = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        g = 0.0
        b = (1.0 * attenuation) ** gamma
    elif 440 <= wavelength < 490:
        r, g, b = 0.0, ((wavelength - 440) / (490 - 440)) ** gamma, 1.0
    elif 490 <= wavelength < 510:
        r, g, b = 0.0, 1.0, (-(wavelength - 510) / (510 - 490)) ** gamma
    elif 510 <= wavelength < 580:
        r, g, b = ((wavelength - 510) / (580 - 510)) ** gamma, 1.0, 0.0
    elif 580 <= wavelength < 645:
        r, g, b = 1.0, (-(wavelength - 645) / (645 - 580)) ** gamma, 0.0
    elif 645 <= wavelength < 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        r, g, b = (1.0 * attenuation) ** gamma, 0.0, 0.0
    else:
        r, g, b = 0.0, 0.0, 0.0
    return f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'