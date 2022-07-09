from pathlib import Path

import matplotlib.colors as colors
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pytz

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


def _keck_one_alt_az_axis(axis: plt.Axes) -> plt.Axes:
    """
    Modify a default polar axis to be set up for altitude-azimuth plotting.
    Be careful! The input axis must be a polar projection already!
    """
    axis.set_theta_zero_location('N')
    axis.set_theta_direction(-1)  # set angle direction to clockwise
    lower_limit_az = np.arange(np.radians(5.3), np.radians(146.3),
                               np.radians(0.1))
    upper_limit_az = np.concatenate((np.arange(np.radians(146.3),
                                               np.radians(360.0),
                                               np.radians(0.1)),
                                     np.arange(np.radians(0.0),
                                               np.radians(5.4),
                                               np.radians(0.1))))
    lower_limit_alt = np.ones_like(lower_limit_az) * 33.3
    upper_limit_alt = np.ones_like(upper_limit_az) * 18
    azimuth_limit = np.concatenate((lower_limit_az, upper_limit_az,
                                    [lower_limit_az[0]]))
    altitude_limit = np.concatenate((lower_limit_alt, upper_limit_alt,
                                     [lower_limit_alt[0]]))
    axis.fill_between(azimuth_limit, altitude_limit, 0, color='k',
                      alpha=0.5, linewidth=0, zorder=2)
    axis.set_rmin(0)
    axis.set_rmax(90)
    axis.set_yticklabels([])
    axis.set_xticks(np.arange(0, 2 * np.pi, np.pi / 6))
    axis.xaxis.set_tick_params(pad=-3)
    axis.yaxis.set_major_locator(ticker.MultipleLocator(15))
    axis.yaxis.set_minor_locator(ticker.NullLocator())
    axis.set_xticklabels(
        ['N', '', '', 'E', '', '', 'S', '', '', 'W', '', ''])
    axis.grid(linewidth=0.5, zorder=1)
    axis.set_xlabel('Keck I Pointing Limits', fontweight='bold')
    return axis


def _format_axis_date_labels(utc_axis: plt.Axes) -> plt.Axes:
    """
    Format axis date labels so that major ticks occur every hour and minor
    ticks occur every 15 minutes. Also creates a new axis with local
    California time as the upper horizontal axis.
    """
    utc_axis.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
    utc_axis.xaxis.set_major_locator(dates.HourLocator(interval=1))
    utc_axis.xaxis.set_minor_locator(
        dates.MinuteLocator(byminute=np.arange(0, 60, 15), interval=1))
    pacific_axis = utc_axis.twiny()
    pacific_axis.set_xlim(utc_axis.get_xlim())
    pacific_axis.xaxis.set_major_formatter(
        dates.DateFormatter('%H:%M', tz=pytz.timezone('US/Pacific')))
    pacific_axis.xaxis.set_major_locator(dates.HourLocator(interval=1))
    pacific_axis.xaxis.set_minor_locator(
        dates.MinuteLocator(byminute=np.arange(0, 60, 15), interval=1))
    return pacific_axis
