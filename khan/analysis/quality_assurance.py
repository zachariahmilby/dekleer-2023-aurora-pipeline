from pathlib import Path

import astropy.units as u
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel

from khan.analysis.brightness_retrieval import OrderData, Background
from khan.graphics import data_cmap, color_dict


def _plot_aperture(axis, order_data: OrderData, y_offset: int = 0):
    """
    Plot the aperture as a dashed white line on a given axis.
    """
    theta = np.linspace(0, 360, 3601) * u.degree
    for wavelength in order_data.aurora_wavelengths:
        x0 = (np.abs(order_data.wavelength_centers - wavelength).argmin()
              * order_data.spectral_bin_scale * u.bin)
        y0 = y_offset * order_data.spatial_bin_scale * u.bin
        r = order_data.target_radius + order_data.seeing
        axis.plot(r*np.cos(theta) + x0, r*np.sin(theta) + y0, color='w',
                  linewidth=0.5, linestyle='--')


def _make_angular_meshgrids(order_data: OrderData):
    """
    Make meshgrids for display of the image data with equal spatial scaling.
    """
    n_obs, ny, nx = order_data.target_images.shape
    horizontal_bins_arcsec = (np.linspace(0, nx, nx + 1)
                              * order_data.spectral_bin_scale.value)
    vertical_bins_arcsec = (np.linspace(-ny/2, ny/2, ny + 1)
                            * order_data.spatial_bin_scale.value)
    x, y = np.meshgrid(horizontal_bins_arcsec, vertical_bins_arcsec)
    return x, y


def make_background_subtraction_quality_assurance_graphic(
        save_path: str | Path, order_data: OrderData, background: Background,
        y_offset: int = 0):
    """
    Save a quality assurance graphic for the background subtraction.
    """
    n_obs, ny, nx = order_data.target_images.shape
    size_scale = order_data.slit_length / (7 * u.arcsec)
    average_wavelength = order_data.aurora_wavelengths.mean()
    xe, ye = _make_angular_meshgrids(order_data)
    for obs in range(n_obs):
        background_subtracted_image = (order_data.target_images[obs]
                                       - background.backgrounds[obs])
        smoothed_image = convolve(background_subtracted_image,
                                  Gaussian2DKernel(x_stddev=1),
                                  boundary='extend')
        norm_image = order_data.target_images[obs].value
        norm = colors.Normalize(vmin=np.percentile(norm_image, 1),
                                vmax=np.percentile(norm_image, 99))
        kwargs = dict(cmap=data_cmap(), norm=norm, rasterized=True)
        fig, axes = plt.subplots(4, 1, figsize=(9, 3*size_scale), sharex='all',
                                 sharey='all', constrained_layout=True)
        axes[0].pcolormesh(xe, ye, order_data.target_images[obs], **kwargs)
        _plot_aperture(axes[0], order_data=order_data, y_offset=y_offset)
        axes[0].set_title(r'Raw Image [electrons/s]')
        axes[1].pcolormesh(xe, ye, background.backgrounds[obs], **kwargs)
        axes[1].set_title(r'Fitted Background [electrons/s]')
        norm_image = background_subtracted_image.value
        norm = colors.Normalize(vmin=0, vmax=np.percentile(norm_image, 99))
        kwargs = dict(cmap=data_cmap(), norm=norm, rasterized=True)
        axes[2].pcolormesh(xe, ye, background_subtracted_image, **kwargs)
        _plot_aperture(axes[2], order_data=order_data, y_offset=y_offset)
        axes[2].set_title(r'Background-Subtracted Image [electrons/s]')
        axes[3].pcolormesh(xe, ye, smoothed_image, **kwargs)
        _plot_aperture(axes[3], order_data=order_data, y_offset=y_offset)
        axes[3].set_title(r'Smoothed Background-Subtracted Image '
                          r'($\sigma =1\,\mathrm{bin}$) [electrons/s]')
        [axis.set_aspect('equal')for axis in axes]
        [axis.set_xticks([]) for axis in axes]
        [axis.set_yticks([]) for axis in axes]
        filename = order_data.filenames[obs].replace('.fits.gz', '.pdf')
        savepath = Path(save_path, f'{average_wavelength:.1f}',
                        'background_subtraction', filename)
        if not savepath.parent.exists():
            savepath.parent.mkdir(parents=True)
        [axis.set_aspect('equal') for axis in axes]
        plt.savefig(savepath, facecolor='white', transparent=False)
        plt.close()

    # average image
    background_subtracted_image = (order_data.average_target_image
                                   - background.average_background)
    smoothed_image = convolve(background_subtracted_image,
                              Gaussian2DKernel(x_stddev=1),
                              boundary='extend')
    norm_image = order_data.average_target_image.value
    norm = colors.Normalize(vmin=np.percentile(norm_image, 1),
                            vmax=np.percentile(norm_image, 99))
    kwargs = dict(cmap=data_cmap(), norm=norm, rasterized=True)
    fig, axes = plt.subplots(4, 1, figsize=(9, 3*size_scale), sharex='all',
                             sharey='all', constrained_layout=True)
    axes[0].pcolormesh(xe, ye, order_data.average_target_image, **kwargs)
    _plot_aperture(axes[0], order_data=order_data, y_offset=y_offset)
    axes[0].set_title(r'Average Raw Image [electrons/s]')
    axes[1].pcolormesh(xe, ye, background.average_background, **kwargs)
    axes[1].set_title(r'Fitted Background [electrons/s]')
    norm_image = background_subtracted_image.value
    norm = colors.Normalize(vmin=0, vmax=np.percentile(norm_image, 99))
    kwargs = dict(cmap=data_cmap(), norm=norm, rasterized=True)
    axes[2].pcolormesh(xe, ye, (order_data.average_target_image
                                - background.average_background), **kwargs)
    _plot_aperture(axes[2], order_data=order_data, y_offset=y_offset)
    axes[2].set_title(r'Background-Subtracted Average Image [electrons/s]')
    axes[3].pcolormesh(xe, ye, smoothed_image, **kwargs)
    _plot_aperture(axes[3], order_data=order_data, y_offset=y_offset)
    axes[3].set_title(r'Smoothed Background-Subtracted Average Image '
                      r'($\sigma =1\,\mathrm{bin}$) [electrons/s]')
    [axis.set_aspect('equal') for axis in axes]
    [axis.set_xticks([]) for axis in axes]
    [axis.set_yticks([]) for axis in axes]
    filename = 'average.pdf'
    savepath = Path(save_path, f'{average_wavelength:.1f}',
                    'background_subtraction', filename)
    if not savepath.parent.exists():
        savepath.parent.mkdir(parents=True)
    [axis.set_aspect('equal') for axis in axes]
    plt.savefig(savepath, facecolor='white', transparent=False)
    plt.close()


def make_1d_spectrum_quality_assurance_graphic(save_path: str | Path,
                                               order_data: OrderData):
    """
    Save a quality assurance graphic for the 1D line spectrum.
    """

    average_wavelength = order_data.aurora_wavelengths.mean()
    files = sorted(Path(save_path, f'{average_wavelength:.1f}',
                        'spectra_1D').glob('*.txt'))
    for file in files:
        wavelength, spectrum = np.genfromtxt(file, unpack=True)
        fig, axis = plt.subplots(1, 1, figsize=(9, 2),
                                 constrained_layout=True)
        smoothed_data = convolve(spectrum, Gaussian1DKernel(stddev=1))
        axis.plot(wavelength, spectrum, color=color_dict['grey'],
                  linewidth=0.5, label='Not Smoothed')
        axis.plot(wavelength, smoothed_data, color='k',
                  label=r'Smoothed ($\sigma = 1\,\mathrm{bin}$)')
        [axis.axvline(wavelength, color=color_dict['red'], linestyle='--',
                      linewidth=0.5)
         for wavelength in order_data.aurora_wavelengths.value]
        axis.legend(loc='upper left', frameon=False)
        axis.set_xlabel('Wavelength [nm]')
        axis.set_xlim(order_data.wavelength_edges[0].value,
                      order_data.wavelength_edges[-1].value)
        axis.set_ylabel(r'Spectral Brightness [$\mathrm{R\,nm^{-1}}$]')
        savepath = str(file).replace('.txt', '.pdf')
        plt.savefig(savepath)
        plt.close(fig)
