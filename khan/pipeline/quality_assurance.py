from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.metrics import r2_score

from khan.graphics import turn_off_ticks, percentile_norm, flat_cmap, \
    data_cmap, color_dict
from khan.pipeline.images import MasterBias, MasterFlat, MasterArc, \
    MasterTrace, CCDImage
from khan.pipeline.rectification import OrderTraces
from khan.pipeline.wavelength_calibration import WavelengthSolution


def make_master_calibration_image_quality_assurance_graphic(
        master_image: MasterBias | MasterFlat | MasterArc | MasterTrace,
        save_path: str, file_name: str, cmap: colors.ListedColormap,
        cbar: bool = True) -> None:
    """
    Make quality-assurance graphics for the master calibration images.
    """
    if cbar:
        horizontal_scale = 1.2
    else:
        horizontal_scale = 1
    if master_image[0].anc['detector_layout'] == 'mosaic':
        n_detectors = 3
        figsize = (4 * horizontal_scale, 6)
    else:
        n_detectors = 1
        figsize = (4 * horizontal_scale, 4)
    fig, axes = plt.subplots(n_detectors, 1, figsize=figsize, sharex='all',
                             squeeze=False, clear=True,
                             constrained_layout=True)
    [turn_off_ticks(axis) for axis in axes.ravel()]
    axes = np.flip(axes, axis=0)
    for ind in range(n_detectors):
        img = axes[ind, 0].pcolormesh(
            master_image[ind].data, cmap=cmap,
            norm=percentile_norm(master_image[ind].data), rasterized=True)
        if cbar:
            cbar = plt.colorbar(img, ax=axes[ind, 0],
                                label='Detector Counts [adu]')
        axes[ind, 0].set_ylabel(f'Detector {ind}')
    file_name = Path(save_path, 'quality_assurance', file_name)
    if not file_name.parent.exists():
        file_name.parent.mkdir(parents=True)
    fig.savefig(file_name, dpi=300)
    plt.close(fig='all')


def make_order_trace_quality_assurance_graphic(
        order_traces: OrderTraces, master_flat: MasterFlat,
        save_path: str) -> None:
    """
    Make a quality-assurance graphic for the order edge traces.
    """
    if master_flat[0].anc['detector_layout'] == 'mosaic':
        n_detectors = 3
        figsize = (4, 6)
    else:
        n_detectors = 1
        figsize = (4, 4)
    fig, axes = plt.subplots(n_detectors, 1, figsize=figsize, sharex='all',
                             squeeze=False, clear=True,
                             constrained_layout=True)
    [turn_off_ticks(axis) for axis in axes.ravel()]
    [axis.set_facecolor(color_dict['grey']) for axis in axes.ravel()]
    axes = np.flip(axes, axis=0)
    for ind in range(n_detectors):
        axes[ind, 0].pcolormesh(master_flat[ind].data, cmap=flat_cmap(),
                                norm=percentile_norm(master_flat[ind].data),
                                rasterized=True)
        top_edges = order_traces.get_top_edges(detector=ind)
        for edges in top_edges:
            axes[ind, 0].plot(np.arange(len(edges))+0.5, edges,
                              color=color_dict['red'], linewidth=0.25)
        bottom_edges = order_traces.get_bottom_edges(detector=ind)
        for edges in bottom_edges:
            axes[ind, 0].plot(np.arange(len(edges))+0.5, edges,
                              color=color_dict['red'], linewidth=0.25)
        axes[ind, 0].set_ylabel(f'Detector {ind}')
    file_name = Path(save_path, 'quality_assurance', 'order_traces.pdf')
    if not file_name.parent.exists():
        file_name.parent.mkdir(parents=True)
    fig.savefig(file_name, dpi=300)
    plt.close(fig='all')


def make_instrument_correction_quality_assurance_graphic(
        images: list[CCDImage], cosmic_ray_cleaned_images: list[CCDImage],
        bias_subtracted_images: list[CCDImage],
        flat_corrected_images: list[CCDImage],
        gain_corrected_images: list[CCDImage],
        save_path: str, sub_directory: str) -> None:
    """
    Make a quality-assurance graphic for each step of the instrument-artifact
    correction procedure.
    """
    if images[0].anc['detector_layout'] == 'mosaic':
        n_detectors = 3
        figsize = (15, 4.5)
    else:
        n_detectors = 1
        figsize = (15, 3)
    fig, axes = plt.subplots(n_detectors, 5, figsize=figsize, sharex='all',
                             squeeze=False, clear=True,
                             constrained_layout=True)
    [turn_off_ticks(axis) for axis in axes.ravel()]
    axes = np.flip(axes, axis=0)
    for ind in range(n_detectors):

        image_data = images[ind].data
        norm = colors.Normalize(vmin=0, vmax=np.nanpercentile(image_data, 99))
        img = axes[ind, 0].pcolormesh(image_data, cmap=data_cmap(), norm=norm,
                                      rasterized=True)
        cbar = plt.colorbar(img, ax=axes[ind, 0])
        cbar.formatter.set_powerlimits((0, 0))

        image_data = cosmic_ray_cleaned_images[ind].data
        norm = colors.Normalize(vmin=0, vmax=np.nanpercentile(image_data, 99))
        img = axes[ind, 1].pcolormesh(image_data, cmap=data_cmap(), norm=norm,
                                      rasterized=True)
        cbar = plt.colorbar(img, ax=axes[ind, 1])
        cbar.formatter.set_powerlimits((0, 0))

        image_data = bias_subtracted_images[ind].data
        norm = colors.Normalize(vmin=0, vmax=np.nanpercentile(image_data, 99))
        img = axes[ind, 2].pcolormesh(image_data, cmap=data_cmap(), norm=norm,
                                      rasterized=True)
        cbar = plt.colorbar(img, ax=axes[ind, 2])
        cbar.formatter.set_powerlimits((0, 0))

        image_data = flat_corrected_images[ind].data
        norm = colors.Normalize(vmin=0, vmax=np.nanpercentile(image_data, 99))
        img = axes[ind, 3].pcolormesh(image_data, cmap=data_cmap(), norm=norm,
                                      rasterized=True)
        cbar = plt.colorbar(img, ax=axes[ind, 3])
        cbar.formatter.set_powerlimits((0, 0))

        image_data = gain_corrected_images[ind].data
        norm = colors.Normalize(vmin=0, vmax=np.nanpercentile(image_data, 99))
        img = axes[ind, 4].pcolormesh(image_data, cmap=data_cmap(), norm=norm,
                                      rasterized=True)
        cbar = plt.colorbar(img, ax=axes[ind, 4])
        cbar.formatter.set_powerlimits((0, 0))

        axes[ind, 0].set_ylabel(f'Detector {ind}')

    axes[-1, 0].set_title('Raw Image [adu]')
    axes[-1, 1].set_title('Cosmic-Ray Cleaned Image [adu]')
    axes[-1, 2].set_title('Bias-Subtracted Image [adu]')
    axes[-1, 3].set_title('Flat-Field-Corrected Image [adu]')
    axes[-1, 4].set_title(r'Gain-Corrected Image [$\mathrm{e^-}$]')

    file_name = Path(save_path, 'quality_assurance',
                     'instrument_artifact_correction',
                     sub_directory,
                     images[0].anc['file_name'].replace('.fits', '.pdf').replace('.gz', ''))
    if not file_name.parent.exists():
        file_name.parent.mkdir(parents=True)
    fig.savefig(file_name, dpi=300)
    plt.close(fig='all')


def make_wavelength_solution_quality_assurance_graphic(
        wavelength_solution: WavelengthSolution, save_path: str) -> None:
    """
    Make quality-assurance graphics for each order's wavelength solution.
    """
    for order in range(wavelength_solution.orders.shape[0]):
        fig, axes = plt.subplots(2, 1, figsize=(5, 4), sharex='all',
                                 gridspec_kw={'height_ratios': [2, 1]},
                                 constrained_layout=True)
        axes[0].scatter(wavelength_solution.input_pixel_centers[order]+0.5,
                        wavelength_solution.input_wavelengths[order],
                        color=color_dict['grey'])
        axes[0].plot(wavelength_solution.pixel_centers,
                     wavelength_solution.pixel_center_wavelengths[order],
                     color='k')
        axes[0].set_ylabel('Wavelength [nm]')
        axes[0].set_title(f'Order {wavelength_solution.orders[order]}')

        y_true = wavelength_solution.input_wavelengths[order]
        fit = wavelength_solution.fit_functions[order]
        y_pred = fit(wavelength_solution.input_pixel_centers[order])
        residual = y_true - y_pred
        axes[1].scatter(wavelength_solution.input_pixel_centers[order],
                        residual, color='k')
        axes[1].axhline(0, color=color_dict['grey'], linestyle='--')
        ylim = np.max(np.abs(axes[1].get_ylim()))
        axes[1].set_ylim(-ylim, ylim)
        axes[1].set_xlabel('Spectral Pixel Number')
        axes[1].xaxis.set_major_locator(ticker.MultipleLocator(512))
        axes[1].set_xlim(0, wavelength_solution.pixel_centers[-1] + 1)
        axes[1].set_ylabel('Residual [nm]')

        r2 = r2_score(y_true, y_pred)
        axes[0].text(0.02, 0.98, fr'$R^2 = {r2:.8f}$', ha='left', va='top',
                     transform=axes[0].transAxes)

        file_name = Path(save_path, 'quality_assurance',
                         'wavelength_solutions',
                         f'order{wavelength_solution.orders[order]}.pdf')
        if not file_name.parent.exists():
            file_name.parent.mkdir(parents=True)
        fig.savefig(file_name, dpi=300)
        plt.close(fig='all')
