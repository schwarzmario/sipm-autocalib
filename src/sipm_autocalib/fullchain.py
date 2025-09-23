import glob
import os
from collections.abc import Callable, Sequence, Iterator, Mapping
from abc import ABC, abstractmethod
from typing import Any
import argparse
import yaml
import re
import math

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import numpy as np
import awkward as ak

import scipy

from lgdo import lh5
from lgdo.lh5.exceptions import LH5DecodeError
from legendmeta import LegendMetadata
from dspeed.processors import get_multi_local_extrema


from .simple_calibration import multi_simple_calibration
from .advanced_calibration import multi_advanced_calibration
from .histograms import plot_all_pe_histograms
from .core import get_calibrated_histograms, get_calibrated_PE_positions, combine_multiple_calibrations

def full_calibration_chain(
        energies_dict: dict[str, np.typing.NDArray[Any]], *,
        gen_hist_defaults: dict[str, Any], 
        peakfinder_defaults: dict[str, Any],
        simple_calibration_defaults: dict[str, Any],
        advanced_calibration_defaults: dict[str, Any],
        gen_hist_overrides: dict[str, dict[str, Any]] = {}, 
        peakfinder_overrides: dict[str, dict[str, Any]] = {},
        simple_calibration_overrides: dict[str, dict[str, Any]] = {},
        advanced_calibration_overrides: dict[str, dict[str, Any]] = {},
        plot_output_dir: str | None = None, # not drawing when None
        plot_interactive: bool = False, # interactive plot, e.g. in jupyter notebook
        verbosity: int = 0
        ) -> dict[str, dict[str, float]]:
    """
    Performs a complete SiPM calibration chain including simple and advanced calibration steps.
    This function executes the full calibration process:
    1. Simple calibration to find initial peak positions
    2. Advanced calibration to refine the calibration parameters
    3. Combines both calibrations into final results
    Parameters
    ----------
    energies_dict : dict[str, np.typing.NDArray[Any]]
        Dictionary mapping SiPM IDs to their raw ADC energy values by event
        (see get_energies())
    gen_hist_defaults : dict[str, Any]
        Default parameters for histogram generation
    peakfinder_defaults : dict[str, Any] 
        Default parameters for peak finding algorithm
    simple_calibration_defaults : dict[str, Any]
        Default parameters for simple calibration step
    advanced_calibration_defaults : dict[str, Any]
        Default parameters for advanced calibration step
    gen_hist_overrides : dict[str, dict[str, Any]], optional
        Per-SiPM overrides for histogram generation parameters
    peakfinder_overrides : dict[str, dict[str, Any]], optional
        Per-SiPM overrides for peak finding parameters  
    simple_calibration_overrides : dict[str, dict[str, Any]], optional
        Per-SiPM overrides for simple calibration parameters
    advanced_calibration_overrides : dict[str, dict[str, Any]], optional
        Per-SiPM overrides for advanced calibration parameters
    plot_output_dir : str | None, optional
        Directory to save plots. If None, no plots are saved
    plot_interactive : bool, optional
        Whether to show interactive plots, defaults to False
    verbosity : int, optional
        Verbosity level for debug output, defaults to 0
    Returns
    -------
    dict[str, dict[str, float]]
        Dictionary containing final calibration parameters for each SiPM
    Raises
    ------
    ValueError
        If simple or advanced calibration fails for any SiPM
    Notes
    -----
    The function generates various diagnostic plots if plot_output_dir is specified:
    - simple_calibration_peaks.pdf: Peak finding results
    - simple_calibration_result.pdf: Histograms after simple calibration
    - advanced_calibration_fits.pdf: Advanced calibration fit results
    - final_calibration_result.pdf: Final calibrated spectra
    """
    draw: bool = plot_output_dir is not None or plot_interactive
    last_fig_path = ""
    def store(figure: Figure | None, filebasename: str) -> str:
        if plot_output_dir is None or figure is None:
            return "NO_PLOT"
        last_fig_path = os.path.join(plot_output_dir, filebasename + ".pdf")
        figure.savefig(last_fig_path)
        return last_fig_path
    def get_hint() -> str:
        if plot_output_dir is not None:
            return f"Please review {last_fig_path} for failures of individual SiPMs."
        elif plot_interactive:
            return f"Please review the last interactive plot for failures of individual SiPMs."
        else:
            return "Please re-run with active plotting to identify failures."

    simple_calib_output, nr_failed_simple_calib, fig = multi_simple_calibration(
        energies_dict, gen_hist_defaults, peakfinder_defaults, simple_calibration_defaults, 
        gen_hist_overrides = gen_hist_overrides,
        peakfinder_overrides = peakfinder_overrides,
        calibration_overrides=simple_calibration_overrides,
        draw=draw,
        verbosity=verbosity
    )
    last_fig_path = store(fig, "simple_calibration_peaks")
    if nr_failed_simple_calib > 0:
        raise ValueError(f"Simple Calibration failed for {nr_failed_simple_calib} out of {len(energies_dict)} SiPMs.\n" + 
                         get_hint() + "\nFailed histograms are drawn in red.")
    
    simple_calibrated_histos = get_calibrated_histograms(
        energies_dict, simple_calib_output, 
        (advanced_calibration_defaults.get("histogram_begin", 0), advanced_calibration_defaults.get("histogram_end", 5)), 
        advanced_calibration_defaults.get("histogram_nbins", 200)
        )
    if draw:
        fig = plot_all_pe_histograms(simple_calibrated_histos, gridx=True)
        last_fig_path = store(fig, "simple_calibration_result")

    advanced_calib_output, nr_failed_advanced_calib, fig = multi_advanced_calibration(
        simple_calibrated_histos, get_calibrated_PE_positions(simple_calib_output), 
        advanced_calibration_defaults, 
        calibration_overrides=advanced_calibration_overrides, 
        draw=draw,
        verbosity=verbosity)
    last_fig_path = store(fig, "advanced_calibration_fits")
    if nr_failed_advanced_calib > 0:
        raise ValueError(f"Advanced Calibration failed for {nr_failed_advanced_calib} out of {len(energies_dict)} SiPMs.\n" + 
                         get_hint() + "\nFailed fits are drawn in red (with initial parameters); Orange indicates failed checks.")
    
    final_calib_output = combine_multiple_calibrations(simple_calib_output, advanced_calib_output)
    if draw:
        adv_calibrated_histos = get_calibrated_histograms(energies_dict, final_calib_output, (0, 5), 200)
        fig = plot_all_pe_histograms(adv_calibrated_histos, gridx=True)
        last_fig_path = store(fig, "final_calibration_result")
    return final_calib_output