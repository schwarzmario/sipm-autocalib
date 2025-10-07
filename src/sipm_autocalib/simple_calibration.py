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


from .core import ResultCheckError
from .histograms import gen_hist_by_quantile, gen_hist_by_range
from .utils import auto_subplots

# - - - - - - SIMPLE CALIBRATION - - - - - - - - -

def find_pe_peaks_in_hist(n, params: Mapping[str, Any]) -> np.typing.NDArray[np.int_]:
    """
    Find PE peak positions in a histogram using local extrema detection.

    Parameters
    ----------
    n : array-like
        Input histogram bin counts array to find peaks in
    params : Mapping[str, Any]
        Dictionary containing peak finding parameters:
        - a_delta_max_in: float, relative threshold for maximum peaks
        - a_delta_min_in: float, relative threshold for minimum peaks
        - search_direction: int, direction to search for peaks
        - a_abs_max_in: float, absolute threshold for maximum peaks
        - a_abs_min_in: float, absolute threshold for minimum peaks
    Returns
    -------
    np.typing.NDArray[np.int_]
        Array of indices corresponding to detected peak positions
    """

    # need to change dtype from int64 to float64 here, since dspeed's get_multi_local_extrema()
    # is only defined for float32 and float64
    n = np.array(n, dtype=np.float64)
    
    # Outputs
    vt_max_out = np.zeros(shape=len(n) - 1)
    vt_min_out = np.zeros(shape=len(n) - 1)
    n_max_out = 0
    n_min_out = 0

    # Call the function with updated parameters
    get_multi_local_extrema(
        n,
        params["a_delta_max_in"] * np.max(n),
        params["a_delta_min_in"] * np.max(n),
        params["search_direction"],
        params["a_abs_max_in"] * np.max(n),
        params["a_abs_min_in"] * np.max(n),
        vt_max_out,
        vt_min_out,
        n_max_out,
        n_min_out,
    ) # type: ignore

    peakpos_indices = vt_max_out[~np.isnan(vt_max_out)].astype(np.int_)
    return peakpos_indices


def check_and_improve_PE_peaks(
        peakpos_indices: np.typing.NDArray[np.int_], 
        n: np.typing.NDArray[Any],
        be: np.typing.NDArray[Any],
        params: Mapping[str, Any]
        ) -> tuple[np.typing.NDArray[np.int_], dict[int, list[Any]]]:
    """
    Checks and improves the indices of detected photoelectron (PE) peaks in a histogram.
    Applies strict or non-strict validation rules to ensure the peaks correspond to expected
    noise and PE peaks, and optionally handles double-peak SiPMs.

    Parameters
    ----------
    peakpos_indices : np.typing.NDArray[np.int_]
    Indices of detected peaks in the histogram.
    n : np.typing.NDArray[Any]
    Histogram bin counts.
    be : np.typing.NDArray[Any]
    Histogram bin edges.
    params : Mapping[str, Any]
    Dictionary of peak finding and checking parameters. Keys include:
        - "max_merge_distance": int, how much bins away two peaks can be to be merged (not double-peak case)
        - "double_peak": bool, whether to handle double-peak SiPMs.
        - "min_peak_dist": int, minimum distance between peaks (in bins).
        - "strict": bool, whether to apply strict validation.
        - "peakdist_compare_margin": int, margin for peak distance comparison.

    Returns
    -------
    tuple[np.typing.NDArray[np.int_], dict[int, list[Any]]]
    Validated and possibly improved peak indices, and a mapping of peak indices to 
    all bin edge values (i.e. two values for 1PE peak if double peak)

    Raises
    ------
    ResultCheckError
    If validation fails under strict mode or insufficient peaks are found.
    """
    max_merge_distance = params.get("max_merge_distance", 0)
    if max_merge_distance > 0:
        new_peakpos_indices = []
        dropnext = False
        dropcurrent = False
        for i in range(len(peakpos_indices)):
            dropcurrent = dropnext
            dropnext = False
            if i < len(peakpos_indices) - 1:
                tooclose = (peakpos_indices[i+1] - peakpos_indices[i]) <= max_merge_distance
                if tooclose:
                    if n[peakpos_indices[i]] < n[peakpos_indices[i+1]]:
                        dropcurrent = True
                    elif tooclose:
                        dropnext = True
            if not dropcurrent:
                new_peakpos_indices.append(peakpos_indices[i])
        peakpos_indices = np.array(new_peakpos_indices, dtype=np.int_)
            
    if params.get("double_peak", False):
        if len(peakpos_indices) < 3:
            raise ResultCheckError(f"Require at least 3 found peaks for double-peak SiPMs; found only {len(peakpos_indices)}.")
        peakpos_map = {0: [be[peakpos_indices[0]]], 1: [be[peakpos_indices[1]], be[peakpos_indices[2]]]} | {i: [be[peakpos_indices[i+1]]] for i in range(2, len(peakpos_indices)-1)}
        mean_1_2 = (peakpos_indices[1] + peakpos_indices[2]) // 2 # small bias but ok
        peakpos_indices = np.concatenate((np.array([peakpos_indices[0], mean_1_2], dtype=np.int_), peakpos_indices[3:]))
    else:
        peakpos_map = {i: [be[peakpos_indices[i]]] for i in range(0, len(peakpos_indices))}

    min_peak_dist = params["min_peak_dist"]
    if params["strict"]:
        if len(peakpos_indices) < 2:
            raise ResultCheckError(f"Only {len(peakpos_indices)} peaks found. Either noise peak or 1pe peak not found.")

        if n[peakpos_indices[1]] > n[peakpos_indices[0]]:
            raise ResultCheckError(f"1pe peak larger than noise peak.")

        if peakpos_indices[1] - peakpos_indices[0] < min_peak_dist:
            raise ResultCheckError(f"Noise peak and 1pe peak too close together (< {min_peak_dist} bins).")

        if len(peakpos_indices) > 2:
            if peakpos_indices[2] - peakpos_indices[1] < min_peak_dist:
                raise ResultCheckError(f"1pe peak and 2pe peak too close together (< {min_peak_dist} bins).")
            if (peakpos_indices[2] - peakpos_indices[1] + params["peakdist_compare_margin"]) < peakpos_indices[1] - peakpos_indices[0]:
                raise ResultCheckError(f"Distance between 1pe and 2pe smaller than 0pe and 1pe (outside peakdist_compare_margin of {params["peakdist_compare_margin"]}).")
    else:
        if len(peakpos_indices) > 2:
            if peakpos_indices[2] - peakpos_indices[1] < min_peak_dist:
                print(f"1pe peak and 2pe peak too close together (< {min_peak_dist} bins). Removing '1pe' peak.")
                peakpos_indices = peakpos_indices[peakpos_indices != peakpos_indices[1]]
    return  peakpos_indices, peakpos_map

def simple_calibration(energies, gen_hist_params: Mapping[str, Any], 
                       peakfinder_params: Mapping[str, Any],
                       calibration_params: Mapping[str, Any], *, 
                       ax = None, verbosity = 0) -> dict[str, float | dict[int, list[Any]]]:
    """
    Perform simple peak-finder based calibration on SiPM energy spectrum.

    Generates a histogram from the input energies and uses peak finding to locate PE peaks.
    The peaks are used to determine calibration parameters (slope and offset).

    Parameters
    ----------
    energies : np.ndarray
        1D array of uncalibrated energy values for a single SiPM
    gen_hist_params : Mapping[str, Any]
        Parameters for histogram generation:
        - "quantile": float, quantile of data to include in histogram
        - "nbins": int, number of histogram bins 
        Or:
        - "range": tuple(float,float), fixed histogram range
        - "nbins": int, number of histogram bins
    peakfinder_params : Mapping[str, Any]
        Parameters controlling peak finding algorithm 
    calibration_params : Mapping[str, Any]
        Parameters for calibration:
        - "use_1pe_0pe_diff_as_fallback": bool, whether to use 0pe-1pe distance as fallback if 2pe peak not found
          (Default & recommended: False)
    ax : matplotlib.axes.Axes, optional
        Axis to plot histogram and peaks on, if visualization desired
    verbosity : int, optional
        Level of debug output, default 0

    Returns
    -------
    dict
        Contains:
        - "slope": float, calibration slope 
        - "offset": float, calibration offset
        - "peaks": dict mapping PE index to list of peak positions in a.u.

    Raises
    ------
    ResultCheckError
        If peak finding or validation fails
    """
    match gen_hist_params:
        case {"quantile": quantile, "nbins": nbins}:
            n, be = gen_hist_by_quantile(energies, quantile, nbins)
        case {"range": r, "nbins": nbins}:
            n, be = gen_hist_by_range(energies, r, nbins)
        case _:
            raise TypeError("gen_hist_params does not match valid histogram type")
    peakpos_indices = find_pe_peaks_in_hist(n, peakfinder_params)
    failed_checks = False
    try:
        peakpos_indices, peakpos_map = check_and_improve_PE_peaks(peakpos_indices, n, be, peakfinder_params)
    except ResultCheckError:
        failed_checks = True
        peaks = be[peakpos_indices] # use old indices for peaks
        raise # runs finally before raise
    else: 
        peaks = be[peakpos_indices]

        if len(peaks) > 2: # use 1PE and 2PE
            gain = peaks[2] - peaks[1]
            c = 1/gain
            offset = 1 - peaks[1] * c # 1pe peak at 1
        else: 
            if calibration_params.get("use_1pe_0pe_diff_as_fallback", False):
                # fallback: use 0 PE and 1 PE - DISCOURAGED because distance usually too small!
                gain = peaks[1] - peaks[0]
                c = 1/gain
                offset = 1 - peaks[1] * c # 1pe peak at 1   
            else:
                # fallback: use only position of 1 PE
                gain = peaks[1]
                c = 1/gain
                offset = 0

        # runs finally (i.e. plot) before return
        return {"slope": c, "offset": offset, "peaks": peakpos_map}
    finally: # draw in any case (for debugging); but choose color
         if ax is not None:
            hist_color = "red" if failed_checks else "blue"
            line_color = "grey" if failed_checks else "red"
            # uncalibrated histogram and peaks
            ax.stairs(n, be, color=hist_color)
            for p in peaks: # type: ignore
                ax.axvline(x=p, color=line_color, ls=":")

def multi_simple_calibration(energies_dict, 
                             gen_hist_defaults: dict[str, Any], 
                             peakfinder_defaults: dict[str, Any],
                             calibration_defaults: dict[str, Any], *,
                             gen_hist_overrides: dict[str, dict[str, Any]] = {}, 
                             peakfinder_overrides: dict[str, dict[str, Any]] = {},
                             calibration_overrides: dict[str, dict[str, Any]] = {},
                             draw = False, 
                             nodraw_axes = False, 
                             verbosity = 0
                             ) -> tuple[dict[str, dict[str, float]], int, Figure | None]:
    """
    Performs calibration for multiple channels using the simple_calibration method.
    Parameters
    ----------
    energies_dict : dict
        Dictionary containing channel names as keys and energy arrays as values
    gen_hist_defaults : dict[str, Any]
        Default parameters for histogram generation
    peakfinder_defaults : dict[str, Any] 
        Default parameters for peak finding algorithm
    calibration_defaults : dict[str, Any]
        Default parameters for calibration procedure
    gen_hist_overrides : dict[str, dict[str, Any]], optional
        Channel-specific overrides for histogram generation parameters
    peakfinder_overrides : dict[str, dict[str, Any]], optional  
        Channel-specific overrides for peak finding parameters
    calibration_overrides : dict[str, dict[str, Any]], optional
        Channel-specific overrides for calibration parameters
    draw : bool, optional
        If True, generates plots of the calibration results
    nodraw_axes : bool, optional
        If True, hides axes in the generated plots
    verbosity : int, optional
        Controls verbosity level of output messages (-1: only warnings, 0: basic info, >0: detailed info)
    Returns
    -------
    tuple[dict[str, dict[str, float]], int, Figure | None]
        - Dictionary containing calibration results for each channel
        - Number of unsuccessful calibrations
        - Matplotlib Figure object if draw=True, None otherwise
    Notes
    -----
    The calibration results dictionary contains for each channel:
    - 'slope': calibration slope
    - 'offset': calibration offset 
    - 'peaks': dictionary of identified peaks
    If a calibration fails for a channel, its results will contain NaN values.
    """
    ret = {}
    fig = None
    if draw:
        fig, ax = auto_subplots(len(energies_dict))
        ax_iter = iter(ax.ravel())
    nr_unsuccessful_calibs = 0
    for name, energies in energies_dict.items():
        if draw:
            ax = next(ax_iter)
            ax.set_yscale("log")
            if nodraw_axes:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            ax.set_title(name, fontsize=10)
        else:
            ax = None
        try:
            calib_results = simple_calibration(
                energies,
                gen_hist_defaults | gen_hist_overrides.get(name, {}),
                peakfinder_defaults | peakfinder_overrides.get(name, {}),
                calibration_defaults | calibration_overrides.get(name, {}),
                ax=ax,  verbosity=verbosity)
            ret[name] = calib_results
        except ResultCheckError as e:
            print(f"Calibration failed for {name}: {e}")
            ret[name] = {"slope": np.nan, "offset": np.nan, "peaks": {}}
            nr_unsuccessful_calibs += 1
    
    if nr_unsuccessful_calibs > 0 and verbosity >= -1:
        print(f"WARNING: {nr_unsuccessful_calibs} calibrations failed!")
    elif verbosity >= 0:
        print("Info: All simple calibrations successful! :)")

    if draw:
        fig.tight_layout()
        if nodraw_axes: # have to do this also for non-drawn plots
            for ax in ax_iter:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            fig.subplots_adjust(wspace=0) # , hspace=0)
    return ret, nr_unsuccessful_calibs, fig