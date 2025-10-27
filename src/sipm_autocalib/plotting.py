
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import numpy as np
import awkward as ak

# Relative imports: only import used functions/classes from relevant files
from .utils import auto_subplots
from .histograms import gen_hist_by_quantile, histo_content_at
from .core import get_calibrated_histograms, calibrate_value

def plot_all_pe_spectra(energies_dict) -> Figure:
    """Simple helper to plot all raw energy spectra."""
    fig, ax = auto_subplots(len(energies_dict))
    ax = ax.ravel()
    for i, (name, data) in enumerate(energies_dict.items()):
        n, be = gen_hist_by_quantile(data, 0.96)
        ax[i].stairs(n, be)
        ax[i].set_yscale("log")
        ax[i].set_title(name, fontsize=10)
    fig.tight_layout()
    return fig

def plot_all_pe_histograms(histos: dict[str, dict[str, Any]], *, gridx = False) -> Figure:
    """Simple helper to draw all available 1-D histograms with labels."""
    fig, ax = auto_subplots(len(histos))
    ax = ax.ravel()
    for i, (name, histo) in enumerate(histos.items()):
        ax[i].set_yscale('log')
        ax[i].stairs(histo["n"], histo["be"])
        ax[i].set_title(name)
        if gridx:
            ax[i].grid(axis='x')
    fig.tight_layout()
    return fig

def plot_all_pe_histograms_and_thresholds(
        histos: dict[str, dict[str, Any]], 
        thresholds: dict[str, np.float64], *, 
        gridx = False,
        logy = True,
        scaleto: tuple[float, float] | None = None, # x position, multiplier for max y
        fig_ax: tuple[Figure, np.ndarray] | None = None) -> Figure:
    """Simple helper to draw all available 1-D histograms with labels and thresholds.

        Optional arguments:
        - gridx: If True, adds x-axis grid lines (red if threshold is NaN)
        - logy: If True, uses logarithmic y-axis scale
        - scaleto: Tuple of (x_position, multiplier) to scale y-axis limits based on histogram content at x_position
        - fig_ax: Tuple of (matplotlib Figure, axes array) for plotting on existing figure
    """
    assert len(histos) <= len(thresholds)
    fig, ax = fig_ax if fig_ax is not None else auto_subplots(len(histos))
    ax = ax.ravel()
    for i, (name, histo) in enumerate(histos.items()):
        if logy:
            ax[i].set_yscale('log')
        if scaleto is not None:
            ref = histo_content_at(histo, scaleto[0])
            ax[i].set_ylim(0.9, ref * scaleto[1])
        ax[i].stairs(histo["n"], histo["be"])
        ax[i].set_title(name)
        if not np.isnan(thresholds[name]):
            ax[i].vlines(thresholds[name], 0.9, np.max(histo['n']), colors=["green"])
        if gridx:
            if not np.isnan(thresholds[name]):
                ax[i].grid(axis='x')
            else:
                ax[i].grid(axis='x', color='red')
    fig.tight_layout()
    return fig


def plot_all_pe_histograms_and_thresholds_twohist(
        energies: dict[str, np.typing.NDArray[Any]],
        calib: dict[str, dict[str, float]],
        thresholds: dict[str, np.float64],
        range: tuple[float, float],
        nbins: int, *,
        gridx = False,
        logy = True,
        scaleto: tuple[float, float] | None = None, # x position, multiplier for max y
        fig_ax: tuple[Figure, np.ndarray] | None = None) -> Figure:
    
    fig, ax = fig_ax if fig_ax is not None else auto_subplots(len(energies))
    ax = ax.ravel()
    
    histos_all = get_calibrated_histograms(energies, calib, range, nbins)
    energies_above_ths = {n: e[calibrate_value(e, calib[n]) > thresholds[n]] for n, e in energies.items()}
    histos_above_ths = get_calibrated_histograms(energies_above_ths, calib, range, nbins)
    for i, (name, histo) in enumerate(histos_all.items()):
        if logy:
            ax[i].set_yscale('log')
        if scaleto is not None:
            ref = histo_content_at(histo, scaleto[0])
            ax[i].set_ylim(0.9, ref * scaleto[1])
        ax[i].stairs(histo["n"], histo["be"], label="All events", color="grey", edgecolor="grey", fill = True, alpha = 0.5)
        if name in histos_above_ths:
            histo_ths = histos_above_ths[name]
            ax[i].stairs(histo_ths["n"], histo_ths["be"], label="Above threshold", color="blue")
        ax[i].set_title(name)
        if gridx:
            ax[i].grid(axis='x')
        #ax[i].legend()

    fig.tight_layout()
    return fig
