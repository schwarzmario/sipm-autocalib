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

# Relative imports: only import used functions/classes from relevant files
from .utils import auto_subplots

def gen_hist_by_quantile(data, quantile=0.99, nbins=200):
    """Generate 1-D histogram starting from 0, encompassing the requested quantile of total events."""
    bins = np.linspace(0, np.round(np.quantile(data, quantile)), nbins+1)
    n, be = np.histogram(data, bins)
    return n, be

def gen_hist_by_range(data, range, nbins=200):
    n, be = np.histogram(data, range=range, bins=nbins)
    return n, be

def histo_content_at(histo: dict[str, Any], x: np.float64) -> np.float64:
    """Return content of bin at position x"""
    if x > histo["be"][-1] or x < histo["be"][0]:
        raise RuntimeError(f"x={x} out of histogram range [{histo['be'][0]}, {histo['be'][-1]}]")
    bin_index = np.searchsorted(histo["be"], x) - 1
    return histo["n"][bin_index]

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