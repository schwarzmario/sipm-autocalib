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

def gen_hist_by_quantile(data, quantile=0.99, nbins=200):
    """Generate 1-D histogram starting from 0, encompassing the requested quantile of total events."""
    bins = np.linspace(0, np.round(np.quantile(data, quantile)), nbins+1)
    n, be = np.histogram(data, bins)
    return n, be

def gen_hist_by_range(data, range, nbins=200):
    n, be = np.histogram(data, range=range, bins=nbins)
    return n, be

def histo_content_at(histo: dict[str, Any], x: np.float64 | float) -> np.float64:
    """Return content of bin at position x"""
    if x > histo["be"][-1] or x < histo["be"][0]:
        raise RuntimeError(f"x={x} out of histogram range [{histo['be'][0]}, {histo['be'][-1]}]")
    bin_index = np.searchsorted(histo["be"], x) - 1
    return histo["n"][bin_index]