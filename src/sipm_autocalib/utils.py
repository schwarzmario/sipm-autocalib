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



def auto_subplots(nr_of_plots: int, figsize_per_fig=(20/6,20/10)) -> tuple[Figure, Any]:
    if nr_of_plots <= 6:
        nr_cols = 2
    elif nr_of_plots <= 18:
        nr_cols = 3
    elif nr_of_plots <= 48:
        nr_cols = 4
    else:
        nr_cols = 6
    nr_rows = math.ceil(nr_of_plots/nr_cols)
    return plt.subplots(nr_rows, nr_cols, figsize=(figsize_per_fig[0]*nr_cols, figsize_per_fig[1]*nr_rows)) # figsize was (20,20)



def output_override_file(calib_output, filename):
    """
    Writes the calibration output to a YAML file in the required nested format.

    Example output:
    S029:
      pars:
        operations:
          energy_in_pe:
            parameters:
              a: 0.0442083928140746
              m: 0.48232985213997065
    """
    data = {}
    for name, vals in calib_output.items():
        data[name] = {
            "pars": {
                "operations": {
                    "energy_in_pe": {
                        "parameters": {
                            "a": float(vals["offset"]),
                            "m": float(vals["slope"])
                        }
                    }
                }
            }
        }
    with open(filename, 'w') as outfile:
        yaml.dump(data, outfile, sort_keys=False)


def load_config_file(filename: str) -> dict[str, Any]:
    # from https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    # required because pyyaml does not read e.g. 5e-2 as float but as string (missing dot)
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(filename, 'r') as f:
        return yaml.load(f, Loader=loader)
    
def store_config_file(filename: str, *,
        gen_hist_defaults: dict[str, Any], 
        peakfinder_defaults: dict[str, Any],
        simple_calibration_defaults: dict[str, Any],
        advanced_calibration_defaults: dict[str, Any],
        gen_hist_overrides: dict[str, dict[str, Any]] = {}, 
        peakfinder_overrides: dict[str, dict[str, Any]] = {},
        simple_calibration_overrides: dict[str, dict[str, Any]] = {},
        advanced_calibration_overrides: dict[str, dict[str, Any]] = {}
    ):
    with open(filename, 'w') as f:
        dumped = {}
        dumped["gen_hist_defaults"] = gen_hist_defaults
        if len(gen_hist_overrides) > 0:
            dumped["gen_hist_overrides"] = gen_hist_overrides
        dumped["peakfinder_defaults"] = peakfinder_defaults
        if len(peakfinder_overrides) > 0:
            dumped["peakfinder_overrides"] = peakfinder_overrides
        if len(simple_calibration_defaults) > 0:
            dumped["simple_calibration_defaults"] = simple_calibration_defaults
        if len(simple_calibration_overrides) > 0:
            dumped["simple_calibration_overrides"] = simple_calibration_overrides
        dumped["advanced_calibration_defaults"] = advanced_calibration_defaults
        if len(advanced_calibration_overrides) > 0:
            dumped["advanced_calibration_overrides"] = advanced_calibration_overrides
        yaml.dump(dumped, f, sort_keys=False, default_flow_style=False)