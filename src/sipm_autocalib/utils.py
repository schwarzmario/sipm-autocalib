import glob
import os
from collections.abc import Callable, Sequence, Iterator, Mapping
from abc import ABC, abstractmethod
from typing import Any, Callable
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
from dbetto import TextDB, AttrsDict

def get_timestamp_from_filename(filename: str) -> str | None:
    match = re.search(r"\d{8}T\d{6}Z", filename)
    return match.group(0) if match else None

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


def deep_map(func: Callable, data: Any) -> Any:
    match data:
        case dict():
            return {k: deep_map(func, v) for k, v in data.items()}
        case list() | tuple():
            return [deep_map(func, x) for x in data]
        case _:
            return func(data)

def floatify(data: Any) -> Any:
    return deep_map(lambda x: float(x) if isinstance(x, np.float64) else x, data)

def output_override_file(
    filename: str, 
    calib_output: dict[str, dict[str, float]] | None = None, 
    thresholds: dict[str, float] | None = None
    ) -> None:
    """
    Writes the calibration output and/or thresholds to a YAML file in the required nested format.
    Handles cases where some channels may have only threshold information without calibration parameters.
    Exports also the (calibrated) width of the 1PE peak if calib_output is not None.

    Example output:
    S029:
      pars:
        operations:
          energy_in_pe:
            parameters:
              a: 0.0442083928140746
              m: 0.48232985213997065
          is_valid_hit:
            parameters:
              a: 0.4346083
      aux:
        sigma_1: 0.0800222
    """
    data = {}
    # Handle calibration parameters
    if calib_output is not None:
        for name, vals in calib_output.items():
            if name not in data:
                data[name] = {"pars": {"operations": {}}, "aux": {}}
            data[name]["pars"]["operations"]["energy_in_pe"] = {
            "parameters": {
                "a": float(vals["offset"]),
                "m": float(vals["slope"])
            }}
            #data[name]["aux"]["sigma_1"] = float(vals["sigma_1"])
            aux = {k: floatify(v) for k, v in vals.items() if k not in ["offset", "slope"]}
            data[name]["aux"] = aux
    # Handle thresholds
    if thresholds is not None:
        for name, thresh in thresholds.items():
            if name not in data:
                data[name] = {"pars": {"operations": {}}}
            data[name]["pars"]["operations"]["is_valid_hit"] = {
            "parameters": {
                "a": float(thresh)
            }}
    with open(filename, 'w') as outfile:
        yaml.dump(data, outfile, sort_keys=False)

def update_override_file(
        oldfilename: str,
        newfilename: str,
        calib_output: dict[str, dict[str, float]] | None = None,
        thresholds: dict[str, float] | None = None
        ) -> None:
    """
    Updates an existing YAML override file with new calibration parameters and/or thresholds.

    Parameters
    ----------
    oldfilename : str
        Path to the existing YAML override file
    newfilename : str
        Path to the new YAML override file to be created
    calib_output : dict[str, dict[str, float]] | None
        A dictionary mapping channel names to their calibration parameters ('slope' and 'offset')
    thresholds : dict[str, float] | None
        A dictionary mapping channel names to their threshold values (in calibrated PE units)
    """
    with open(oldfilename, 'r') as f:
        data = yaml.safe_load(f)

    if calib_output is not None:
        for name, vals in calib_output.items():
            if name not in data:
                data[name] = {"pars": {}}
            if "pars" not in data[name]:
                data[name]["pars"] = {"operations": {}}
            if "operations" not in data[name]["pars"]:
                data[name]["pars"]["operations"] = {}
            data[name]["pars"]["operations"]["energy_in_pe"] = {
                "parameters": {
                    "a": float(vals["offset"]),
                    "m": float(vals["slope"])
                }
            }
    if thresholds is not None:
        for name, thresh in thresholds.items():
            if name not in data:
                data[name] = {"pars": {}}
            if "pars" not in data[name]:
                data[name]["pars"] = {"operations": {}}
            if "operations" not in data[name]["pars"]:
                data[name]["pars"]["operations"] = {}
            data[name]["pars"]["operations"]["is_valid_hit"] = {
                "parameters": {
                    "a": float(thresh)
                }
            }
    with open(newfilename, 'w') as outfile:
        yaml.dump(data, outfile, sort_keys=False)

def read_override_file(filename: str) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """
    Reads a YAML override file and extracts calibration parameters and thresholds.

    Parameters
    ----------
    filename : str
        Path to the YAML override file

    Returns
    -------
    tuple[dict[str, dict[str, float]], dict[str, float]]
        A tuple containing:
        - A dictionary mapping channel names to their calibration parameters ('slope' and 'offset')
        - A dictionary mapping channel names to their threshold values (in calibrated PE units)
    """
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    return read_dict(data)


_overrides_cache = {}
def read_overrides_from_metadata(
        metadata_dir: str, 
        timestamp: str,
        *,
        include_aux: bool = False,
        use_cache: bool = True
        ) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """
    Reads overrides from legend-metadata (legend-dataflow-overrides).

    Parameters
    ----------
    metadata_dir : str
        Path to the metadata directory
    timestamp : str
        Timestamp for the overrides to read

    Returns
    -------
    tuple[dict[str, dict[str, Any]], dict[str, float]]
        A tuple containing:
        - A dictionary mapping channel names to their calibration parameters ('slope' and 'offset')
        - A dictionary mapping channel names to their threshold values (in calibrated PE units)
    """
    cache_key = (metadata_dir, timestamp)
    if use_cache and cache_key in _overrides_cache:
        sandict = _overrides_cache[cache_key]
    else:
        db = TextDB(os.path.join(metadata_dir, "dataprod/overrides/hit"))
        rawdict = db.on(timestamp) # contains also ged, pmt, ...
        assert isinstance(rawdict, AttrsDict)
        sandict = {k: v for k, v in rawdict.items() if re.match(r'^S\d{3}$', k)}
        if use_cache:
            _overrides_cache[cache_key] = sandict
    return read_dict(sandict, include_aux)

    
def read_dict(data: dict[str, Any], include_aux: bool = False) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """
    Reads a dict formatted as override file and extracts calibration parameters and thresholds.
    """
    calib_output = {}
    thresholds = {}
    
    for name, content in data.items():
        if "pars" not in content or "operations" not in content["pars"]:
            continue
        operations = content["pars"]["operations"]
        
        if "energy_in_pe" in operations:
            energy_in_pe = operations["energy_in_pe"]
            if "parameters" in energy_in_pe:
                params = energy_in_pe["parameters"]
                if "a" in params and "m" in params:
                    calib_output[name] = {
                        "offset": float(params["a"]),
                        "slope": float(params["m"])
                    }
        
        if "is_valid_hit" in operations:
            is_valid_hit = operations["is_valid_hit"]
            if "parameters" in is_valid_hit:
                params = is_valid_hit["parameters"]
                if "a" in params:
                    thresholds[name] = float(params["a"])

        if include_aux and "aux" in content:
            calib_output[name] |= content["aux"]
    
    return calib_output, thresholds

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


def check_calibration_change(
    calib_output_previous: dict[str, dict[str, float]], 
    calib_output_current: dict[str, dict[str, float]], 
    thresholds_previous: dict[str, float] | None = None,
    thresholds_current: dict[str, float] | None = None,
    tolerances: dict[str, float] = {}
) -> list[str]:
    """Checks if calibration outputs deviate beyond specified tolerances.
    Returns a list of warning strings describing the deviations for each channel.
    Missing channels in either previous or current outputs are ignored (no warnings).
    Checks threshold changes only if both previous and current thresholds are provided.
    """

    tolerances = {
        "offset_rel": 0.0,
        "offset_abs": 0.01,
        "slope_rel": 0.0,
        "slope_abs": 0.01,
        "threshold_rel": 0.0,
        "threshold_abs":  0.2
    } | tolerances

    warnings = []
    for name, current_calib in calib_output_current.items():
        if name not in calib_output_previous:
            continue
        previous_calib = calib_output_previous[name]
        # Check slope
        if not math.isclose(previous_calib["slope"], current_calib["slope"], rel_tol=tolerances["slope_rel"], abs_tol=tolerances["slope_abs"]):
            warnings.append(f"Channel {name}: Slope changed from {previous_calib['slope']:.4f} to {current_calib['slope']:.4f} exceeding tolerances.")
        # Check offset
        if not math.isclose(previous_calib["offset"], current_calib["offset"], rel_tol=tolerances["offset_rel"], abs_tol=tolerances["offset_abs"]):
            warnings.append(f"Channel {name}: Offset changed from {previous_calib['offset']:.4f} to {current_calib['offset']:.4f} exceeding tolerances.")
    # Check thresholds if both provided
    if thresholds_previous is not None and thresholds_current is not None:
        for name, current_thresh in thresholds_current.items():
            if name not in thresholds_previous:
                continue
            previous_thresh = thresholds_previous[name]
            if not math.isclose(previous_thresh, current_thresh, rel_tol=tolerances["threshold_rel"], abs_tol=tolerances["threshold_abs"]):
                warnings.append(f"Channel {name}: Threshold changed from {previous_thresh:.4f} to {current_thresh:.4f} exceeding tolerances.")
    return warnings