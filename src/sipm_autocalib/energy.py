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



def get_nopulser_mask(orig_dsp_file: Sequence[str] | str, chmap) -> ak.Array:
    """
    Generate a boolean mask indicating events with no pulser signal in the specified DSP file(s).
    """
    # 
    trap_puls = lh5.read_as(f"ch{chmap['PULS01'].daq.rawid}/dsp/trapTmax", orig_dsp_file, "ak")
    return trap_puls < 100

def get_t0_selection(dsp_file: Sequence[str] | str, raw_key: int, name_key: str,
                     min_t0: float, max_t0: float) -> ak.Array:
    """Returns a mask which cuts all SiPM pulses for a given SiPM with t0 < min or t0 > max"""
    fcns = [lambda rawid, name: f"ch{rawid}/dsp/trigger_pos", lambda rawid, name: f"{name}/dsp/trigger_pos"]
    fcn = None
    for try_fcn in fcns:
        try:
            _ = lh5.read_as(try_fcn(raw_key, name_key), dsp_file if isinstance(dsp_file, str) else dsp_file[0], "ak")
            fcn = try_fcn
            break
        except LH5DecodeError:
            continue
    else:
        raise RuntimeError("Have no clue how to extract energy info")

    t0 = lh5.read_as(fcn(raw_key, name_key), dsp_file, "ak")
    return (t0 >= min_t0) & (t0 <= max_t0)


def get_energies(dsp_file: Sequence[str] | str, keys: Iterator[int], chmap, *, 
                 orig_dsp_file: Sequence[str] | str | None = None,
                 take_pulser_from_normal: bool = False,
                 t0_selection: tuple[float, float] | None = None
                 ):
    """
    Extracts energy arrays for specified channels from DSP files, with optional pulser event removal.

    If `orig_dsp_file` is provided, pulser events are removed based on the mask obtained from the original DSP file
    using `get_nopulser_mask`. If `orig_dsp_file` is not provided but take_pulser_from_normal is True, 
    the function attempts to remove pulser events from `dsp_file` itself. 
    Otherwise, pulser events are retained.

    The function determines the correct energy object path in the DSP file for each channel, reads the energy data,
    optionally removes pulser events, and returns a dictionary mapping channel names to their corresponding energy arrays.

    Parameters
    ----------
    dsp_file : Sequence[str] or str
        Path(s) to the DSP file(s) containing energy data.
    keys : Iterator[int]
        Iterable of raw channel IDs to extract energies for.
    chmap : object
        Channel map object (LEGEND metadata)
    orig_dsp_file : Sequence[str] or str or None, optional
        Path(s) to the original DSP file(s) containing pulser information for mask generation. 
        If None but take_pulser_from_normal is True, pulser removal is attempted from `dsp_file` itself.
    take_pulser_from_normal : bool, optional
        If True, attempts to remove pulser events from `dsp_file` even if `orig_dsp_file` is not provided.
    t0_selection : tuple[float, float] or None, optional
        If provided, a tuple specifying (min_t0, max_t0) to filter events based on their pulse
        arrival times in the SiPMs (trigger_pos). 

    Returns
    -------
    energies_dict : dict
        Dictionary mapping channel names to numpy arrays of energies, with pulser events removed if possible.

    Notes
    -----
    - The function tries several possible energy object paths in the DSP file for compatibility.
    - The returned dictionary is sorted by channel name.
    """

    _keys: list[int] = list(keys) # I need to access element 0 separately

    def get_energy_object_name_function(dsp_file: str, raw_key: int, name_key: str) -> Callable[[int, str], str]:
        if not os.path.isfile(dsp_file):
            raise RuntimeError(f"ERROR: no file: {dsp_file}")
        fcns = [lambda rawid, name: f"ch{rawid}/dsp/energy", lambda rawid, name: f"ch{rawid}/dsp/energies",
                lambda rawid, name: f"{name}/dsp/energy", lambda rawid, name: f"{name}/dsp/energies"]
        for fcn in fcns:
            try:
                _ = lh5.read_as(fcn(raw_key, name_key), dsp_file, "ak")
                return fcn
            except LH5DecodeError:
                continue
        raise RuntimeError("Have no clue how to extract energy info")

    energy_object_name_fcn = get_energy_object_name_function(dsp_file if isinstance(dsp_file, str) else dsp_file[0], _keys[0], chmap.map("daq.rawid")[_keys[0]].name)
    energies_dict = {}

    for ch in _keys:
        name = chmap.map("daq.rawid")[ch].name
        #energy = lh5.read_as(f"{name}/dsp/energy", f_dsp, "ak")
        energy = lh5.read_as(energy_object_name_fcn(ch, name), dsp_file, "ak")
        # remove pulser if we have original DSP files (containing pulser info)
        # TODO perf: cache nopulser_mask
        if orig_dsp_file is not None or take_pulser_from_normal:
            if orig_dsp_file is not None:
                nopulser_mask = get_nopulser_mask(orig_dsp_file, chmap)
            else:
                nopulser_mask = get_nopulser_mask(dsp_file, chmap)
            if len(nopulser_mask) < len(energy):
                raise RuntimeError("Nopulser mask too short")
            elif len(nopulser_mask) > len(energy):
                nopulser_mask = nopulser_mask[:len(energy)]
            energy = ak.mask(energy, nopulser_mask)

        if t0_selection is not None:
            t0_mask = get_t0_selection(dsp_file, ch, name, t0_selection[0], t0_selection[1])
            if len(t0_mask) < len(energy):
                raise RuntimeError("t0 mask too short")
            elif len(t0_mask) > len(energy):
                t0_mask = t0_mask[:len(energy)]
            energy = ak.mask(energy, t0_mask)

        energy = ak.drop_none(energy)
        energies = np.array(ak.flatten(energy))

        energies_dict[name] = energies
        
    energies_dict = dict(sorted(energies_dict.items()))
    
    return energies_dict