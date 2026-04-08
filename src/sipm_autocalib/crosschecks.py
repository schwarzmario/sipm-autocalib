"""Cross-checks with other tiers (apart from DSP)"""


import glob
from collections.abc import Sequence
from collections import defaultdict

import matplotlib.pyplot as plt

import numpy as np
import awkward as ak

from lgdo import read_as
from dbetto import AttrsDict

from .utils import auto_subplots


def check_evt(path: str, channel: str, chmap: AttrsDict):
    """Plot an energy spectrum for a given channel from the evt tier.
    path should point to a directory containing lh5 files from the evt tier."""
    files = glob.glob(path+"/*.lh5")
    evt_e = read_as(f"evt/spms/energy", files, "ak")
    evt_rawid = read_as(f"evt/spms/rawid", files, "ak")
    raw_id = chmap[channel].daq.rawid
    evt_e = ak.flatten(evt_e[evt_rawid==raw_id], axis=2)
    hist, edges = np.histogram(ak.to_numpy(ak.flatten(evt_e)), bins=100, range=(0, 5))
    fig, ax = plt.subplots()
    ax.stairs(hist, edges)
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="x")
    ax.set_title(channel)

def check_evt_separate(path: str, channel: str | list[str] | None, chmap: AttrsDict, *,
                       chunk_size: int = 0, file_selection: Sequence[int] | None = None, plot_range=(0, 5)):
    """ Plot energy spectra for different files in the evt tier separately; useful for identifying drifts 
    in calibration and rates over time. path should point to a directory containing lh5 files from the evt tier.
    channel can be a single channel, a list or all (=None),
    chunk_size: how many files should be chnunked together. chunk_size=0 uses an automatic range.
    """
    if isinstance(channel, str):
        channel = [channel]
    if channel is None:
        channel = sorted(list(chmap.keys()))
    channel.append(channel[-1]) # so we can plot the legend in the last plot
    fig, ax = auto_subplots(len(channel))
    ax = ax.ravel()
    files = glob.glob(path+"/*.lh5")
    indices = list(range(0,len(files)))
    if file_selection is not None:
        files = [files[i] for i in file_selection]
        indices = list(file_selection)
    if chunk_size == 0:
        chunk_size = len(files) // 5
    files = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    indices = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
    histos = defaultdict(dict)
    for file, index in zip(files, indices):
        evt_e = read_as(f"evt/spms/energy", file, "ak")
        evt_rawid = read_as(f"evt/spms/rawid", file, "ak")
        for ch in channel:
            raw_id = chmap[ch].daq.rawid
            evt_e_x = ak.flatten(evt_e[evt_rawid==raw_id], axis=2)
            hist, edges = np.histogram(ak.to_numpy(ak.flatten(evt_e_x)), bins=100, range=plot_range)
            label = f"file {index}" if isinstance (index, int) else f"files {index[0]}-{index[-1]}"
            fileid = ";".join(file)
            histos[fileid][ch] = (hist, edges, label)
        #ax.stairs(hist, edges, label=label)
    for ch, axx in zip(channel, ax):
        for file in files:
            fileid = ";".join(file)
            axx.stairs(*(histos[fileid][ch][:2]), label = histos[fileid][ch][2])
        axx.set_yscale("log")
        axx.grid(True, which="both", axis="x")
        if axx is ax[len(channel)-1]:
            axx.legend()
        axx.set_title(f"{ch}")
    fig.suptitle(f"{'/'.join(path.split('/')[-3:])}")