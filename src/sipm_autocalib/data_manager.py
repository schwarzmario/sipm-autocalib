import glob
import os
from collections.abc import Iterable
from abc import ABC, abstractmethod
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import numpy as np
import awkward as ak
from legendmeta import LegendMetadata

from .utils import get_timestamp_from_filename

class DataManager:
    def __init__(
            self, *,
            project_dir: str = "/mnt/atlas02/projects/legend/sipm_qc",
            dsp_subpath: str, # e.g. "data/tier/dsp/ssc/p16/r009" on sator
            dsp_glob: str = "/l200-*-tier_dsp.lh5",
            dsp_file_limit: int | None = None,
            metadata_dir: str | None = None,
            metadata_subpath: str | None = "metadata/legend-metadata-schwarz",
            timestamp_override: str | None = None,
            key_selection: Iterable[str] | None = None
        ):

        self.project_dir = project_dir
        self.dsp_dir = os.path.join(self.project_dir, dsp_subpath)
        self.dsp_files = glob.glob(self.dsp_dir+dsp_glob)
        self.dsp_files.sort()
        if len(self.dsp_files) == 0:
            raise RuntimeError(f"Found no files in {self.dsp_dir}")
        if dsp_file_limit is not None:
            self.dsp_files = self.dsp_files[:dsp_file_limit]
        if metadata_dir is not None:
            self.metadata_dir = metadata_dir
        elif metadata_subpath is not None:
            self.metadata_dir = os.path.join(self.project_dir, metadata_subpath)
        else:
            raise RuntimeError("No metadata directory specified")
        self.lmeta  = LegendMetadata(self.metadata_dir)
        self.chmap = self.lmeta.channelmap(get_timestamp_from_filename(self.dsp_files[0]))
        self.chmap_sipm = self.chmap.map("system", unique=False).spms
        #requires recent legend-datasets
        self.raw_keys: Iterable[int] = self.chmap_sipm.map(
            "analysis.usability", unique=False).on.map("daq.rawid").keys()
        if key_selection is not None:
            self.raw_keys = [self.chmap[k].daq.rawid for k in key_selection]
