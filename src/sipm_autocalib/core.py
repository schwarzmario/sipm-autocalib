from typing import Any

import numpy as np

from .histograms import gen_hist_by_range

class ResultCheckError(ValueError):
    """Indicates a failed check of an automated fit or similar."""
    def __init__(self, *args):
        super().__init__(*args)



def get_calibrated_histograms(energies, calib_output, range: tuple[float, float], nbins:int):
    """
    Generates calibrated histograms (linear) from raw energies using calibration parameters.

    Parameters
    ----------
    energies : dict[str, np.ndarray]
        Dictionary mapping channel names to arrays of raw energy values
    calib_output : dict[str, dict[str, float]]
        Dictionary containing calibration parameters ('slope' and 'offset') for each channel
    range : tuple[float, float]
        Tuple specifying (min, max) range for the histogram bins
    nbins : int
        Number of bins for the histogram

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Dictionary mapping channel names to dictionaries containing:
            - 'n': histogram counts
            - 'be': bin edges
        Only includes channels with valid calibration parameters (non-NaN values)
    """
    ret: dict[str, dict[str, np.typing.NDArray[Any]]] = {}
    for name, energy in energies.items():
        if name not in calib_output:
            continue
        c = calib_output[name]["slope"]
        offset = calib_output[name]["offset"]
        if np.isnan(c) or np.isnan(offset):
            continue
        energy_calibrated = energy * c + offset
        n, be = gen_hist_by_range(energy_calibrated, range, nbins)
        ret[name] = {"n": n, "be": be}
    return ret

def get_calibrated_PE_positions(calib_output) -> dict[str, dict[int, list[Any]]]:
    """
    Applies the same calibration as in get_calibrated_histograms() for calib_output 
    (returned by multi_simple_calibration()).
    """
    ret = {}
    for name, calib in calib_output.items():
        ret[name] = {}
        c = calib["slope"]
        offset = calib["offset"]
        for pe_id, peaks in calib["peaks"].items():
            ret[name][pe_id] = [p * c + offset for p in peaks]
    return ret



# - - - - - - - - - - - - COMBINE CALIBRATIONS - - - - - - - - - 

def combine_calibration(calib1: dict[str, Any], calib2: dict[str, Any]) -> dict[str, Any]:
    """Combines two calibration outputs. calib1 has to be the initial one (from uncalibrated to intermediate),
    while calib2 is a recalibration, going from the intermediate to the final calibration.
    The resulting calibration curve takes us from the uncalibrated to the final calibration."""
    return {
        "slope": calib1["slope"] * calib2["slope"],
        "offset": calib2["slope"] * calib1["offset"] + calib2["offset"]
    }
def combine_multiple_calibrations(calib1: dict[str, dict[str, Any]], calib2: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """same as combine_calibration, but loops over all SiPMs"""
    ret = {}
    for name in sorted(calib1.keys() & calib2.keys()):
        ret[name] = combine_calibration(calib1[name], calib2[name])
    return ret