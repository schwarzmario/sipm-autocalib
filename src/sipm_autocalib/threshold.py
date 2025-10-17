
import numpy as np
from typing import Any
from .core import ResultCheckError


def find_valley_minimum(
        calibrated_histo: dict[str, np.typing.NDArray[Any]],
        maximum_position: float = 0.8
        ) -> np.float64:
    """Returns the center position of the bin with minimum height between the 0pe and 1pe peak"""
    left_borders = calibrated_histo["be"][:-1]
    pe_zero_index = np.argmax(calibrated_histo["n"][left_borders < maximum_position])
    valley_index = np.argmin(calibrated_histo["n"][
        (left_borders >= left_borders[pe_zero_index]) & (left_borders < maximum_position)
        ]) + pe_zero_index
    if left_borders[left_borders < maximum_position][-1] == left_borders[valley_index]:
        raise ResultCheckError(f"No valley minimum found before maximum_position {maximum_position}")
    return (calibrated_histo["be"][valley_index] + calibrated_histo["be"][valley_index+1]) / 2

def multi_valley_minima(
        calibrated_histos: dict[str, dict[str, np.typing.NDArray[Any]]], *, 
        mimimum_position: float = 0.0,
        maximum_position: float = 0.8,
        overrides: dict[str, dict[str, float]] = {}
        ) -> dict[str, np.float64]:
    """Returns the center positions of the bins with minimum height between the 0pe and 1pe peak for multiple histograms"""
    config = {"minimum_position": mimimum_position, "maximum_position": maximum_position}
    ret = {}
    for sipm, histo in calibrated_histos.items():
        config_here = config.copy()
        if sipm in overrides:
            config_here.update(overrides[sipm])
        try:
            ret[sipm] = max(
                find_valley_minimum(histo, maximum_position=config_here["maximum_position"]), 
                np.float64(config_here["minimum_position"])
                )
        except ResultCheckError:
            print(f"Warning: No valley minimum found for {sipm}")
            ret[sipm] = np.nan
    return ret
    #return {sipm: max(find_valley_minimum(histo), np.float64(mimimum_position)) for sipm, histo in calibrated_histos.items()}

