
import numpy as np
from typing import Any


def find_valley_minimum(calibrated_histo: dict[str, np.typing.NDArray[Any]]) -> np.float64:
    """Returns the center position of the bin with minimum height between the 0pe and 1pe peak"""
    left_borders = calibrated_histo["be"][:-1]
    pe_zero_index = np.argmax(calibrated_histo["n"][left_borders < 0.8])
    valley_index = np.argmin(calibrated_histo["n"][
        (left_borders >= left_borders[pe_zero_index]) & (left_borders < 0.8)
        ]) + pe_zero_index
    return (calibrated_histo["be"][valley_index] + calibrated_histo["be"][valley_index+1]) / 2

def multi_valley_minima(
        calibrated_histos: dict[str, dict[str, np.typing.NDArray[Any]]], *, 
        mimimum_position: float = 0.0
        ) -> dict[str, np.float64]:
    """Returns the center positions of the bins with minimum height between the 0pe and 1pe peak for multiple histograms"""
    return {sipm: max(find_valley_minimum(histo), np.float64(mimimum_position)) for sipm, histo in calibrated_histos.items()}

