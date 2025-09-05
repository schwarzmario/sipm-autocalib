import glob
import os
from collections.abc import Callable, Sequence, Iterator, Mapping
from abc import ABC, abstractmethod
from typing import Any
import argparse
import yaml
import re

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import hist

import numpy as np
import awkward as ak

import scipy

from lgdo import lh5
from lgdo.lh5.exceptions import LH5DecodeError
from legendmeta import LegendMetadata
from dspeed.processors import get_multi_local_extrema




def get_nopulser_mask(orig_dsp_file: Sequence[str] | str) -> ak.Array:
    trap_puls = lh5.read_as(f"ch{chmap['PULS01'].daq.rawid}/dsp/trapTmax", orig_dsp_file, "ak")
    return trap_puls < 100

def get_energies(dsp_file: Sequence[str] | str, keys: Iterator[int], chmap, *, 
                 orig_dsp_file: Sequence[str] | str | None = None,
                 take_pulser_from_normal: bool = False
                 ):
    """if orig_dsp_file is given: remove the pulser based on get_nopulser_mask"""
    keys: list[int] = list(keys) # I need to access element 0 separately

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

    energy_object_name_fcn = get_energy_object_name_function(dsp_file if isinstance(dsp_file, str) else dsp_file[0], keys[0], chmap.map("daq.rawid")[keys[0]].name)
    energies_dict = {}
    #print(f"{len(keys)} keys in dsp files")
    for ch in keys:
        name = chmap.map("daq.rawid")[ch].name
        #energy = lh5.read_as(f"{name}/dsp/energy", f_dsp, "ak")
        energy = lh5.read_as(energy_object_name_fcn(ch, name), dsp_file, "ak")
        # remove pulser if we have original DSP files (containing pulser info)
        # TODO perf: cache nopulser_mask
        if orig_dsp_file is not None or take_pulser_from_normal:
            if orig_dsp_file is not None:
                nopulser_mask = get_nopulser_mask(orig_dsp_file)
            else:
                nopulser_mask = get_nopulser_mask(dsp_file)
            if len(nopulser_mask) < len(energy):
                raise RuntimeError("Nopulser mask too short")
            elif len(nopulser_mask) > len(energy):
                nopulser_mask = nopulser_mask[:len(energy)]
            energy = energy[nopulser_mask]

        energies = np.array(ak.flatten(energy))

        energies_dict[name] = energies
        
    energies_dict = dict(sorted(energies_dict.items()))
    
    return energies_dict

def gen_hist_by_quantile(data, quantile=0.99, nbins=200):
    bins = np.linspace(0, np.round(np.quantile(data, quantile)), nbins+1)
    n, be = np.histogram(data, bins)
    return n, be

def gen_hist_by_range(data, range, nbins=200):
    n, be = np.histogram(data, range=range, bins=nbins)
    return n, be


def plot_all_pe_spectra(energies_dict) -> Figure:
    fig, ax = plt.subplots(10, 6, figsize=(20,20))
    ax = ax.ravel()
    for i, (name, data) in enumerate(energies_dict.items()):
        n, be = gen_hist_by_quantile(data, 0.96)
        ax[i].stairs(n, be)
        ax[i].set_yscale("log")
        ax[i].set_title(name, fontsize=10)
    fig.tight_layout()
    return fig

def plot_all_pe_histograms(histos: dict[str, dict[str, Any]], *, gridx = False) -> Figure:
    fig, ax = plt.subplots(10, 6, figsize=(20, 20))
    ax = ax.ravel()
    for i, (name, histo) in enumerate(histos.items()):
        ax[i].set_yscale('log')
        ax[i].stairs(histo["n"], histo["be"])
        ax[i].set_title(name)
        if gridx:
            ax[i].grid(axis='x')
    fig.tight_layout()
    return fig


# - - - - - - SIMPLE CALIBRATION - - - - - - - - -

def find_pe_peaks_in_hist(n, be, params: Mapping[str, Any]) -> np.typing.NDArray[np.int_]:
    n = np.array(n)
    be = np.array(be)

    # Outputs
    vt_max_out = np.zeros(shape=len(n) - 1)
    vt_min_out = np.zeros(shape=len(n) - 1)
    n_max_out = 0
    n_min_out = 0

    # Call the function with updated parameters
    get_multi_local_extrema(
        n,
        params["a_delta_max_in"] * np.max(n),
        params["a_delta_min_in"] * np.max(n),
        params["search_direction"],
        params["a_abs_max_in"] * np.max(n),
        params["a_abs_min_in"] * np.max(n),
        vt_max_out,
        vt_min_out,
        n_max_out,
        n_min_out,
    ) # type: ignore

    peakpos_indices = vt_max_out[~np.isnan(vt_max_out)].astype(np.int_)
    return peakpos_indices

class ResultCheckError(ValueError):
    def __init__(self, *args):
        super().__init__(*args)

def check_and_improve_PE_peaks(
        peakpos_indices: np.typing.NDArray[np.int_], 
        n: np.typing.NDArray[Any],
        be: np.typing.NDArray[Any],
        params: Mapping[str, Any]
        ) -> tuple[np.typing.NDArray[np.int_], dict[int, list[Any]]]:
    """
    Checks and improves the indices of detected photoelectron (PE) peaks in a histogram.
    Applies strict or non-strict validation rules to ensure the peaks correspond to expected
    noise and PE peaks, and optionally handles double-peak SiPMs.

    Parameters
    ----------
    peakpos_indices : np.typing.NDArray[np.int_]
    Indices of detected peaks in the histogram.
    n : np.typing.NDArray[Any]
    Histogram bin counts.
    be : np.typing.NDArray[Any]
    Histogram bin edges.
    params : Mapping[str, Any]
    Dictionary of peak finding and checking parameters. Keys include:
        - "double_peak": bool, whether to handle double-peak SiPMs.
        - "min_peak_dist": int, minimum distance between peaks (in bins).
        - "strict": bool, whether to apply strict validation.
        - "peakdist_compare_margin": int, margin for peak distance comparison.

    Returns
    -------
    tuple[np.typing.NDArray[np.int_], dict[int, list[Any]]]
    Validated and possibly improved peak indices, and a mapping of peak indices to 
    all bin edge values (i.e. two values for 1PE peak if double peak)

    Raises
    ------
    ResultCheckError
    If validation fails under strict mode or insufficient peaks are found.
    """
    if params.get("double_peak", False):
        if len(peakpos_indices) < 3:
            raise ResultCheckError(f"Require at least 3 found peaks for double-peak SiPMs; found only {len(peakpos_indices)}.")
        peakpos_map = {0: [be[peakpos_indices[0]]], 1: [be[peakpos_indices[1]], be[peakpos_indices[2]]]} | {i: [be[peakpos_indices[i+1]]] for i in range(2, len(peakpos_indices)-1)}
        mean_1_2 = (peakpos_indices[1] + peakpos_indices[2]) // 2 # small bias but ok
        peakpos_indices = np.concatenate((np.array([peakpos_indices[0], mean_1_2], dtype=np.int_), peakpos_indices[3:]))
    else:
        peakpos_map = {i: [be[peakpos_indices[i]]] for i in range(0, len(peakpos_indices))}

    min_peak_dist = params["min_peak_dist"]
    if params["strict"]:
        if len(peakpos_indices) < 2:
            raise ResultCheckError(f"Only {len(peakpos_indices)} peaks found. Either noise peak or 1pe peak not found.")

        if n[peakpos_indices[1]] > n[peakpos_indices[0]]:
            raise ResultCheckError(f"1pe peak larger than noise peak.")

        if peakpos_indices[1] - peakpos_indices[0] < min_peak_dist:
            raise ResultCheckError(f"Noise peak and 1pe peak too close together (< {min_peak_dist} bins).")

        if len(peakpos_indices) > 2:
            if peakpos_indices[2] - peakpos_indices[1] < min_peak_dist:
                raise ResultCheckError(f"1pe peak and 2pe peak too close together (< {min_peak_dist} bins).")
            if (peakpos_indices[2] - peakpos_indices[1] + params["peakdist_compare_margin"]) < peakpos_indices[1] - peakpos_indices[0]:
                raise ResultCheckError(f"Distance between 1pe and 2pe smaller than 0pe and 1pe (outside peakdist_compare_margin of {params["peakdist_compare_margin"]}).")
    else:
        if len(peakpos_indices) > 2:
            if peakpos_indices[2] - peakpos_indices[1] < min_peak_dist:
                print(f"1pe peak and 2pe peak too close together (< {min_peak_dist} bins). Removing '1pe' peak.")
                peakpos_indices = peakpos_indices[peakpos_indices != peakpos_indices[1]]
    return  peakpos_indices, peakpos_map

def simple_calibration(energies, gen_hist_params: Mapping[str, Any], 
                       peakfinder_params: Mapping[str, Any],
                       calibration_params: Mapping[str, Any], *, 
                       ax = None, verbosity = 0) -> dict[str, float | dict[int, list[Any]]]:
    """Generate histogram and perform a simple peakfinder-based calibration. 
    If an axis is provided: plot on that (otherwise don't plot)
    Does this for 1 SiPM; i.e. energies has to be a 1-d array of energies
    Returns calibration and dict of PE-indices to list of peak positions (in a.u.) found for these"""
    match gen_hist_params:
        case {"quantile": quantile, "nbins": nbins}:
            n, be = gen_hist_by_quantile(energies, quantile, nbins)
        case {"range": r, "nbins": nbins}:
            n, be = gen_hist_by_range(energies, r, nbins)
        case _:
            raise TypeError("gen_hist_params does not match valid histogram type")
    peakpos_indices = find_pe_peaks_in_hist(n, be, peakfinder_params)
    failed_checks = False
    try:
        peakpos_indices, peakpos_map = check_and_improve_PE_peaks(peakpos_indices, n, be, peakfinder_params)
    except ResultCheckError:
        failed_checks = True
        peaks = be[peakpos_indices] # use old indices for peaks
        raise # runs finally before raise
    else: 
        peaks = be[peakpos_indices]

        if len(peaks) > 2: # use 1PE and 2PE
            gain = peaks[2] - peaks[1]
            c = 1/gain
            offset = 1 - peaks[1] * c # 1pe peak at 1
        else: 
            if calibration_params.get("use_1pe_0pe_diff_as_fallback", False):
                # fallback: use 0 PE and 1 PE - DISCOURAGED because distance usually too small!
                gain = peaks[1] - peaks[0]
                c = 1/gain
                offset = 1 - peaks[1] * c # 1pe peak at 1   
            else:
                # fallback: use only position of 1 PE
                gain = peaks[1]
                c = 1/gain
                offset = 0

        # runs finally (i.e. plot) before return
        return {"slope": c, "offset": offset, "peaks": peakpos_map}
    finally: # draw in any case (for debugging); but choose color
         if ax is not None:
            hist_color = "red" if failed_checks else "blue"
            line_color = "grey" if failed_checks else "red"
            # uncalibrated histogram and peaks
            ax.stairs(n, be, color=hist_color)
            for p in peaks: # type: ignore
                ax.axvline(x=p, color=line_color, ls=":")

def multi_simple_calibration(energies_dict, 
                             gen_hist_defaults: dict[str, Any], 
                             peakfinder_defaults: dict[str, Any],
                             calibration_defaults: dict[str, Any], *,
                             gen_hist_overrides: dict[str, dict[str, Any]] = {}, 
                             peakfinder_overrides: dict[str, dict[str, Any]] = {},
                             calibration_overrides: dict[str, dict[str, Any]] = {},
                             draw = False, 
                             nodraw_axes = False, 
                             verbosity = 0
                             ) -> tuple[dict[str, dict[str, float]], int, Figure | None]:
    """Performs simple_calibration for all channels present in energies_dict"""
    ret = {}
    fig = None
    if draw:
        fig, ax = plt.subplots(10, 6, figsize=(20,20))
        ax_iter = iter(ax.ravel())
    nr_unsuccessful_calibs = 0
    for name, energies in energies_dict.items():
        if draw:
            ax = next(ax_iter)
            ax.set_yscale("log")
            if nodraw_axes:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            ax.set_title(name, fontsize=10)
        else:
            ax = None
        try:
            calib_results = simple_calibration(
                energies,
                gen_hist_defaults | gen_hist_overrides.get(name, {}),
                peakfinder_defaults | peakfinder_overrides.get(name, {}),
                calibration_defaults | calibration_overrides.get(name, {}),
                ax=ax,  verbosity=verbosity)
            ret[name] = calib_results
        except ResultCheckError as e:
            print(f"Calibration failed for {name}: {e}")
            ret[name] = {"slope": np.nan, "offset": np.nan, "peaks": {}}
            nr_unsuccessful_calibs += 1
    
    if nr_unsuccessful_calibs > 0 and verbosity >= -1:
        print(f"WARNING: {nr_unsuccessful_calibs} calibrations failed!")
    elif verbosity >= 0:
        print("Info: All simple calibrations successful! :)")

    if draw:
        fig.tight_layout()
        if nodraw_axes: # have to do this also for non-drawn plots
            for ax in ax_iter:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            fig.subplots_adjust(wspace=0) # , hspace=0)
    return ret, nr_unsuccessful_calibs, fig


def get_calibrated_histograms(energies, calib_output, range: tuple[float, float], nbins:int):
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
    ret = {}
    for name, calib in calib_output.items():
        ret[name] = {}
        c = calib["slope"]
        offset = calib["offset"]
        for pe_id, peaks in calib["peaks"].items():
            ret[name][pe_id] = [p * c + offset for p in peaks]
    return ret


# - - - - - - - - ADVANCED CALIBRATION - - - - - - - - - - -

class ModelParameter:
    def __init__(self, init: tuple[float,float,float] | float):
        if isinstance(init, tuple):
            self.init = init[0]
            self.min = init[1]
            self.max = init[2]
        else:
            self.init = init
            self.min = -np.inf
            self.max = np.inf
        self.result: np.float64 = np.float64(np.nan)
    def set_result(self, result):
        self.result = result

class ModelComponent(ABC):
    def __init__(self, params: dict[str, ModelParameter]):
        self.params = params
    def nr_params(self) -> int:
        return len(self.params)
    @abstractmethod
    def eval(self, x, params) -> np.float64:
        pass
    def set_result_params(self, params):
        for res, par in zip(params, self.params.values()):
            par.set_result(res)
    def get_result_params(self) -> Sequence[np.float64]:
        return [p.result for p in self.params.values()]
    def print_results(self) -> None:
        for name, param in self.params.items():
            print(name, param.result)
        
class Gauss(ModelComponent):
    def __init__(self, mean, sigma, scale):
        super().__init__({
            "mean": ModelParameter(mean),
            "sigma": ModelParameter(sigma),
            "scale": ModelParameter(scale)
        })
    def eval(self, x, params):
        return params[2] * scipy.stats.norm.pdf(x, loc=params[0], scale=params[1])
    
class ExpoDec(ModelComponent):
    def __init__(self, lamb, scale):
        super().__init__({
            "lamb": ModelParameter(lamb),
            "scale": ModelParameter(scale)
        })
    def eval(self, x, params):
        return params[1]*np.exp(-1*params[0]*x)
    
class TwoHyperbole(ModelComponent):
    def __init__(self, p0, p1, p2):
        super().__init__({
            "p0": ModelParameter(p0),
            "p1": ModelParameter(p1),
            "p2": ModelParameter(p2),
        })
    def eval(self, x, params):
        return params[0] + params[1]/x + params[2]/(x*x)
    
class Linear(ModelComponent):
    def __init__(self, p0, p1):
        super().__init__({
            "p0": ModelParameter(p0),
            "p1": ModelParameter(p1),
        })
    def eval(self, x, params):
        return params[0] + params[1]*x
    
class SumModel(ModelComponent):
    def __init__(self, components: dict[str, ModelComponent]):
        passed_params = {}
        for m_name, model in components.items():
            for p_name, param in model.params.items():
                passed_params[m_name+"."+p_name] = param
        super().__init__(passed_params)
        self.components = components
    def eval(self, x, params) -> np.float64:
        curr = 0
        ret = np.float64(0)
        for comp in self.components.values():
            ret += comp.eval(x, params[curr:curr+comp.nr_params()])
            curr += comp.nr_params()
        return ret

def evaluate(model_component: ModelComponent, x, param_values: list[Any]):
    if not isinstance(param_values, list):
        raise ValueError("param_values has to be a list so it can be modified")
    ret = model_component.eval(x, param_values[:model_component.nr_params()])
    del param_values[:model_component.nr_params()]
    return ret

def evaluate_at_result(model_component: ModelComponent, x):
    return model_component.eval(x, model_component.get_result_params())

def get_inits(model_components: list[ModelComponent]) -> list[float]:
    ret = []
    for mc in model_components:
        for p in mc.params.values():
            ret.append(p.init)
    return ret

def get_upper_bounds(model_components: list[ModelComponent]) -> list[float]:
    ret = []
    for mc in model_components:
        for p in mc.params.values():
            ret.append(p.max)
    return ret

def get_lower_bounds(model_components: list[ModelComponent]) -> list[float]:
    ret = []
    for mc in model_components:
        for p in mc.params.values():
            ret.append(p.min)
    return ret

class Fittable:
    def __init__(self, model: ModelComponent, fit_range: tuple[float, float]):
        self.model = model
        self.fit_range = fit_range
    def fit(self, bin_weights, bin_centers): # raises RuntimeError if fit failed
        fit_range_mask = (bin_centers >= self.fit_range[0]) & (bin_centers <= self.fit_range[1])
        range_bin_weights = bin_weights[fit_range_mask]
        range_bin_centers = bin_centers[fit_range_mask]
        def model_fcn(x, *p):
            params = list(p)
            ret = evaluate(self.model, x, params)
            assert len(params) == 0
            return ret
        return scipy.optimize.curve_fit(
                model_fcn, range_bin_centers, range_bin_weights, p0=get_inits([self.model]), 
                bounds=(get_lower_bounds([self.model]), get_upper_bounds([self.model])))
    def draw(self, ax, params, color):
        xx = np.linspace(self.fit_range[0], self.fit_range[1], 1000)
        ax.plot(xx, self.model.eval(xx, params), color=color)

def check_gauss_fit_results(gausses: list[Gauss]) -> None:
    gauss_means = [g.params["mean"].result for g in gausses]
    if len(gauss_means) < 2:
        #raise ResultCheckError(f"Too little nr of gausses {len(gauss_means)}")
        pass # there are SiPMs with only 1 gauss...
    for i, mean in enumerate(gauss_means):
        peak_expect = i+1
        if mean < gausses[i].params["mean"].min+0.05 or mean > gausses[i].params["mean"].max-0.05:
            raise ResultCheckError(f"Mean of PE peak #{peak_expect} out of range: {mean}")
        if i > 0:
            if abs(gauss_means[i] - gauss_means[i-1] - 1) > 0.2:
                raise ResultCheckError(f"Distance between mean of PE peaks {peak_expect-1},{peak_expect} too far off 1: {gauss_means[i] - gauss_means[i-1]}")
    for i, gauss in enumerate(gausses):
        if gauss.params["sigma"].result > 10:
            raise ResultCheckError(f"Sigma of PE peak {i+1} way too large: {gauss.params["sigma"].result}")
    
def check_bkg_fit_results(bkg_models: list[ModelComponent], fit_range: tuple[float, float]) -> None:
    if len(bkg_models) == 0:
        return
    val_at_range_begin = np.sum([evaluate_at_result(mc, fit_range[0]) for mc in bkg_models])
    val_at_range_end = np.sum([evaluate_at_result(mc, fit_range[1]) for mc in bkg_models])
    if val_at_range_end > val_at_range_begin:
        # probably trying to fit following peak
        raise ResultCheckError(f"Background model rises over range; from {val_at_range_begin} to {val_at_range_end}.")

def advanced_calibration(
        precalibrated_histo: dict[str, np.typing.NDArray[Any]],
        calibrated_PE_positions: dict[int, list[Any]],
        params: Mapping[str, Any], *,
        ax = None, nofit=False, verbosity = 0
        ) -> dict[str, float]:
    
    n = precalibrated_histo["n"]
    be = precalibrated_histo["be"]
    be_mid = (be[:-1] + be[1:]) / 2
    assert len(be_mid) == len(be) - 1

    gaussians = {}
    for pe_id in range(1, params["max_nr_gausspeaks"]+1):
        try:
            peaks = calibrated_PE_positions[pe_id]
        except KeyError:
            break # don't fit more gaussians than PE found
        peakdiff = None
        if len(peaks) >= 2:
            peakdiff = np.max(peaks) - np.min(peaks)
        max_in_range = np.max(n[(be_mid >= pe_id-0.5) & (be_mid <= pe_id+0.5)])
        gauss = Gauss(
            (pe_id, pe_id-params["gauss_mean_range_low"], pe_id+params["gauss_mean_range_high"]),
            (peakdiff*2, peakdiff, np.inf) if peakdiff else 0.1,
            (max_in_range/(5), 0, np.inf) # was (max_in_range/(3+pe_id*2), 0, np.inf)
        )
        gaussians[f"gauss{pe_id}"] = gauss

    fit_range = (1 - params["fit_range_prePE"], len(gaussians) + params["fit_range_pastPE"])
    fit_range_mask = (be_mid >= fit_range[0]) & (be_mid <= fit_range[1])
    max_in_range = np.max(n[fit_range_mask])

    fittables: list[Fittable] = []
    
    if params["model"] == "combo":
        expodec = ExpoDec((2, 0, np.inf), (max_in_range/2, 0, np.inf))
        linear = Linear((max_in_range/100, 0, np.inf), -10)
        #th = TwoHyperbole(max_in_range/2, 100, (0,-1,1))
        backgrounds = {"expodec": expodec, "linear": linear}
        
        fittables.append(Fittable(SumModel(gaussians | backgrounds), fit_range))
    elif params["model"] == "individual":
        backgrounds = {}
        for i, gauss in enumerate(gaussians.values(), 1):
            fittables.append(Fittable(gauss, (i-params["fit_range_prePE"], i+params["fit_range_pastPE"])))
    else:
        raise ValueError(f"Invalid \"model\" parameter: {params['model']}")

    failure: str = ""
    try:
        try:
            if nofit:
                raise RuntimeError("No fit performed, as requested")
            for fi in fittables:
                fitted_params, pcov = fi.fit(n, be_mid)
                fi.model.set_result_params(fitted_params)
        except RuntimeError as e:
            for fi in fittables:
                fi.model.set_result_params(get_inits([fi.model]))
            failure = "fit"
            raise ResultCheckError(e) from e
        try:
            check_gauss_fit_results(list(gaussians.values()))
            check_bkg_fit_results(list(backgrounds.values()), fit_range)
        except ResultCheckError as e:
            failure = "check"
            raise
    except ResultCheckError:
        raise
    else:
        #TODO: do calibration in this case!
        if len(gaussians) >= 2: # proper calibration using 1 PE and 2 PE; getting offset
            gain = gaussians["gauss2"].params["mean"].result - gaussians["gauss1"].params["mean"].result
            c = 1/gain
            offset = 1 - gaussians["gauss1"].params["mean"].result * c # 1pe peak at 1
        else: # fallback: use 1 PE only
            gain = gaussians["gauss1"].params["mean"].result
            c = 1/gain
            offset = 0
        # runs finally (i.e. plot) before return
        return {"slope": c, "offset": offset}
    finally: # runs in any case; exception or not
        if verbosity > 2:
            for fi in fittables:
                fi.model.print_results()
        if ax is not None:
            ax.stairs(n, be)
            match failure:
                case "":
                    color="green"
                case "fit":
                    color="red"
                case "check":
                    color="orange"
            for fi in fittables:
                fi.draw(ax, fi.model.get_result_params(), color)
            ax.set_ylim(((np.min(n) if np.min(n) > 0 else 0.5)*0.9, np.max(n)*1.1))

def multi_advanced_calibration(calibrated_histo_dict, 
                               calibrated_PE_positions: dict[str, dict[int, list[Any]]],
                             calibration_defaults: dict[str, Any], *,
                             calibration_overrides: dict[str, dict[str, Any]] = {},
                             draw = False, 
                             nodraw_axes = False, 
                             verbosity = 0
                             ) -> tuple[dict[str, dict[str, float]], int, Figure | None]:
    """Performs advanced_calibration for all channels present in calibrated_histo_dict"""
    ret = {}
    fig = None
    if draw:
        fig, ax = plt.subplots(10, 6, figsize=(20,20))
        ax_iter = iter(ax.ravel())
    nr_unsuccessful_calibs = 0
    for name, calibrated_histo in calibrated_histo_dict.items():
        if draw:
            ax = next(ax_iter)
            ax.set_yscale("log")
            if nodraw_axes:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            ax.set_title(name, fontsize=10)
        else:
            ax = None
        try:
            calib_results = advanced_calibration(
                calibrated_histo,
                calibrated_PE_positions[name],
                calibration_defaults | calibration_overrides.get(name, {}),
                ax=ax,  verbosity=verbosity)
            ret[name] = calib_results
        except ResultCheckError as e:
            print(f"Calibration failed for {name}: {e}")
            ret[name] = {"slope": np.nan, "offset": np.nan}
            nr_unsuccessful_calibs += 1
    
    if nr_unsuccessful_calibs > 0 and verbosity >= -1:
        print(f"WARNING: {nr_unsuccessful_calibs} calibrations failed!")
    elif verbosity >= 0:
        print("Info: All advanced calibrations successful! :)")

    if draw:
        fig.tight_layout()
        if nodraw_axes: # have to do this also for non-drawn plots
            for ax in ax_iter:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            fig.subplots_adjust(wspace=0) # , hspace=0)
    return ret, nr_unsuccessful_calibs, fig


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
    for name in calib1.keys() & calib2.keys():
        ret[name] = combine_calibration(calib1[name], calib2[name])
    return ret

def full_calibration_chain(
        energies_dict: dict[str, np.typing.NDArray[Any]], *,
        gen_hist_defaults: dict[str, Any], 
        peakfinder_defaults: dict[str, Any],
        simple_calibration_defaults: dict[str, Any],
        advanced_calibration_defaults: dict[str, Any],
        gen_hist_overrides: dict[str, dict[str, Any]] = {}, 
        peakfinder_overrides: dict[str, dict[str, Any]] = {},
        simple_calibration_overrides: dict[str, dict[str, Any]] = {},
        advanced_calibration_overrides: dict[str, dict[str, Any]] = {},
        plot_output_dir: str | None = None, # not drawing when None
        plot_interactive: bool = False, # interactive plot, e.g. in jupyter notebook
        verbosity: int = 0
        ) -> dict[str, dict[str, float]]:
    draw: bool = plot_output_dir is not None or plot_interactive
    last_fig_path = ""
    def store(figure: Figure | None, filebasename: str):
        if plot_output_dir is None or figure is None:
            return
        last_fig_path = os.path.join(plot_output_dir, filebasename + ".pdf")
        figure.savefig(last_fig_path)
    def get_hint() -> str:
        if plot_output_dir is not None:
            return f"Please review {last_fig_path} for failures of individual SiPMs."
        elif plot_interactive:
            return f"Please review the last interactive plot for failures of individual SiPMs."
        else:
            return "Please re-run with active plotting to identify failures."

    simple_calib_output, nr_failed_simple_calib, fig = multi_simple_calibration(
        energies_dict, gen_hist_defaults, peakfinder_defaults, simple_calibration_defaults, 
        gen_hist_overrides = gen_hist_overrides,
        peakfinder_overrides = peakfinder_overrides,
        calibration_overrides=simple_calibration_overrides,
        draw=draw,
        verbosity=verbosity
    )
    store(fig, "simple_calibration_peaks")
    if nr_failed_simple_calib > 0:
        raise ValueError(f"Simple Calibration failed for {nr_failed_simple_calib} out of {len(energies_dict)} SiPMs.\n" + 
                         get_hint() + "\nFailed histograms are drawn in red.")
    
    simple_calibrated_histos = get_calibrated_histograms(
        energies, simple_calib_output, 
        (advanced_calibration_defaults.get("histogram_begin", 0), advanced_calibration_defaults.get("histogram_end", 5)), 
        advanced_calibration_defaults.get("histogram_nbins", 200)
        )
    if draw:
        fig = plot_all_pe_histograms(simple_calibrated_histos, gridx=True)
        store(fig, "simple_calibration_result")

    advanced_calib_output, nr_failed_advanced_calib, fig = multi_advanced_calibration(
        simple_calibrated_histos, get_calibrated_PE_positions(simple_calib_output), 
        advanced_calibration_defaults, 
        calibration_overrides=advanced_calibration_overrides, 
        draw=draw,
        verbosity=verbosity)
    store(fig, "advanced_calibration_fits")
    if nr_failed_advanced_calib > 0:
        raise ValueError(f"Advanced Calibration failed for {nr_failed_simple_calib} out of {len(energies_dict)} SiPMs.\n" + 
                         get_hint() + "\nFailed fits are drawn in red (with initial parameters); Orange indicates failed checks.")
    
    final_calib_output = combine_multiple_calibrations(simple_calib_output, advanced_calib_output)
    if draw:
        adv_calibrated_histos = get_calibrated_histograms(energies, final_calib_output, (0, 5), 200)
        fig = plot_all_pe_histograms(adv_calibrated_histos, gridx=True)
        store(fig, "final_calibration_result")
    return final_calib_output

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


if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (10, 4)

    parser = argparse.ArgumentParser(
            prog='autocalibrator',
            description='Calibrate LEGEND SiPM spectra')
    parser.add_argument('--dsp-files', help="Input dsp tier file(s) or directory", nargs='+')
    parser.add_argument('--pulser-dsp-dir', help="DSP files containing pulser data; for removing pulser events")
    parser.add_argument('--metadata-dir', help="LEGEND metadata directory", required=True)
    parser.add_argument('--config', help="YAML config used for setting autocalibration parameters", required=True)
    parser.add_argument('--output-dir', help="Directory to store output plots and override file", required=True)

    args = parser.parse_args()

    if os.path.isdir(args.dsp_files):
        dsp_dir = args.dsp_files
        dsp_files = glob.glob(dsp_dir+"/l200-*-tier_dsp.lh5")
    elif isinstance(args.dsp_files, str):
        dsp_dir = os.path.dirname(args.dsp_files)
        dsp_files = [args.dsp_files]
    else:
        dsp_files = args.dsp_files
        dsp_dir = os.path.dirname(dsp_files[0])
    dsp_files.sort()
    if len(dsp_files) == 0:
        raise ValueError(f"No dsp files found in {dsp_dir}")
    if not os.path.isdir(args.metadata_dir):
        raise ValueError(f"Metadata dir {args.metadata_dir} not found")
    if not os.path.isfile(args.config):
        raise ValueError(f"Config file {args.config} not found")

    def get_timestamp_from_filename(filename: str) -> str | None:
        match = re.search(r"\d{8}T\d{6}Z", filename)
        return match.group(0) if match else None

    lmeta  = LegendMetadata(args.metadata_dir)
    chmap = lmeta.channelmap(get_timestamp_from_filename(dsp_files[0]))
    chmap_sipm = chmap.map("system", unique=False).spms
    #requires recent legend-datasets
    raw_keys = chmap_sipm.map("analysis.usability", unique=False).on.map("daq.rawid").keys()

    def gimme_orig_dsp_filename(dspfilename: str):
        # get the original dsp files so I get pulser info
        return dspfilename.replace(dsp_dir, orig_dsp_dir)

    orig_dsp_files = None
    if args.pulser_dsp_dir is not None:
        orig_dsp_dir = args.pulser_dsp_dir
        if not os.path.isdir(orig_dsp_dir):
            raise ValueError(f"Pulser DSP dir {orig_dsp_dir} not found")
        orig_dsp_files = [gimme_orig_dsp_filename(f) for f in dsp_files]
        for f in orig_dsp_files:
            if not os.path.isfile(f):
                raise ValueError(f"Pulser DSP file {f} not found")
    take_pulser_from_normal = False
    if orig_dsp_files is None:
        try:
            get_nopulser_mask(dsp_files[0])
        except LH5DecodeError:
            print("WARNING: Cannot remove pulser; Pulser will be present in data!")
        else:
            take_pulser_from_normal = True
    energies = get_energies(dsp_files, raw_keys, chmap, 
                            orig_dsp_file=orig_dsp_files, take_pulser_from_normal=take_pulser_from_normal)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    final_calib_output = full_calibration_chain(
        energies, 
        gen_hist_defaults = config["gen_hist_defaults"],
        peakfinder_defaults = config["peakfinder_defaults"],
        simple_calibration_defaults = config.get("simple_calibration_defaults", {}),
        advanced_calibration_defaults = config["advanced_calibration_defaults"],
        gen_hist_overrides = config.get("gen_hist_overrides", {}),
        peakfinder_overrides = config.get("peakfinder_overrides", {}),
        simple_calibration_overrides = config.get("simple_calibration_overrides", {}),
        advanced_calibration_overrides = config.get("advanced_calibration_overrides", {}),
        plot_output_dir = args.output_dir,
        verbosity = 0
    )

    output_override_file(final_calib_output, os.path.join(args.output_dir, "autocalibration_overrides.yaml"))

    print("Done!")

    