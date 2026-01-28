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


from .utils import auto_subplots
from .simple_calibration import ResultCheckError

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

def check_gauss_fit_results(gausses: list[Gauss], *, max_peak_distance=0.2) -> None:
    """
    Run through all Gauss fit results (agnostic of actual models; has to be 1PE, 2PE, ...).
    raises a ResultCheckError if a value is out of bounds.
    """
    gauss_means = [g.params["mean"].result for g in gausses]
    for i, mean in enumerate(gauss_means):
        peak_expect = i+1
        if mean < gausses[i].params["mean"].min+0.05 or mean > gausses[i].params["mean"].max-0.05:
            raise ResultCheckError(f"Mean of PE peak #{peak_expect} out of range: {mean}")
        if i > 0:
            if abs(gauss_means[i] - gauss_means[i-1] - 1) > max_peak_distance:
                raise ResultCheckError(f"Distance between mean of PE peaks {peak_expect-1},{peak_expect} too far off 1: {gauss_means[i] - gauss_means[i-1]}")
    for i, gauss in enumerate(gausses):
        if gauss.params["sigma"].result > 10:
            raise ResultCheckError(f"Sigma of PE peak {i+1} way too large: {gauss.params["sigma"].result}")
    
def check_bkg_fit_results(bkg_models: list[ModelComponent], fit_range: tuple[float, float]) -> None:
    """
    Checks a background model (agnostic of actual model used) and raises ResultCheckError
    if there is something fishy.
    """
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
    """
    Perform advanced calibration of SiPM charge spectra using gaussian fits.
    This function fits gaussian peaks to the pre-calibrated charge spectrum to determine
    the calibration parameters (slope and offset) that convert ADC values to PE.
    Parameters
    ----------
    precalibrated_histo : dict[str, np.typing.NDArray[Any]]
        Dictionary containing the histogram data with keys:
        - 'n': bin contents (counts)
        - 'be': bin edges
    calibrated_PE_positions : dict[int, list[Any]]
        Dictionary mapping PE numbers to lists of peak positions 
        (from simple_calibration(); has to be piped through get_calibrated_PE_positions())
    params : Mapping[str, Any]
        Configuration parameters dictionary with keys:
        - 'max_nr_gausspeaks': Maximum number of gaussian peaks to fit 
          (will do less if simple_calibration() found less)
        - 'gauss_mean_range_low': Lower range for gaussian mean
        - 'gauss_mean_range_high': Upper range for gaussian mean
        - 'fit_range_prePE': Range to fit before first PE peak
        - 'fit_range_pastPE': Range to fit after last PE peak
        - 'model': Fitting model to use:
            - 'combo': Combined fit of all gaussians plus background
            - 'individual': Individual fits for each gaussian (No background)
    ax : matplotlib.axes.Axes, optional
        If provided, the fit results will be plotted on this axis
    nofit : bool, optional
        If True, skip the fitting process and use initial parameters
    verbosity : int, optional
        Controls the amount of output (0: none, >2: print fit results)
    Returns
    -------
    dict[str, float]
        Dictionary containing calibration parameters:
        - 'slope': Conversion factor from ADC to PE
        - 'offset': Offset to align 1 PE peak at 1
    Raises
    ------
    ValueError
        If invalid model parameter is provided
    ResultCheckError
        If fitting fails or results don't meet quality criteria
    """
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
        return {"slope": c, "offset": offset, "sigma_1": gaussians["gauss1"].params["sigma"].result * c}
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
    """Performs advanced calibration for multiple channels and returns calibration results.
    Parameters
    ----------
    calibrated_histo_dict : dict
        Dictionary containing pre-calibrated histograms for each channel (multi_simple_calibration()).
    calibrated_PE_positions : dict[str, dict[int, list[Any]]]
        Dictionary containing PE positions for each channel, calibrated using get_calibrated_PE_positions()
    calibration_defaults : dict[str, Any]
        Default calibration parameters.
    calibration_overrides : dict[str, dict[str, Any]], optional
        Channel-specific calibration parameter overrides.
    draw : bool, optional
        If True, creates plots for each channel calibration, by default False.
    nodraw_axes : bool, optional
        If True, hides axis labels in plots, by default False.
    verbosity : int, optional
        Controls the level of output messages, by default 0.
    Returns
    -------
    tuple[dict[str, dict[str, float]], int, Figure | None]
        A tuple containing:
        - Dictionary of calibration results for each channel
        - Number of unsuccessful calibrations
        - Matplotlib Figure object if draw=True, None otherwise
    Notes
    -----
    The calibration results dictionary contains 'slope' and 'offset' values for each channel.
    Failed calibrations will have NaN values for both parameters.
    """
    ret = {}
    fig = None
    if draw:
        fig, ax = auto_subplots(len(calibrated_histo_dict))
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