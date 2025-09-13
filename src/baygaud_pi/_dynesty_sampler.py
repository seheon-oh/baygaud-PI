#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _dynesty_sampler.py
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|


#|-----------------------------------------|
import os
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from numba import njit, config, set_num_threads


# Fix Numba threading layer / thread count BEFORE compilation.
try:
    config.THREADING_LAYER = 'workqueue'
    set_num_threads(1)
except Exception:
    pass

_NUMBA_WARMED = False  # One-time JIT warm-up flag per worker process
#|-----------------------------------------|


#|-----------------------------------------|
from re import A, I
import sys
import numpy as np
from numpy import sum, exp, log, pi
from numpy import linalg, array, sum, log, exp, pi, std, diag, concatenate
import matplotlib.pyplot as plt

#|-----------------------------------------|
# TEST
from numba import njit, prange

#|-----------------------------------------|
import dynesty
from dynesty import NestedSampler
from dynesty import DynamicNestedSampler
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

import gc
import ray
import multiprocessing as mp

from _set_init_priors import search_gaussian_seeds_matched_filter_norm, \
                            set_sgfit_bounds_from_matched_filter_seeds_norm, \
                            print_priors_both, \
                            print_gaussian_seeds_matched_filter, \
                            pin_threads_single, \
                            set_init_priors_multiple_gaussians, \
                            prev_fit_from_results_slice


from _stats import _gaussian_sum_norm, \
                    _rms_of_residual, \
                    _little_derive_rms_core, \
                    little_derive_rms, \
                    robust_bg_rms_from_seed_dict_norm, \
                    update_bg_rms_to_seeds, \
                    convert_units_norm_to_phys


from _multi_gaussmodels import _multi_gaussian_model_norm_core, \
                                multi_gaussian_model_d, \
                                f_gaussian_model, \
                                multi_gaussian_model_d_vectorization, \
                                multi_gaussian_model_d_classic

from _handle_ray import _mat

from _handle_yaml import _get_threading_env_from_params, \
                        _sec, \
                        build_dynesty_run_config_dynesty_v2_1_5



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# set minimum nlive if cur_nlive is too small 
def enforce_min_nlive(ndim: int, _params: dict, verbose: bool = True):
    """
    Guarantee dynesty's recommended minimum for nlive (= 1 + ndim*(ndim+1)//2).
    This function returns the minimum-safe nlive. The provided config is not
    mutated except by the caller assigning the return value.
    """
    # optimal minimum nlive
    min_nlive = 1 + (ndim * (ndim + 1) // 2)

    cur = int(_params['nlive'])
    if cur < min_nlive:
        return min_nlive
        # if verbose:
        #     print(f"[dynesty] bumping nlive: {cur} --> {min_nlive} (ndim={ndim})")
        #     return min_nlive
    else:
        return cur


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# run dynesty for each line profile
def dynamic_baygaud_nested_sampling(num_cpus_nested_sampling, _params):

    # YAML --> environment variables dict (supports both 'threading' section and top-level keys)
    env_vars = _get_threading_env_from_params(_params)

    @ray.remote(num_cpus=num_cpus_nested_sampling)
    def baygaud_nested_sampling(
        _inputDataCube_id, _x_id,
        _peak_sn_map_id, _sn_int_map_id,
        _params_id,
        _is, _ie, i, _js, _je,
        _cube_mask_2d_id):

        # Pin all thread counts inside this worker process (Numba included).
        # Doing it once per process is sufficient; re-calling is harmless.
        pin_threads_single()

        # Materialize objects from Ray object refs
        _inputDataCube = _mat(_inputDataCube_id)
        _x_norm        = _mat(_x_id)
        _peak_sn_map   = _mat(_peak_sn_map_id)
        _sn_int_map    = _mat(_sn_int_map_id)
        _params        = _mat(_params_id)
        _cube_mask_2d  = _mat(_cube_mask_2d_id) 
        
        _max_ngauss = _params['max_ngauss']
        v_min_phys = _params['vel_min']   # defined by the calling code
        v_max_phys = _params['vel_max']   # defined by the calling code
        _cdelt3 = _params['cdelt3']

        gfit_results = np.zeros(((_je-_js), _max_ngauss, 2*(2+3*_max_ngauss) + 7 + _max_ngauss), dtype=np.float32)

        nparams_g1 = 3*1 + 2  # for initial single-Gaussian fit
        gfit_priors_init_g1 = np.zeros(1*nparams_g1, dtype=np.float32)
        gfit_peak_sn = np.zeros(_max_ngauss, dtype=np.float32)

        # print("CHECK S/N: %d %d | peak S/N: %.1f < %.1f | integrated S/N: %.1f < %.1f" \
        #     % (i, 0+_js, _peak_sn_map[0+_js, i], _params['peak_sn_limit'], _sn_int_map[0+_js, i], _params['int_sn_limit']))


        global _NUMBA_WARMED
        if not _NUMBA_WARMED:
            # Warm up each JIT-compiled function once using dummy data
            x_d = np.linspace(0.0, 1.0, 8, dtype=np.float64)
            prof_d = np.zeros_like(x_d)
            theta_d = np.zeros(3*1+2, dtype=np.float64)  # ngauss=1
            theta_d[1] = 0.0  # bg
            theta_d[2] = 0.5; theta_d[3] = 0.1; theta_d[4] = 1.0
            m = _gaussian_sum_norm(x_d, theta_d, 1)
            _ = _rms_of_residual(prof_d, m)
            # _ = _neg_half_chi2(prof_d, m, 1.0)
            _ = _little_derive_rms_core(prof_d, x_d, 0.0, 1.0, 1, theta_d)
            _NUMBA_WARMED = True


        # Physical velocity range of THIS spectrum
        # (e.g., assume v_min, v_max have been precomputed from FITS)
        v_min_phys = _params['vel_min']   # defined by the calling code
        v_max_phys = _params['vel_max']   # defined by the calling code

        # Handle YAML parameters
        mf = _sec(_params, "matched_filter")
        rr = _sec(_params, "robust_rms")
        sb = _sec(_params, "sgfit_bounds")

        for j in range(0, _je - _js):
            _flux_phys = _inputDataCube[:, j+_js, i]
            _f_min = float(np.min(_flux_phys))  # lowest flux --> used for normalization
            _f_max = float(np.max(_flux_phys))  # peak flux   --> used for normalization
            denom  = (_f_max - _f_min) if (_f_max > _f_min) else 1.0
            _f_norm = (_flux_phys - _f_min) / denom  # normalization to [0, 1]


            # --- (1) Find matched-filter seeds using YAML parameters ---
            _gaussian_seeds = search_gaussian_seeds_matched_filter_norm(
                _x_norm, _f_norm,                       # _x_norm already normalized to [0, 1]
                rms=None, bg=None,
                sigma_list_ch = mf.get("sigma_list_ch", [1.5,2,3,4,5]),
                k_sigma       = float(mf.get("k_sigma", 3.0)),
                thres_sigma   = float(mf.get("thres_sigma", 2.0)),
                amp_sigma_thres = float(mf.get("amp_sigma_thres", 2.0)),
                sep_channels  = int(mf.get("sep_channels", 5)),
                max_components= mf.get("max_components", None),
                refine_center = bool(mf.get("refine_center", True)),
                detrend_local = bool(mf.get("detrend_local", False)),
                detrend_halfwin = int(mf.get("detrend_halfwin", 8)),
                numba_threads = int(mf.get("numba_threads", 1)),
            )

            # --- (2) Robust background/RMS using YAML parameters ---
            bg_norm, rms_norm = robust_bg_rms_from_seed_dict_norm(
                _x_norm, _f_norm, _gaussian_seeds,
                exclude_k       = float(rr.get("exclude_k", 5.0)),
                k_min           = float(rr.get("k_min", 2.0)),
                shrink_factor   = float(rr.get("shrink_factor", 0.85)),
                max_shrink_steps= int(rr.get("max_shrink_steps", 6)),
                min_bg_frac     = float(rr.get("min_bg_frac", 0.25)),
                clip_sigma      = float(rr.get("clip_sigma", 3.0)),
                max_iter        = int(rr.get("max_iter", 8)),
                emission_positive = bool(rr.get("emission_positive", True)),
            )


            # Convert to physical units
            bg_phys  = bg_norm  * (_f_max - _f_min) + _f_min
            rms_phys = rms_norm * (_f_max - _f_min)

            # Update _gaussian_seeds in-place with bg/rms
            _gaussian_seeds = update_bg_rms_to_seeds(_gaussian_seeds, bg_norm, rms_norm, inplace=True)

            # Derive peak-flux S/N
            _peakflux_sn = (_f_max - bg_phys) / (rms_phys)

            # Single-Gaussian bounds (normalized units)

            # --- (3) Build single-Gaussian bounds using YAML parameters (normalized units) ---
            # gfit_priors_init = [sig1, bg1, x1, std1, p1, sig2, bg2, x2, std2, p2]
            _seed_priors_using_matched_filter = set_sgfit_bounds_from_matched_filter_seeds_norm(
                _gaussian_seeds,
                use_phys_for_x_bounds = bool(sb.get("use_phys_for_x_bounds", True)),
                v_min = v_min_phys, v_max = v_max_phys,
                cdelt3 = _cdelt3,
                model_sigma_bounds = tuple(sb.get("model_sigma_bounds", (0.0, 0.7))),
                k_bg = float(sb.get("k_bg", 3.0)),
                k_x  = float(sb.get("k_x", 5.0)),
                sigma_scale_bounds = tuple(sb.get("sigma_scale_bounds", (0.1, 3.0))),
                peak_scale_bounds  = tuple(sb.get("peak_scale_bounds",  (0.3, 1.5))),
                min_x_span   = float(sb.get("min_x_span", 1e-5)),
                min_sigma    = float(sb.get("min_sigma", 1e-4)),
                clip_sigma_hi= float(sb.get("clip_sigma_hi", 0.999)),
            )



#            _seed_priors_using_matched_filter = set_sgfit_bounds_from_matched_filter_seeds_norm(
#                    _gaussian_seeds,
#                    use_phys_for_x_bounds=True,
#                    v_min=v_min_phys, v_max=v_max_phys,
#                    cdelt3=_cdelt3,   # if negative, it will automatically anchor with reversed sign
#                    # Adjust factors if necessary
#                    model_sigma_bounds=(0.0, 0.9),
#                    k_bg=3.0, k_x=5.0,
#                    sigma_scale_bounds=(0.1, 3.0),
#                    peak_scale_bounds=(0.3, 1.5)
#                    )
            

            gfit_priors_init = _seed_priors_using_matched_filter

            # __________________________________________________________________ #
            # CHECK POINT
            # if i == 472 and j+_js == 407:
            #     print("-------------------")
            #     print("pixel:", i, j+_js, "ncomp:", _gaussian_seeds['ncomp'])
            #     print("model_sigma(norm):", _gaussian_seeds['rms'], "rms(phys):", _gaussian_seeds['rms'] * denom)
            #     print("bg(norm):", _gaussian_seeds['bg'], "bg(phys):", _gaussian_seeds['bg'] * denom + _f_min)
            #     print("components (x_norm, sigma_norm, peakflux_norm):\n", _gaussian_seeds['components'])
            #     print("priors(single-gauss, normalized):", gfit_priors_init)
            #     print("-------------------")
            #     print_priors_both(gfit_priors_init, _f_min, _f_max, v_min_phys, v_max_phys,
            #       cdelt3=_cdelt3, unit_flux="Jy/beam", unit_vel="km/s")
            #
            #     print("-------------------")
            #     print_gaussian_seeds_matched_filter(_gaussian_seeds, _f_min, _f_max, v_min_phys, v_max_phys, cdelt3=_cdelt3,
            #         unit_flux="Jy/beam", unit_vel="km/s") 
            #     print("-------------------")
            # __________________________________________________________________ #


            # Before attempting the first single-Gaussian fit,
            # check peak S/N and basic validity.

            if _cube_mask_2d[j+_js, i] <= 0:  # if masked, skip. NOTE: mask value must be zero or negative.
                # print("mask filtered: %d %d | peak S/N: %.1f :: S/N limit: %.1f | integrated S/N: %.1f :: S/N limit: %.1f :: f_min: %e :: f_max: %e" \
                #     % (i, j+_js, _peak_sn_map[j+_js, i], _params['peak_sn_limit'], _sn_int_map[j+_js, i], _params['int_sn_limit'], _f_min, _f_max))

                # Save current profile location
                l_range = np.arange(_max_ngauss)

                gfit_results[j, l_range, 2*(3*_max_ngauss+2) + l_range] = rms_phys

                constant_indices = np.array([2*(3*_max_ngauss+2) + _max_ngauss + offset for offset in range(7)])

                gfit_results[j, l_range[:, np.newaxis], constant_indices] = np.array([
                    -1E11,       # for sgfit: log-Z
                    _is,         # start index
                    _ie,         # end index
                    _js,         # start index in j
                    _je,         # end index in j
                    i,           # current i index
                    j + _js      # adjusted j index
                ])[np.newaxis, :]
                continue

            # elif _sn_int_map[j+_js, i] < _params['int_sn_limit'] or _peak_sn_map[j+_js, i] < _params['peak_sn_limit'] \  # ... previous version ...
            elif _sn_int_map[j+_js, i] < _params['int_sn_limit'] or _peakflux_sn < _params['peak_sn_limit'] \
                or np.isnan(_f_max) or np.isnan(_f_min) \
                or np.isinf(_f_min) or np.isinf(_f_min):

                # print("low S/N: %d %d | peak S/N: %.1f :: S/N limit: %.1f | integrated S/N: %.1f :: S/N limit: %.1f :: f_min: %e :: f_max: %e" \
                #     % (i, j+_js, _peak_sn_map[j+_js, i], _params['peak_sn_limit'], _sn_int_map[j+_js, i], _params['int_sn_limit'], _f_min, _f_max))
                
                # Save current profile location
                l_range = np.arange(_max_ngauss)

                gfit_results[j, l_range, 2*(3*_max_ngauss+2) + l_range] = rms_phys

                constant_indices = np.array([2*(3*_max_ngauss+2) + _max_ngauss + offset for offset in range(7)])

                gfit_results[j, l_range[:, np.newaxis], constant_indices] = np.array([
                    -1E11,       # for sgfit: log-Z
                    _is,         # start index
                    _ie,         # end index
                    _js,         # start index in j
                    _je,         # end index in j
                    i,           # current i index
                    j + _js      # adjusted j index
                ])[np.newaxis, :]
                continue


            for k in range(0, _max_ngauss):
                ngauss = k+1  # set the number of Gaussian components
                ndim = 3*ngauss + 2
                nparams = ndim

                # print("processing: %d %d | peak s/n: %.1f | integrated s/n: %.1f | gauss-%d" % (i, j+_js, _peak_sn_map[j+_js, i], _sn_int_map[j+_js, i], ngauss))
                # Ensure a minimum-safe nlive given common_kwargs['nlive']
                _params['nlive'] = enforce_min_nlive(ndim, _params, verbose=True)

                # Run dynesty 2.1.15
                if _params['_dynesty_class_'] == 'static':
                    _queue_size = int(_params['num_cpus_nested_sampling'])
                    rstate = np.random.default_rng(2)

                    # logl_args  --> loglike_d args[1:] (args[0] is set by dynesty)
                    # ptform_args --> optimal_prior args[1:] (args[0] is set by dynesty)
                    sampler = NestedSampler(loglike_d, optimal_prior, ndim,
                        nlive=_params['nlive'],
                        update_interval=_params['update_interval'],
                        sample=_params['sample'],
                        rstate=rstate,
                        first_update={
                            'min_eff': 10,
                            'min_ncall': 200},
                        bound=_params['bound'],
                        facc=_params['facc'],
                        fmove=_params['fmove'],
                        max_move=_params['max_move'],
                        logl_args=[_f_norm, _x_norm, ngauss], ptform_args=[ngauss, gfit_priors_init])
                        # logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x_norm, ngauss], ptform_args=[ngauss, gfit_priors_init])
                    sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False)

                elif _params['_dynesty_class_'] == 'dynamic':
                    _queue_size = int(_params['num_cpus_nested_sampling'])
                    rstate = np.random.default_rng(2)
                    sampler = DynamicNestedSampler(loglike_d, optimal_prior, ndim,
                        nlive=_params['nlive'],
                        rstate=rstate,
                        sample=_params['sample'],
                        # pool=None,
                        queue_size=_queue_size,
                        bound=_params['bound'],
                        facc=_params['facc'],
                        fmove=_params['fmove'],
                        max_move=_params['max_move'],
                        logl_args=[_f_norm, _x_norm, ngauss], ptform_args=[ngauss, gfit_priors_init])
                        # logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x_norm, ngauss], ptform_args=[ngauss, gfit_priors_init])
                    sampler.run_nested(dlogz_init=_params['dlogz'], maxiter_init=_params['maxiter'], maxcall_init=_params['maxcall'], print_progress=False)

                else:
                    raise ValueError(f"Unknown dynesty class: {_params['_dynesty_class_']}")   


                _gfit_results_temp, _logz = get_dynesty_sampler_results(sampler)  # normalized units

                # ---------------------------------------------------------
                # param1, param2, param3, ... , param1-e, param2-e, param3-e
                # gfit_results[j][k][0..2*nparams] = _gfit_results_temp[0..2*nparams]
                # ---------------------------------------------------------
                gfit_results[j, k, :2*nparams] = _gfit_results_temp  # normalized units
                # ---------------------------------------------------------

                # ---------------------------------------------------------
                # Derive RMS of the current profile for this ngauss fit
                # ---------------------------------------------------------
                _rms_ngfit = little_derive_rms(_inputDataCube, i, j+_js, _x_norm, _f_min, _f_max, ngauss, _gfit_results_temp)  # normalized [0..1]

                # ---------------------------------------------------------
                if ngauss == 1:  # Check peak S/N from the single-Gaussian fit
                    # Load the normalized single-Gaussian result --> derive RMS for S/N
                    # _bg_sgfit = _gfit_results_temp[1]
                    # _x_sgfit = _gfit_results_temp[2]
                    # _std_sgfit = _gfit_results_temp[3]
                    # _p_sgfit = _gfit_results_temp[4]
                    # Peak flux of the single-Gaussian model:
                    # _f_sgfit = _p_sgfit * exp( -0.5*((_x - _x_sgfit) / _std_sgfit)**2) + _bg_sgfit

                    # Update priors for subsequent fits based on the single-Gaussian fit
                    gfit_priors_init_g1 = _gfit_results_temp[:nparams_g1]

                    # More accurate peak S/N from the first single-Gaussian fit
                    _bg_sgfit = _gfit_results_temp[1]  # bg
                    _sigma_sgfit = _gfit_results_temp[3] * (v_max_phys - v_min_phys)  # sigma in physical units
                    _p_sgfit = _gfit_results_temp[4]  # bg already subtracted: normalized units
                    _peak_sn_sgfit = _p_sgfit/_rms_ngfit


                    if _peak_sn_sgfit < _params['peak_sn_limit'] or _sigma_sgfit < _params['g_sigma_lower'] or _sigma_sgfit > _params['g_sigma_upper']: 

                        # print("skip the rest of Gaussian fits: %d %d | rms:%.5f | bg:%.5f | peak:%.5f | peak_sgfit s/n: %.3f < %.3f" % (i, j+_js, _rms_ngfit, _bg_sgfit, _p_sgfit, _peak_sn_sgfit, _params['peak_sn_limit']))

                        # Save current profile location (vectorized)
                        l_indices = np.arange(_max_ngauss)

                        gfit_results[j, l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 1] = _is
                        gfit_results[j, l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 2] = _ie
                        gfit_results[j, l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 3] = _js
                        gfit_results[j, l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 4] = _je
                        gfit_results[j, l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 5] = i
                        gfit_results[j, l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 6] = _js + j

                        gfit_results[j, 0, 2 * (3 * _max_ngauss + 2)] = _rms_ngfit
                        gfit_results[j, 0, 2 * (3 * _max_ngauss + 2) + _max_ngauss] = _logz

                        gfit_results[j, 1:, 2 * (3 * _max_ngauss + 2) + l_indices[1:]] = 0   # proper indexing: other RMS = 0
                        gfit_results[j, 1:, 2 * (3 * _max_ngauss + 2) + _max_ngauss] = -1E11  # other log-Z = sentinel

                        # ________________________________________________________________________________________|
                        # UNIT CONVERSION
                        # ________________________________________________________________________________________|
                        convert_units_norm_to_phys(
                            gfit_results, j, k,
                            f_min=_f_min, f_max=_f_max,
                            vel_min=v_min_phys, vel_max=v_max_phys,
                            cdelt3=_cdelt3,
                            max_ngauss=_max_ngauss)

                        break
                    # |_______________________________________________________________________________________|
                    # |---------------------------------------------------------------------------------------|


                # ---------------------------------------------------------
                # Update optimal priors based on the current (ngauss) fit results
                if ngauss < _max_ngauss:  # Update gfit_priors_init for the next Gaussian count
                    nparams_n = 3*(ngauss+1) + 2  # (+1) for the next model (e.g., ngauss=2, 3, 4 ...)

                    gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)

                    # 2) Convert the previous stage's result into prev_fit
                    # Extract parameters from gfit_results[j, ngauss-1] slice
                    prev_fit = prev_fit_from_results_slice(
                        gfit_results[j, ngauss - 1],  # <-- result from the immediately previous ngauss
                        n_prev=ngauss
                    )

                    # 3) Build multi-Gaussian priors for this stage (e.g., M=3)
                    gfit_priors_init = set_init_priors_multiple_gaussians(
                        M=ngauss+1,  # for the next Gaussian model
                        seed_bounds=_seed_priors_using_matched_filter,
                        seed_out=_gaussian_seeds,     # used to compose bg±k_bg*rms
                        prev_fit=prev_fit,            # pass previous result if present
                        k_x=3.0,
                        sigma_scale_bounds=(0.5, 2.0),
                        peak_scale_bounds=(0.5, 1.5),
                        k_bg=3.0,
                        k_msig=3.0,
                        peak_upper_cap=1.0
                    )
                    # nsigma_prior_range_gfit=3.0 (default)
                # |_______________________________________________________________________________________|
                # |---------------------------------------------------------------------------------------|



                # ---------------------------------------------------------
                # Update the tail part of gfit_results 
                gfit_results[j, k, 2*(3*_max_ngauss+2)+k] = _rms_ngfit  # rms_(k+1)gfit : normalized units (from new little_derive_rms)
                gfit_results[j, k, 2*(3*_max_ngauss+2)+_max_ngauss+0] = _logz
                gfit_results[j, k, 2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
                gfit_results[j, k, 2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
                gfit_results[j, k, 2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
                gfit_results[j, k, 2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
                gfit_results[j, k, 2*(3*_max_ngauss+2)+_max_ngauss+5] = i
                gfit_results[j, k, 2*(3*_max_ngauss+2)+_max_ngauss+6] = _js + j
                # print(gfit_results[j][k])


                # |-----------------------------------------|
                # |-----------------------------------------|
                # example: 3 Gaussians : bg + 3 * (x, std, peak) 
                #  ______________________________________  |
                # |_G1___________________________________| |
                # |_000000000000000000000000000000000000_| |
                # |_000000000000000000000000000000000000_| |
                # |_G1-rms : 0 : 0 : log-Z : xs-xe-ys-ye_| |
                #  ______________________________________  |
                # |_G1___________________________________| |
                # |_G2___________________________________| |
                # |_000000000000000000000000000000000000_| |
                # |_0 : G2-rms : 0 : log-Z : xs-xe-ys-ye_| |
                #  ______________________________________  |
                # |_G1___________________________________| |
                # |_G2___________________________________| |
                # |_G3___________________________________| |
                # |_0 : 0 : G3-rms : log-Z : xs-xe-ys-ye_| |

                # |-----------------------------------------|
                # gfit_results[j][k][0] : dist-sig
                # gfit_results[j][k][1] : bg
                # gfit_results[j][k][2] : g1-x --> *(vel_max-vel_min) + vel_min
                # gfit_results[j][k][3] : g1-s --> *(vel_max-vel_min)
                # gfit_results[j][k][4] : g1-p
                # gfit_results[j][k][5] : g2-x --> *(vel_max-vel_min) + vel_min
                # gfit_results[j][k][6] : g2-s --> *(vel_max-vel_min)
                # gfit_results[j][k][7] : g2-p
                # gfit_results[j][k][8] : g3-x --> *(vel_max-vel_min) + vel_min
                # gfit_results[j][k][9] : g3-s --> *(vel_max-vel_min)
                # gfit_results[j][k][10] : g3-p

                # gfit_results[j][k][11] : dist-sig-e
                # gfit_results[j][k][12] : bg-e
                # gfit_results[j][k][13] : g1-x-e --> *(vel_max-vel_min)
                # gfit_results[j][k][14] : g1-s-e --> *(vel_max-vel_min)
                # gfit_results[j][k][15] : g1-p-e
                # gfit_results[j][k][16] : g2-x-e --> *(vel_max-vel_min)
                # gfit_results[j][k][17] : g2-s-e --> *(vel_max-vel_min)
                # gfit_results[j][k][18] : g2-p-e
                # gfit_results[j][k][19] : g3-x-e --> *(vel_max-vel_min)
                # gfit_results[j][k][20] : g3-s-e --> *(vel_max-vel_min)
                # gfit_results[j][k][21] : g3-p-e --> *(f_max-bg_flux)

                # gfit_results[j][k][22] : g1-rms --> *(f_max-bg_flux) : bg RMS for single-Gaussian fitting
                # gfit_results[j][k][23] : g2-rms --> *(f_max-bg_flux) : bg RMS for double-Gaussian fitting
                # gfit_results[j][k][24] : g3-rms --> *(f_max-bg_flux) : bg RMS for triple-Gaussian fitting

                # gfit_results[j][k][25] : log-Z : log-evidence : log-marginalized likelihood

                # gfit_results[j][k][26] : xs
                # gfit_results[j][k][27] : xe
                # gfit_results[j][k][28] : ys
                # gfit_results[j][k][29] : ye
                # gfit_results[j][k][30] : x
                # gfit_results[j][k][31] : y
                # |-----------------------------------------|

                # ________________________________________________________________________________________|
                # UNIT CONVERSION
                # ________________________________________________________________________________________|
                convert_units_norm_to_phys(
                    gfit_results, j, k,
                    f_min=_f_min, f_max=_f_max,
                    vel_min=v_min_phys, vel_max=v_max_phys,
                    cdelt3=_cdelt3,
                    max_ngauss=_max_ngauss)


                # ________________________________________________________________________________________|
                # SKIP CONDITION
                # ________________________________________________________________________________________|
                _m_indices = np.arange(k+1)
                _peak_flux_indices = 4 + 3*_m_indices
                gfit_peak_sn = (gfit_results[j, k, _peak_flux_indices] - gfit_results[j][k][1]) / gfit_results[j, k, 2*(3*_max_ngauss+2) + k]

                if np.all(gfit_peak_sn < _params['peak_sn_limit']) and k != (_max_ngauss-1):  # If all peaks are below threshold and not at max components
                #     print("")
                #     print(
                #         "|--> skip the rest of Gaussian fits: %d %d | rms:%.5f | bg:%.5f | "
                #         "%d-Gaussian fit's peak S/N (all): %s < %.3f <--|"
                #         % (
                #             i,
                #             j + _js,
                #             _rms_ngfit,
                #             _bg_sgfit,
                #             k+1,
                #             np.array2string(gfit_peak_sn, precision=3, separator=", "),
                #             _params['peak_sn_limit']
                #         )
                #     )

                    # Vectorized location bookkeeping
                    _l_indices = np.arange(_max_ngauss)

                    gfit_results[j, _l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 1] = _is
                    gfit_results[j, _l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 2] = _ie
                    gfit_results[j, _l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 3] = _js
                    gfit_results[j, _l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 4] = _je
                    gfit_results[j, _l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 5] = i
                    gfit_results[j, _l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 6] = _js + j

                    gfit_results[j, k, 2 * (3 * _max_ngauss + 2)] = gfit_results[j, k, 2*(3*_max_ngauss+2) + k]  # rms
                    gfit_results[j, k, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 0] = _logz  # log-Z

                    gfit_results[j, k+1:, 2 * (3 * _max_ngauss + 2) + _l_indices[k+1:]] = 0      # other rms = 0
                    gfit_results[j, k+1:, 2 * (3 * _max_ngauss + 2) + _max_ngauss] = -1E11       # other log-Z = sentinel

                    break  # !!! SKIP THE REMAINING GAUSSIANS !!!

        return gfit_results
    return baygaud_nested_sampling
    # return baygaud_nested_sampling.options(runtime_env={"env_vars": env_vars})
    #-- END OF SUB-ROUTINE____________________________________________________________#




    
    
#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# run dynesty for each line profile
@ray.remote(num_cpus=1)
def run_dynesty_sampler_uniform_priors(_x, _inputDataCube, _is, _ie, i, _js, _je, _max_ngauss, _vel_min, _vel_max, _params):

    gfit_results = np.zeros(((_je-_js), _max_ngauss, 2*(2+3*_max_ngauss)+7), dtype=np.float32)
    for j in range(0, _je -_js):

        _f_max = np.max(_inputDataCube[:,j+_js,i])  # peak flux --> used for normalization
        _f_min = np.min(_inputDataCube[:,j+_js,i])  # lowest flux --> used for normalization

        gfit_priors_init = np.zeros(2*5, dtype=np.float32)
        # gfit_priors_init = [sig1, bg1, x1, std1, p1, sig2, bg2, x2, std2, p2]
        gfit_priors_init = [0.0, 0.0, 0.01, 0.01, 0.01, 0.5, 0.5, 0.9, 0.6, 1.01]
        for k in range(0, _max_ngauss):
            ngauss = k+1  # set the number of Gaussian components
            ndim = 3*ngauss + 2
            nparams = ndim

            # print("processing: %d %d gauss-%d" % (i, j+_js, ngauss))

            if _params['_dynesty_class_'] == 'static':
                sampler = NestedSampler(loglike_d, uniform_prior, ndim,
                    nlive=_params['nlive'],
                    update_interval=_params['update_interval'],
                    sample=_params['sample'],
                    bound=_params['bound'],
                    facc=_params['facc'],
                    fmove=_params['fmove'],
                    max_move=_params['max_move'],
                    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False)

            elif _params['_dynesty_class_'] == 'dynamic':
                sampler = DynamicNestedSampler(loglike_d, uniform_prior, ndim,
                    nlive=_params['nlive'],
                    sample=_params['sample'],
                    update_interval=_params['update_interval'],
                    bound=_params['bound'],
                    facc=_params['facc'],
                    fmove=_params['fmove'],
                    max_move=_params['max_move'],
                    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                sampler.run_nested(dlogz_init=_params['dlogz'], maxiter_init=_params['maxiter'], maxcall_init=_params['maxcall'], print_progress=False)

            _gfit_results_temp, _logz = get_dynesty_sampler_results(sampler)

            gfit_results[j][k][:2*nparams] = _gfit_results_temp
            gfit_results[j][k][2*(3*_max_ngauss+2)+0] = _logz
            gfit_results[j][k][2*(3*_max_ngauss+2)+1] = _is
            gfit_results[j][k][2*(3*_max_ngauss+2)+2] = _ie
            gfit_results[j][k][2*(3*_max_ngauss+2)+3] = _js
            gfit_results[j][k][2*(3*_max_ngauss+2)+4] = _je
            gfit_results[j][k][2*(3*_max_ngauss+2)+5] = i
            gfit_results[j][k][2*(3*_max_ngauss+2)+6] = _js + j


            # |-----------------------------------------|
            # example: 3 Gaussians : bg + 3 * (x, std, peak) 
            # gfit_results[j][k][0]  : dist-sig
            # gfit_results[j][k][1]  : bg
            # gfit_results[j][k][2]  : g1-x1 --> *(vel_max-vel_min) + vel_min
            # gfit_results[j][k][3]  : g1-s1 --> *(vel_max-vel_min)
            # gfit_results[j][k][4]  : g1-p1
            # gfit_results[j][k][5]  : g2-x2 --> *(vel_max-vel_min) + vel_min
            # gfit_results[j][k][6]  : g2-s2 --> *(vel_max-vel_min)
            # gfit_results[j][k][7]  : g2-p2
            # gfit_results[j][k][8]  : g3-x3 --> *(vel_max-vel_min) + vel_min
            # gfit_results[j][k][9]  : g3-s3 --> *(vel_max-vel_min)
            # gfit_results[j][k][10] : g3-p3

            # gfit_results[j][k][11] : dist-sig-e
            # gfit_results[j][k][12] : bg-e
            # gfit_results[j][k][13] : g1-x1-e --> *(vel_max-vel_min)
            # gfit_results[j][k][14] : g1-s1-e --> *(vel_max-vel_min)
            # gfit_results[j][k][15] : g1-p1-e
            # gfit_results[j][k][16] : g2-x2-e --> *(vel_max-vel_min)
            # gfit_results[j][k][17] : g2-s2-e --> *(vel_max-vel_min)
            # gfit_results[j][k][18] : g2-p2-e
            # gfit_results[j][k][19] : g3-x3-e --> *(vel_max-vel_min)

            # gfit_results[j][k][22] : log-Z : log-evidence : log-marginalization likelihood

            # gfit_results[j][k][23] : xs
            # gfit_results[j][k][24] : xe
            # gfit_results[j][k][25] : ys
            # gfit_results[j][k][26] : ye
            # gfit_results[j][k][27] : x
            # gfit_results[j][k][28] : y
            # |-----------------------------------------|


            gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min  # bg-flux
            gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min)  # bg-flux-e

            for m in range(0, k+1):
                # Unit conversion
                # peak flux --> data cube units
                # velocity, velocity-dispersion --> km/s
                gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min  # velocity
                gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min)            # velocity-dispersion
                gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) + _f_min       # flux

                gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min)    # velocity-e
                gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min)    # velocity-dispersion-e
                gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min)        # flux-e

    
    del(ndim, nparams, ngauss, sampler)
    gc.collect()

    return gfit_results


    # Plot a summary of the run.
#    rfig, raxes = dyplot.runplot(sampler.results)
#    rfig.savefig("r.pdf")
    
    # Plot traces and 1-D marginalized posteriors.
#    tfig, taxes = dyplot.traceplot(sampler.results)
#    tfig.savefig("t.pdf")
    
    # Plot the 2-D marginalized posteriors.
    # cfig, caxes = dyplot.cornerplot(sampler.results)
    # cfig.savefig("c.pdf")
#-- END OF SUB-ROUTINE____________________________________________________________#

    
    


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def get_dynesty_sampler_results(_sampler):
    # Extract sampling results.
    samples = _sampler.results.samples  # samples
    weights = exp(_sampler.results.logwt - _sampler.results.logz[-1])  # normalized weights

    # print(_sampler.results.samples[-1, :]) 
    # print(_sampler.results.logwt.shape) 

    # Compute 10%-90% quantiles.
    quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
                for samps in samples.T]

    # Compute weighted mean and covariance. 
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    bestfit_results = _sampler.results.samples[-1, :]
    log_Z = _sampler.results.logz[-1]

    # print(bestfit_results, log_Z)
    # print(concatenate((bestfit_results, diag(cov)**0.5)))

    # Resample weighted samples.
    # samples_equal = dyfunc.resample_equal(samples, weights)
    
    # Generate a new set of results with statistical+sampling uncertainties.
    # results_sim = dyfunc.simulate_run(_sampler.results)

    # mean_std = np.concatenate((mean, diag(cov)**0.5))
    # return mean_std  # mean + std of each parameter: std array follows the mean array

    del(samples, weights, quantiles)
    gc.collect()

    return concatenate((bestfit_results, diag(cov)**0.5)), log_Z
    # return concatenate((bestfit_results, diag(cov)**0.5)), _sampler.results.logz[-1]
#-- END OF SUB-ROUTINE____________________________________________________________#





# ----------------------------- #
# JIT core: simple loop, in-place
# ----------------------------- #
@njit(cache=True, fastmath=True)
def _optimal_prior_core(u, ngauss, bounds):
    """
    Internal JIT core:
      - u, bounds: 1D float64 arrays
      - ngauss: int
    In-place transforms u using lower/upper bounds, then returns u.
    """
    nparams = 3 * ngauss + 2

    # sigma
    # args[2][0]      : _sigma0
    # args[2][nparams]: _sigma1
    s0 = bounds[0]
    s1 = bounds[nparams]
    u[0] = s0 + u[0] * (s1 - s0)  # sigma: uniform prior between 0..1

    # bg
    # args[2][1]          : _bg0
    # args[2][nparams+1]  : _bg1
    b0 = bounds[1]
    b1 = bounds[nparams + 1]
    u[1] = b0 + u[1] * (b1 - b0)  # bg: uniform prior between 0..1

    # n Gaussians
    # x / std / p
    # lower: args[2][2:nparams:3], [3:nparams:3], [4:nparams:3]
    # upper: args[2][nparams+2:2*nparams:3], [nparams+3:2*nparams:3], [nparams+4:2*nparams:3]
    base_lo = 2
    base_hi = nparams + 2
    for m in range(ngauss):
        i_x = base_lo + 3 * m
        i_s = i_x + 1
        i_p = i_x + 2

        lx = bounds[i_x]
        ux = bounds[base_hi + 3 * m]
        ls = bounds[i_s]
        us = bounds[base_hi + 3 * m + 1]
        lp = bounds[i_p]
        up = bounds[base_hi + 3 * m + 2]

        # x: uniform prior between xn0..xn1
        u[i_x] = lx + u[i_x] * (ux - lx)
        # std: uniform prior between stdn0..stdn1
        u[i_s] = ls + u[i_s] * (us - ls)
        # p: uniform prior between pn0..pn1
        u[i_p] = lp + u[i_p] * (up - lp)

    return u


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# parameters are sigma, bg, _x01, _std01, _p01, _x02, _std02, _p02...
def optimal_prior(*args):
    #---------------------
    # args[0][0] : sigma
    # args[0][1] : bg0
    # args[0][2] : x0
    # args[0][3] : std0
    # args[0][4] : p0
    # ...
    #_____________________
    #---------------------
    # args[1] : ngauss
    # e.g., if ngauss == 3
    #_____________________
    #---------------------
    # args[2][0] : _sigma0
    # args[2][1] : _bg0
    #.....................
    # args[2][2] : _x10
    # args[2][3] : _std10
    # args[2][4] : _p10
    #.....................
    # args[2][5] : _x20
    # args[2][6] : _std20
    # args[2][7] : _p20
    #.....................
    # args[2][8] : _x30
    # args[2][9] : _std30
    # args[2][10] : _p30
    #_____________________
    #---------------------
    # args[2][11] : _sigma1
    # args[2][12] : _bg1
    #.....................
    # args[2][13] : _x11
    # args[2][14] : _std11
    # args[2][15] : _p11
    #.....................
    # args[2][16] : _x21
    # args[2][17] : _std21
    # args[2][18] : _p21
    #.....................
    # args[2][19] : _x31
    # args[2][20] : _std31
    # args[2][21] : _p31
    #---------------------

    # Safety & performance: normalize dtype and memory layout
    u      = np.ascontiguousarray(np.asarray(args[0], dtype=np.float64))
    ngauss = int(args[1])
    bounds = np.ascontiguousarray(np.asarray(args[2], dtype=np.float64))

    # Call the core (in-place transform and return the same array)
    return _optimal_prior_core(u, ngauss, bounds)






#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# parameters are sigma, bg, _x01, _std01, _p01, _x02, _std02, _p02...
def optimal_prior_native(*args):

    #---------------------
    # args[0][0] : sigma
    # args[0][1] : bg0
    # args[0][2] : x0
    # args[0][3] : std0
    # args[0][4] : p0
    # ...
    #_____________________
    #---------------------
    # args[1] : ngauss
    # e.g., if ngauss == 3
    #_____________________
    #---------------------
    # args[2][0] : _sigma0
    # args[2][1] : _bg0
    #.....................
    # args[2][2] : _x10
    # args[2][3] : _std10
    # args[2][4] : _p10
    #.....................
    # args[2][5] : _x20
    # args[2][6] : _std20
    # args[2][7] : _p20
    #.....................
    # args[2][8] : _x30
    # args[2][9] : _std30
    # args[2][10] : _p30
    #_____________________
    #---------------------
    # args[2][11] : _sigma1
    # args[2][12] : _bg1
    #.....................
    # args[2][13] : _x11
    # args[2][14] : _std11
    # args[2][15] : _p11
    #.....................
    # args[2][16] : _x21
    # args[2][17] : _std21
    # args[2][18] : _p21
    #.....................
    # args[2][19] : _x31
    # args[2][20] : _std31
    # args[2][21] : _p31
    #---------------------

    nparams = 3*args[1] + 2

    # sigma
    _sigma0 = np.array(args[2][0])
    _sigma1 = np.array(args[2][nparams])  # args[1] = ngauss

    # bg
    _bg0 = np.array(args[2][1])
    _bg1 = np.array(args[2][nparams+1])  # args[1] = ngauss

    # x
    _xn_0 = np.array(args[2][2:nparams:3])
    _xn_1 = np.array(args[2][nparams+2:2*nparams:3])

    # std
    _stdn_0 = np.array(args[2][3:nparams:3])
    _stdn_1 = np.array(args[2][nparams+3:2*nparams:3])

    # p
    _pn_0 = np.array(args[2][4:nparams:3])
    _pn_1 = np.array(args[2][nparams+4:2*nparams:3])

    # Vectorized transform
    # sigma and bg
    args[0][0] = _sigma0 + args[0][0]*(_sigma1 - _sigma0)   # sigma: uniform prior between 0..1
    args[0][1] = _bg0 + args[0][1]*(_bg1 - _bg0)            # bg: uniform prior between 0..1

    # n Gaussians
    # x
    args[0][2:nparams:3] = _xn_0 +   args[0][2:nparams:3]*(_xn_1 - _xn_0)            # x: uniform prior between xn0..xn1
    # std
    args[0][3:nparams:3] = _stdn_0 + args[0][3:nparams:3]*(_stdn_1 - _stdn_0)        # std: uniform prior between stdn0..stdn1
    # p
    args[0][4:nparams:3] = _pn_0 + args[0][4:nparams:3]*(_pn_1 - _pn_0)              # p: uniform prior between pn0..pn1

    return args[0]

    # old version
    # _sigma0 = np.array(args[2][0])
    # _sigma1 = np.array(args[2][2+3*args[1]])  # args[1]=ngauss
    # _bg0 = np.array(args[2][1])
    # _bg1 = np.array(args[2][3+3*args[1]])     # args[1]=ngauss


    # _xn_0 = np.zeros(args[1])
    # _xn_1 = np.zeros(args[1])
    # _stdn_0 = np.zeros(args[1])
    # _stdn_1 = np.zeros(args[1])
    # _pn_0 = np.zeros(args[1])
    # _pn_1 = np.zeros(args[1])


    # nparams = 3*args[1] + 2

    # _xn_0 = np.array(args[2][2:nparams:3])
    # _xn_1 = np.array(args[2][nparams+2:2*nparams:3])

    # _stdn_0 = np.array(args[2][3:nparams:3])
    # _stdn_1 = np.array(args[2][nparams+3:2*nparams:3])

    # _pn_0 = np.array(args[2][4:nparams:3])
    # _pn_1 = np.array(args[2][nparams+4:2*nparams:3])


    # args[0][0] = _sigma0 + args[0][0]*(_sigma1 - _sigma0)   # sigma: uniform prior between 0..1
    # args[0][1] = _bg0 + args[0][1]*(_bg1 - _bg0)            # bg: uniform prior between 0..1

    # args[0][2:nparams:3] = _xn_0 +   args[0][2:nparams:3]*(_xn_1 - _xn_0)
    # args[0][3:nparams:3] = _stdn_0 + args[0][3:nparams:3]*(_stdn_1 - _stdn_0)
    # args[0][4:nparams:3] = _pn_0 + args[0][4:nparams:3]*(_pn_1 - _pn_0)

    # return args[0]

#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# parameters are sigma, bg, _x01, _std01, _p01, _x02, _std02, _p02...
def uniform_prior(*args):

    # args[0][0] : sigma
    # args[0][1] : bg0
    # args[0][2] : x0
    # args[0][3] : std0
    # args[0][4] : p0
    # ...
    # args[1] : ngauss
    # args[2][0] : _sigma0
    # args[2][1] : _bg0
    # args[2][2] : _x0
    # args[2][3] : _std0
    # args[2][4] : _p0
    # args[2][5] : _sigma1
    # args[2][6] : _bg1
    # args[2][7] : _x1
    # args[2][8] : _std1
    # args[2][9] : _p1

    # sigma
    _sigma0 = args[2][0]
    _sigma1 = args[2][5]
    # bg
    _bg0 = args[2][1]
    _bg1 = args[2][6]
    # x
    _x0 = args[2][2]
    _x1 = args[2][7]
    # std
    _std0 = args[2][3]
    _std1 = args[2][8]
    # p
    _p0 = args[2][4]
    _p1 = args[2][9]

    # sigma (example bounds)
    # _sigma0 = 0
    # _sigma1 = 0.03 
    # bg
    # _bg0 = -0.02
    # _bg1 = 0.02
    # _x0
    # _x0 = 0
    # _x1 = 0.8
    # std
    # _std0 = 0.0
    # _std1 = 0.5
    # p
    # _p0 = 0.0
    # _p1 = 0.5

    # partial[2:] copy to a (3 x ngauss) cube --> x, std, p
    params_t = args[0][2:].reshape(args[1], 3).T

    # vectorization
    # sigma and bg
    args[0][0] = _sigma0 + args[0][0]*(_sigma1 - _sigma0)            # sigma: uniform prior 0..1
    args[0][1] = _bg0 + args[0][1]*(_bg1 - _bg0)                      # bg:    uniform prior 0..1

    # n Gaussians
    # x
    params_t[0] = (_x0 + params_t[0].T*(_x1 - _x0)).T
    # std
    params_t[1] = (_std0 + params_t[1].T*(_std1 - _std0)).T
    # p
    params_t[2] = (_p0 + params_t[2].T*(_p1 - _p0)).T


    params_t_conc = np.hstack((params_t[0].reshape(args[1], 1), params_t[1].reshape(args[1], 1), params_t[2].reshape(args[1], 1))).reshape(1, 3*args[1])
    args[0][2:] = params_t_conc

    # print(args[0])
    # del(_bg0, _bg1, _x0, _x1, _std0, _std1, _p0, _p1, _sigma0, _sigma1, params_t, params_t_conc)
    return args[0]
#-- END OF SUB-ROUTINE____________________________________________________________#






#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# numba version
@njit(cache=True, fastmath=True)
def _loglike_core_numba(params, spect, x, ngauss):
    """
    Log-likelihood core (Numba-accelerated).

    params = [sigma, bg, g1_x, g1_std, g1_p, g2_x, g2_std, g2_p, ...]  (normalized scale)
    spect  = observed spectrum (normalized data)
    x      = channel axis (normalized)
    ngauss = number of Gaussian components

    Returns logL including the normalization constant.
    """

    n = x.size
    sigma = params[0]
    if sigma <= 0.0:
        sigma = 1e-12
    bg = params[1]
    log_n_sigma = -0.5 * n * np.log(2.0 * np.pi) - n * np.log(sigma)
    inv_sigma2  = 1.0 / (sigma * sigma)

    # Model = constant bg + sum of Gaussian components
    model = np.empty_like(x, dtype=np.float64)
    model[:] = bg

    for m in range(ngauss):
        mu  = params[2 + 3*m]
        sig = params[3 + 3*m]
        if sig <= 1e-12:
            sig = 1e-12
        amp = params[4 + 3*m]
        invs = 1.0 / sig
        dx = (x - mu) * invs
        model += amp * np.exp(-0.5 * dx * dx)

    r   = model - spect
    sse = np.dot(r, r)            # Use BLAS (recommended to keep threads=1)

    return log_n_sigma - 0.5 * sse * inv_sigma2
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# python wrapper --> _loglike_core_numba
def loglike_d(*args):
    """
     args[0] : params : dynesty default
     args[1] : _spect : input velocity profile array [N channels] --> normalized (F - f_min)/(f_max - f_min)
     args[2] : _x
     args[3] : ngauss

     Mapping:
       bg, x0, std0, p0, .... = params[1], params[2], params[3], params[4]
       sigma = params[0]  # log-likelihood noise scale

     Note: 'print(args[1])' was used for debugging.
    """

    params = np.ascontiguousarray(args[0], dtype=np.float64)
    spect  = np.ascontiguousarray(args[1], dtype=np.float64)
    x      = np.ascontiguousarray(args[2], dtype=np.float64)
    ngauss = int(args[3])

    return float(_loglike_core_numba(params, spect, x, ngauss))

#-- END OF SUB-ROUTINE____________________________________________________________#
