#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _dynesty_sampler.py
#|-----------------------------------------|
#|
#| version history
#| v1.0 (2022 Dec 25)
#|
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|


#|-----------------------------------------|
from re import A, I
import sys
import numpy as np
from numpy import sum, exp, log, pi
from numpy import linalg, array, sum, log, exp, pi, std, diag, concatenate

#|-----------------------------------------|
# TEST 
import numba
#from numba import jit

#|-----------------------------------------|
import dynesty
from dynesty import NestedSampler
from dynesty import DynamicNestedSampler
from dynesty import plotting as dyplot
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

import gc
import ray

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# derive rms of a profile via ngfit 
def derive_rms_npoints(_inputDataCube, _cube_mask_2d, _x, _params, ngauss):

    ndim = 3*ngauss + 2
    nparams = ndim

    naxis1 = int(_params['naxis1'])
    naxis2 = int(_params['naxis2'])

    naxis1_s0 = int(_params['naxis1_s0'])
    naxis1_e0 = int(_params['naxis1_e0'])
    naxis2_s0 = int(_params['naxis2_s0'])
    naxis2_e0 = int(_params['naxis2_e0'])

    naxis1_seg = naxis1_e0 - naxis1_s0
    naxis2_seg = naxis2_e0 - naxis2_s0

    nsteps_x = int(_params['nsteps_x_rms'])
    nsteps_y = int(_params['nsteps_y_rms'])

    _rms = np.zeros(nsteps_x*nsteps_y+1, dtype=np.float32)
    _bg = np.zeros(nsteps_x*nsteps_y+1, dtype=np.float32)
    # prior arrays for the single Gaussian fit
    gfit_priors_init = np.zeros(2*5, dtype=np.float32)
    _x_boundaries = np.full(2*ngauss, fill_value=-1E11, dtype=np.float32)
    #gfit_priors_init = [sig1, bg1, x1, std1, p1, sig2, bg2, x2, std2, p2]
    gfit_priors_init = [0.0, 0.0, 0.01, 0.01, 0.01, 0.5, 0.6, 0.99, 0.6, 1.01]

    k=0
    for x in range(0, nsteps_x):
        for y in range(0, nsteps_y):

            i = int(0.5*(naxis1_seg/nsteps_x) + x*(naxis1_seg/nsteps_x)) + naxis1_s0
            j = int(0.5*(naxis2_seg/nsteps_y) + y*(naxis2_seg/nsteps_y)) + naxis2_s0

            print("[--> measure background rms at (i:%d j:%d)...]" % (i, j))

            if(_cube_mask_2d[j, i] > 0 and not np.isnan(_inputDataCube[:, j, i]).any()): # if not masked: 

                _f_max = np.max(_inputDataCube[:, j, i]) # peak flux : being used for normalization
                _f_min = np.min(_inputDataCube[:, j, i]) # lowest flux : being used for normalization
    
                #---------------------------------------------------------
                if(ndim * (ndim + 1) // 2 > _params['nlive']):
                    _params['nlive'] = 1 + ndim * (ndim + 1) // 2 # optimal minimum nlive
    
                # run dynesty 1.1
                #sampler = NestedSampler(loglike_d, optimal_prior, ndim,
                #    vol_dec=_params['vol_dec'],
                #    vol_check=_params['vol_check'],
                #    facc=_params['facc'],
                #    nlive=_params['nlive'],
                #    sample=_params['sample'],
                #    bound=_params['bound'],
                #    #rwalk=_params['rwalk'],
                #    max_move=_params['max_move'],
                #    logl_args=[(_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])


                if _params['_dynesty_class_'] == 'static':
                    #---------------------------------------------------------
                    # run dynesty 2.0.3
                    sampler = NestedSampler(loglike_d, optimal_prior, ndim,
                        nlive=_params['nlive'],
                        update_interval=_params['update_interval'],
                        sample=_params['sample'],
                        #walks=_params['walks'],
                        bound=_params['bound'],
                        facc=_params['facc'],
                        fmove=_params['fmove'],
                        max_move=_params['max_move'],
                        logl_args=[(_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])

                    sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=True)
                    #_run_nested = jit(sampler.run_nested(dlogz=1000, maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=True), nopython=True, cache=True, nogil=True, fastmath=True)
                    #sampler.reset()
                    #_run_nested = jit(sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=True), nopython=True, cache=True, nogil=True, fastmath=True)
                    #_run_nested = jit(sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=True), nopython=True, cache=True, nogil=True, parallel=True)
                    #_run_nested()

                elif _params['_dynesty_class_'] == 'dynamic':
                    #---------------------------------------------------------
                    # run dynesty 2.0.3
                    sampler = DynamicNestedSampler(loglike_d, optimal_prior, ndim,
                        nlive=_params['nlive'],
                        update_interval=_params['update_interval'],
                        sample=_params['sample'],
                        bound=_params['bound'],
                        facc=_params['facc'],
                        fmove=_params['fmove'],
                        max_move=_params['max_move'],
                        logl_args=[(_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                    sampler.run_nested(dlogz_init=_params['dlogz'], maxiter_init=_params['maxiter'], maxcall_init=_params['maxcall'], print_progress=True)

                    #---------------------------------------------------------
                _gfit_results_temp, _logz = get_dynesty_sampler_results(sampler)
    
    
                #---------------------------------------------------------
                # lower bounds : x1-3*std1, x2-3*std2, ...  | x:_gfit_results_temp[2, 5, 8, ...], std:_gfit_results_temp[3, 6, 9, ...]
                #_x_boundaries[0:ngauss] = _gfit_results_temp[2:nparams:3] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[3:nparams:3] # x - 3*std
                _x_boundaries[0:ngauss] = _gfit_results_temp[2:nparams:3] - 5*_gfit_results_temp[3:nparams:3] # x - 3*std
                #print("g:", ngauss, "lower bounds:", _x_boundaries)
    
                #---------------------------------------------------------
                # upper bounds : x1+3*std1, x2+3*std2, ...  | x:_gfit_results_temp[2, 5, 8, ...], std:_gfit_results_temp[3, 6, 9, ...]
                #_x_boundaries[ngauss:2*ngauss] = _gfit_results_temp[2:nparams:3] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[3:nparams:3] # x + 3*std
                _x_boundaries[ngauss:2*ngauss] = _gfit_results_temp[2:nparams:3] + 5*_gfit_results_temp[3:nparams:3] # x + 3*std
                #print("g:", ngauss, "upper bounds:", _x_boundaries)
    
                #---------------------------------------------------------
                # lower/upper bounds
                _x_boundaries_ft = np.delete(_x_boundaries, np.where(_x_boundaries < -1E3))
                _x_lower = np.sort(_x_boundaries_ft)[0]
                _x_upper = np.sort(_x_boundaries_ft)[-1]
                _x_lower = _x_lower if _x_lower > 0 else 0
                _x_upper = _x_upper if _x_upper < 1 else 1
                #print(_x_lower, _x_upper)
    
                #---------------------------------------------------------
                # derive the rms given the current ngfit
                _f_ngfit = f_gaussian_model(_x, _gfit_results_temp, ngauss)
                # residual : input_flux - ngfit_flux
                _res_spect = ((_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min) - _f_ngfit)
                # rms
                #print(np.where(_x > _x_lower and _x < _x_upper))
                _index_t = np.argwhere((_x > _x_lower) & (_x < _x_upper))
                #print(_index_t)
                _res_spect_ft = np.delete(_res_spect, _index_t)
    
                # rms
                _rms[k] = np.std(_res_spect_ft)*(_f_max - _f_min)
                # bg
                _bg[k] = _gfit_results_temp[1]*(_f_max - _f_min) + _f_min # bg
                print(i, j, _rms[k], _bg[k])
                k += 1

    # median values
    # first replace 0.0 (zero) to NAN value to use numpy nanmedian function instead of using numpy median
    zero_to_nan_rms = np.where(_rms == 0.0, np.nan, _rms)
    zero_to_nan_bg = np.where(_bg == 0.0, np.nan, _bg)

    _rms_med = np.nanmedian(zero_to_nan_rms)
    _bg_med = np.nanmedian(zero_to_nan_bg)
    # update _rms_med, _bg_med in _params
    _params['_rms_med'] = _rms_med
    _params['_bg_med'] = _bg_med
    print("rms_med:_", _rms_med)
    print("bg_med:_", _bg_med)
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# derive rms of a profile using _gfit_results_temp derived from ngfit
def little_derive_rms_npoints(_inputDataCube, i, j, _x, _f_min, _f_max, ngauss, _gfit_results_temp):

    ndim = 3*ngauss + 2
    nparams = ndim

    _x_boundaries = np.full(2*ngauss, fill_value=-1E11, dtype=np.float32)
    #---------------------------------------------------------
    # lower bounds : x1-3*std1, x2-3*std2, ...  | x:_gfit_results_temp[2, 5, 8, ...], std:_gfit_results_temp[3, 6, 9, ...]
    #_x_boundaries[0:ngauss] = _gfit_results_temp[2:nparams:3] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[3:nparams:3] # x - 3*std
    _x_boundaries[0:ngauss] = _gfit_results_temp[2:nparams:3] - 5*_gfit_results_temp[3:nparams:3] # x - 3*std
    #print("g:", ngauss, "lower bounds:", _x_boundaries)

    #---------------------------------------------------------
    # upper bounds : x1+3*std1, x2+3*std2, ...  | x:_gfit_results_temp[2, 5, 8, ...], std:_gfit_results_temp[3, 6, 9, ...]
    #_x_boundaries[ngauss:2*ngauss] = _gfit_results_temp[2:nparams:3] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[3:nparams:3] # x + 3*std
    _x_boundaries[ngauss:2*ngauss] = _gfit_results_temp[2:nparams:3] + 5*_gfit_results_temp[3:nparams:3] # x + 3*std
    #print("g:", ngauss, "upper bounds:", _x_boundaries)

    #---------------------------------------------------------
    # lower/upper bounds
    _x_boundaries_ft = np.delete(_x_boundaries, np.where(_x_boundaries < -1E3))
    _x_lower = np.sort(_x_boundaries_ft)[0]
    _x_upper = np.sort(_x_boundaries_ft)[-1]
    _x_lower = _x_lower if _x_lower > 0 else 0
    _x_upper = _x_upper if _x_upper < 1 else 1
    #print(_x_lower, _x_upper)

    #---------------------------------------------------------
    # derive the rms given the current ngfit
    _f_ngfit = f_gaussian_model(_x, _gfit_results_temp, ngauss)
    # residual : input_flux - ngfit_flux
    _res_spect = ((_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min) - _f_ngfit)
    # rms
    #print(np.where(_x > _x_lower and _x < _x_upper))
    _index_t = np.argwhere((_x > _x_lower) & (_x < _x_upper))
    #print(_index_t)
    _res_spect_ft = np.delete(_res_spect, _index_t)

    # rms
    #_rms_ngfit = np.std(_res_spect_ft)*(_f_max - _f_min)
    _rms_ngfit = np.std(_res_spect_ft) # normalised
    # bg
    #_bg_ngfit = _gfit_results_temp[1]*(_f_max - _f_min) + _f_min # bg

    del(_x_boundaries, _x_boundaries_ft, _index_t, _res_spect_ft)
    gc.collect()

    return _rms_ngfit # resturn normalised _rms
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# run dynesty for each line profile
#@jit(nopython=True)
@ray.remote(num_cpus=1)
#@ray.remote
def run_dynesty_sampler_optimal_priors(_inputDataCube, _x, _peak_sn_map, _sn_int_map, _params, _is, _ie, i, _js, _je, _cube_mask_2d):

    _max_ngauss = _params['max_ngauss']
    _vel_min = _params['vel_min']
    _vel_max = _params['vel_max']
    _cdelt3 = _params['cdelt3']

    gfit_results = np.zeros(((_je-_js), _max_ngauss, 2*(2+3*_max_ngauss) + 7 + _max_ngauss), dtype=np.float32)
    _x_boundaries = np.full(2*_max_ngauss, fill_value=-1E11, dtype=np.float32)

    #print("CHECK S/N: %d %d | peak S/N: %.1f < %.1f | integrated S/N: %.1f < %.1f" \
    #    % (i, 0+_js, _peak_sn_map[0+_js, i], _params['peak_sn_limit'], _sn_int_map[0+_js, i], _params['int_sn_limit']))
    for j in range(0, _je -_js):

        _f_max = np.max(_inputDataCube[:,j+_js,i]) # peak flux : being used for normalization
        _f_min = np.min(_inputDataCube[:,j+_js,i]) # lowest flux : being used for normalization
        #print(_f_max, _f_min)

        # prior arrays for the 1st single Gaussian fit
        gfit_priors_init = np.zeros(2*5, dtype=np.float32)
        #gfit_priors_init = [sig1, bg1, x1, std1, p1, sig2, bg2, x2, std2, p2]
        gfit_priors_init = [0.0, 0.0, 0.01, 0.01, 0.01, 0.5, 0.6, 0.99, 0.6, 1.01]

        if _cube_mask_2d[j+_js, i] <= 0 : # if masked, then skip : NOTE THE MASK VALUE SHOULD BE negative.
            print("mask filtered: %d %d | peak S/N: %.1f :: S/N limit: %.1f | integrated S/N: %.1f :: S/N limit: %.1f :: f_min: %e :: f_max: %e" \
                % (i, j+_js, _peak_sn_map[j+_js, i], _params['peak_sn_limit'], _sn_int_map[j+_js, i], _params['int_sn_limit'], _f_min, _f_max))

            # save the current profile location
            for l in range(0, _max_ngauss):
                gfit_results[j][l][2*(3*_max_ngauss+2)+l] = _params['_rms_med'] # rms: the one derived from derive_rms_npoints_sgfit
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+0] = -1E11 # this is for sgfit: log-Z
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+5] = i
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+6] = j + _js
            continue

        elif _sn_int_map[j+_js, i] < _params['int_sn_limit'] or _peak_sn_map[j+_js, i] < _params['peak_sn_limit'] \
            or np.isnan(_f_max) or np.isnan(_f_min) \
            or np.isinf(_f_min) or np.isinf(_f_min):

            print("low S/N: %d %d | peak S/N: %.1f :: S/N limit: %.1f | integrated S/N: %.1f :: S/N limit: %.1f :: f_min: %e :: f_max: %e" \
                % (i, j+_js, _peak_sn_map[j+_js, i], _params['peak_sn_limit'], _sn_int_map[j+_js, i], _params['int_sn_limit'], _f_min, _f_max))

            # save the current profile location
            for l in range(0, _max_ngauss):
                gfit_results[j][l][2*(3*_max_ngauss+2)+l] = _params['_rms_med'] # rms: the one derived from derive_rms_npoints_sgfit
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+0] = -1E11 # this is for sgfit: log-Z
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+5] = i
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+6] = j + _js
            continue


        for k in range(0, _max_ngauss):
            ngauss = k+1  # set the number of gaussian
            ndim = 3*ngauss + 2
            nparams = ndim

            if(ndim * (ndim + 1) // 2 > _params['nlive']):
                _params['nlive'] = 1 + ndim * (ndim + 1) // 2 # optimal minimum nlive

            print("processing: %d %d | peak s/n: %.1f | integrated s/n: %.1f | gauss-%d" % (i, j+_js, _peak_sn_map[j+_js, i], _sn_int_map[j+_js, i], ngauss))

            #sampler = NestedSampler(loglike_d, uniform_prior_d, ndim, sample='unif',
            #    vol_dec = 0.2, vol_check = 2, facc=0.5, rwalk=1000, nlive=200, max_move=100,
            #    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss])

            #---------------------------------------------------------
            # run dynesty 1.1
            #sampler = NestedSampler(loglike_d, optimal_prior, ndim,
            #    vol_dec=_params['vol_dec'],
            #    vol_check=_params['vol_check'],
            #    facc=_params['facc'],
            #    sample=_params['sample'],
            #    nlive=_params['nlive'],
            #    bound=_params['bound'],
            #    #rwalk=_params['rwalk'],
            #    max_move=_params['max_move'],
            #    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])

            if _params['_dynesty_class_'] == 'static':
                #---------------------------------------------------------
                # run dynesty 2.0.3
                sampler = NestedSampler(loglike_d, optimal_prior, ndim,
                    nlive=_params['nlive'],
                    update_interval=_params['update_interval'],
                    sample=_params['sample'],
                    bound=_params['bound'],
                    facc=_params['facc'],
                    fmove=_params['fmove'],
                    max_move=_params['max_move'],
                    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False)
                #numba.jit(sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False), nopython=True, cache=True, nogil=True, parallel=True)
                #_run_nested = jit(sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False), nopython=True, cache=True, nogil=True, fastmath=True)

            elif _params['_dynesty_class_'] == 'dynamic':
                #---------------------------------------------------------
                # run dynesty 2.0.3
                sampler = DynamicNestedSampler(loglike_d, optimal_prior, ndim,
                    nlive=_params['nlive'],
                    sample=_params['sample'],
                    #update_interval=_params['update_interval'],
                    bound=_params['bound'],
                    facc=_params['facc'],
                    fmove=_params['fmove'],
                    max_move=_params['max_move'],
                    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                #sampler.reset()
                #numba.jit(sampler.run_nested(dlogz=0.1, maxiter=5000000, maxcall=50000000, print_progress=False), nopython=True, cache=True, nogil=True, parallel=True)
                sampler.run_nested(dlogz_init=_params['dlogz'], maxiter_init=_params['maxiter'], maxcall_init=_params['maxcall'], print_progress=False)

            #---------------------------------------------------------
            _gfit_results_temp, _logz = get_dynesty_sampler_results(sampler)

            #---------------------------------------------------------
            # param1, param2, param3 ....param1-e, param2-e, param3-e
            #gfit_results[j][k][0~2*nparams] = _gfit_results_temp[0~2*nparams]
            gfit_results[j][k][:2*nparams] = _gfit_results_temp
            #---------------------------------------------------------

            #---------------------------------------------------------
            # derive rms of the profile given the current ngfit
            _rms_ngfit = little_derive_rms_npoints(_inputDataCube, i, j+_js, _x, _f_min, _f_max, ngauss, _gfit_results_temp)
            #---------------------------------------------------------

            #---------------------------------------------------------
            if ngauss == 1: # check the peak s/n
                # load the normalised sgfit results : --> derive rms for s/n
                #_bg_sgfit = _gfit_results_temp[1]
                #_x_sgfit = _gfit_results_temp[2]
                #_std_sgfit = _gfit_results_temp[3]
                #_p_sgfit = _gfit_results_temp[4]
                # peak flux of the sgfit
                #_f_sgfit =_p_sgfit * exp( -0.5*((_x - _x_sgfit) / _std_sgfit)**2) + _bg_sgfit

                #---------------------------------------------------------
                # update gfit_priors_init
                nparams_n = 3*(ngauss+1) + 2
                gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)
                # lower bound : the parameters for the current ngaussian components
                # nsigma_prior_range_gfit=3.0 (default)
                gfit_priors_init[:nparams] = _gfit_results_temp[:nparams] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
                # upper bound : the parameters for the current ngaussian components
                gfit_priors_init[nparams+3:2*nparams+3] = _gfit_results_temp[:nparams] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
                #---------------------------------------------------------


                # peak s/n : more accurate peak s/n from the first sgfit
                _bg_sgfit = _gfit_results_temp[1]
                _p_sgfit = _gfit_results_temp[4] # bg already subtracted
                _peak_sn_sgfit = _p_sgfit/_rms_ngfit

                if _peak_sn_sgfit < _params['peak_sn_limit']: 
                    print("skip the rest of Gaussian fits: %d %d | rms:%.1f | bg:%.1f | peak:%.1f | peak_sgfit s/n: %.1f < %.1f" % (i, j+_js, _rms_ngfit, _bg_sgfit, _p_sgfit, _peak_sn_sgfit, _params['peak_sn_limit']))

                    # save the current profile location
                    for l in range(0, _max_ngauss):
                        if l == 0:
                        # for sgfit
                            gfit_results[j][l][2*(3*_max_ngauss+2)+l] = _rms_ngfit # this is for sgfit : rms
                            gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+0] = _logz # this is for sgfit: log-Z
                        else:
                            gfit_results[j][l][2*(3*_max_ngauss+2)+l] = 0 # put a blank value
                            gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+0] = -1E11 # put a blank value

                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+5] = i
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+6] = _js + j

                    #________________________________________________________________________________________|
                    #|---------------------------------------------------------------------------------------|
                    # unit conversion
                    # sigma-flux --> data cube units
                    gfit_results[j][k][0] = gfit_results[j][k][0]*(_f_max - _f_min) # sigma-flux
                    # background --> data cube units
                    gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
                    gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e
                    _bg_flux = gfit_results[j][k][1]
        
                    for m in range(0, k+1):
                        #________________________________________________________________________________________|
                        # UNIT CONVERSION
                        #________________________________________________________________________________________|
                        # velocity, velocity-dispersion --> km/s
                        if _cdelt3 > 0: # if velocity axis is with increasing order
                            gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min # velocity
                        elif _cdelt3 < 0: # if velocity axis is with decreasing order
                            gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_min - _vel_max) + _vel_max # velocity

                        gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min) # velocity-dispersion

                        #________________________________________________________________________________________|
                        # peak flux --> data cube units
                        #gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) + _f_min # flux
                        gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _bg_flux) # peak flux
        
                        #________________________________________________________________________________________|
                        # velocity-e, velocity-dispersion-e --> km/s
                        gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-e
                        gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-dispersion-e
                        #________________________________________________________________________________________|
                        #gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # flux-e
                        gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _bg_flux) # flux-e

                    # lastly put rms 
                    gfit_results[j][k][2*(3*_max_ngauss+2)+k] = gfit_results[j][k][2*(3*_max_ngauss+2)+k]*(_f_max - _f_min) # rms-(k+1)gfit
                    #________________________________________________________________________________________|
                    #|---------------------------------------------------------------------------------------|
                    continue
            #---------------------------------------------------------


            # update optimal priors based on the current ngaussian fit results
            if ngauss < _max_ngauss:
                nparams_n = 3*(ngauss+1) + 2
                gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)
                # lower bound : the parameters for the current ngaussian components
                # nsigma_prior_range_gfit=3.0 (default)
                gfit_priors_init[:nparams] = _gfit_results_temp[:nparams] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
                # upper bound : the parameters for the current ngaussian components
                gfit_priors_init[nparams+3:2*nparams+3] = _gfit_results_temp[:nparams] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
    
                # the parameters for the next gaussian component: based on the current ngaussians
                _x_min_t = _gfit_results_temp[2:nparams:3].min()
                _x_max_t = _gfit_results_temp[2:nparams:3].max()
                _std_min_t = _gfit_results_temp[3:nparams:3].min()
                _std_max_t = _gfit_results_temp[3:nparams:3].max()
                _p_min_t = _gfit_results_temp[4:nparams:3].min()
                _p_max_t = _gfit_results_temp[4:nparams:3].max()

                # sigma_prior_lowerbound_factor=0.2 (default), sigma_prior_upperbound_factor=2.0 (default)
                gfit_priors_init[0] = _params['sigma_prior_lowerbound_factor']*_gfit_results_temp[0]
                gfit_priors_init[nparams_n] = _params['sigma_prior_upperbound_factor']*_gfit_results_temp[0]

                # bg_prior_lowerbound_factor=0.2 (defaut), bg_prior_upperbound_factor=2.0 (default)
                gfit_priors_init[1] = _params['bg_prior_lowerbound_factor']*_gfit_results_temp[1]
                gfit_priors_init[nparams_n + 1] = _params['bg_prior_upperbound_factor']*_gfit_results_temp[1]

                #print("x:", _x_min_t, _x_max_t, "std:", _std_min_t, _std_max_t, "p:",_p_min_t, _p_max_t)

                #____________________________________________
                # x: lower bound
                if ngauss == 1:
                    # x_lowerbound_gfit=0.1 (default), x_upperbound_gfit=0.9 (default)
                    gfit_priors_init[nparams] = _params['x_lowerbound_gfit']
                    gfit_priors_init[2*nparams+3] = _params['x_upperbound_gfit']
                    #if gfit_priors_init[nparams] < 0 : gfit_priors_init[nparams] = 0
                else:
                    # x_prior_lowerbound_factor=5 (default), x_prior_upperbound_factor=5 (default)
                    gfit_priors_init[nparams] = _x_min_t - _params['x_prior_lowerbound_factor']*_std_max_t
                    gfit_priors_init[2*nparams+3] = _x_max_t + _params['x_prior_upperbound_factor']*_std_max_t
                    #if gfit_priors_init[2*nparams+3] > 1 : gfit_priors_init[2*nparams+3] = 1

                #____________________________________________
                # std: lower bound
                # std_prior_lowerbound_factor=0.1 (default)
                gfit_priors_init[nparams+1] = _params['std_prior_lowerbound_factor']*_std_min_t
                #gfit_priors_init[nparams+1] = 0.01
                #if gfit_priors_init[nparams+1] < 0 : gfit_priors_init[nparams+1] = 0
                # std: upper bound
                # std_prior_upperbound_factor=3.0 (default)
                gfit_priors_init[2*nparams+4] = _params['std_prior_upperbound_factor']*_std_max_t
                #gfit_priors_init[2*nparams+4] = 0.9
                #if gfit_priors_init[2*nparams+4] > 1 : gfit_priors_init[2*nparams+4] = 1
    
                #____________________________________________
                # p: lower bound
                # p_prior_lowerbound_factor=0.05 (default)
                gfit_priors_init[nparams+2] = _params['p_prior_lowerbound_factor']*_p_max_t # 5% of the maxium flux
                # p: upper bound
                # p_prior_upperbound_factor=1.0 (default)
                gfit_priors_init[2*nparams+5] = _params['p_prior_upperbound_factor']*_p_max_t

                gfit_priors_init = np.where(gfit_priors_init<0, 0, gfit_priors_init)
                gfit_priors_init = np.where(gfit_priors_init>1, 1, gfit_priors_init)


            gfit_results[j][k][2*(3*_max_ngauss+2)+k] = _rms_ngfit # rms_(k+1)gfit
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+0] = _logz
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+5] = i
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+6] = _js + j
            #print(gfit_results[j][k])

            #|-----------------------------------------|
            #|-----------------------------------------|
            # example: 3 gaussians : bg + 3 * (x, std, peak) 
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

            #|-----------------------------------------|
            #gfit_results[j][k][0] : dist-sig
            #gfit_results[j][k][1] : bg
            #gfit_results[j][k][2] : g1-x --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][3] : g1-s --> *(vel_max-vel_min)
            #gfit_results[j][k][4] : g1-p
            #gfit_results[j][k][5] : g2-x --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][6] : g2-s --> *(vel_max-vel_min)
            #gfit_results[j][k][7] : g2-p
            #gfit_results[j][k][8] : g3-x --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][9] : g3-s --> *(vel_max-vel_min)
            #gfit_results[j][k][10] : g3-p

            #gfit_results[j][k][11] : dist-sig-e
            #gfit_results[j][k][12] : bg-e
            #gfit_results[j][k][13] : g1-x-e --> *(vel_max-vel_min)
            #gfit_results[j][k][14] : g1-s-e --> *(vel_max-vel_min)
            #gfit_results[j][k][15] : g1-p-e
            #gfit_results[j][k][16] : g2-x-e --> *(vel_max-vel_min)
            #gfit_results[j][k][17] : g2-s-e --> *(vel_max-vel_min)
            #gfit_results[j][k][18] : g2-p-e
            #gfit_results[j][k][19] : g3-x-e --> *(vel_max-vel_min)
            #gfit_results[j][k][20] : g3-s-e --> *(vel_max-vel_min)
            #gfit_results[j][k][21] : g3-p-e --> *(f_max-bg_flux)

            #gfit_results[j][k][22] : g1-rms --> *(f_max-bg_flux) : the bg rms for the case with single gaussian fitting
            #gfit_results[j][k][23] : g2-rms --> *(f_max-bg_flux) : the bg rms for the case with double gaussian fitting
            #gfit_results[j][k][24] : g3-rms --> *(f_max-bg_flux) : the bg rms for the case with triple gaussian fitting

            #gfit_results[j][k][25] : log-Z : log-evidence : log-marginalization likelihood

            #gfit_results[j][k][26] : xs
            #gfit_results[j][k][27] : xe
            #gfit_results[j][k][28] : ys
            #gfit_results[j][k][29] : ye
            #gfit_results[j][k][30] : x
            #gfit_results[j][k][31] : y
            #|-----------------------------------------|

            #________________________________________________________________________________________|
            #|---------------------------------------------------------------------------------------|
            # UNIT CONVERSION
            # sigma-flux --> data cube units
            gfit_results[j][k][0] = gfit_results[j][k][0]*(_f_max - _f_min) # sigma-flux
            # background --> data cube units
            gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
            gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e
            _bg_flux = gfit_results[j][k][1]

            for m in range(0, k+1):
                #________________________________________________________________________________________|
                # UNIT CONVERSION

                #________________________________________________________________________________________|
                # velocity, velocity-dispersion --> km/s
                if _cdelt3 > 0: # if velocity axis is with increasing order
                    gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min # velocity
                elif _cdelt3 < 0: # if velocity axis is with decreasing order
                    gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_min - _vel_max) + _vel_max # velocity

                gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min) # velocity-dispersion

                #________________________________________________________________________________________|
                # peak flux --> data cube units : (_f_max - _bg_flux) should be used for scaling as the normalised peak flux is from the bg
                #gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) + _f_min # flux
                gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _bg_flux) # flux

                #________________________________________________________________________________________|
                # velocity-e, velocity-dispersion-e --> km/s
                gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-e
                gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-dispersion-e

                #gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # flux-e
                gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _bg_flux) # peak flux-e

            gfit_results[j][k][2*(3*_max_ngauss+2)+k] = gfit_results[j][k][2*(3*_max_ngauss+2)+k]*(_f_max - _f_min) # rms-(k+1)gfit
            #________________________________________________________________________________________|
            #|---------------------------------------------------------------------------------------|

    #del(_gfit_results_temp, gfit_priors_init)
    #gc.collect()

    return gfit_results

    # Plot a summary of the run.
#    rfig, raxes = dyplot.runplot(sampler.results)
#    rfig.savefig("r.pdf")
    
    # Plot traces and 1-D marginalized posteriors.
#    tfig, taxes = dyplot.traceplot(sampler.results)
#    tfig.savefig("t.pdf")
    
    # Plot the 2-D marginalized posteriors.
    #cfig, caxes = dyplot.cornerplot(sampler.results)
    #cfig.savefig("c.pdf")
#-- END OF SUB-ROUTINE____________________________________________________________#

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# run dynesty for each line profile
#@jit(nopython=True)
@ray.remote(num_cpus=1)
#@ray.remote
def run_dynesty_sampler_optimal_priors_org(_inputDataCube, _x, _peak_sn_map, _sn_int_map, _params, _is, _ie, i, _js, _je):

    _max_ngauss = _params['max_ngauss']
    _vel_min = _params['vel_min']
    _vel_max = _params['vel_max']
    gfit_results = np.zeros(((_je-_js), _max_ngauss, 2*(2+3*_max_ngauss) + 7 + _max_ngauss), dtype=np.float32)
    _x_boundaries = np.full(2*_max_ngauss, fill_value=-1E11, dtype=np.float32)

    #print("CHECK S/N: %d %d | peak S/N: %.1f < %.1f | integrated S/N: %.1f < %.1f" \
    #    % (i, 0+_js, _peak_sn_map[0+_js, i], _params['peak_sn_limit'], _sn_int_map[0+_js, i], _params['int_sn_limit']))
    for j in range(0, _je -_js):

        _f_max = np.max(_inputDataCube[:,j+_js,i]) # peak flux : being used for normalization
        _f_min = np.min(_inputDataCube[:,j+_js,i]) # lowest flux : being used for normalization
        #print(_f_max, _f_min)

        # prior arrays for the 1st single Gaussian fit
        gfit_priors_init = np.zeros(2*5, dtype=np.float32)
        #gfit_priors_init = [sig1, bg1, x1, std1, p1, sig2, bg2, x2, std2, p2]
        gfit_priors_init = [0.0, 0.0, 0.01, 0.01, 0.01, 0.5, 0.6, 0.99, 0.6, 1.01]

        if _sn_int_map[j+_js, i] < _params['int_sn_limit'] or _peak_sn_map[j+_js, i] < _params['peak_sn_limit'] \
            or np.isnan(_f_max) or np.isnan(_f_min) \
            or np.isinf(_f_min) or np.isinf(_f_min):

            print("low S/N: %d %d | peak S/N: %.1f :: S/N limit: %.1f | integrated S/N: %.1f :: S/N limit: %.1f :: f_min: %e :: f_max: %e" \
                % (i, j+_js, _peak_sn_map[j+_js, i], _params['peak_sn_limit'], _sn_int_map[j+_js, i], _params['int_sn_limit'], _f_min, _f_max))
            # save the current profile location
            for l in range(0, _max_ngauss):
                gfit_results[j][l][2*(3*_max_ngauss+2)+l] = _params['_rms_med'] # rms: the one derived from derive_rms_npoints_sgfit
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+0] = -1E11 # this is for sgfit: log-Z
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+5] = i
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+6] = j + _js
            continue

        for k in range(0, _max_ngauss):
            ngauss = k+1  # set the number of gaussian
            ndim = 3*ngauss + 2
            nparams = ndim

            if(ndim * (ndim + 1) // 2 > _params['nlive']):
                _params['nlive'] = 1 + ndim * (ndim + 1) // 2 # optimal minimum nlive

            print("processing: %d %d | peak s/n: %.1f | integrated s/n: %.1f | gauss-%d" % (i, j+_js, _peak_sn_map[j+_js, i], _sn_int_map[j+_js, i], ngauss))

            #sampler = NestedSampler(loglike_d, uniform_prior_d, ndim, sample='unif',
            #    vol_dec = 0.2, vol_check = 2, facc=0.5, rwalk=1000, nlive=200, max_move=100,
            #    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss])

            #---------------------------------------------------------
            # run dynesty 1.1
            #sampler = NestedSampler(loglike_d, optimal_prior, ndim,
            #    vol_dec=_params['vol_dec'],
            #    vol_check=_params['vol_check'],
            #    facc=_params['facc'],
            #    sample=_params['sample'],
            #    nlive=_params['nlive'],
            #    bound=_params['bound'],
            #    #rwalk=_params['rwalk'],
            #    max_move=_params['max_move'],
            #    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])

            if _params['_dynesty_class_'] == 'static':
                #---------------------------------------------------------
                # run dynesty 2.0.3
                sampler = NestedSampler(loglike_d, optimal_prior, ndim,
                    nlive=_params['nlive'],
                    update_interval=_params['update_interval'],
                    sample=_params['sample'],
                    bound=_params['bound'],
                    facc=_params['facc'],
                    fmove=_params['fmove'],
                    max_move=_params['max_move'],
                    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False)
                #numba.jit(sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False), nopython=True, cache=True, nogil=True, parallel=True)
                #_run_nested = jit(sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False), nopython=True, cache=True, nogil=True, fastmath=True)

            elif _params['_dynesty_class_'] == 'dynamic':
                #---------------------------------------------------------
                # run dynesty 2.0.3
                sampler = DynamicNestedSampler(loglike_d, optimal_prior, ndim,
                    nlive=_params['nlive'],
                    sample=_params['sample'],
                    update_interval=_params['update_interval'],
                    bound=_params['bound'],
                    facc=_params['facc'],
                    fmove=_params['fmove'],
                    max_move=_params['max_move'],
                    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                sampler.run_nested(dlogz_init=_params['dlogz'], maxiter_init=_params['maxiter'], maxcall_init=_params['maxcall'], print_progress=False)

                #sampler.reset()
                #numba.jit(sampler.run_nested(dlogz=0.1, maxiter=5000000, maxcall=50000000, print_progress=False), nopython=True, cache=True, nogil=True, parallel=True)

            #---------------------------------------------------------
            _gfit_results_temp, _logz = get_dynesty_sampler_results(sampler)

            #---------------------------------------------------------
            # param1, param2, param3 ....param1-e, param2-e, param3-e
            #gfit_results[j][k][0~2*nparams] = _gfit_results_temp[0~2*nparams]
            gfit_results[j][k][:2*nparams] = _gfit_results_temp
            #---------------------------------------------------------

            #---------------------------------------------------------
            # derive rms of the profile given the current ngfit
            _rms_ngfit = little_derive_rms_npoints(_inputDataCube, i, j+_js, _x, _f_min, _f_max, ngauss, _gfit_results_temp)
            #---------------------------------------------------------

            #---------------------------------------------------------
            if ngauss == 1: # check the peak s/n
                # load the normalised sgfit results : --> derive rms for s/n
                #_bg_sgfit = _gfit_results_temp[1]
                #_x_sgfit = _gfit_results_temp[2]
                #_std_sgfit = _gfit_results_temp[3]
                #_p_sgfit = _gfit_results_temp[4]
                # peak flux of the sgfit
                #_f_sgfit =_p_sgfit * exp( -0.5*((_x - _x_sgfit) / _std_sgfit)**2) + _bg_sgfit

                #---------------------------------------------------------
                # update gfit_priors_init
                nparams_n = 3*(ngauss+1) + 2
                gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)
                # lower bound : the parameters for the current ngaussian components
                # nsigma_prior_range_gfit=3.0 (default)
                gfit_priors_init[:nparams] = _gfit_results_temp[:nparams] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
                # upper bound : the parameters for the current ngaussian components
                gfit_priors_init[nparams+3:2*nparams+3] = _gfit_results_temp[:nparams] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
                #---------------------------------------------------------


                # peak s/n : more accurate peak s/n from the first sgfit
                _bg_sgfit = _gfit_results_temp[1]
                _p_sgfit = _gfit_results_temp[4] # bg already subtracted
                _peak_sn_sgfit = _p_sgfit/_rms_ngfit

                if _peak_sn_sgfit < _params['peak_sn_limit']: 
                    print("skip the rest of Gaussian fits: %d %d | rms:%.1f | bg:%.1f | peak:%.1f | peak_sgfit s/n: %.1f < %.1f" % (i, j+_js, _rms_ngfit, _bg_sgfit, _p_sgfit, _peak_sn_sgfit, _params['peak_sn_limit']))

                    # save the current profile location
                    for l in range(0, _max_ngauss):
                        if l == 0:
                        # for sgfit
                            gfit_results[j][l][2*(3*_max_ngauss+2)+l] = _rms_ngfit # this is for sgfit : rms
                            gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+0] = _logz # this is for sgfit: log-Z
                        else:
                            gfit_results[j][l][2*(3*_max_ngauss+2)+l] = 0 # put a blank value
                            gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+0] = -1E11 # put a blank value

                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+5] = i
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+6] = _js + j

                    #________________________________________________________________________________________|
                    #|---------------------------------------------------------------------------------------|
                    # unit conversion
                    # sigma-flux --> data cube units
                    gfit_results[j][k][0] = gfit_results[j][k][0]*(_f_max - _f_min) # sigma-flux
                    # background --> data cube units
                    gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
                    gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e
                    _bg_flux = gfit_results[j][k][1]
        
                    for m in range(0, k+1):
                        # unit conversion
                        # peak flux --> data cube units
                        # velocity, velocity-dispersion --> km/s
                        gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min # velocity
                        gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min) # velocity-dispersion
                        #gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) + _f_min # flux
                        gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _bg_flux) # peak flux
        
                        gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-e
                        gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-dispersion-e
                        #gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # flux-e
                        gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _bg_flux) # flux-e

                    # lastly put rms 
                    gfit_results[j][k][2*(3*_max_ngauss+2)+k] = gfit_results[j][k][2*(3*_max_ngauss+2)+k]*(_f_max - _f_min) # rms-(k+1)gfit
                    #________________________________________________________________________________________|
                    #|---------------------------------------------------------------------------------------|
                    continue
            #---------------------------------------------------------


            # update optimal priors based on the current ngaussian fit results
            if ngauss < _max_ngauss:
                nparams_n = 3*(ngauss+1) + 2
                gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)
                # lower bound : the parameters for the current ngaussian components
                # nsigma_prior_range_gfit=3.0 (default)
                gfit_priors_init[:nparams] = _gfit_results_temp[:nparams] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
                # upper bound : the parameters for the current ngaussian components
                gfit_priors_init[nparams+3:2*nparams+3] = _gfit_results_temp[:nparams] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
    
                # the parameters for the next gaussian component: based on the current ngaussians
                _x_min_t = _gfit_results_temp[2:nparams:3].min()
                _x_max_t = _gfit_results_temp[2:nparams:3].max()
                _std_min_t = _gfit_results_temp[3:nparams:3].min()
                _std_max_t = _gfit_results_temp[3:nparams:3].max()
                _p_min_t = _gfit_results_temp[4:nparams:3].min()
                _p_max_t = _gfit_results_temp[4:nparams:3].max()

                # sigma_prior_lowerbound_factor=0.2 (default), sigma_prior_upperbound_factor=2.0 (default)
                gfit_priors_init[0] = _params['sigma_prior_lowerbound_factor']*_gfit_results_temp[0]
                gfit_priors_init[nparams_n] = _params['sigma_prior_upperbound_factor']*_gfit_results_temp[0]

                # bg_prior_lowerbound_factor=0.2 (defaut), bg_prior_upperbound_factor=2.0 (default)
                gfit_priors_init[1] = _params['bg_prior_lowerbound_factor']*_gfit_results_temp[1]
                gfit_priors_init[nparams_n + 1] = _params['bg_prior_upperbound_factor']*_gfit_results_temp[1]

                #print("x:", _x_min_t, _x_max_t, "std:", _std_min_t, _std_max_t, "p:",_p_min_t, _p_max_t)

                #____________________________________________
                # x: lower bound
                if ngauss == 1:
                    # x_lowerbound_gfit=0.1 (default), x_upperbound_gfit=0.9 (default)
                    gfit_priors_init[nparams] = _params['x_lowerbound_gfit']
                    gfit_priors_init[2*nparams+3] = _params['x_upperbound_gfit']
                    #if gfit_priors_init[nparams] < 0 : gfit_priors_init[nparams] = 0
                else:
                    # x_prior_lowerbound_factor=5 (default), x_prior_upperbound_factor=5 (default)
                    gfit_priors_init[nparams] = _x_min_t - _params['x_prior_lowerbound_factor']*_std_max_t
                    gfit_priors_init[2*nparams+3] = _x_max_t + _params['x_prior_upperbound_factor']*_std_max_t
                    #if gfit_priors_init[2*nparams+3] > 1 : gfit_priors_init[2*nparams+3] = 1

                #____________________________________________
                # std: lower bound
                # std_prior_lowerbound_factor=0.1 (default)
                gfit_priors_init[nparams+1] = _params['std_prior_lowerbound_factor']*_std_min_t
                #gfit_priors_init[nparams+1] = 0.01
                #if gfit_priors_init[nparams+1] < 0 : gfit_priors_init[nparams+1] = 0
                # std: upper bound
                # std_prior_upperbound_factor=3.0 (default)
                gfit_priors_init[2*nparams+4] = _params['std_prior_upperbound_factor']*_std_max_t
                #gfit_priors_init[2*nparams+4] = 0.9
                #if gfit_priors_init[2*nparams+4] > 1 : gfit_priors_init[2*nparams+4] = 1
    
                #____________________________________________
                # p: lower bound
                # p_prior_lowerbound_factor=0.05 (default)
                gfit_priors_init[nparams+2] = _params['p_prior_lowerbound_factor']*_p_max_t # 5% of the maxium flux
                # p: upper bound
                # p_prior_upperbound_factor=1.0 (default)
                gfit_priors_init[2*nparams+5] = _params['p_prior_upperbound_factor']*_p_max_t

                gfit_priors_init = np.where(gfit_priors_init<0, 0, gfit_priors_init)
                gfit_priors_init = np.where(gfit_priors_init>1, 1, gfit_priors_init)


            gfit_results[j][k][2*(3*_max_ngauss+2)+k] = _rms_ngfit # rms_(k+1)gfit
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+0] = _logz
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+5] = i
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+6] = _js + j
            #print(gfit_results[j][k])

            #|-----------------------------------------|
            # example: 3 gaussians : bg + 3 * (x, std, peak) 
            #gfit_results[j][k][0] : dist-sig
            #gfit_results[j][k][1] : bg
            #gfit_results[j][k][2] : g1-x --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][3] : g1-s --> *(vel_max-vel_min)
            #gfit_results[j][k][4] : g1-p
            #gfit_results[j][k][5] : g2-x --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][6] : g2-s --> *(vel_max-vel_min)
            #gfit_results[j][k][7] : g2-p
            #gfit_results[j][k][8] : g3-x --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][9] : g3-s --> *(vel_max-vel_min)
            #gfit_results[j][k][10] : g3-p

            #gfit_results[j][k][11] : dist-sig-e
            #gfit_results[j][k][12] : bg-e
            #gfit_results[j][k][13] : g1-x-e --> *(vel_max-vel_min)
            #gfit_results[j][k][14] : g1-s-e --> *(vel_max-vel_min)
            #gfit_results[j][k][15] : g1-p-e
            #gfit_results[j][k][16] : g2-x-e --> *(vel_max-vel_min)
            #gfit_results[j][k][17] : g2-s-e --> *(vel_max-vel_min)
            #gfit_results[j][k][18] : g2-p-e
            #gfit_results[j][k][19] : g3-x-e --> *(vel_max-vel_min)
            #gfit_results[j][k][20] : g3-s-e --> *(vel_max-vel_min)
            #gfit_results[j][k][21] : g3-p-e --> *(f_max-bg_flux)

            #gfit_results[j][k][22] : g1-rms --> *(f_max-bg_flux)
            #gfit_results[j][k][23] : g2-rms --> *(f_max-bg_flux)
            #gfit_results[j][k][24] : g3-rms --> *(f_max-bg_flux)

            #gfit_results[j][k][25] : log-Z : log-evidence : log-marginalization likelihood

            #gfit_results[j][k][26] : xs
            #gfit_results[j][k][27] : xe
            #gfit_results[j][k][28] : ys
            #gfit_results[j][k][29] : ye
            #gfit_results[j][k][30] : x
            #gfit_results[j][k][31] : y
            #|-----------------------------------------|

            #________________________________________________________________________________________|
            #|---------------------------------------------------------------------------------------|
            # unit conversion
            # sigma-flux --> data cube units
            gfit_results[j][k][0] = gfit_results[j][k][0]*(_f_max - _f_min) # sigma-flux
            # background --> data cube units
            gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
            gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e
            _bg_flux = gfit_results[j][k][1]

            for m in range(0, k+1):
                # unit conversion
                # peak flux --> data cube units
                # velocity, velocity-dispersion --> km/s
                gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min # velocity
                gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min) # velocity-dispersion
                #gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) + _f_min # flux
                gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _bg_flux) # flux

                gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-e
                gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-dispersion-e
                #gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # flux-e
                gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _bg_flux) # peak flux-e

            gfit_results[j][k][2*(3*_max_ngauss+2)+k] = gfit_results[j][k][2*(3*_max_ngauss+2)+k]*(_f_max - _f_min) # rms-(k+1)gfit
            #________________________________________________________________________________________|
            #|---------------------------------------------------------------------------------------|

    #del(_gfit_results_temp, gfit_priors_init)
    #gc.collect()

    return gfit_results

    # Plot a summary of the run.
#    rfig, raxes = dyplot.runplot(sampler.results)
#    rfig.savefig("r.pdf")
    
    # Plot traces and 1-D marginalized posteriors.
#    tfig, taxes = dyplot.traceplot(sampler.results)
#    tfig.savefig("t.pdf")
    
    # Plot the 2-D marginalized posteriors.
    #cfig, caxes = dyplot.cornerplot(sampler.results)
    #cfig.savefig("c.pdf")
#-- END OF SUB-ROUTINE____________________________________________________________#

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# run dynesty for each line profile
@ray.remote(num_cpus=1)
#@ray.remote
def run_dynesty_sampler_uniform_priors(_x, _inputDataCube, _is, _ie, i, _js, _je, _max_ngauss, _vel_min, _vel_max):

    gfit_results = np.zeros(((_je-_js), _max_ngauss, 2*(2+3*_max_ngauss)+7), dtype=np.float32)
    for j in range(0, _je -_js):

        _f_max = np.max(_inputDataCube[:,j+_js,i]) # peak flux : being used for normalization
        _f_min = np.min(_inputDataCube[:,j+_js,i]) # lowest flux : being used for normalization

        #print(_f_max, _f_min)
        gfit_priors_init = np.zeros(2*5, dtype=np.float32)
        #gfit_priors_init = [sig1, bg1, x1, std1, p1, sig2, bg2, x2, std2, p2]
        gfit_priors_init = [0.0, 0.0, 0.01, 0.01, 0.01, 0.5, 0.5, 0.9, 0.6, 1.01]
        for k in range(0, _max_ngauss):
            ngauss = k+1  # set the number of gaussian
            ndim = 3*ngauss + 2
            nparams = ndim

#            if(ndim * (ndim + 1) // 2 > 100):
#                _nlive = 1 + ndim * (ndim + 1) // 2 # optimal minimum nlive
#            else:
#                _nlive = 100
    
            # run dynesty
            print("processing: %d %d gauss-%d" % (i, j+_js, ngauss))

            #sampler = NestedSampler(loglike_d, uniform_prior_d, ndim, sample='unif',
            #    vol_dec = 0.2, vol_check = 2, facc=0.5, rwalk=1000, nlive=200, max_move=100,
            #    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss])

            #sampler = NestedSampler(loglike_d, uniform_prior, ndim,
            #    vol_dec = 0.2, vol_check = 2, facc=0.5, sample='hslice', nlive=100, bound='multi', rwalk=1000, max_move=100,
            #    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])

            # run dynesty 1.1   
            #sampler = NestedSampler(loglike_d, uniform_prior, ndim,
            #    vol_dec = 0.2, vol_check = 2, facc=0.5, sample='auto', nlive=100, bound='multi', max_move=100,
            #    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
#
            if _params['_dynesty_class_'] == 'static':
                #---------------------------------------------------------
                # run dynesty 2.0.3
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
                #numba.jit(sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False), nopython=True, cache=True, nogil=True, parallel=True)
                #_run_nested = jit(sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False), nopython=True, cache=True, nogil=True, fastmath=True)

            elif _params['_dynesty_class_'] == 'dynamic':
                #---------------------------------------------------------
                # run dynesty 2.0.3
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

                #sampler.reset()
                #numba.jit(sampler.run_nested(dlogz=0.1, maxiter=5000000, maxcall=50000000, print_progress=False), nopython=True, cache=True, nogil=True, parallel=True)
            #---------------------------------------------------------
            _gfit_results_temp, _logz = get_dynesty_sampler_results(sampler)

            # param1, param2, param3 ....param1-e, param2-e, param3-e
            #gfit_results[j][k][0~2*nparams] = _gfit_results_temp[0~2*nparams]
            gfit_results[j][k][:2*nparams] = _gfit_results_temp

            print(gfit_priors_init)

            gfit_results[j][k][2*(3*_max_ngauss+2)+0] = _logz
            gfit_results[j][k][2*(3*_max_ngauss+2)+1] = _is
            gfit_results[j][k][2*(3*_max_ngauss+2)+2] = _ie
            gfit_results[j][k][2*(3*_max_ngauss+2)+3] = _js
            gfit_results[j][k][2*(3*_max_ngauss+2)+4] = _je
            gfit_results[j][k][2*(3*_max_ngauss+2)+5] = i
            gfit_results[j][k][2*(3*_max_ngauss+2)+6] = _js + j
            #print(gfit_results[j][k])

            #|-----------------------------------------|
            # example: 3 gaussians : bg + 3 * (x, std, peak) 
            #gfit_results[j][k][0] : dist-sig
            #gfit_results[j][k][1] : bg
            #gfit_results[j][k][2] : g1-x1 --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][3] : g1-s1 --> *(vel_max-vel_min)
            #gfit_results[j][k][4] : g1-p1
            #gfit_results[j][k][5] : g2-x2 --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][6] : g2-s2 --> *(vel_max-vel_min)
            #gfit_results[j][k][7] : g2-p2
            #gfit_results[j][k][8] : g3-x3 --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][9] : g3-s3 --> *(vel_max-vel_min)
            #gfit_results[j][k][10] : g3-p3

            #gfit_results[j][k][11] : dist-sig-e
            #gfit_results[j][k][12] : bg-e
            #gfit_results[j][k][13] : g1-x1-e --> *(vel_max-vel_min)
            #gfit_results[j][k][14] : g1-s1-e --> *(vel_max-vel_min)
            #gfit_results[j][k][15] : g1-p1-e
            #gfit_results[j][k][16] : g2-x2-e --> *(vel_max-vel_min)
            #gfit_results[j][k][17] : g2-s2-e --> *(vel_max-vel_min)
            #gfit_results[j][k][18] : g2-p2-e
            #gfit_results[j][k][19] : g3-x3-e --> *(vel_max-vel_min)

            #gfit_results[j][k][22] : log-Z : log-evidence : log-marginalization likelihood

            #gfit_results[j][k][23] : xs
            #gfit_results[j][k][24] : xe
            #gfit_results[j][k][25] : ys
            #gfit_results[j][k][26] : ye
            #gfit_results[j][k][27] : x
            #gfit_results[j][k][28] : y
            #|-----------------------------------------|

            # unit conversion
            # background --> data cube units
            gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
            gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e

            for m in range(0, k+1):
                # unit conversion
                # peak flux --> data cube units
                # velocity, velocity-dispersion --> km/s
                gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min # velocity
                gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min) # velocity-dispersion
                gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) + _f_min # flux

                gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-e
                gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-dispersion-e
                gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # flux-e

            #if(gfit_results[j][k][4] < 2E-3):
            #    break;
    
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
    #cfig, caxes = dyplot.cornerplot(sampler.results)
    #cfig.savefig("c.pdf")
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def get_dynesty_sampler_results(_sampler):
    # Extract sampling results.
    samples = _sampler.results.samples  # samples
    weights = exp(_sampler.results.logwt - _sampler.results.logz[-1])  # normalized weights

    #print(_sampler.results.samples[-1, :]) 
    #print(_sampler.results.logwt.shape) 

    # Compute 10%-90% quantiles.
    quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
                for samps in samples.T]
    
    # Compute weighted mean and covariance.
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    bestfit_results = _sampler.results.samples[-1, :]
    log_Z = log(exp(_sampler.results.logz[-1]) - exp(_sampler.results.logz[-2]))

    #print(bestfit_results, log_Z)
    #print(concatenate((bestfit_results, diag(cov)**0.5)))

    # Resample weighted samples.
    #samples_equal = dyfunc.resample_equal(samples, weights)
    
    # Generate a new set of results with statistical+sampling uncertainties.
    #results_sim = dyfunc.simulate_run(_sampler.results)

    #mean_std = np.concatenate((mean, diag(cov)**0.5))
    #return mean_std # meand + std of each parameter: std array is followed by the mean array
    del(samples, weights, quantiles)
    gc.collect()

    return concatenate((bestfit_results, diag(cov)**0.5)), log_Z
    #return concatenate((bestfit_results, diag(cov)**0.5)), _sampler.results.logz[-1]
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def multi_gaussian_model_d(_x, _params, ngauss): # params: cube
    #_bg0 : _params[1]
    try:
        g = ((_params[3*i+4] * exp( -0.5*((_x - _params[3*i+2]) / _params[3*i+3])**2)) \
            for i in range(0, ngauss) \
            if _params[3*i+3] != 0 and not np.isnan(_params[3*i+3]) and not np.isinf(_params[3*i+3]))
    except:
        g = 1E9 * exp( -0.5*((_x - 0) / 1)**2)
        print(g)

    return sum(g, axis=1) + _params[1]
#-- END OF SUB-ROUTINE____________________________________________________________#

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def f_gaussian_model(_x, gfit_results, ngauss):
    #_bg0 : gfit_results[1]
    try:
        g = ((gfit_results[3*i+4] * exp( -0.5*((_x - gfit_results[3*i+2]) / gfit_results[3*i+3])**2)) \
            for i in range(0, ngauss) \
            if gfit_results[3*i+3] != 0 and not np.isnan(gfit_results[3*i+3]) and not np.isinf(gfit_results[3*i+3]))
    except:
        g = 1E9 * exp( -0.5*((_x - 0) / 1)**2)
        print(g)

    return sum(g, axis=1) + gfit_results[1]
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def multi_gaussian_model_d_new(_x, _params, ngauss): # _x: global array, params: cube

    _gparam = _params[2:].reshape(ngauss, 3).T
    #_bg0 : _params[1]
    return (_gparam[2].reshape(ngauss, 1)*exp(-0.5*((_x-_gparam[0].reshape(ngauss, 1)) / _gparam[1].reshape(ngauss, 1))**2)).sum(axis=0) + _params[1]
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def multi_gaussian_model_d_classic(_x, _params, ngauss): # params: cube
    _bg0 = _params[1]
    _y = np.zeros_like(_x, dtype=np.float32)
    for i in range(0, ngauss):
        _x0 = _params[3*i+2]
        _std0 = _params[3*i+3]
        _p0 = _params[3*i+4]

        _y += _p0 * exp( -0.5*((_x - _x0) / _std0)**2)
        #y += _p0 * (scipy.stats.norm.pdf(_x, loc=_x0, scale=_std0))
    _y += _bg0
    return _y
#-- END OF SUB-ROUTINE____________________________________________________________#



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

    # sigma
    _sigma0 = args[2][0]
    _sigma1 = args[2][2+3*args[1]] # args[1]=ngauss
    # bg
    _bg0 = args[2][1]
    _bg1 = args[2][3+3*args[1]] # args[1]=ngauss

    # partial[2:] copy cube to params_t --> x, std, p ....
    #params_t = args[0][2:].reshape(args[1], 3).T
    params_t = args[0][2:].reshape(args[1], 3).T

    _xn_0 = np.zeros(args[1])
    _xn_1 = np.zeros(args[1])
    _stdn_0 = np.zeros(args[1])
    _stdn_1 = np.zeros(args[1])
    _pn_0 = np.zeros(args[1])
    _pn_1 = np.zeros(args[1])

    for i in range(0, args[1]):
        _xn_0[i] = args[2][2+3*i]
        _xn_1[i] = args[2][4+3*args[1]+3*i]

        _stdn_0[i] = args[2][3+3*i]
        _stdn_1[i] = args[2][5+3*args[1]+3*i]

        _pn_0[i] = args[2][4+3*i]
        _pn_1[i] = args[2][6+3*args[1]+3*i]


    # vectorization
    # sigma and bg
    args[0][0] = _sigma0 + args[0][0]*(_sigma1 - _sigma0)   # sigma: uniform prior between 0:1
    args[0][1] = _bg0 + args[0][1]*(_bg1 - _bg0)            # bg: uniform prior betwargs[1]een 0:1

    # n-gaussians
    # x
    params_t[0] = (_xn_0 + params_t[0].T*(_xn_1 - _xn_0)).T
    # std
    params_t[1] = (_stdn_0 + params_t[1].T*(_stdn_1 - _stdn_0)).T
    # p
    params_t[2] = (_pn_0 + params_t[2].T*(_pn_1 - _pn_0)).T

    params_t_conc = np.hstack((params_t[0].reshape(args[1], 1), params_t[1].reshape(args[1], 1), params_t[2].reshape(args[1], 1))).reshape(1, 3*args[1])
    #print(params_t_conc)
    args[0][2:] = params_t_conc

    #print(args[0])
    #del(_bg0, _bg1, _x0, _x1, _std0, _std1, _p0, _p1, _sigma0, _sigma1, params_t, params_t_conc)
    return args[0]
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
    # _x0
    _x0 = args[2][2]
    _x1 = args[2][7]
    # _std0
    _std0 = args[2][3]
    _std1 = args[2][8]
    # _p0
    _p0 = args[2][4]
    _p1 = args[2][9]

    # sigma
    #_sigma0 = 0
    #_sigma1 = 0.03 
    ## bg
    #_bg0 = -0.02
    #_bg1 = 0.02
    ## _x0
    #_x0 = 0
    #_x1 = 0.8
    ## _std0
    #_std0 = 0.0
    #_std1 = 0.5
    ## _p0
    #_p0 = 0.0
    #_p1 = 0.5

    # partial[2:] copy cube to params_t --> x, std, p ....
    #params_t = args[0][2:].reshape(args[1], 3).T
    params_t = args[0][2:].reshape(args[1], 3).T

    # vectorization
    # sigma and bg
    args[0][0] = _sigma0 + args[0][0]*(_sigma1 - _sigma0)            # bg: uniform prior between 0:1
    args[0][1] = _bg0 + args[0][1]*(_bg1 - _bg0)            # bg: uniform prior betwargs[1]een 0:1

    # n-gaussians
    # x
    params_t[0] = (_x0 + params_t[0].T*(_x1 - _x0)).T
    #params_t[0] = _x0 + params_t[0]*(_x1 - _x0)
    #print(params_t[0])
    # std
    params_t[1] = (_std0 + params_t[1].T*(_std1 - _std0)).T
    #params_t[1] = _std0 + params_t[1]*(_std1 - _std0)
    # p
    params_t[2] = (_p0 + params_t[2].T*(_p1 - _p0)).T
    #params_t[2] = _p0 + params_t[2]*(_p1 - _p0)

    #print(params_t[1].reshape(args[1], 1))

    params_t_conc = np.hstack((params_t[0].reshape(args[1], 1), params_t[1].reshape(args[1], 1), params_t[2].reshape(args[1], 1))).reshape(1, 3*args[1])
    #params_t_conc1 = np.concatenate((params_t[0], params_t[1], params_t[2]), axis=0)
    #print(params_t_conc)
    args[0][2:] = params_t_conc

    #print(args[0])
    #del(_bg0, _bg1, _x0, _x1, _std0, _std1, _p0, _p1, _sigma0, _sigma1, params_t, params_t_conc)
    return args[0]
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# parameters are sigma, bg, _x01, _std01, _p01, _x02, _std02, _p02...
def uniform_prior_d_pre(*args):

    # args[0][0] : sigma
    # args[0][1] : bg0
    # args[0][2] : x0
    # args[0][3] : std0
    # args[0][4] : p0
    # ...
    # args[1] : ngauss
    # args[2][0] : _sigma0
    # args[2][1] : _sigma1
    # args[2][2] : _bg0
    # args[2][3] : _bg1
    # args[2][4] : _x0
    # args[2][5] : _x1
    # args[2][6] : _std0
    # args[2][7] : _std1
    # args[2][8] : _p0
    # args[2][9] : _p1

    # sigma
    _sigma0 = args[2][0]
    _sigma1 = args[2][5]
    # bg
    _bg0 = args[2][1]
    _bg1 = args[2][6]
    # _x0
    _x0 = args[2][2]
    _x1 = args[2][7]
    # _std0
    _std0 = args[2][3]
    _std1 = args[2][8]
    # _p0
    _p0 = args[2][4]
    _p1 = args[2][9]

    # sigma
    #_sigma0 = 0
    #_sigma1 = 0.03 
    ## bg
    #_bg0 = -0.02
    #_bg1 = 0.02
    ## _x0
    #_x0 = 0
    #_x1 = 0.8
    ## _std0
    #_std0 = 0.0
    #_std1 = 0.5
    ## _p0
    #_p0 = 0.0
    #_p1 = 0.5

    # partial[2:] copy cube to params_t --> x, std, p ....
    #params_t = args[0][2:].reshape(args[1], 3).T
    params_t = args[0][2:].reshape(args[1], 3).T

    # vectorization
    # sigma and bg
    args[0][0] = _sigma0 + args[0][0]*(_sigma1 - _sigma0)            # bg: uniform prior between 0:1
    args[0][1] = _bg0 + args[0][1]*(_bg1 - _bg0)            # bg: uniform prior betwargs[1]een 0:1

    # n-gaussians
    # x
    params_t[0] = _x0 + params_t[0]*(_x1 - _x0)
    # std
    params_t[1] = _std0 + params_t[1]*(_std1 - _std0)
    # p
    params_t[2] = _p0 + params_t[2]*(_p1 - _p0)

    #print(params_t[1].reshape(args[1], 1))

    params_t_conc = np.hstack((params_t[0].reshape(args[1], 1), params_t[1].reshape(args[1], 1), params_t[2].reshape(args[1], 1))).reshape(1, 3*args[1])
    #params_t_conc1 = np.concatenate((params_t[0], params_t[1], params_t[2]), axis=0)
    #print(params_t_conc)
    args[0][2:] = params_t_conc

    #print(args[0])
    #del(_bg0, _bg1, _x0, _x1, _std0, _std1, _p0, _p1, _sigma0, _sigma1, params_t, params_t_conc)
    return args[0]
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def loglike_d(*args):
    # args[0] : params
    # args[1] : _spect : input velocity profile array [N channels] <-- normalized (F-f_max)/(f_max-f_min)
    # args[2] : _x
    # args[3] : ngauss
    # _bg, _x0, _std, _p0, .... = params[1], params[2], params[3], params[4]
    # sigma = params[0] # loglikelihoood sigma
    #print(args[1])

    npoints = args[2].size
    sigma = args[0][0] # loglikelihoood sigma

    gfit = multi_gaussian_model_d(args[2], args[0], args[3])

    log_n_sigma = -0.5*npoints*log(2.0*pi) - 1.0*npoints*log(sigma)
    chi2 = sum((-1.0 / (2*sigma**2)) * ((gfit - args[1])**2))

    return log_n_sigma + chi2


#-- END OF SUB-ROUTINE____________________________________________________________#



