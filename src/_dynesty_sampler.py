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


from re import A, I
import sys
import numpy as np
from numpy import sum, exp, log, pi
from numpy import linalg, array, sum, log, exp, pi, std, diag, concatenate

import numba
import matplotlib.pyplot as plt

import dynesty
from dynesty import NestedSampler
from dynesty import DynamicNestedSampler
from dynesty import plotting as dyplot
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

import gc
import ray
import multiprocessing as mp

import argparse
from _baygaud_params import read_configfile










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
    gfit_priors_init = np.zeros(2*5, dtype=np.float32)
    _x_boundaries = np.full(2*ngauss, fill_value=-1E11, dtype=np.float32)
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
    
                if(ndim * (ndim + 1) // 2 > _params['nlive']):
                    _params['nlive'] = 1 + ndim * (ndim + 1) // 2 # optimal minimum nlive
    


                if _params['_dynesty_class_'] == 'static':
                    sampler = NestedSampler(loglike_d, optimal_prior, ndim,
                        nlive=_params['nlive'],
                        update_interval=_params['update_interval'],
                        sample=_params['sample'],
                        bound=_params['bound'],
                        facc=_params['facc'],
                        fmove=_params['fmove'],
                        max_move=_params['max_move'],
                        logl_args=[(_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])

                    sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=True)

                elif _params['_dynesty_class_'] == 'dynamic':
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

                _gfit_results_temp, _logz = get_dynesty_sampler_results(sampler)
    
    
                _x_boundaries[0:ngauss] = _gfit_results_temp[2:nparams:3] - 5*_gfit_results_temp[3:nparams:3] # x - 3*std
    
                _x_boundaries[ngauss:2*ngauss] = _gfit_results_temp[2:nparams:3] + 5*_gfit_results_temp[3:nparams:3] # x + 3*std
    
                _x_boundaries_ft = np.delete(_x_boundaries, np.where(_x_boundaries < -1E3))
                _x_lower = np.sort(_x_boundaries_ft)[0]
                _x_upper = np.sort(_x_boundaries_ft)[-1]
                _x_lower = _x_lower if _x_lower > 0 else 0
                _x_upper = _x_upper if _x_upper < 1 else 1
    
                _f_ngfit = f_gaussian_model(_x, _gfit_results_temp, ngauss)
                _res_spect = ((_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min) - _f_ngfit)
                _index_t = np.argwhere((_x < _x_lower) | (_x > _x_upper))
                _res_spect_ft = np.delete(_res_spect, _index_t)
    
                _rms[k] = np.std(_res_spect_ft)*(_f_max - _f_min)
                _bg[k] = _gfit_results_temp[1]*(_f_max - _f_min) + _f_min # bg
                print(i, j, _rms[k], _bg[k])
                k += 1

    zero_to_nan_rms = np.where(_rms == 0.0, np.nan, _rms)
    zero_to_nan_bg = np.where(_bg == 0.0, np.nan, _bg)

    _rms_med = np.nanmedian(zero_to_nan_rms)
    _bg_med = np.nanmedian(zero_to_nan_bg)
    _params['_rms_med'] = _rms_med
    _params['_bg_med'] = _bg_med
    print("rms_med:_", _rms_med)
    print("bg_med:_", _bg_med)


def little_derive_rms_npoints(_inputDataCube, i, j, _x, _f_min, _f_max, ngauss, _gfit_results_temp):

    ndim = 3*ngauss + 2
    nparams = ndim

    _x_boundaries = np.full(2*ngauss, fill_value=-1E11, dtype=np.float32)
    _x_boundaries[0:ngauss] = _gfit_results_temp[2:nparams:3] - 5*_gfit_results_temp[3:nparams:3] # x - 3*std

    _x_boundaries[ngauss:2*ngauss] = _gfit_results_temp[2:nparams:3] + 5*_gfit_results_temp[3:nparams:3] # x + 3*std

    _x_boundaries_ft = np.delete(_x_boundaries, np.where(_x_boundaries < -1E3))
    _x_lower = np.sort(_x_boundaries_ft)[0]
    _x_upper = np.sort(_x_boundaries_ft)[-1]
    _x_lower = _x_lower if _x_lower > 0 else 0
    _x_upper = _x_upper if _x_upper < 1 else 1

    _f_ngfit = f_gaussian_model(_x, _gfit_results_temp, ngauss)
    _res_spect = ((_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min) - _f_ngfit)
    _index_t = np.argwhere((_x < _x_lower) | (_x > _x_upper))
    _res_spect_ft = np.delete(_res_spect, _index_t)

    _rms_ngfit = np.std(_res_spect_ft) # normalised


    del(_x_boundaries, _x_boundaries_ft, _index_t, _res_spect_ft)
    gc.collect()

    return _rms_ngfit # resturn normalised _rms




def dynamic_baygaud_nested_sampling(num_cpus_nested_sampling):

    @ray.remote(num_cpus=num_cpus_nested_sampling)
    def baygaud_nested_sampling(_inputDataCube, _x, _peak_sn_map, _sn_int_map, _params, _is, _ie, i, _js, _je, _cube_mask_2d):

        _max_ngauss = _params['max_ngauss']
        _vel_min = _params['vel_min']
        _vel_max = _params['vel_max']
        _cdelt3 = _params['cdelt3']

        gfit_results = np.zeros(((_je-_js), _max_ngauss, 2*(2+3*_max_ngauss) + 7 + _max_ngauss), dtype=np.float32)

        nparams_g1 = 3*1 + 2
        gfit_priors_init_g1 = np.zeros(nparams_g1, dtype=np.float32)




        for j in range(0, _je -_js):

            _f_max = np.max(_inputDataCube[:,j+_js,i]) # peak flux : being used for normalization
            _f_min = np.min(_inputDataCube[:,j+_js,i]) # lowest flux : being used for normalization

            gfit_priors_init = np.zeros(2*5, dtype=np.float32)



            gfit_priors_init = [0.0, 0.0, \
                                0.001, 0.001, 0.001, \
                                0.9, 0.6, \
                                0.999, 0.999, 1.0]

            if _cube_mask_2d[j+_js, i] <= 0 : # if masked, then skip : NOTE THE MASK VALUE SHOULD BE zero or negative.
                print("mask filtered: %d %d | peak S/N: %.1f :: S/N limit: %.1f | integrated S/N: %.1f :: S/N limit: %.1f :: f_min: %e :: f_max: %e" \
                    % (i, j+_js, _peak_sn_map[j+_js, i], _params['peak_sn_limit'], _sn_int_map[j+_js, i], _params['int_sn_limit'], _f_min, _f_max))


                l_range = np.arange(_max_ngauss)

                gfit_results[j, l_range, 2*(3*_max_ngauss+2) + l_range] = _params['_rms_med']

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


            elif _sn_int_map[j+_js, i] < _params['int_sn_limit'] or _peak_sn_map[j+_js, i] < _params['peak_sn_limit'] \
                or np.isnan(_f_max) or np.isnan(_f_min) \
                or np.isinf(_f_min) or np.isinf(_f_min):

                print("low S/N: %d %d | peak S/N: %.1f :: S/N limit: %.1f | integrated S/N: %.1f :: S/N limit: %.1f :: f_min: %e :: f_max: %e" \
                    % (i, j+_js, _peak_sn_map[j+_js, i], _params['peak_sn_limit'], _sn_int_map[j+_js, i], _params['int_sn_limit'], _f_min, _f_max))
                

                l_range = np.arange(_max_ngauss)

                gfit_results[j, l_range, 2*(3*_max_ngauss+2) + l_range] = _params['_rms_med']

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
                ngauss = k+1  # set the number of gaussian
                ndim = 3*ngauss + 2
                nparams = ndim

                if(ndim * (ndim + 1) // 2 > _params['nlive']):
                    _params['nlive'] = 1 + ndim * (ndim + 1) // 2 # optimal minimum nlive

                print("processing: %d %d | peak s/n: %.1f | integrated s/n: %.1f | gauss-%d" % (i, j+_js, _peak_sn_map[j+_js, i], _sn_int_map[j+_js, i], ngauss))



                if _params['_dynesty_class_'] == 'static':
                    rstate = np.random.default_rng(2)

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
                        logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])

                    sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False)


                elif _params['_dynesty_class_'] == 'dynamic':
                    _queue_size = int(_params['num_cpus_nested_sampling'])
                    rstate = np.random.default_rng(2)
                    sampler = DynamicNestedSampler(loglike_d, optimal_prior, ndim,
                        nlive=_params['nlive'],
                        rstate=rstate,
                        sample=_params['sample'],
                        pool=pool,
                        queue_size=_queue_size,
                        bound=_params['bound'],
                        facc=_params['facc'],
                        fmove=_params['fmove'],
                        max_move=_params['max_move'],
                        logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                    sampler.run_nested(dlogz_init=_params['dlogz'], maxiter_init=_params['maxiter'], maxcall_init=_params['maxcall'], print_progress=False)

                _gfit_results_temp, _logz = get_dynesty_sampler_results(sampler)

                gfit_results[j][k][:2*nparams] = _gfit_results_temp

                _rms_ngfit = little_derive_rms_npoints(_inputDataCube, i, j+_js, _x, _f_min, _f_max, ngauss, _gfit_results_temp)

                if ngauss == 1: # check the peak s/n





                    gfit_priors_init_g1 = _gfit_results_temp[:nparams_g1]



                    _bg_sgfit = _gfit_results_temp[1]
                    _p_sgfit = _gfit_results_temp[4] # bg already subtracted
                    _peak_sn_sgfit = _p_sgfit/_rms_ngfit

                    if _peak_sn_sgfit < _params['peak_sn_limit']: 
                        print("skip the rest of Gaussian fits: %d %d | rms:%.5f | bg:%.5f | peak:%.5f | peak_sgfit s/n: %.3f < %.3f" % (i, j+_js, _rms_ngfit, _bg_sgfit, _p_sgfit, _peak_sn_sgfit, _params['peak_sn_limit']))

                        l_indices = np.arange(_max_ngauss)

                        gfit_results[j, l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 1] = _is
                        gfit_results[j, l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 2] = _ie
                        gfit_results[j, l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 3] = _js
                        gfit_results[j, l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 4] = _je
                        gfit_results[j, l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 5] = i
                        gfit_results[j, l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 6] = _js + j

                        gfit_results[j, 0, 2 * (3 * _max_ngauss + 2)] = _rms_ngfit
                        gfit_results[j, 0, 2 * (3 * _max_ngauss + 2) + _max_ngauss] = _logz

                        gfit_results[j, 1:, 2 * (3 * _max_ngauss + 2) + l_indices[1:]] = 0  # Adjust for proper indexing
                        gfit_results[j, 1:, 2 * (3 * _max_ngauss + 2) + _max_ngauss] = -1E11






                        gfit_results[j, k, 0] *= (_f_max - _f_min)  # sigma-flux to data cube units
                        gfit_results[j, k, 1] = gfit_results[j, k, 1] * (_f_max - _f_min) + _f_min  # background to data cube units
                        gfit_results[j, k, 6 + 3*k] *= (_f_max - _f_min)  # background error to data cube units

                        m_indices = np.arange(k+1)

                        velocity_indices = 2 + 3*m_indices
                        velocity_dispersion_indices = 3 + 3*m_indices
                        peak_flux_indices = 4 + 3*m_indices
                        velocity_e_indices = 7 + 3*(m_indices+k)
                        velocity_dispersion_e_indices = 8 + 3*(m_indices+k)
                        flux_e_indices = 9 + 3*(m_indices+k)

                        if _cdelt3 > 0:
                            gfit_results[j, k, velocity_indices] = gfit_results[j, k, velocity_indices] * (_vel_max - _vel_min) + _vel_min
                        else:
                            gfit_results[j, k, velocity_indices] = gfit_results[j, k, velocity_indices] * (_vel_min - _vel_max) + _vel_max

                        gfit_results[j, k, velocity_dispersion_indices] *= (_vel_max - _vel_min)
                        gfit_results[j, k, peak_flux_indices] *= (_f_max - _f_min)

                        gfit_results[j, k, velocity_e_indices] *= (_vel_max - _vel_min)
                        gfit_results[j, k, velocity_dispersion_e_indices] *= (_vel_max - _vel_min)
                        gfit_results[j, k, flux_e_indices] *= (_f_max - _f_min)

                        gfit_results[j, k, 2*(3*_max_ngauss+2)+k] *= (_f_max - _f_min)

                        break





                if ngauss < _max_ngauss: # update gfit_priors_init with the g1fit results for the rest of the gaussians of the current profile
                    nparams_n = 3*(ngauss+1) + 2 # <-- ( + 1)
                    gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)

                    g1fit_x = gfit_priors_init_g1[2]
                    g1fit_std = gfit_priors_init_g1[3]
                    g1fit_p = gfit_priors_init_g1[4]

                    gfit_priors_init[:nparams_n] = 0.01

                    gfit_priors_init[2:nparams_n:3] = g1fit_x - g1fit_std * _params['x_prior_lowerbound_factor']
                    gfit_priors_init[3:nparams_n:3] = _params['std_prior_lowerbound_factor'] * g1fit_std
                    gfit_priors_init[3:nparams_n:3] = np.where( (gfit_priors_init[3:nparams_n:3]*(_vel_max - _vel_min) < (_cdelt3/1000.)), (_cdelt3/1000.)/(_vel_max - _vel_min), gfit_priors_init[3:nparams_n:3])
                    gfit_priors_init[4:nparams_n:3] = _params['p_prior_lowerbound_factor'] * g1fit_p


                    gfit_priors_init[nparams_n:2*nparams_n] = 0.99

                    gfit_priors_init[nparams_n+2:2*nparams_n:3] = g1fit_x + g1fit_std * _params['x_prior_upperbound_factor']
                    gfit_priors_init[nparams_n+3:2*nparams_n:3] = _params['std_prior_upperbound_factor'] * g1fit_std
                    gfit_priors_init[nparams_n+4:2*nparams_n:3] = _params['p_prior_upperbound_factor'] * g1fit_p

                    gfit_priors_init = np.where(gfit_priors_init<0, 0.01, gfit_priors_init)
                    gfit_priors_init = np.where(gfit_priors_init>1, 0.99, gfit_priors_init)






                gfit_results[j][k][2*(3*_max_ngauss+2)+k] = _rms_ngfit # rms_(k+1)gfit
                gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+0] = _logz
                gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
                gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
                gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
                gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
                gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+5] = i
                gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+6] = _js + j









                gfit_results[j][k][0] = gfit_results[j][k][0]*(_f_max - _f_min) # sigma-flux
                gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
                gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e
                _bg_flux = gfit_results[j][k][1]

                velocity_indices = 2 + 3*np.arange(k+1)
                velocity_dispersion_indices = 3 + 3*np.arange(k+1)
                peak_flux_indices = 4 + 3*np.arange(k+1)

                velocity_errors_indices = 7 + 3*np.arange(k+1) + 3*k  # Adjusted for the layout of your results array
                velocity_dispersion_errors_indices = 8 + 3*np.arange(k+1) + 3*k
                flux_errors_indices = 9 + 3*np.arange(k+1) + 3*k

                if _cdelt3 > 0:  # Velocity axis with increasing order
                    gfit_results[j, k, velocity_indices] = gfit_results[j, k, velocity_indices] * (_vel_max - _vel_min) + _vel_min
                else:  # Velocity axis with decreasing order
                    gfit_results[j, k, velocity_indices] = gfit_results[j, k, velocity_indices] * (_vel_min - _vel_max) + _vel_max

                gfit_results[j, k, velocity_dispersion_indices] *= (_vel_max - _vel_min)

                gfit_results[j, k, peak_flux_indices] *= (_f_max - _f_min)
                gfit_results[j, k, velocity_errors_indices] *= (_vel_max - _vel_min)
                gfit_results[j, k, velocity_dispersion_errors_indices] *= (_vel_max - _vel_min)
                gfit_results[j, k, flux_errors_indices] *= (_f_max - _f_min)

                gfit_results[j, k, 2*(3*_max_ngauss+2) + k] *= (_f_max - _f_min)






        return gfit_results
    return baygaud_nested_sampling

    
    

@ray.remote(num_cpus=1)
def run_dynesty_sampler_optimal_priors_org(_inputDataCube, _x, _peak_sn_map, _sn_int_map, _params, _is, _ie, i, _js, _je):

    _max_ngauss = _params['max_ngauss']
    _vel_min = _params['vel_min']
    _vel_max = _params['vel_max']
    gfit_results = np.zeros(((_je-_js), _max_ngauss, 2*(2+3*_max_ngauss) + 7 + _max_ngauss), dtype=np.float32)
    _x_boundaries = np.full(2*_max_ngauss, fill_value=-1E11, dtype=np.float32)

    for j in range(0, _je -_js):

        _f_max = np.max(_inputDataCube[:,j+_js,i]) # peak flux : being used for normalization
        _f_min = np.min(_inputDataCube[:,j+_js,i]) # lowest flux : being used for normalization

        gfit_priors_init = np.zeros(2*5, dtype=np.float32)
        gfit_priors_init = [0.0, 0.0, 0.01, 0.01, 0.01, 0.5, 0.6, 0.99, 0.6, 1.01]

        if _sn_int_map[j+_js, i] < _params['int_sn_limit'] or _peak_sn_map[j+_js, i] < _params['peak_sn_limit'] \
            or np.isnan(_f_max) or np.isnan(_f_min) \
            or np.isinf(_f_min) or np.isinf(_f_min):

            print("low S/N: %d %d | peak S/N: %.1f :: S/N limit: %.1f | integrated S/N: %.1f :: S/N limit: %.1f :: f_min: %e :: f_max: %e" \
                % (i, j+_js, _peak_sn_map[j+_js, i], _params['peak_sn_limit'], _sn_int_map[j+_js, i], _params['int_sn_limit'], _f_min, _f_max))
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



            if _params['_dynesty_class_'] == 'static':
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

            elif _params['_dynesty_class_'] == 'dynamic':
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


            _gfit_results_temp, _logz = get_dynesty_sampler_results(sampler)

            gfit_results[j][k][:2*nparams] = _gfit_results_temp

            _rms_ngfit = little_derive_rms_npoints(_inputDataCube, i, j+_js, _x, _f_min, _f_max, ngauss, _gfit_results_temp)

            if ngauss == 1: # check the peak s/n

                nparams_n = 3*(ngauss+1) + 2
                gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)
                gfit_priors_init[:nparams] = _gfit_results_temp[:nparams] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
                gfit_priors_init[nparams+3:2*nparams+3] = _gfit_results_temp[:nparams] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]


                _bg_sgfit = _gfit_results_temp[1]
                _p_sgfit = _gfit_results_temp[4] # bg already subtracted
                _peak_sn_sgfit = _p_sgfit/_rms_ngfit

                if _peak_sn_sgfit < _params['peak_sn_limit']: 
                    print("skip the rest of Gaussian fits: %d %d | rms:%.1f | bg:%.1f | peak:%.1f | peak_sgfit s/n: %.1f < %.1f" % (i, j+_js, _rms_ngfit, _bg_sgfit, _p_sgfit, _peak_sn_sgfit, _params['peak_sn_limit']))

                    for l in range(0, _max_ngauss):
                        if l == 0:
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

                    gfit_results[j][k][0] = gfit_results[j][k][0]*(_f_max - _f_min) # sigma-flux
                    gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
                    gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e
                    _bg_flux = gfit_results[j][k][1]
        
                    for m in range(0, k+1):
                        gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min # velocity
                        gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min) # velocity-dispersion
                        gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) # flux
        
                        gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-e
                        gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-dispersion-e
                        gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # flux-e

                    gfit_results[j][k][2*(3*_max_ngauss+2)+k] = gfit_results[j][k][2*(3*_max_ngauss+2)+k]*(_f_max - _f_min) # rms-(k+1)gfit
                    continue


            if ngauss < _max_ngauss:
                nparams_n = 3*(ngauss+1) + 2
                gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)
                gfit_priors_init[:nparams] = _gfit_results_temp[:nparams] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
                gfit_priors_init[nparams+3:2*nparams+3] = _gfit_results_temp[:nparams] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
    
                _x_min_t = _gfit_results_temp[2:nparams:3].min()
                _x_max_t = _gfit_results_temp[2:nparams:3].max()
                _std_min_t = _gfit_results_temp[3:nparams:3].min()
                _std_max_t = _gfit_results_temp[3:nparams:3].max()
                _p_min_t = _gfit_results_temp[4:nparams:3].min()
                _p_max_t = _gfit_results_temp[4:nparams:3].max()

                gfit_priors_init[0] = _params['sigma_prior_lowerbound_factor']*_gfit_results_temp[0]
                gfit_priors_init[nparams_n] = _params['sigma_prior_upperbound_factor']*_gfit_results_temp[0]

                gfit_priors_init[1] = _params['bg_prior_lowerbound_factor']*_gfit_results_temp[1]
                gfit_priors_init[nparams_n + 1] = _params['bg_prior_upperbound_factor']*_gfit_results_temp[1]


                if ngauss == 1:
                    gfit_priors_init[nparams] = _params['x_lowerbound_gfit']
                    gfit_priors_init[2*nparams+3] = _params['x_upperbound_gfit']
                else:
                    gfit_priors_init[nparams] = _x_min_t - _params['x_prior_lowerbound_factor']*_std_max_t
                    gfit_priors_init[2*nparams+3] = _x_max_t + _params['x_prior_upperbound_factor']*_std_max_t

                gfit_priors_init[nparams+1] = _params['std_prior_lowerbound_factor']*_std_min_t
                gfit_priors_init[2*nparams+4] = _params['std_prior_upperbound_factor']*_std_max_t
    
                gfit_priors_init[nparams+2] = _params['p_prior_lowerbound_factor']*_p_max_t # 5% of the maxium flux
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






            gfit_results[j][k][0] = gfit_results[j][k][0]*(_f_max - _f_min) # sigma-flux
            gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
            gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e
            _bg_flux = gfit_results[j][k][1]

            for m in range(0, k+1):
                gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min # velocity
                gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min) # velocity-dispersion
                gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min)  # peak flux

                gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-e
                gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-dispersion-e
                gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # peak flux-e

            gfit_results[j][k][2*(3*_max_ngauss+2)+k] = gfit_results[j][k][2*(3*_max_ngauss+2)+k]*(_f_max - _f_min) # rms-(k+1)gfit


    return gfit_results

    
    

@ray.remote(num_cpus=1)
def run_dynesty_sampler_uniform_priors(_x, _inputDataCube, _is, _ie, i, _js, _je, _max_ngauss, _vel_min, _vel_max):

    gfit_results = np.zeros(((_je-_js), _max_ngauss, 2*(2+3*_max_ngauss)+7), dtype=np.float32)
    for j in range(0, _je -_js):

        _f_max = np.max(_inputDataCube[:,j+_js,i]) # peak flux : being used for normalization
        _f_min = np.min(_inputDataCube[:,j+_js,i]) # lowest flux : being used for normalization

        gfit_priors_init = np.zeros(2*5, dtype=np.float32)
        gfit_priors_init = [0.0, 0.0, 0.01, 0.01, 0.01, 0.5, 0.5, 0.9, 0.6, 1.01]
        for k in range(0, _max_ngauss):
            ngauss = k+1  # set the number of gaussian
            ndim = 3*ngauss + 2
            nparams = ndim

    
            print("processing: %d %d gauss-%d" % (i, j+_js, ngauss))



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





            gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
            gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e

            for m in range(0, k+1):
                gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min # velocity
                gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min) # velocity-dispersion
                gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) + _f_min # flux

                gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-e
                gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-dispersion-e
                gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # flux-e

    
    del(ndim, nparams, ngauss, sampler)
    gc.collect()

    return gfit_results

    
    



def get_dynesty_sampler_results(_sampler):
    samples = _sampler.results.samples  # samples
    weights = exp(_sampler.results.logwt - _sampler.results.logz[-1])  # normalized weights


    quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
                for samps in samples.T]
    
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    bestfit_results = _sampler.results.samples[-1, :]
    log_Z = _sampler.results.logz[-1]


    

    del(samples, weights, quantiles)
    gc.collect()

    return concatenate((bestfit_results, diag(cov)**0.5)), log_Z



def multi_gaussian_model_d(_x, _params, ngauss): # params: cube
    try:
        g = ((_params[3*i+4] * exp( -0.5*((_x - _params[3*i+2]) / _params[3*i+3])**2)) \
            for i in range(0, ngauss) \
            if _params[3*i+3] != 0 and not np.isnan(_params[3*i+3]) and not np.isinf(_params[3*i+3]))
    except:
        g = 1E9 * exp( -0.5*((_x - 0) / 1)**2)
        print(g)

    return sum(g, axis=1) + _params[1]

def f_gaussian_model(_x, gfit_results, ngauss):
    try:
        g = ((gfit_results[3*i+4] * exp( -0.5*((_x - gfit_results[3*i+2]) / gfit_results[3*i+3])**2)) \
            for i in range(0, ngauss) \
            if gfit_results[3*i+3] != 0 and not np.isnan(gfit_results[3*i+3]) and not np.isinf(gfit_results[3*i+3]))
    except:
        g = 1E9 * exp( -0.5*((_x - 0) / 1)**2)
        print(g)

    return sum(g, axis=1) + gfit_results[1]


def multi_gaussian_model_d_new(_x, _params, ngauss): # _x: global array, params: cube

    _gparam = _params[2:].reshape(ngauss, 3).T
    return (_gparam[2].reshape(ngauss, 1)*exp(-0.5*((_x-_gparam[0].reshape(ngauss, 1)) / _gparam[1].reshape(ngauss, 1))**2)).sum(axis=0) + _params[1]


def multi_gaussian_model_d_classic(_x, _params, ngauss): # params: cube
    _bg0 = _params[1]
    _y = np.zeros_like(_x, dtype=np.float32)
    for i in range(0, ngauss):
        _x0 = _params[3*i+2]
        _std0 = _params[3*i+3]
        _p0 = _params[3*i+4]

        _y += _p0 * exp( -0.5*((_x - _x0) / _std0)**2)
    _y += _bg0
    return _y



def optimal_prior(*args):


    nparams = 3*args[1] + 2

    _sigma0 = np.array(args[2][0])
    _sigma1 = np.array(args[2][2+3*args[1]]) # args[1]=ngauss

    _bg0 = np.array(args[2][1])
    _bg1 = np.array(args[2][3+3*args[1]]) # args[1]=ngauss

    _xn_0 = np.array(args[2][2:nparams:3])
    _xn_1 = np.array(args[2][nparams+2:2*nparams:3])

    _stdn_0 = np.array(args[2][3:nparams:3])
    _stdn_1 = np.array(args[2][nparams+3:2*nparams:3])

    _pn_0 = np.array(args[2][4:nparams:3])
    _pn_1 = np.array(args[2][nparams+4:2*nparams:3])

    args[0][0] = _sigma0 + args[0][0]*(_sigma1 - _sigma0)   # sigma: uniform prior between 0:1
    args[0][1] = _bg0 + args[0][1]*(_bg1 - _bg0)            # bg: uniform prior betwargs[1]een 0:1

    args[0][2:nparams:3] = _xn_0 +   args[0][2:nparams:3]*(_xn_1 - _xn_0)            # bg: uniform prior betwargs[1]een 0:1
    args[0][3:nparams:3] = _stdn_0 + args[0][3:nparams:3]*(_stdn_1 - _stdn_0)            # bg: uniform prior betwargs[1]een 0:1
    args[0][4:nparams:3] = _pn_0 + args[0][4:nparams:3]*(_pn_1 - _pn_0)            # bg: uniform prior betwargs[1]een 0:1

    return args[0]











    _sigma0 = np.array(args[2][0])
    _sigma1 = np.array(args[2][2+3*args[1]]) # args[1]=ngauss
    _bg0 = np.array(args[2][1])
    _bg1 = np.array(args[2][3+3*args[1]]) # args[1]=ngauss


    _xn_0 = np.zeros(args[1])
    _xn_1 = np.zeros(args[1])
    _stdn_0 = np.zeros(args[1])
    _stdn_1 = np.zeros(args[1])
    _pn_0 = np.zeros(args[1])
    _pn_1 = np.zeros(args[1])


    nparams = 3*args[1] + 2

    _xn_0 = np.array(args[2][2:nparams:3])
    _xn_1 = np.array(args[2][nparams+2:2*nparams:3])

    _stdn_0 = np.array(args[2][3:nparams:3])
    _stdn_1 = np.array(args[2][nparams+3:2*nparams:3])

    _pn_0 = np.array(args[2][4:nparams:3])
    _pn_1 = np.array(args[2][nparams+4:2*nparams:3])


    args[0][0] = _sigma0 + args[0][0]*(_sigma1 - _sigma0)   # sigma: uniform prior between 0:1
    args[0][1] = _bg0 + args[0][1]*(_bg1 - _bg0)            # bg: uniform prior betwargs[1]een 0:1

    args[0][2:nparams:3] = _xn_0 +   args[0][2:nparams:3]*(_xn_1 - _xn_0)            # bg: uniform prior betwargs[1]een 0:1
    args[0][3:nparams:3] = _stdn_0 + args[0][3:nparams:3]*(_stdn_1 - _stdn_0)            # bg: uniform prior betwargs[1]een 0:1
    args[0][4:nparams:3] = _pn_0 + args[0][4:nparams:3]*(_pn_1 - _pn_0)            # bg: uniform prior betwargs[1]een 0:1

    return args[0]





def uniform_prior(*args):


    _sigma0 = args[2][0]
    _sigma1 = args[2][5]
    _bg0 = args[2][1]
    _bg1 = args[2][6]
    _x0 = args[2][2]
    _x1 = args[2][7]
    _std0 = args[2][3]
    _std1 = args[2][8]
    _p0 = args[2][4]
    _p1 = args[2][9]


    params_t = args[0][2:].reshape(args[1], 3).T

    args[0][0] = _sigma0 + args[0][0]*(_sigma1 - _sigma0)            # bg: uniform prior between 0:1
    args[0][1] = _bg0 + args[0][1]*(_bg1 - _bg0)            # bg: uniform prior betwargs[1]een 0:1

    params_t[0] = (_x0 + params_t[0].T*(_x1 - _x0)).T
    params_t[1] = (_std0 + params_t[1].T*(_std1 - _std0)).T
    params_t[2] = (_p0 + params_t[2].T*(_p1 - _p0)).T


    params_t_conc = np.hstack((params_t[0].reshape(args[1], 1), params_t[1].reshape(args[1], 1), params_t[2].reshape(args[1], 1))).reshape(1, 3*args[1])
    args[0][2:] = params_t_conc

    return args[0]



def uniform_prior_d_pre(*args):


    _sigma0 = args[2][0]
    _sigma1 = args[2][5]
    _bg0 = args[2][1]
    _bg1 = args[2][6]
    _x0 = args[2][2]
    _x1 = args[2][7]
    _std0 = args[2][3]
    _std1 = args[2][8]
    _p0 = args[2][4]
    _p1 = args[2][9]


    params_t = args[0][2:].reshape(args[1], 3).T

    args[0][0] = _sigma0 + args[0][0]*(_sigma1 - _sigma0)            # bg: uniform prior between 0:1
    args[0][1] = _bg0 + args[0][1]*(_bg1 - _bg0)            # bg: uniform prior betwargs[1]een 0:1

    params_t[0] = _x0 + params_t[0]*(_x1 - _x0)
    params_t[1] = _std0 + params_t[1]*(_std1 - _std0)
    params_t[2] = _p0 + params_t[2]*(_p1 - _p0)


    params_t_conc = np.hstack((params_t[0].reshape(args[1], 1), params_t[1].reshape(args[1], 1), params_t[2].reshape(args[1], 1))).reshape(1, 3*args[1])
    args[0][2:] = params_t_conc

    return args[0]



def loglike_d(*args):

    npoints = args[2].size
    sigma = args[0][0] # loglikelihoood sigma

    gfit = multi_gaussian_model_d(args[2], args[0], args[3])


    log_n_sigma = -0.5*npoints*np.log(2.0*np.pi) - 1.0*npoints*np.log(sigma)
    chi2 = np.nansum((-1.0 / (2*sigma**2)) * ((gfit - args[1])**2))
    return log_n_sigma + chi2







