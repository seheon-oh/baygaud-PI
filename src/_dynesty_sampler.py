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


# Numba 스레딩 레이어/스레드수 고정 (컴파일 전에)
try:
    config.THREADING_LAYER = 'workqueue'
    set_num_threads(1)
except Exception:
    pass

_NUMBA_WARMED = False  # 워커별 1회 JIT 워밍업 플래그
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
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

import gc
import ray
import multiprocessing as mp

from _baygaud_params import read_configfile

from _set_init_priors import find_gaussian_seeds_matched_filter_norm, \
                            set_sgfit_bounds_from_matched_filter_seeds_norm, \
                            print_priors_both, \
                            print_gaussian_seeds_matched_filter_out, \
                            pin_threads_single, \
                            set_init_priors_multiple_gaussians, \
                            prev_fit_from_results_slice




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

                # run dynesty 2.0.3
                #sampler = NestedSampler(loglike_d, optimal_prior, ndim,
                #    nlive=_params['nlive'],
                #    sample=_params['sample'],
                #    bound=_params['bound'],
                #    facc=_params['facc'],
                #    fmove=_params['fmove'],
                #    max_move=_params['max_move'],
                #    logl_args=[(_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                #sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False)

                # run dynesty 2.1.15
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

                #---------------------------------------------------------
                # lower bounds : x1-3*std1, x2-3*std2, ...  | x:_gfit_results_temp[2, 5, 8, ...], std:_gfit_results_temp[3, 6, 9, ...]
                #_x_boundaries[0:ngauss] = _gfit_results_temp[2:nparams:3] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[3:nparams:3] # x - 3*std
                _x_boundaries[0:ngauss] = _gfit_results_temp[2:nparams:3] - 5*_gfit_results_temp[3:nparams:3] # x - 3*std
                #print("g:", ngauss, "upper bounds:", _x_boundaries)
    
                _x_boundaries[ngauss:2*ngauss] = _gfit_results_temp[2:nparams:3] + 5*_gfit_results_temp[3:nparams:3] # x + 3*std

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
                _index_t = np.argwhere((_x < _x_lower) | (_x > _x_upper))
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
def little_derive_rms_npoints_org(_inputDataCube, i, j, _x, _f_min, _f_max, ngauss, _gfit_results_temp):

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
    _index_t = np.argwhere((_x < _x_lower) | (_x > _x_upper))
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




@njit(cache=True, fastmath=True)  # parallel 제거
def _gaussian_sum_norm(x, theta, ngauss):
    n = x.size
    model = np.empty(n, dtype=np.float64)
    bg = float(theta[1])
    for t in range(n):
        model[t] = bg
    for m in range(ngauss):
        mu  = float(theta[2 + 3*m])
        sig = float(theta[3 + 3*m])
        amp = float(theta[4 + 3*m])
        invs = 1.0 / sig
        for t in range(n):
            dx = (x[t] - mu) * invs
            model[t] += amp * np.exp(-0.5 * dx * dx)
    return model

@njit(cache=True, fastmath=True)  # parallel 제거
def _rms_of_residual(data_norm, model):
    n = data_norm.size
    acc = 0.0
    for t in range(n):
        d = data_norm[t] - model[t]
        acc += d * d
    return np.sqrt(acc / n)

@njit(cache=True, fastmath=True)  # parallel 제거
def _neg_half_chi2(data_norm, model, inv_sigma2):
    n = data_norm.size
    s = 0.0
    for t in range(n):
        r = data_norm[t] - model[t]
        s += r * r
    return -0.5 * s * inv_sigma2


@njit(cache=True, fastmath=True)
def _little_derive_rms_npoints_core(profile, x, f_min, f_max, ngauss, theta):
    """
    profile: 원시 스펙트럼 (cube[:, j, i])
    x: 채널 축 (정규화 동일 축)
    f_min, f_max: 정규화에 사용된 min/max
    ngauss: 사용 가우시안 개수
    theta: dynesty가 반환한 (정규화 스케일) 파라미터 벡터 (길이 >= 3*ngauss+2)
    """
    scale = (f_max - f_min)
    data_norm = (profile - f_min) / scale
    model = _gaussian_sum_norm(x, theta, ngauss)
    return _rms_of_residual(data_norm, model)


def little_derive_rms_npoints(input_cube, i, j, x, f_min, f_max, ngauss, theta):
    """
    기존 코드가 호출하던 시그니처 유지. 내부는 JIT 코어 호출.
    theta는 정규화 스케일 파라미터(단위변환 전)를 넣어야 합니다.
    """
    # C-contiguous 보장(필요 시)
    prof = np.ascontiguousarray(input_cube[:, j, i], dtype=np.float64)
    x64  = np.ascontiguousarray(x, dtype=np.float64)
    th64 = np.ascontiguousarray(theta[:(3*ngauss+2)], dtype=np.float64)
    return _little_derive_rms_npoints_core(prof, x64, float(f_min), float(f_max), int(ngauss), th64)



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# _materialize function
def _mat(x):
    # if x is ObjectRef then ray.get it otherwise just return it
    try:
        if isinstance(x, ray.ObjectRef):
            return ray.get(x)
    except Exception:
        pass
    return x




#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# UNIT CONVERSION
def convert_units_norm_to_phys(
    gfit_results: np.ndarray,
    j: int,                     # 프로파일 index (y축 index 등)
    k: int,                     # 다중 가우스 모델 차수-1 (즉, 가우스 개수 = k+1)
    f_min: float, f_max: float, # 해당 프로파일의 플럭스 정규화 기준
    vel_min: float, vel_max: float,  # 물리 속도 범위 (항상 vel_min < vel_max 권장)
    cdelt3: float,              # 채널 축 증감 방향 (음수면 내림차순)
    max_ngauss: int             # 전체 최대 가우스 개수 (_max_ngauss)
) -> None:
    """
    gfit_results[j, k, :]의 정규화 결과를 물리 단위로 *제자리(in-place)* 변환한다.
    - 파라미터 구역: 길이 nparams = 2 + 3*(k+1)
      [model_sigma, bg, g1_x, g1_sigma, g1_peak, g2_x, g2_sigma, g2_peak, ...]
    - 에러 구역: 동일 길이 nparams, 바로 뒤에 이어짐.
      [model_sigma_e, bg_e, g1_x_e, g1_sigma_e, g1_peak_e, ...]
    - 그 뒤 부가정보들이 있으나 여기서는 배경/피크/속도/분산/에러 및 rms만 변환.

    주의: 함수는 gfit_results를 in-place로 수정하며 반환값은 없습니다.
    """
    #________________________________________________________________________________________|
    #|---------------------------------------------------------------------------------------|
    # unit conversion
    # sigma-flux --> data cube units
    gfit_results[j, k, 0] *= (f_max - f_min)  # sigma-flux to data cube units

    # background --> data cube units
    gfit_results[j, k, 1] = gfit_results[j, k, 1] * (f_max - f_min) + f_min  # background to data cube units

    # background error --> data cube units
    gfit_results[j, k, 6 + 3*k] *= (f_max - f_min)  # background error to data cube units

    # vectorization
    m_indices = np.arange(k + 1)

    velocity_indices               = 2 + 3 * m_indices
    velocity_dispersion_indices    = 3 + 3 * m_indices
    peak_flux_indices              = 4 + 3 * m_indices

    velocity_e_indices             = 7 + 3 * (m_indices + k)
    velocity_dispersion_e_indices  = 8 + 3 * (m_indices + k)
    flux_e_indices                 = 9 + 3 * (m_indices + k)

    #________________________________________________________________________________________|
    # UNIT CONVERSION
    #________________________________________________________________________________________|
    # velocity, velocity-dispersion --> km/s
    if cdelt3 > 0:  # if velocity axis is with increasing order
        gfit_results[j, k, velocity_indices] = (
            gfit_results[j, k, velocity_indices] * (vel_max - vel_min) + vel_min  # velocity
        )
    else:  # if velocity axis is with decreasing order
        gfit_results[j, k, velocity_indices] = (
            gfit_results[j, k, velocity_indices] * (vel_min - vel_max) + vel_max  # velocity
        )

    gfit_results[j, k, velocity_dispersion_indices] *= (vel_max - vel_min)  # velocity-dispersion

    #________________________________________________________________________________________|
    # peak flux --> data cube units
    # peak flux --> data cube units : (_f_max - _bg_flux) should be used for scaling as the normalised peak flux is from the bg
    #gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) + _f_min # flux
    gfit_results[j, k, peak_flux_indices] *= (f_max - f_min)  # flux

    #________________________________________________________________________________________|
    # velocity-e, velocity-dispersion-e --> km/s
    gfit_results[j, k, velocity_e_indices]            *= (vel_max - vel_min)  # velocity-e
    gfit_results[j, k, velocity_dispersion_e_indices] *= (vel_max - vel_min)  # velocity-dispersion-e

    # flux-e --> data cube units
    gfit_results[j, k, flux_e_indices] *= (f_max - f_min)  # flux-e

    # lastly put rms 
    # 위치: 2*(3*max_ngauss + 2) + k  (각 k 모델의 rms 저장 슬롯)
    gfit_results[j, k, 2 * (3 * max_ngauss + 2) + k] *= (f_max - f_min)  # rms-(k+1)gfit
#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
    

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# run dynesty for each line profile
def dynamic_baygaud_nested_sampling(num_cpus_nested_sampling, _params):

   # 워커 환경변수(스레드 1 고정). _params에 없으면 기본값 사용
    env_vars = {
        "OMP_NUM_THREADS":        "1",
        "MKL_NUM_THREADS":        "1",
        "OPENBLAS_NUM_THREADS":   "1",
        "BLIS_NUM_THREADS":       "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS":    "1",
        "NUMBA_THREADING_LAYER":  str(_params.get("numba_threading_layer", "workqueue")),
        "NUMBA_NUM_THREADS":      str(_params.get("numba_num_threads", 1)),
    }

    @ray.remote(num_cpus=num_cpus_nested_sampling)
    def baygaud_nested_sampling(
        _inputDataCube_id, _x_id,
        _peak_sn_map_id, _sn_int_map_id,
        _params_id,
        _is, _ie, i, _js, _je,
        _cube_mask_2d_id):

        # 워커 프로세스 내 스레드 고정(Numba 포함) — 프로세스당 1번이면 충분하지만, 재호출해도 무해
        pin_threads_single()

        # materialize first from ray id
        _inputDataCube = _mat(_inputDataCube_id)
        _x             = _mat(_x_id)
        _peak_sn_map   = _mat(_peak_sn_map_id)
        _sn_int_map    = _mat(_sn_int_map_id)
        _params        = _mat(_params_id)
        _cube_mask_2d  = _mat(_cube_mask_2d_id) 
        
        _max_ngauss = _params['max_ngauss']
        _vel_min = _params['vel_min']
        _vel_max = _params['vel_max']
        _cdelt3 = _params['cdelt3']

        gfit_results = np.zeros(((_je-_js), _max_ngauss, 2*(2+3*_max_ngauss) + 7 + _max_ngauss), dtype=np.float32)

        nparams_g1 = 3*1 + 2 # for initial sgfit
        gfit_priors_init_g1 = np.zeros(1*nparams_g1, dtype=np.float32)
        gfit_peak_sn = np.zeros(_max_ngauss, dtype=np.float32)

        #print("CHECK S/N: %d %d | peak S/N: %.1f < %.1f | integrated S/N: %.1f < %.1f" \
        #    % (i, 0+_js, _peak_sn_map[0+_js, i], _params['peak_sn_limit'], _sn_int_map[0+_js, i], _params['int_sn_limit']))


        global _NUMBA_WARMED
        if not _NUMBA_WARMED:
            # 더미 데이터로 각 JIT 함수 1회 컴파일
            x_d = np.linspace(0.0, 1.0, 8, dtype=np.float64)
            prof_d = np.zeros_like(x_d)
            theta_d = np.zeros(3*1+2, dtype=np.float64)  # ngauss=1
            theta_d[1] = 0.0  # bg
            theta_d[2] = 0.5; theta_d[3] = 0.1; theta_d[4] = 1.0
            m = _gaussian_sum_norm(x_d, theta_d, 1)
            _ = _rms_of_residual(prof_d, m)
            _ = _neg_half_chi2(prof_d, m, 1.0)
            _ = _little_derive_rms_npoints_core(prof_d, x_d, 0.0, 1.0, 1, theta_d)
            _NUMBA_WARMED = True


        # 이 스펙트럼의 물리 속도 범위 (예: FITS에서 v_min, v_max를 계산해두었다고 가정)
        v_min_phys = _params['vel_min']   # 사용자 코드에서 정의
        v_max_phys = _params['vel_max']   # 사용자 코드에서 정의

        for j in range(0, _je - _js):
            fl = _inputDataCube[:, j+_js, i]
            _f_min = float(np.min(fl))  # lowest flux : being used for normalization
            _f_max = float(np.max(fl))  # peak flux : being used for normalization
            denom  = (_f_max - _f_min) if (_f_max > _f_min) else 1.0
            f = (fl - _f_min) / denom  # normalization [0,1]

            _gaussian_seeds = find_gaussian_seeds_matched_filter_norm(
                _x, f,                       # _x는 이미 [0,1] 정규화된 속도축
                rms=None, bg=None,
                sigma_list_ch=[1.5,2,3,4,5],
                k_sigma=3.0,
                thres_sigma=1.5,
                amp_sigma_thres=1.5,
                sep_channels=5,
                max_components=None,
                refine_center=True,
                detrend_local=False,
                detrend_halfwin=8, 
                numba_threads=1 # ← 단일-스레드 JIT 경로(오버헤드 최소화)
            )


            # 단일 가우시안 경계(정규화 단위)
            #gfit_priors_init = [sig1, bg1, x1, std1, p1, sig2,bg2, x2, std2, p2]
            _seed_priors_using_matched_filter = set_sgfit_bounds_from_matched_filter_seeds_norm(
                    _gaussian_seeds,
                    use_phys_for_x_bounds=True,
                    v_min=v_min_phys, v_max=v_max_phys,
                    cdelt3=_cdelt3,   # 음수면 자동 반전 앵커
                    # 필요시 팩터 조정
                    model_sigma_bounds=(0.0, 0.5),
                    k_bg=3.0, k_x=5.0,
                    sigma_scale_bounds=(0.1, 3.0),
                    peak_scale_bounds=(0.3, 1.5)
                    )

            gfit_priors_init = _seed_priors_using_matched_filter


            # __________________________________________________________________ #
            # CHECK POINT
            if i == 499 and j+_js == 577:
                print("-------------------")
                print("pixel:", i, j+_js, "ncomp:", _gaussian_seeds['ncomp'])
                print("model_sigma(norm):", _gaussian_seeds['rms'], "rms(phys):", _gaussian_seeds['rms'] * denom)
                print("bg(norm):", _gaussian_seeds['bg'], "bg(phys):", _gaussian_seeds['bg'] * denom + _f_min)
                print("components (x_norm, sigma_norm, peakflux_norm):\n", _gaussian_seeds['components'])
                print("priors(single-gauss, normalized):", gfit_priors_init)
                print("-------------------")
                print_priors_both(gfit_priors_init, _f_min, _f_max, v_min_phys, v_max_phys,
                  cdelt3=_cdelt3, unit_flux="Jy/beam", unit_vel="km/s")

                print("-------------------")
                print_gaussian_seeds_matched_filter_out(_gaussian_seeds, _f_min, _f_max, v_min_phys, v_max_phys, cdelt3=_cdelt3,
                    unit_flux="Jy/beam", unit_vel="km/s") 
                print("-------------------")
            # __________________________________________________________________ #



            if _cube_mask_2d[j+_js, i] <= 0 : # if masked, then skip : NOTE THE MASK VALUE SHOULD BE zero or negative.
                #print("mask filtered: %d %d | peak S/N: %.1f :: S/N limit: %.1f | integrated S/N: %.1f :: S/N limit: %.1f :: f_min: %e :: f_max: %e" \
                #    % (i, j+_js, _peak_sn_map[j+_js, i], _params['peak_sn_limit'], _sn_int_map[j+_js, i], _params['int_sn_limit'], _f_min, _f_max))

                # save the current profile location
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

                #print("low S/N: %d %d | peak S/N: %.1f :: S/N limit: %.1f | integrated S/N: %.1f :: S/N limit: %.1f :: f_min: %e :: f_max: %e" \
                #    % (i, j+_js, _peak_sn_map[j+_js, i], _params['peak_sn_limit'], _sn_int_map[j+_js, i], _params['int_sn_limit'], _f_min, _f_max))
                
                # save the current profile location
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

                #print("processing: %d %d | peak s/n: %.1f | integrated s/n: %.1f | gauss-%d" % (i, j+_js, _peak_sn_map[j+_js, i], _sn_int_map[j+_js, i], ngauss))


                # run dynesty 2.1.15
                if _params['_dynesty_class_'] == 'static':
                    _queue_size = int(_params['num_cpus_nested_sampling'])
                    rstate = np.random.default_rng(2)

                    # logl_args : loglike_d args[1:] (args[0] is set for dynesty)
                    # ptform_args : optimal_prior args[1:] (args[0] is set for dynesty)
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
                        #pool=None,
                        queue_size=_queue_size,
                        bound=_params['bound'],
                        facc=_params['facc'],
                        fmove=_params['fmove'],
                        max_move=_params['max_move'],
                        logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                    sampler.run_nested(dlogz_init=_params['dlogz'], maxiter_init=_params['maxiter'], maxcall_init=_params['maxcall'], print_progress=False)

                _gfit_results_temp, _logz = get_dynesty_sampler_results(sampler)

                #---------------------------------------------------------
                # param1, param2, param3 ....param1-e, param2-e, param3-e
                # gfit_results[j][k][0~2*nparams] = _gfit_results_temp[0~2*nparams]
                #---------------------------------------------------------
                gfit_results[j][k][:2*nparams] = _gfit_results_temp
                #---------------------------------------------------------

                #---------------------------------------------------------
                # derive rms of the profile given the current ngfit
                #---------------------------------------------------------
                _rms_ngfit = little_derive_rms_npoints(_inputDataCube, i, j+_js, _x, _f_min, _f_max, ngauss, _gfit_results_temp)
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
                    # update gfit_priors_init based on single gaussian fit
                    gfit_priors_init_g1 = _gfit_results_temp[:nparams_g1]

                    # peak s/n : more accurate peak s/n from the first sgfit 
                    _bg_sgfit = _gfit_results_temp[1]
                    _p_sgfit = _gfit_results_temp[4] # bg already subtracted
                    _peak_sn_sgfit = _p_sgfit/_rms_ngfit

                    if _peak_sn_sgfit < _params['peak_sn_limit']: 
                        #print("skip the rest of Gaussian fits: %d %d | rms:%.5f | bg:%.5f | peak:%.5f | peak_sgfit s/n: %.3f < %.3f" % (i, j+_js, _rms_ngfit, _bg_sgfit, _p_sgfit, _peak_sn_sgfit, _params['peak_sn_limit']))

                        # save the current profile location
                        # vectorization
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

                        #________________________________________________________________________________________|
                        # UNIT CONVERSION
                        #________________________________________________________________________________________|
                        convert_units_norm_to_phys(
                            gfit_results, j, k,
                            f_min=_f_min, f_max=_f_max,
                            vel_min=_vel_min, vel_max=_vel_max,
                            cdelt3=_cdelt3,
                            max_ngauss=_max_ngauss)

                        break
                    #|_______________________________________________________________________________________|
                    #|---------------------------------------------------------------------------------------|


                # update optimal priors based on the current ngaussian fit results
                if ngauss < _max_ngauss: # update gfit_priors_init with the g1fit results for the rest of the gaussians of the current profile
                    nparams_n = 3*(ngauss+1) + 2 # <-- (+ 1) for the next gaussians (e.g., ngauss=2, 3, 4 ...)

                    gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)


                    # 2) 직전 단계 결과를 prev_fit로 변환
                    # gfit_results[j, k_prev-1] 슬라이스에서 파라미터 추출
                    prev_fit = prev_fit_from_results_slice(
                        gfit_results[j, ngauss - 1],  # ← (the very before ngauss)의 결과
                        n_prev=ngauss
                    )

                    # 3) 이번 단계(예: M=3) 다중 가우스 priors 만들기
                    gfit_priors_init = set_init_priors_multiple_gaussians(
                        M=ngauss+1, # for next gaussian model
                        seed_bounds=_seed_priors_using_matched_filter,
                        seed_out=_gaussian_seeds,     # bg±k_bg*rms 구성에 활용
                        prev_fit=prev_fit,            # 이전 결과가 있으면 전달
                        k_x=3.0,
                        sigma_scale_bounds=(0.5, 2.0),
                        peak_scale_bounds=(0.5, 1.5),
                        k_bg=3.0,
                        k_msig=3.0,
                        peak_upper_cap=1.0
                    )
                    # nsigma_prior_range_gfit=3.0 (default)

                gfit_results[j, k, 2*(3*_max_ngauss+2)+k] = _rms_ngfit # rms_(k+1)gfit
                gfit_results[j, k, 2*(3*_max_ngauss+2)+_max_ngauss+0] = _logz
                gfit_results[j, k, 2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
                gfit_results[j, k, 2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
                gfit_results[j, k, 2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
                gfit_results[j, k, 2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
                gfit_results[j, k, 2*(3*_max_ngauss+2)+_max_ngauss+5] = i
                gfit_results[j, k, 2*(3*_max_ngauss+2)+_max_ngauss+6] = _js + j
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
                # UNIT CONVERSION
                #________________________________________________________________________________________|
                convert_units_norm_to_phys(
                    gfit_results, j, k,
                    f_min=_f_min, f_max=_f_max,
                    vel_min=_vel_min, vel_max=_vel_max,
                    cdelt3=_cdelt3,
                    max_ngauss=_max_ngauss)


                # SKIP CONDITION
                _m_indices = np.arange(k+1)
                _peak_flux_indices = 4 + 3*_m_indices
                gfit_peak_sn = (gfit_results[j, k, _peak_flux_indices] - gfit_results[j][k][1]) / gfit_results[j, k, 2*(3*_max_ngauss+2) + k]

                if np.all(gfit_peak_sn < _params['peak_sn_limit']) and k != (_max_ngauss-1): # if all the peak sn < threshold and k is not the max gaussian
                #    print("")
                #    print(
                #        "|-> skip the rest of Gaussian fits: %d %d | rms:%.5f | bg:%.5f | "
                #        "%d-Gaussians fit's peak SN (all): %s < %.3f <-|"
                #        % (
                #            i,
                #            j + _js,
                #            _rms_ngfit,
                #            _bg_sgfit,
                #            k+1,
                #            np.array2string(gfit_peak_sn, precision=3, separator=", "),
                #            _params['peak_sn_limit']
                #        )
                #    )                                                                                                                                                                                                                            

                    # vectorization
                    _l_indices = np.arange(_max_ngauss)

                    gfit_results[j, _l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 1] = _is
                    gfit_results[j, _l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 2] = _ie
                    gfit_results[j, _l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 3] = _js
                    gfit_results[j, _l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 4] = _je
                    gfit_results[j, _l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 5] = i
                    gfit_results[j, _l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 6] = _js + j

                    gfit_results[j, k, 2 * (3 * _max_ngauss + 2)] = gfit_results[j, k, 2*(3*_max_ngauss+2) + k] # rms
                    gfit_results[j, k, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 0] = _logz # log-Z

                    gfit_results[j, k+1:, 2 * (3 * _max_ngauss + 2) + _l_indices[k+1:]] = 0  # Adjust for proper indexing  | other rms = 0
                    gfit_results[j, k+1:, 2 * (3 * _max_ngauss + 2) + _max_ngauss] = -1E11  # other log-Z = nan

                    break # !!! SKIP THE REST GAUSSIANS !!!

        return gfit_results
    return baygaud_nested_sampling
    #-- END OF SUB-ROUTINE____________________________________________________________#





#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# run dynesty for each line profile
def dynamic_baygaud_nested_sampling0(num_cpus_nested_sampling):

    @ray.remote(num_cpus=num_cpus_nested_sampling)
    def baygaud_nested_sampling(_inputDataCube, _x, _peak_sn_map, _sn_int_map, _params, _is, _ie, i, _js, _je, _cube_mask_2d):

        _max_ngauss = _params['max_ngauss']
        _vel_min = _params['vel_min']
        _vel_max = _params['vel_max']
        _cdelt3 = _params['cdelt3']

        gfit_results = np.zeros(((_je-_js), _max_ngauss, 2*(2+3*_max_ngauss) + 7 + _max_ngauss), dtype=np.float32)

        nparams_g1 = 3*1 + 2
        gfit_priors_init_g1 = np.zeros(nparams_g1, dtype=np.float32)
        gfit_peak_sn = np.zeros(_max_ngauss, dtype=np.float32)

        #print("CHECK S/N: %d %d | peak S/N: %.1f < %.1f | integrated S/N: %.1f < %.1f" \
        #    % (i, 0+_js, _peak_sn_map[0+_js, i], _params['peak_sn_limit'], _sn_int_map[0+_js, i], _params['int_sn_limit']))

        for j in range(0, _je -_js):

            _f_max = np.max(_inputDataCube[:,j+_js,i]) # peak flux : being used for normalization
            _f_min = np.min(_inputDataCube[:,j+_js,i]) # lowest flux : being used for normalization
            #print(_f_max, _f_min)

            # prior arrays for the 1st single Gaussian fit
            gfit_priors_init = np.zeros(2*5, dtype=np.float32)

            #gfit_priors_init = [sig1, bg1, x1, std1, p1, sig2, bg2, x2, std2, p2]
            gfit_priors_init = [0.0, 0.0, \
                                0.001, 0.001, 0.001, \
                                0.9, 0.6, \
                                0.999, 0.999, 1.0]

            if _cube_mask_2d[j+_js, i] <= 0 : # if masked, then skip : NOTE THE MASK VALUE SHOULD BE zero or negative.
                #print("mask filtered: %d %d | peak S/N: %.1f :: S/N limit: %.1f | integrated S/N: %.1f :: S/N limit: %.1f :: f_min: %e :: f_max: %e" \
                #    % (i, j+_js, _peak_sn_map[j+_js, i], _params['peak_sn_limit'], _sn_int_map[j+_js, i], _params['int_sn_limit'], _f_min, _f_max))

                # save the current profile location
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

                #print("low S/N: %d %d | peak S/N: %.1f :: S/N limit: %.1f | integrated S/N: %.1f :: S/N limit: %.1f :: f_min: %e :: f_max: %e" \
                #    % (i, j+_js, _peak_sn_map[j+_js, i], _params['peak_sn_limit'], _sn_int_map[j+_js, i], _params['int_sn_limit'], _f_min, _f_max))
                
                # save the current profile location
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

                #print("processing: %d %d | peak s/n: %.1f | integrated s/n: %.1f | gauss-%d" % (i, j+_js, _peak_sn_map[j+_js, i], _sn_int_map[j+_js, i], ngauss))

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

                #---------------------------------------------------------
                # run dynesty 2.0.3
                #sampler = NestedSampler(loglike_d, optimal_prior, ndim,
                #    nlive=_params['nlive'],
                #    sample=_params['sample'],
                #    bound=_params['bound'],
                #    facc=_params['facc'],
                #    fmove=_params['fmove'],
                #    max_move=_params['max_move'],
                #    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])

                # run dynesty 2.1.15
                if _params['_dynesty_class_'] == 'static':
                    _queue_size = int(_params['num_cpus_nested_sampling'])
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

                #---------------------------------------------------------
                # param1, param2, param3 ....param1-e, param2-e, param3-e
                # gfit_results[j][k][0~2*nparams] = _gfit_results_temp[0~2*nparams]
                #---------------------------------------------------------
                gfit_results[j][k][:2*nparams] = _gfit_results_temp
                #---------------------------------------------------------

                #---------------------------------------------------------
                # derive rms of the profile given the current ngfit
                #---------------------------------------------------------
                _rms_ngfit = little_derive_rms_npoints(_inputDataCube, i, j+_js, _x, _f_min, _f_max, ngauss, _gfit_results_temp)
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
                    # update gfit_priors_init based on single gaussian fit
                    gfit_priors_init_g1 = _gfit_results_temp[:nparams_g1]

                    # peak s/n : more accurate peak s/n from the first sgfit 
                    _bg_sgfit = _gfit_results_temp[1]
                    _p_sgfit = _gfit_results_temp[4] # bg already subtracted
                    _peak_sn_sgfit = _p_sgfit/_rms_ngfit

                    if _peak_sn_sgfit < _params['peak_sn_limit']: 
                        #print("skip the rest of Gaussian fits: %d %d | rms:%.5f | bg:%.5f | peak:%.5f | peak_sgfit s/n: %.3f < %.3f" % (i, j+_js, _rms_ngfit, _bg_sgfit, _p_sgfit, _peak_sn_sgfit, _params['peak_sn_limit']))

                        # save the current profile location
                        # vectorization
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

                        #________________________________________________________________________________________|
                        #|---------------------------------------------------------------------------------------|
                        # unit conversion
                        # sigma-flux --> data cube units
                        gfit_results[j, k, 0] *= (_f_max - _f_min)  # sigma-flux to data cube units
                        # background --> data cube units
                        gfit_results[j, k, 1] = gfit_results[j, k, 1] * (_f_max - _f_min) + _f_min  # background to data cube units
                        gfit_results[j, k, 6 + 3*k] *= (_f_max - _f_min)  # background error to data cube units


                        # vectorization
                        m_indices = np.arange(k+1)

                        velocity_indices = 2 + 3*m_indices
                        velocity_dispersion_indices = 3 + 3*m_indices
                        peak_flux_indices = 4 + 3*m_indices
                        velocity_e_indices = 7 + 3*(m_indices+k)
                        velocity_dispersion_e_indices = 8 + 3*(m_indices+k)
                        flux_e_indices = 9 + 3*(m_indices+k)

                        #________________________________________________________________________________________|
                        # UNIT CONVERSION
                        #________________________________________________________________________________________|
                        # velocity, velocity-dispersion --> km/s
                        if _cdelt3 > 0: # if velocity axis is with increasing order
                            gfit_results[j, k, velocity_indices] = gfit_results[j, k, velocity_indices] * (_vel_max - _vel_min) + _vel_min # velocity
                        else: # if velocity axis is with decreasing order
                            gfit_results[j, k, velocity_indices] = gfit_results[j, k, velocity_indices] * (_vel_min - _vel_max) + _vel_max  # velocity

                        gfit_results[j, k, velocity_dispersion_indices] *= (_vel_max - _vel_min) # velocity-dispersion

                        #________________________________________________________________________________________|
                        # peak flux --> data cube units
                        #gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) + _f_min # flux
                        gfit_results[j, k, peak_flux_indices] *= (_f_max - _f_min)

                        #________________________________________________________________________________________|
                        # velocity-e, velocity-dispersion-e --> km/s
                        gfit_results[j, k, velocity_e_indices] *= (_vel_max - _vel_min) # velocity-e
                        gfit_results[j, k, velocity_dispersion_e_indices] *= (_vel_max - _vel_min) # velocity-dispersion-e

                        #gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # flux-e
                        gfit_results[j, k, flux_e_indices] *= (_f_max - _f_min) # flux-e

                        # lastly put rms 
                        gfit_results[j, k, 2*(3*_max_ngauss+2)+k] *= (_f_max - _f_min) # rms-(k+1)gfit

                        break
                    #|_______________________________________________________________________________________|
                    #|---------------------------------------------------------------------------------------|


                # update optimal priors based on the current ngaussian fit results
                if ngauss < _max_ngauss: # update gfit_priors_init with the g1fit results for the rest of the gaussians of the current profile
                    nparams_n = 3*(ngauss+1) + 2 # <-- ( + 1)
                    gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)

                    # nsigma_prior_range_gfit=3.0 (default)

                    # single Gaussian fit results
                    g1fit_x = gfit_priors_init_g1[2]
                    g1fit_std = gfit_priors_init_g1[3]
                    g1fit_p = gfit_priors_init_g1[4]

                    # lower bound : the parameters for the current ngaussian components
                    gfit_priors_init[:nparams_n] = 0.01

                    #____________________________________________
                    # the parameters for the next gaussian component: based on the current ngaussians
                    #____________________________________________
                    # x: lower bound
                    gfit_priors_init[2:nparams_n:3] = g1fit_x - g1fit_std * _params['x_prior_lowerbound_factor']
                    # std: lower bound
                    gfit_priors_init[3:nparams_n:3] = _params['std_prior_lowerbound_factor'] * g1fit_std
                    # x: lower bound
                    gfit_priors_init[3:nparams_n:3] = np.where( (gfit_priors_init[3:nparams_n:3]*(_vel_max - _vel_min) < (_cdelt3/1000.)), (_cdelt3/1000.)/(_vel_max - _vel_min), gfit_priors_init[3:nparams_n:3])
                    # p: lower bound
                    gfit_priors_init[4:nparams_n:3] = _params['p_prior_lowerbound_factor'] * g1fit_p

                    gfit_priors_init[nparams_n:2*nparams_n] = 0.99

                    #____________________________________________
                    # x: upper bound
                    gfit_priors_init[nparams_n+2:2*nparams_n:3] = g1fit_x + g1fit_std * _params['x_prior_upperbound_factor']
                    # std: upper bound
                    gfit_priors_init[nparams_n+3:2*nparams_n:3] = _params['std_prior_upperbound_factor'] * g1fit_std
                    # p: upper bound
                    gfit_priors_init[nparams_n+4:2*nparams_n:3] = _params['p_prior_upperbound_factor'] * g1fit_p

                    # set hard lower/upper boundaries
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
                # background-e --> data cube units
                gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e
                _bg_flux = gfit_results[j][k][1]

                # vectorization
                velocity_indices = 2 + 3*np.arange(k+1)
                velocity_dispersion_indices = 3 + 3*np.arange(k+1)
                peak_flux_indices = 4 + 3*np.arange(k+1)

                velocity_errors_indices = 7 + 3*np.arange(k+1) + 3*k  # Adjusted for the layout of your results array
                velocity_dispersion_errors_indices = 8 + 3*np.arange(k+1) + 3*k
                flux_errors_indices = 9 + 3*np.arange(k+1) + 3*k

                # velocity, velocity-dispersion --> km/s
                if _cdelt3 > 0:  # Velocity axis with increasing order
                    gfit_results[j, k, velocity_indices] = gfit_results[j, k, velocity_indices] * (_vel_max - _vel_min) + _vel_min # velocity
                else:  # Velocity axis with decreasing order
                    gfit_results[j, k, velocity_indices] = gfit_results[j, k, velocity_indices] * (_vel_min - _vel_max) + _vel_max # velocity

                # velocity dispersion --> km/s
                gfit_results[j, k, velocity_dispersion_indices] *= (_vel_max - _vel_min) # velocity-dispersion

                #________________________________________________________________________________________|
                # peak flux --> data cube units : (_f_max - _bg_flux) should be used for scaling as the normalised peak flux is from the bg
                #gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) + _f_min # flux
                gfit_results[j, k, peak_flux_indices] *= (_f_max - _f_min)

                #________________________________________________________________________________________|
                # velocity-e, velocity-dispersion-e --> km/s
                gfit_results[j, k, velocity_errors_indices] *= (_vel_max - _vel_min) # velocity-e
                gfit_results[j, k, velocity_dispersion_errors_indices] *= (_vel_max - _vel_min) # velocity-dispersion-e

                #________________________________________________________________________________________|
                # flux-e, rms --> cube units 
                gfit_results[j, k, flux_errors_indices] *= (_f_max - _f_min) # flux-e
                gfit_results[j, k, 2*(3*_max_ngauss+2) + k] *= (_f_max - _f_min) # rms-(k+1)gfit


                # SKIP CONDITION
                _m_indices = np.arange(k+1)
                _peak_flux_indices = 4 + 3*_m_indices
                gfit_peak_sn = (gfit_results[j, k, _peak_flux_indices] - gfit_results[j][k][1]) / gfit_results[j, k, 2*(3*_max_ngauss+2) + k]

                if np.all(gfit_peak_sn < _params['peak_sn_limit']) and k != (_max_ngauss-1): # if all the peak sn < threshold and k is not the max gaussian
                #    print("")
                #    print(
                #        "|-> skip the rest of Gaussian fits: %d %d | rms:%.5f | bg:%.5f | "
                #        "%d-Gaussians fit's peak SN (all): %s < %.3f <-|"
                #        % (
                #            i,
                #            j + _js,
                #            _rms_ngfit,
                #            _bg_sgfit,
                #            k+1,
                #            np.array2string(gfit_peak_sn, precision=3, separator=", "),
                #            _params['peak_sn_limit']
                #        )
                #    )                                                                                                                                                                                                                            

                    # vectorization
                    _l_indices = np.arange(_max_ngauss)

                    gfit_results[j, _l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 1] = _is
                    gfit_results[j, _l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 2] = _ie
                    gfit_results[j, _l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 3] = _js
                    gfit_results[j, _l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 4] = _je
                    gfit_results[j, _l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 5] = i
                    gfit_results[j, _l_indices, 2 * (3 * _max_ngauss + 2) + _max_ngauss + 6] = _js + j

                    gfit_results[j, k, 2 * (3 * _max_ngauss + 2)] = gfit_results[j, k, 2*(3*_max_ngauss+2) + k] # rms
                    gfit_results[j, k, 2 * (3 * _max_ngauss + 2) + _max_ngauss] = _logz # log-Z

                    gfit_results[j, k+1:, 2 * (3 * _max_ngauss + 2) + _l_indices[k+1:]] = 0  # Adjust for proper indexing  | other rms = 0
                    gfit_results[j, k+1:, 2 * (3 * _max_ngauss + 2) + _max_ngauss] = -1E11  # other log-Z = nan

                    break # !!! SKIP THE REST GAUSSIANS !!!

                # Plot a summary of the run.
                # rfig, raxes = dyplot.runplot(sampler.results)
                # rfig.savefig("r.pdf")
                
                # Plot traces and 1-D marginalized posteriors.
                # tfig, taxes = dyplot.traceplot(sampler.results)
                # tfig.savefig("t.pdf")
                
                # Plot the 2-D marginalized posteriors.
                #cfig, caxes = dyplot.cornerplot(sampler.results)
                #cfig.savefig("c.pdf")

        return gfit_results
    return baygaud_nested_sampling
    #-- END OF SUB-ROUTINE____________________________________________________________#


    
    
#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# run dynesty for each line profile
@ray.remote(num_cpus=1)
def run_dynesty_sampler_uniform_priors(_x, _inputDataCube, _is, _ie, i, _js, _je, _max_ngauss, _vel_min, _vel_max):

    gfit_results = np.zeros(((_je-_js), _max_ngauss, 2*(2+3*_max_ngauss)+7), dtype=np.float32)
    for j in range(0, _je -_js):

        _f_max = np.max(_inputDataCube[:,j+_js,i]) # peak flux : being used for normalization
        _f_min = np.min(_inputDataCube[:,j+_js,i]) # lowest flux : being used for normalization

        gfit_priors_init = np.zeros(2*5, dtype=np.float32)
        #gfit_priors_init = [sig1, bg1, x1, std1, p1, sig2, bg2, x2, std2, p2]
        gfit_priors_init = [0.0, 0.0, 0.01, 0.01, 0.01, 0.5, 0.5, 0.9, 0.6, 1.01]
        for k in range(0, _max_ngauss):
            ngauss = k+1  # set the number of gaussian
            ndim = 3*ngauss + 2
            nparams = ndim

            #print("processing: %d %d gauss-%d" % (i, j+_js, ngauss))

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
    log_Z = _sampler.results.logz[-1]

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
@njit(cache=True, fastmath=True)
def _multi_gaussian_model_norm_core(x, params, ngauss):
    """
    x      : 정규화 채널 축 (1D)
    params : [sigma, bg, g1_x, g1_std, g1_p, g2_x, g2_std, g2_p, ...]
             (정규화 스케일; sigma는 여기서 사용하지 않음)
    ngauss : 가우시안 개수
    return : model spectrum (정규화 스케일)
    """
    n = x.size
    out = np.empty(n, dtype=np.float64)

    # 배경부터 채움
    bg = params[1]
    for t in range(n):
        out[t] = bg

    # 가우시안 성분 합산
    for m in range(ngauss):
        mu  = params[2 + 3*m]
        sig = params[3 + 3*m]
        amp = params[4 + 3*m]
        if sig <= 1e-12:  # 안전 하한
            sig = 1e-12
        invs = 1.0 / sig
        for t in range(n):
            dx = (x[t] - mu) * invs
            out[t] += amp * np.exp(-0.5 * dx * dx)
    return out
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def multi_gaussian_model_d(x, params, ngauss):
    """
    드롭인 교체용 파이썬 래퍼 (원본 시그니처 유지).
    x, params를 float64 C-contiguous로 정리해 JIT 코어 호출.
    """
    x64 = np.ascontiguousarray(x, dtype=np.float64)
    # 필요한 길이까지만 잘라 전달 (3*ngauss+2)
    p64 = np.ascontiguousarray(params[:(3*int(ngauss)+2)], dtype=np.float64)
    return _multi_gaussian_model_norm_core(x64, p64, int(ngauss))
#-- END OF SUB-ROUTINE____________________________________________________________#





#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def multi_gaussian_model_d_org(_x, _params, ngauss): # params: cube : non_njit version
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
def multi_gaussian_model_d_vectorization(_x, _params, ngauss): # _x: global array, params: cube

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



# ----------------------------- #
# JIT 코어: 단순 루프, in-place  #
# ----------------------------- #
@njit(cache=True, fastmath=True)
def _optimal_prior_core(u, ngauss, bounds):
    """
    내부 코어: u, bounds는 1D float64 배열, ngauss는 int.
    u를 in-place로 변환 후 반환.
    """
    nparams = 3 * ngauss + 2

    # sigma
    # args[2][0] : _sigma0
    # args[2][nparams] : _sigma1
    s0 = bounds[0]
    s1 = bounds[nparams]
    u[0] = s0 + u[0] * (s1 - s0)  # sigma: uniform prior between 0:1

    # bg
    # args[2][1] : _bg0
    # args[2][nparams+1] : _bg1
    b0 = bounds[1]
    b1 = bounds[nparams + 1]
    u[1] = b0 + u[1] * (b1 - b0)  # bg: uniform prior between 0:1

    # n-gaussians
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

        # x: uniform prior between xn0:xn1
        u[i_x] = lx + u[i_x] * (ux - lx)
        # std: uniform prior between stdn0:stdn1
        u[i_s] = ls + u[i_s] * (us - ls)
        # p: uniform prior between pn0:pn1
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

    # 안전·성능: dtype·연속성 정리
    u      = np.ascontiguousarray(np.asarray(args[0], dtype=np.float64))
    ngauss = int(args[1])
    bounds = np.ascontiguousarray(np.asarray(args[2], dtype=np.float64))

    # 코어 호출 (in-place 변환 후 같은 배열 반환)
    return _optimal_prior_core(u, ngauss, bounds)






#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# parameters are sigma, bg, _x01, _std01, _p01, _x02, _std02, _p02...
def optimal_prior0(*args):

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
    _sigma1 = np.array(args[2][nparams]) # args[1]=ngauss

    # bg
    _bg0 = np.array(args[2][1])
    _bg1 = np.array(args[2][nparams+1]) # args[1]=ngauss

    # x
    _xn_0 = np.array(args[2][2:nparams:3])
    _xn_1 = np.array(args[2][nparams+2:2*nparams:3])

    # std
    _stdn_0 = np.array(args[2][3:nparams:3])
    _stdn_1 = np.array(args[2][nparams+3:2*nparams:3])

    # p
    _pn_0 = np.array(args[2][4:nparams:3])
    _pn_1 = np.array(args[2][nparams+4:2*nparams:3])

    # vectorization
    # sigma and bg
    args[0][0] = _sigma0 + args[0][0]*(_sigma1 - _sigma0)   # sigma: uniform prior between 0:1
    args[0][1] = _bg0 + args[0][1]*(_bg1 - _bg0)            # bg: uniform prior between 0:1

    # n-gaussians
    # x
    args[0][2:nparams:3] = _xn_0 +   args[0][2:nparams:3]*(_xn_1 - _xn_0)            # x: uniform prior between xn0:xn1
    # std
    args[0][3:nparams:3] = _stdn_0 + args[0][3:nparams:3]*(_stdn_1 - _stdn_0)            # std: uniform prior between stdn0:stdn1
    # p
    args[0][4:nparams:3] = _pn_0 + args[0][4:nparams:3]*(_pn_1 - _pn_0)            # p: uniform prior between pn0:pn1

    return args[0]

    # old version
    #_sigma0 = np.array(args[2][0])
    #_sigma1 = np.array(args[2][2+3*args[1]]) # args[1]=ngauss
    #_bg0 = np.array(args[2][1])
    #_bg1 = np.array(args[2][3+3*args[1]]) # args[1]=ngauss


    #_xn_0 = np.zeros(args[1])
    #_xn_1 = np.zeros(args[1])
    #_stdn_0 = np.zeros(args[1])
    #_stdn_1 = np.zeros(args[1])
    #_pn_0 = np.zeros(args[1])
    #_pn_1 = np.zeros(args[1])


    #nparams = 3*args[1] + 2

    #_xn_0 = np.array(args[2][2:nparams:3])
    #_xn_1 = np.array(args[2][nparams+2:2*nparams:3])

    #_stdn_0 = np.array(args[2][3:nparams:3])
    #_stdn_1 = np.array(args[2][nparams+3:2*nparams:3])

    #_pn_0 = np.array(args[2][4:nparams:3])
    #_pn_1 = np.array(args[2][nparams+4:2*nparams:3])


    #args[0][0] = _sigma0 + args[0][0]*(_sigma1 - _sigma0)   # sigma: uniform prior between 0:1
    #args[0][1] = _bg0 + args[0][1]*(_bg1 - _bg0)            # bg: uniform prior betwargs[1]een 0:1

    #args[0][2:nparams:3] = _xn_0 +   args[0][2:nparams:3]*(_xn_1 - _xn_0)            # bg: uniform prior betwargs[1]een 0:1
    #args[0][3:nparams:3] = _stdn_0 + args[0][3:nparams:3]*(_stdn_1 - _stdn_0)            # bg: uniform prior betwargs[1]een 0:1
    #args[0][4:nparams:3] = _pn_0 + args[0][4:nparams:3]*(_pn_1 - _pn_0)            # bg: uniform prior betwargs[1]een 0:1

    #return args[0]

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
    params_t = args[0][2:].reshape(args[1], 3).T

    # vectorization
    # sigma and bg
    args[0][0] = _sigma0 + args[0][0]*(_sigma1 - _sigma0)            # bg: uniform prior between 0:1
    args[0][1] = _bg0 + args[0][1]*(_bg1 - _bg0)            # bg: uniform prior betwargs[1]een 0:1

    # n-gaussians
    # x
    params_t[0] = (_x0 + params_t[0].T*(_x1 - _x0)).T
    # std
    params_t[1] = (_std0 + params_t[1].T*(_std1 - _std0)).T
    # p
    params_t[2] = (_p0 + params_t[2].T*(_p1 - _p0)).T


    params_t_conc = np.hstack((params_t[0].reshape(args[1], 1), params_t[1].reshape(args[1], 1), params_t[2].reshape(args[1], 1))).reshape(1, 3*args[1])
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

    # x
    _x0 = args[2][2]
    _x1 = args[2][7]

    # std
    _std0 = args[2][3]
    _std1 = args[2][8]

    # p
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


    params_t_conc = np.hstack((params_t[0].reshape(args[1], 1), params_t[1].reshape(args[1], 1), params_t[2].reshape(args[1], 1))).reshape(1, 3*args[1])
    args[0][2:] = params_t_conc

    return args[0]
#-- END OF SUB-ROUTINE____________________________________________________________#





#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
@njit(cache=True, fastmath=True)
def _loglike_core_numba_serial(params, spect, x, ngauss):
    """
    params = [sigma, bg, g1_x, g1_std, g1_p, g2_x, g2_std, g2_p, ...]  (정규화 스케일)
    spect  = 정규화된 관측 스펙트럼 (data_norm)
    x      = 정규화된 채널 축
    ngauss = 가우시안 개수
    반환   = logL (정규화 상수 포함)
    """
    n = x.size
    sigma = params[0]
    # 안전장치(0 또는 음수 sigma 방지)
    if sigma <= 0.0:
        sigma = 1e-12

    # 가우시안 파라미터를 한 번만 읽어둠
    mu  = np.empty(ngauss, dtype=np.float64)
    sig = np.empty(ngauss, dtype=np.float64)
    amp = np.empty(ngauss, dtype=np.float64)
    for m in range(ngauss):
        mu[m]  = params[2 + 3*m]
        sigm   = params[3 + 3*m]
        # 표준편차 안전 하한
        sig[m] = sigm if sigm > 1e-12 else 1e-12
        amp[m] = params[4 + 3*m]

    bg = params[1]
    inv_sigma2 = 1.0 / (sigma * sigma)

    # 정규화 상수(모델 비교 시 상쇄되지만, 원본 함수와 동일하게 포함)
    log_n_sigma = -0.5 * n * np.log(2.0 * np.pi) - n * np.log(sigma)

    sse = 0.0
    for t in range(n):
        yt = bg
        xt = x[t]
        # 가우시안 합
        for m in range(ngauss):
            d = (xt - mu[m]) / sig[m]
            yt += amp[m] * np.exp(-0.5 * d * d)
        r = (yt - spect[t])
        sse += (r * r)

    return log_n_sigma - 0.5 * sse * inv_sigma2
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# numba version
@njit(cache=True, fastmath=True)
def _loglike_core_numba(params, spect, x, ngauss):
    """
    params = [sigma, bg, g1_x, g1_std, g1_p, g2_x, g2_std, g2_p, ...]  (정규화 스케일)
    spect  = 정규화된 관측 스펙트럼 (data_norm)
    x      = 정규화된 채널 축
    ngauss = 가우시안 개수
    반환   = logL (정규화 상수 포함)
    """

    n = x.size
    sigma = params[0]
    if sigma <= 0.0:
        sigma = 1e-12
    bg = params[1]
    log_n_sigma = -0.5 * n * np.log(2.0 * np.pi) - n * np.log(sigma)
    inv_sigma2  = 1.0 / (sigma * sigma)

    # 모델 = bg로 채우고, 성분별로 벡터 더하기
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
    sse = np.dot(r, r)            # BLAS 호출(스레드=1로 고정 권장)

    return log_n_sigma - 0.5 * sse * inv_sigma2
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# python wrapper -> _loglike_core_numba
def loglike_d(*args):
    """
     args[0] : params : dynesty default
     args[1] : _spect : input velocity profile array [N channels] <-- normalized (F-f_max)/(f_max-f_min)
     args[2] : _x
     args[3] : ngauss
     _bg, _x0, _std, _p0, .... = params[1], params[2], params[3], params[4]
     sigma = params[0] # loglikelihoood sigma
     print(args[1])
    """

    params = np.ascontiguousarray(args[0], dtype=np.float64)
    spect  = np.ascontiguousarray(args[1], dtype=np.float64)
    x      = np.ascontiguousarray(args[2], dtype=np.float64)
    ngauss = int(args[3])

    return float(_loglike_core_numba(params, spect, x, ngauss))
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def loglike_d_org(*args): # non-njit version
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


    log_n_sigma = -0.5*npoints*np.log(2.0*np.pi) - 1.0*npoints*np.log(sigma)
    chi2 = np.nansum((-1.0 / (2*sigma**2)) * ((gfit - args[1])**2))
    return log_n_sigma + chi2
#-- END OF SUB-ROUTINE____________________________________________________________#











