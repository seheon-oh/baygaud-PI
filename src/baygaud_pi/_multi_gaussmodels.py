#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _multi_gaussmodels.py
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|

import numpy as np
from numpy import sum, exp, log, pi
from numba import njit

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
@njit(cache=True, fastmath=True)
def _multi_gaussian_model_norm_core(x, params, ngauss):
    """
    x      : normalized channel axis (1D)
    params : [sigma, bg, g1_x, g1_std, g1_p, g2_x, g2_std, g2_p, ...]
             (all values in normalized scale; note: sigma is NOT used here)
    ngauss : number of Gaussian components
    return : model spectrum (normalized scale)
    """
    n = x.size
    out = np.empty(n, dtype=np.float64)

    # Fill with background first
    bg = params[1]
    for t in range(n):
        out[t] = bg

    # Add Gaussian components
    for m in range(ngauss):
        mu  = params[2 + 3*m]
        sig = params[3 + 3*m]
        amp = params[4 + 3*m]
        if sig <= 1e-12:  # safety lower bound
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
    Drop-in Python wrapper (keep original signature).
    Convert x and params to float64 C-contiguous arrays and call the JIT core.
    """
    x64 = np.ascontiguousarray(x, dtype=np.float64)
    # Pass only the required length (3*ngauss + 2)
    p64 = np.ascontiguousarray(params[:(3*int(ngauss)+2)], dtype=np.float64)
    return _multi_gaussian_model_norm_core(x64, p64, int(ngauss))
#-- END OF SUB-ROUTINE____________________________________________________________#





#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def multi_gaussian_model_d_org(_x, _params, ngauss): # params: cube : non_njit version
    # _bg0 : _params[1]
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
    # _bg0 : gfit_results[1]
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
    # _bg0 : _params[1]
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
        # y += _p0 * (scipy.stats.norm.pdf(_x, loc=_x0, scale=_std0))

    _y += _bg0
    return _y

#-- END OF SUB-ROUTINE____________________________________________________________#


