#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| baygaud_clasify.py
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|

#|-----------------------------------------|
# Python 3 compatability

from __future__ import division, print_function
from re import A, I, L
from six.moves import range

import time, sys, os
from datetime import datetime
import shutil

import numpy as np
from numpy import linalg, array, sum, log, exp, pi, std, diag, concatenate
import gc
import operator
import copy as cp

import itertools as itt

from _dirs_files import make_dirs

import json
import sys
import scipy.stats, scipy
import matplotlib.pyplot as plt

import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from dynesty import NestedSampler

import fitsio
import astropy.units as u
from astropy.io import fits
from spectral_cube import SpectralCube
from _handle_yaml import read_configfile

from numba import njit, prange


from _fits_io import update_header_cube_to_2d

global _inputDataCube
global _x
global _spect
global _is, _ie, _js, _je
global parameters
global nparams
global ngauss
_spect = None
global ndim
global max_ngauss


def find_nearest_values_along_axis(nparray, nparray_ref, axis):
    argmin_index = np.abs(nparray - nparray_ref).argmin(axis=axis)
    _shape = nparray.shape
    index = list(np.ix_(*[np.arange(i) for i in _shape]))
    index[axis] = np.expand_dims(argmin_index, axis=axis)

    return np.squeeze(nparray[index])


def find_nearest_index_along_axis(nparray, nparray_ref, axis):
    argmin_index = np.abs(nparray - nparray_ref).argmin(axis=axis)
    _shape = nparray.shape
    index = list(np.ix_(*[np.arange(i) for i in _shape]))
    index[axis] = np.expand_dims(argmin_index, axis=axis)

    return index[axis][0]


def extract_maps_bulk(_fitsarray_gfit_results2, params, _output_dir, _kin_comp, ng_opt, _bulk_ref_vf, _bulk_delv_limit, _hdu):

    max_ngauss = params['max_ngauss']
    nparams = (3*max_ngauss+2)
    peak_sn_limit = params['peak_sn_limit']

    if _kin_comp == 'bulk':
        _g_vlos_lower = params['g_vlos_lower']
        _g_vlos_upper = params['g_vlos_upper']
        _g_sigma_lower = params['g_sigma_lower']
        _g_sigma_upper = params['g_sigma_upper']





    nparams_step = 2*(3*max_ngauss+2) + (max_ngauss + 7)


    sn_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    x_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    std_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    p_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    bg_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    rms_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)


    for i in range(0, max_ngauss):
        for j in range(0, max_ngauss):
            sn_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i) & \
                                                    (_fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :] > 0), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0]
            sn_ng_opt_t[i, :, :] += sn_ng_opt_slice[i, j, :, :]
            sn_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    0.0, 0.0)])[0] # put zero to rms-error as it is not available
            sn_ng_opt_e_t[i, :, :] += sn_ng_opt_slice_e[i, j, :, :]

            x_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*i, :, :], 1E-7)])[0] # otherwise put 1E-7 used for summing up Xs below
            x_ng_opt_t[i, :, :] += x_ng_opt_slice[i, j, :, :]
            x_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 0 + 3*i, :, :], 1E-7)])[0] # otherwise put 1E-7 used for summing up Xs below
            x_ng_opt_e_t[i, :, :] += x_ng_opt_slice_e[i, j, :, :]

            std_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 3 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up stds below
            std_ng_opt_t[i, :, :] += std_ng_opt_slice[i, j, :, :]
            std_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 1 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up stds below
            std_ng_opt_e_t[i, :, :] += std_ng_opt_slice_e[i, j, :, :]

            p_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up ps below
            p_ng_opt_t[i, :, :] += p_ng_opt_slice[i, j, :, :]
            p_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 2 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up ps below
            p_ng_opt_e_t[i, :, :] += p_ng_opt_slice_e[i, j, :, :]

            bg_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 1, :, :], 0.0)])[0] # otherwise put 1E-7 used for summing up bgs below
            bg_ng_opt_t[i, :, :] += bg_ng_opt_slice[i, j, :, :]
            bg_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 1, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up bgs below
            bg_ng_opt_e_t[i, :, :] += bg_ng_opt_slice_e[i, j, :, :]

            rms_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0] # otherwise put 0.0
            rms_ng_opt_t[i, :, :] += rms_ng_opt_slice[i, j, :, :]
            rms_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    0.0, 0.0)])[0] # put zero to rms-error as it is not available
            rms_ng_opt_e_t[i, :, :] += rms_ng_opt_slice_e[i, j, :, :]


    sn_ng_opt = np.where(sn_ng_opt_t != 3*0.0, sn_ng_opt_t, -1E9)
    sn_ng_opt_e = np.where(sn_ng_opt_t != 3*0.0, sn_ng_opt_t, -1E9)

    x_ng_opt = np.where(x_ng_opt_t != 3*1E-7, x_ng_opt_t, -1E9)
    x_ng_opt_e = np.where(x_ng_opt_e_t != 3*1E-7, x_ng_opt_e_t, -1E9)

    std_ng_opt = np.where(std_ng_opt_t != 3*0.0, std_ng_opt_t, -1E9)
    std_ng_opt_e = np.where(std_ng_opt_e_t != 3*0.0, std_ng_opt_e_t, -1E9)

    p_ng_opt = np.where(p_ng_opt_t != 3*0.0, p_ng_opt_t, -1E9)
    p_ng_opt_e = np.where(p_ng_opt_e_t != 3*0.0, p_ng_opt_e_t, -1E9)

    bg_ng_opt = np.where(bg_ng_opt_t != 3*0.0, bg_ng_opt_t, -1E9)
    bg_ng_opt_e = np.where(bg_ng_opt_e_t != 3*0.0, bg_ng_opt_e_t, -1E9)

    rms_ng_opt = np.where(rms_ng_opt_t != 3*0.0, rms_ng_opt_t, -1E9)
    rms_ng_opt_e = np.where(rms_ng_opt_e_t != 3*0.0, rms_ng_opt_e_t, -1E9)

    if _kin_comp == 'sgfit' or _kin_comp == 'psgfit':
        _ng = 1 
    else:
        _ng = max_ngauss

    bk_index = find_nearest_index_along_axis(x_ng_opt, _bulk_ref_vf, axis=0)






    _ax1 = np.arange(x_ng_opt.shape[1])[:, None]
    _ax2 = np.arange(x_ng_opt.shape[2])[None, :]
    sn_ng_opt_bulk = sn_ng_opt[bk_index, _ax1, _ax2]
    x_ng_opt_bulk = x_ng_opt[bk_index, _ax1, _ax2]
    x_ng_opt_bulk_e = x_ng_opt_e[bk_index, _ax1, _ax2]
    std_ng_opt_bulk = std_ng_opt[bk_index, _ax1, _ax2]
    std_ng_opt_bulk_e = std_ng_opt_e[bk_index, _ax1, _ax2]
    p_ng_opt_bulk = p_ng_opt[bk_index, _ax1, _ax2]
    p_ng_opt_bulk_e = p_ng_opt_e[bk_index, _ax1, _ax2]
    bg_ng_opt_bulk = bg_ng_opt[bk_index, _ax1, _ax2]
    bg_ng_opt_bulk_e = bg_ng_opt_e[bk_index, _ax1, _ax2]
    rms_ng_opt_bulk = rms_ng_opt[bk_index, _ax1, _ax2]
    rms_ng_opt_bulk_e = rms_ng_opt_e[bk_index, _ax1, _ax2]


    i1 = params['_i0']
    j1 = params['_j0']

    print("sn:", sn_ng_opt_bulk[j1, i1])
    print("x:", x_ng_opt_bulk[j1, i1])
    print("ref_x:", _bulk_ref_vf[j1, i1])
    print("delv:", _bulk_delv_limit[j1, i1])
    print("std:", std_ng_opt_bulk[j1, i1])

    print("_g_sigma_lower:", _g_sigma_lower)
    print("g_sigma_upper:", _g_sigma_upper)
    print("vlos_lower:", _g_vlos_lower)
    print("vlos_upper:", _g_vlos_upper)
    



    _filter_bulk = ( \
            (sn_ng_opt_bulk[:, :] > peak_sn_limit) & \
            (x_ng_opt_bulk[:, :] >= _bulk_ref_vf[:, :] - _bulk_delv_limit[:, :]) & \
            (x_ng_opt_bulk[:, :] < _bulk_ref_vf[:, :] + _bulk_delv_limit[:, :]) & \
            (std_ng_opt_bulk[:, :] >= _g_sigma_lower) & \
            (std_ng_opt_bulk[:, :] < _g_sigma_upper))


    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    np.sqrt(2*np.pi)* std_ng_opt_bulk * p_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #update_header_cube_to_2d(_hdulist_nparray, _hdu)
    update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=0)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.0.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()

    _nparray_t = np.array([np.where( \
                                    (_filter_bulk) & \
                                    (p_ng_opt_bulk > 0.0) & \
                                    (std_ng_opt_bulk > 0.0), \
                                    np.sqrt(2*np.pi) * \
                                    p_ng_opt_bulk * \
                                    std_ng_opt_bulk * \
                                    ((p_ng_opt_bulk_e/p_ng_opt_bulk)**2 + \
                                    (std_ng_opt_bulk_e/std_ng_opt_bulk)**2)**0.5, \
                                   np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #update_header_cube_to_2d(_hdulist_nparray, _hdu)
    update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=0)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.0.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()


    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    x_ng_opt_bulk, np.nan)])

    _nparray_bulk_x_extracted = np.array([np.where( \
                                    _filter_bulk, \
                                    x_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #update_header_cube_to_2d(_hdulist_nparray, _hdu)
    update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=1)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.1.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    x_ng_opt_bulk_e, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #update_header_cube_to_2d(_hdulist_nparray, _hdu)
    update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=1)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.1.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()


    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    std_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #update_header_cube_to_2d(_hdulist_nparray, _hdu)
    update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=2)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.2.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    std_ng_opt_bulk_e, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #update_header_cube_to_2d(_hdulist_nparray, _hdu)
    update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=2)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.2.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()

    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    bg_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=3)
   # update_header_cube_to_2d(_hdulist_nparray, _hdu)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.3.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    bg_ng_opt_bulk_e, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #update_header_cube_to_2d(_hdulist_nparray, _hdu)
    update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=3)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.3.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()


    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    rms_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #update_header_cube_to_2d(_hdulist_nparray, _hdu)
    update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=4)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.4.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    0.0, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #update_header_cube_to_2d(_hdulist_nparray, _hdu)
    update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=4)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.4.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()


    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    p_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #update_header_cube_to_2d(_hdulist_nparray, _hdu)
    update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=5)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.5.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    p_ng_opt_bulk_e, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #update_header_cube_to_2d(_hdulist_nparray, _hdu)
    update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=5)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.5.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()

    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    p_ng_opt_bulk / rms_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #update_header_cube_to_2d(_hdulist_nparray, _hdu)
    update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=6)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.6.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    0.0, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #update_header_cube_to_2d(_hdulist_nparray, _hdu)
    update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=6)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.6.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()

    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    ng_opt+1, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #update_header_cube_to_2d(_hdulist_nparray, _hdu)
    update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=7)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.7.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    0.0, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #update_header_cube_to_2d(_hdulist_nparray, _hdu)
    update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=7)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.7.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()




    for i in range(0, _ng):



        _filter_non_bulk = ( \
            (ng_opt[:, :] >= i) & \
            (sn_ng_opt[i, :, :] > peak_sn_limit) & \
            (x_ng_opt[i,:, :] >= _g_vlos_lower) & \
            (x_ng_opt[i,:, :] < _g_vlos_upper) & \
            (np.absolute(x_ng_opt[i,:, :] - _nparray_bulk_x_extracted[0, :, :]) > 0.1) & \
            (std_ng_opt[i,:, :] >= _g_sigma_lower) & \
            (std_ng_opt[i,:, :] < _g_sigma_upper))

        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        np.sqrt(2*np.pi)*_fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] * \
                                        _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=0)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.0.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk & \
                                        (_fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :] > 0.0) & \
                                        (_fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] > 0.0), \
                                        np.sqrt(2*np.pi) * \
                                        _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] * \
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :] * \
                                        ((_fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :])**2 + \
                                         (_fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 1 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :])**2)**0.5, \
                                        np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=0)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.0.e.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=1)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.1.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()


        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 0 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=1)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.1.e.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=2)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.2.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 1 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=2)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.2.e.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        _fitsarray_gfit_results2[nparams_step*i + 1 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=3)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.3.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 1 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=3)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.3.e.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        _fitsarray_gfit_results2[nparams_step*(i+1)-max_ngauss-7+i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=4)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.4.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        0.0, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=4)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.4.e.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=5)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.5.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=5)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.5.e.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()


        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :] / \
                                        _fitsarray_gfit_results2[nparams_step*(i+1)-max_ngauss-7+i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=6)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.6.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        0.0, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=6)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.6.e.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        ng_opt+1, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=7)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.7.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        0.0, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=7)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.7.e.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

    print('[—> fits written ..', _nparray_t.shape)






def extract_maps_bulk_org(_fitsarray_gfit_results2, params, _output_dir, _kin_comp, ng_opt, _bulk_ref_vf, _bulk_delv_limit):

    max_ngauss = params['max_ngauss']
    nparams = (3*max_ngauss+2)
    peak_sn_limit = params['peak_sn_limit']

    if _kin_comp == 'sgfit':
        _g_vlos_lower = params['g_vlos_lower']
        _g_vlos_upper = params['g_vlos_upper']
        _g_sigma_lower = params['g_sigma_lower']
        _g_sigma_upper = params['g_sigma_upper']
        print("")
        print("| ... extracting sgfit results ... |")
        print("| vlos-lower: %1.f" % _g_vlos_lower)
        print("| vlos-upper: %1.f" % _g_vlos_upper)
        print("| vdisp-lower: %1.f" % _g_sigma_lower)
        print("| vdisp-upper: %1.f" % _g_sigma_upper)
        print("")
    elif _kin_comp == 'psgfit':
        _g_vlos_lower = params['g_vlos_lower']
        _g_vlos_upper = params['g_vlos_upper']
        _g_sigma_lower = params['g_sigma_lower']
        _g_sigma_upper = params['g_sigma_upper']
        print("")
        print("| ... extracting sgfit results ... |")
        print("| vlos-lower: %1.f" % _g_vlos_lower)
        print("| vlos-upper: %1.f" % _g_vlos_upper)
        print("| vdisp-lower: %1.f" % _g_sigma_lower)
        print("| vdisp-upper: %1.f" % _g_sigma_upper)
        print("")
    elif _kin_comp == 'cool':
        _g_vlos_lower = params['g_vlos_lower_cool']
        _g_vlos_upper = params['g_vlos_upper_cool']
        _g_sigma_lower = params['g_sigma_lower_cool']
        _g_sigma_upper = params['g_sigma_upper_cool']
        print("")
        print("| ... extracting kinematically cool results ... |")
        print("| vlos-lower: %1.f" % _g_vlos_lower)
        print("| vlos-upper: %1.f" % _g_vlos_upper)
        print("| vdisp-lower: %1.f" % _g_sigma_lower)
        print("| vdisp-upper: %1.f" % _g_sigma_upper)
        print("")
    elif _kin_comp == 'warm':
        _g_vlos_lower = params['g_vlos_lower_warm']
        _g_vlos_upper = params['g_vlos_upper_warm']
        _g_sigma_lower = params['_g_sigma_lower_warm']
        _g_sigma_upper = params['g_sigma_upper_warm']
        print("")
        print("| ... extracting kinematically warm results ... |")
        print("| vlos-lower: %1.f" % _g_vlos_lower)
        print("| vlos-upper: %1.f" % _g_vlos_upper)
        print("| vdisp-lower: %1.f" % _g_sigma_lower)
        print("| vdisp-upper: %1.f" % _g_sigma_upper)
        print("")
    elif _kin_comp == 'hot':
        _g_vlos_lower = params['g_vlos_lower_hot']
        _g_vlos_upper = params['g_vlos_upper_hot']
        _g_sigma_lower = params['_g_sigma_lower_hot']
        _g_sigma_upper = params['g_sigma_upper_hot']
        print("")
        print("| ... extracting kinematically hot results ... |")
        print("| vlos-lower: %1.f" % _g_vlos_lower)
        print("| vlos-upper: %1.f" % _g_vlos_upper)
        print("| vdisp-lower: %1.f" % _g_sigma_lower)
        print("| vdisp-upper: %1.f" % _g_sigma_upper)
        print("")
    else: # including bulk
        _g_vlos_lower = params['g_vlos_lower']
        _g_vlos_upper = params['g_vlos_upper']
        _g_sigma_lower = params['g_sigma_lower']
        _g_sigma_upper = params['g_sigma_upper']






    nparams_step = 2*(3*max_ngauss+2) + (max_ngauss + 7)


    sn_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    x_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    std_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    p_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    bg_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    rms_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)


    for i in range(0, max_ngauss):
        for j in range(0, max_ngauss):
            sn_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i) & \
                                                    (_fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :] > 0), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0]
            sn_ng_opt_t[i, :, :] += sn_ng_opt_slice[i, j, :, :]
            sn_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    0.0, 0.0)])[0] # put zero to rms-error as it is not available
            sn_ng_opt_e_t[i, :, :] += sn_ng_opt_slice_e[i, j, :, :]

            x_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*i, :, :], 1E-7)])[0] # otherwise put 1E-7 used for summing up Xs below
            x_ng_opt_t[i, :, :] += x_ng_opt_slice[i, j, :, :]
            x_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 0 + 3*i, :, :], 1E-7)])[0] # otherwise put 1E-7 used for summing up Xs below
            x_ng_opt_e_t[i, :, :] += x_ng_opt_slice_e[i, j, :, :]

            std_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 3 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up stds below
            std_ng_opt_t[i, :, :] += std_ng_opt_slice[i, j, :, :]
            std_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 1 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up stds below
            std_ng_opt_e_t[i, :, :] += std_ng_opt_slice_e[i, j, :, :]

            p_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up ps below
            p_ng_opt_t[i, :, :] += p_ng_opt_slice[i, j, :, :]
            p_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 2 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up ps below
            p_ng_opt_e_t[i, :, :] += p_ng_opt_slice_e[i, j, :, :]

            bg_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 1, :, :], 0.0)])[0] # otherwise put 1E-7 used for summing up bgs below
            bg_ng_opt_t[i, :, :] += bg_ng_opt_slice[i, j, :, :]
            bg_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 1, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up bgs below
            bg_ng_opt_e_t[i, :, :] += bg_ng_opt_slice_e[i, j, :, :]

            rms_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0] # otherwise put 0.0
            rms_ng_opt_t[i, :, :] += rms_ng_opt_slice[i, j, :, :]
            rms_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    0.0, 0.0)])[0] # put zero to rms-error as it is not available
            rms_ng_opt_e_t[i, :, :] += rms_ng_opt_slice_e[i, j, :, :]


    sn_ng_opt = np.where(sn_ng_opt_t != 3*0.0, sn_ng_opt_t, -1E9)
    sn_ng_opt_e = np.where(sn_ng_opt_t != 3*0.0, sn_ng_opt_t, -1E9)

    x_ng_opt = np.where(x_ng_opt_t != 3*1E-7, x_ng_opt_t, -1E9)
    x_ng_opt_e = np.where(x_ng_opt_e_t != 3*1E-7, x_ng_opt_e_t, -1E9)

    std_ng_opt = np.where(std_ng_opt_t != 3*0.0, std_ng_opt_t, -1E9)
    std_ng_opt_e = np.where(std_ng_opt_e_t != 3*0.0, std_ng_opt_e_t, -1E9)

    p_ng_opt = np.where(p_ng_opt_t != 3*0.0, p_ng_opt_t, -1E9)
    p_ng_opt_e = np.where(p_ng_opt_e_t != 3*0.0, p_ng_opt_e_t, -1E9)

    bg_ng_opt = np.where(bg_ng_opt_t != 3*0.0, bg_ng_opt_t, -1E9)
    bg_ng_opt_e = np.where(bg_ng_opt_e_t != 3*0.0, bg_ng_opt_e_t, -1E9)

    rms_ng_opt = np.where(rms_ng_opt_t != 3*0.0, rms_ng_opt_t, -1E9)
    rms_ng_opt_e = np.where(rms_ng_opt_e_t != 3*0.0, rms_ng_opt_e_t, -1E9)

    if _kin_comp == 'sgfit' or _kin_comp == 'psgfit':
        _ng = 1 
    else:
        _ng = max_ngauss

    bk_index = find_nearest_index_along_axis(x_ng_opt, _bulk_ref_vf, axis=0)






    _ax1 = np.arange(x_ng_opt.shape[1])[:, None]
    _ax2 = np.arange(x_ng_opt.shape[2])[None, :]
    sn_ng_opt_bulk = sn_ng_opt[bk_index, _ax1, _ax2]
    x_ng_opt_bulk = x_ng_opt[bk_index, _ax1, _ax2]
    x_ng_opt_bulk_e = x_ng_opt_e[bk_index, _ax1, _ax2]
    std_ng_opt_bulk = std_ng_opt[bk_index, _ax1, _ax2]
    std_ng_opt_bulk_e = std_ng_opt_e[bk_index, _ax1, _ax2]
    p_ng_opt_bulk = p_ng_opt[bk_index, _ax1, _ax2]
    p_ng_opt_bulk_e = p_ng_opt_e[bk_index, _ax1, _ax2]
    bg_ng_opt_bulk = bg_ng_opt[bk_index, _ax1, _ax2]
    bg_ng_opt_bulk_e = bg_ng_opt_e[bk_index, _ax1, _ax2]
    rms_ng_opt_bulk = rms_ng_opt[bk_index, _ax1, _ax2]
    rms_ng_opt_bulk_e = rms_ng_opt_e[bk_index, _ax1, _ax2]

    i1 = params['_i0']
    j1 = params['_j0']

    print("sn:", sn_ng_opt_bulk[j1, i1])
    print("x:", x_ng_opt_bulk[j1, i1])
    print("ref_x:", _bulk_ref_vf[j1, i1])
    print("delv:", _bulk_delv_limit[j1, i1])
    print("std:", std_ng_opt_bulk[j1, i1])

    print("_g_sigma_lower:", _g_sigma_lower)
    print("g_sigma_upper:", _g_sigma_upper)
    print("vlos_lower:", _g_vlos_lower)
    print("vlos_upper:", _g_vlos_upper)

    _filter_bulk = ( \
            (sn_ng_opt_bulk[:, :] > peak_sn_limit) & \
            (x_ng_opt_bulk[:, :] >= _bulk_ref_vf[:, :] - _bulk_delv_limit[:, :]) & \
            (x_ng_opt_bulk[:, :] < _bulk_ref_vf[:, :] + _bulk_delv_limit[:, :]) & \
            (std_ng_opt_bulk[:, :] >= _g_sigma_lower) & \
            (std_ng_opt_bulk[:, :] < _g_sigma_upper))


    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    np.sqrt(2*np.pi)* std_ng_opt_bulk * p_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.0.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    _nparray_t = np.array([np.where( \
                                    (_filter_bulk) & \
                                    (p_ng_opt_bulk > 0.0) & \
                                    (std_ng_opt_bulk > 0.0), \
                                    np.sqrt(2*np.pi) * \
                                    p_ng_opt_bulk * \
                                    std_ng_opt_bulk * \
                                    ((p_ng_opt_bulk_e/p_ng_opt_bulk)**2 + \
                                    (std_ng_opt_bulk_e/std_ng_opt_bulk)**2)**0.5, \
                                   np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.0.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()


    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    x_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.1.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    x_ng_opt_bulk_e, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.1.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()


    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    std_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.2.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    std_ng_opt_bulk_e, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.2.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()

    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    bg_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.3.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    bg_ng_opt_bulk_e, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.3.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()


    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    rms_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.4.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    0.0, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.4.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()


    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    p_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.5.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    p_ng_opt_bulk_e, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.5.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()

    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    p_ng_opt_bulk / rms_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.6.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    0.0, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.6.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()

    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    ng_opt+1, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.7.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    0.0, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.7.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()




def extract_maps_gparam_sorted(_fitsarray_gfit_results2, params, _output_dir, _kin_comp, ng_opt, _hdu):

    max_ngauss = params['max_ngauss']
    nparams = (3*max_ngauss+2)
    peak_sn_limit = params['peak_sn_limit']

    if _kin_comp == 'sgfit':
        _g_vlos_lower = params['g_vlos_lower']
        _g_vlos_upper = params['g_vlos_upper']
        _g_sigma_lower = params['g_sigma_lower']
        _g_sigma_upper = params['g_sigma_upper']
        print("")
        print("| ... extracting sgfit results ... |")
        print("| vlos-lower: %1.f [km/s]" % _g_vlos_lower)
        print("| vlos-upper: %1.f [km/s]" % _g_vlos_upper)
        print("| vdisp-lower: %1.f [km/s]" % _g_sigma_lower)
        print("| vdisp-upper: %1.f [km/s]" % _g_sigma_upper)
        print("")
    elif _kin_comp == 'psgfit':
        _g_vlos_lower = params['g_vlos_lower']
        _g_vlos_upper = params['g_vlos_upper']
        _g_sigma_lower = params['g_sigma_lower']
        _g_sigma_upper = params['g_sigma_upper']
        print("")
        print("| ... extracting psgfit results ... |")
        print("| vlos-lower: %1.f [km/s]" % _g_vlos_lower)
        print("| vlos-upper: %1.f [km/s]" % _g_vlos_upper)
        print("| vdisp-lower: %1.f [km/s]" % _g_sigma_lower)
        print("| vdisp-upper: %1.f [km/s]" % _g_sigma_upper)
        print("")
    elif _kin_comp == 'cool':
        _g_vlos_lower = params['g_vlos_lower_cool']
        _g_vlos_upper = params['g_vlos_upper_cool']
        _g_sigma_lower = params['g_sigma_lower_cool']
        _g_sigma_upper = params['g_sigma_upper_cool']
        print("")
        print("| ... extracting kinematically cool components ... |")
        print("| vlos-lower: %1.f [km/s]" % _g_vlos_lower)
        print("| vlos-upper: %1.f [km/s]" % _g_vlos_upper)
        print("| vdisp-lower: %1.f [km/s]" % _g_sigma_lower)
        print("| vdisp-upper: %1.f [km/s]" % _g_sigma_upper)
        print("")
    elif _kin_comp == 'warm':
        _g_vlos_lower = params['g_vlos_lower_warm']
        _g_vlos_upper = params['g_vlos_upper_warm']
        _g_sigma_lower = params['_g_sigma_lower_warm']
        _g_sigma_upper = params['g_sigma_upper_warm']
        print("")
        print("| ... extracting kinematically warm components ... |")
        print("| vlos-lower: %1.f [km/s]" % _g_vlos_lower)
        print("| vlos-upper: %1.f [km/s]" % _g_vlos_upper)
        print("| vdisp-lower: %1.f [km/s]" % _g_sigma_lower)
        print("| vdisp-upper: %1.f [km/s]" % _g_sigma_upper)
        print("")
    elif _kin_comp == 'hot':
        _g_vlos_lower = params['g_vlos_lower_hot']
        _g_vlos_upper = params['g_vlos_upper_hot']
        _g_sigma_lower = params['_g_sigma_lower_hot']
        _g_sigma_upper = params['g_sigma_upper_hot']
        print("")
        print("| ... extracting kinematically hot components ... |")
        print("| vlos-lower: %1.f [km/s]" % _g_vlos_lower)
        print("| vlos-upper: %1.f [km/s]" % _g_vlos_upper)
        print("| vdisp-lower: %1.f [km/s]" % _g_sigma_lower)
        print("| vdisp-upper: %1.f [km/s]" % _g_sigma_upper)
        print("")
    else:
        _g_vlos_lower = params['vel_min']
        _g_vlos_upper = params['vel_max']
        _g_sigma_lower = params['g_sigma_lower']
        _g_sigma_upper = params['g_sigma_upper']





    i1 = params['_i0']
    j1 = params['_j0']

    nparams_step = 2*(3*max_ngauss+2) + (max_ngauss + 7)

    sn_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    x_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    std_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    p_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    bg_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    rms_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)


    for i in range(0, max_ngauss):
        for j in range(0, max_ngauss):

            sn_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i) & \
                                                    (_fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :] > 0), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0]
            sn_ng_opt_t[i, :, :] += sn_ng_opt_slice[i, j, :, :]
            sn_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    0.0, 0.0)])[0] # put zero to rms-error as it is not available
            sn_ng_opt_e_t[i, :, :] += sn_ng_opt_slice_e[i, j, :, :]

            x_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*i, :, :], 1E-7)])[0] # otherwise put 1E-7 used for summing up Xs below
            x_ng_opt_t[i, :, :] += x_ng_opt_slice[i, j, :, :]
            x_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 0 + 3*i, :, :], 1E-7)])[0] # otherwise put 1E-7 used for summing up Xs below
            x_ng_opt_e_t[i, :, :] += x_ng_opt_slice_e[i, j, :, :]

            std_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 3 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up stds below
            std_ng_opt_t[i, :, :] += std_ng_opt_slice[i, j, :, :]
            std_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 1 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up stds below
            std_ng_opt_e_t[i, :, :] += std_ng_opt_slice_e[i, j, :, :]

            p_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up ps below
            p_ng_opt_t[i, :, :] += p_ng_opt_slice[i, j, :, :]
            p_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 2 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up ps below
            p_ng_opt_e_t[i, :, :] += p_ng_opt_slice_e[i, j, :, :]

            bg_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 1, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up bgs below
            bg_ng_opt_t[i, :, :] += bg_ng_opt_slice[i, j, :, :]
            bg_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 1, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up bgs below
            bg_ng_opt_e_t[i, :, :] += bg_ng_opt_slice_e[i, j, :, :]

            rms_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0] # otherwise put 0.0
            rms_ng_opt_t[i, :, :] += rms_ng_opt_slice[i, j, :, :]
            rms_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    0.0, 0.0)])[0] # put zero to rms-error as it is not available
            rms_ng_opt_e_t[i, :, :] += rms_ng_opt_slice_e[i, j, :, :]


    sn_ng_opt = np.where(sn_ng_opt_t != 3*0.0, sn_ng_opt_t, -1E9)
    sn_ng_opt_e = np.where(sn_ng_opt_t != 3*0.0, sn_ng_opt_t, -1E9)

    x_ng_opt = np.where(x_ng_opt_t != 3*1E-7, x_ng_opt_t, -1E9)
    x_ng_opt_e = np.where(x_ng_opt_e_t != 3*1E-7, x_ng_opt_e_t, -1E9)

    std_ng_opt = np.where(std_ng_opt_t != 3*0.0, std_ng_opt_t, -1E9)
    std_ng_opt_e = np.where(std_ng_opt_e_t != 3*0.0, std_ng_opt_e_t, -1E9)

    p_ng_opt = np.where(p_ng_opt_t != 3*0.0, p_ng_opt_t, -1E9)
    p_ng_opt_e = np.where(p_ng_opt_e_t != 3*0.0, p_ng_opt_e_t, -1E9)

    bg_ng_opt = np.where(bg_ng_opt_t != 3*0.0, bg_ng_opt_t, -1E9)
    bg_ng_opt_e = np.where(bg_ng_opt_e_t != 3*0.0, bg_ng_opt_e_t, -1E9)

    rms_ng_opt = np.where(rms_ng_opt_t != 3*0.0, rms_ng_opt_t, -1E9)
    rms_ng_opt_e = np.where(rms_ng_opt_e_t != 3*0.0, rms_ng_opt_e_t, -1E9)


    i1 = params['_i0']
    j1 = params['_j0']


    if _kin_comp == 'sgfit' or _kin_comp == 'psgfit':
        _ng = 1 
    else:
        _ng = max_ngauss

    for i in range(0, _ng):



        if _kin_comp == 'psgfit': # if ng_opt == 0 <-- ngauss==1
            _filter = ( \
                    (ng_opt[:, :] == 0) & \
                    (sn_ng_opt[i, :, :] > peak_sn_limit) & \
                    (x_ng_opt[i,:, :] >= _g_vlos_lower) & \
                    (x_ng_opt[i,:, :] < _g_vlos_upper) & \
                    (std_ng_opt[i,:, :] >= _g_sigma_lower) & \
                    (std_ng_opt[i,:, :] < _g_sigma_upper))
        else:
            _filter = ( \
                    (ng_opt[:, :] >= i) & \
                    (sn_ng_opt[i, :, :] > peak_sn_limit) & \
                    (x_ng_opt[i,:, :] >= _g_vlos_lower) & \
                    (x_ng_opt[i,:, :] < _g_vlos_upper) & \
                    (std_ng_opt[i,:, :] >= _g_sigma_lower) & \
                    (std_ng_opt[i,:, :] < _g_sigma_upper))

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                        _filter, \
                                        np.sqrt(2*np.pi)*_fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] * \
                                        _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :], np.nan)])

        else: # if others
            _nparray_t = np.array([np.where( \
                                        _filter, \
                                        np.sqrt(2*np.pi)*std_ng_opt[i, :, :] * p_ng_opt[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.0.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()


        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                        _filter & \
                                        (_fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :] > 0.0) & \
                                        (_fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] > 0.0), \
                                        np.sqrt(2*np.pi) * \
                                        _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] * \
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :] * \
                                        ((_fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :])**2 + \
                                         (_fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 1 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :])**2)**0.5, \
                                        np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                        _filter & \
                                        (p_ng_opt[i, :, :] > 0.0) & \
                                        (std_ng_opt[i, :, :] > 0.0), \
                                        np.sqrt(2*np.pi) * \
                                        std_ng_opt[i, :, :] * \
                                        p_ng_opt_e[i, :, :] * \
                                        ((p_ng_opt_e[i, :, :] / p_ng_opt[i, :, :])**2 + \
                                         (std_ng_opt_e[i, :, :] / std_ng_opt[i, :, :])**2)**0.5, \
                                        np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.0.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            _fitsarray_gfit_results2[nparams_step*i + 2 + 3*i, :, :], np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            x_ng_opt[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.1.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()


        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 0 + 3*i, :, :], np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            x_ng_opt_e[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.1.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :], np.nan)])

        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            std_ng_opt[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.2.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 1 + 3*i, :, :], np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            std_ng_opt_e[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.2.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            _fitsarray_gfit_results2[nparams_step*i + 1 + 3*i, :, :], np.nan)]) #: this is wrong as bg is always at 1 

        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            bg_ng_opt[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.3.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()


        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 1, :, :], np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            bg_ng_opt_e[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.3.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            _fitsarray_gfit_results2[nparams_step*(i+1)-max_ngauss-7+i, :, :], np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            rms_ng_opt[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.4.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            0.0, np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            rms_ng_opt_e[i, :, :]  , np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.4.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :], np.nan)])

        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            p_ng_opt[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.5.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :], np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            p_ng_opt_e[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.5.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :] / \
                                            _fitsarray_gfit_results2[nparams_step*(i+1)-max_ngauss-7+i, :, :], np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            sn_ng_opt[i, :, :],  np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.6.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            0.0, np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            sn_ng_opt_e[i, :, :],  np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.6.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            ng_opt+1, np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            ng_opt+1, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.7.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            0.0, np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            0.0, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.7.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

    print('[—> fits written ..', _nparray_t.shape)



def extract_maps(_fitsarray_gfit_results2, params, _output_dir, _kin_comp, ng_opt, _hdu):

    max_ngauss = params['max_ngauss']
    nparams = (3*max_ngauss+2)
    peak_sn_limit = params['peak_sn_limit']

    if _kin_comp == 'sgfit':
        _g_vlos_lower = params['g_vlos_lower']
        _g_vlos_upper = params['g_vlos_upper']
        _g_sigma_lower = params['g_sigma_lower']
        _g_sigma_upper = params['g_sigma_upper']
        print("")
        print("| ... extracting sgfit results ... |")
        print("| vlos-lower: %1.f [km/s]" % _g_vlos_lower)
        print("| vlos-upper: %1.f [km/s]" % _g_vlos_upper)
        print("| vdisp-lower: %1.f [km/s]" % _g_sigma_lower)
        print("| vdisp-upper: %1.f [km/s]" % _g_sigma_upper)
        print("")
    elif _kin_comp == 'psgfit':
        _g_vlos_lower = params['g_vlos_lower']
        _g_vlos_upper = params['g_vlos_upper']
        _g_sigma_lower = params['g_sigma_lower']
        _g_sigma_upper = params['g_sigma_upper']
        print("")
        print("| ... extracting psgfit results ... |")
        print("| vlos-lower: %1.f [km/s]" % _g_vlos_lower)
        print("| vlos-upper: %1.f [km/s]" % _g_vlos_upper)
        print("| vdisp-lower: %1.f [km/s]" % _g_sigma_lower)
        print("| vdisp-upper: %1.f [km/s]" % _g_sigma_upper)
        print("")
    elif _kin_comp == 'cool':
        _g_vlos_lower = params['g_vlos_lower_cool']
        _g_vlos_upper = params['g_vlos_upper_cool']
        _g_sigma_lower = params['g_sigma_lower_cool']
        _g_sigma_upper = params['g_sigma_upper_cool']
        print("")
        print("| ... extracting kinematically cool components ... |")
        print("| vlos-lower: %1.f [km/s]" % _g_vlos_lower)
        print("| vlos-upper: %1.f [km/s]" % _g_vlos_upper)
        print("| vdisp-lower: %1.f [km/s]" % _g_sigma_lower)
        print("| vdisp-upper: %1.f [km/s]" % _g_sigma_upper)
        print("")
    elif _kin_comp == 'warm':
        _g_vlos_lower = params['g_vlos_lower_warm']
        _g_vlos_upper = params['g_vlos_upper_warm']
        _g_sigma_lower = params['g_sigma_lower_warm']
        _g_sigma_upper = params['g_sigma_upper_warm']
        print("")
        print("| ... extracting kinematically warm components ... |")
        print("| vlos-lower: %1.f [km/s]" % _g_vlos_lower)
        print("| vlos-upper: %1.f [km/s]" % _g_vlos_upper)
        print("| vdisp-lower: %1.f [km/s]" % _g_sigma_lower)
        print("| vdisp-upper: %1.f [km/s]" % _g_sigma_upper)
        print("")
    elif _kin_comp == 'hot':
        _g_vlos_lower = params['g_vlos_lower_hot']
        _g_vlos_upper = params['g_vlos_upper_hot']
        _g_sigma_lower = params['g_sigma_lower_hot']
        _g_sigma_upper = params['g_sigma_upper_hot']
        print("")
        print("| ... extracting kinematically hot components ... |")
        print("| vlos-lower: %1.f [km/s]" % _g_vlos_lower)
        print("| vlos-upper: %1.f [km/s]" % _g_vlos_upper)
        print("| vdisp-lower: %1.f [km/s]" % _g_sigma_lower)
        print("| vdisp-upper: %1.f [km/s]" % _g_sigma_upper)
        print("")
    elif _kin_comp == 'hvc':
        _g_vlos_lower = params['g_vlos_lower_hvc']
        _g_vlos_upper = params['g_vlos_upper_hvc']
        _g_sigma_lower = params['g_sigma_lower_hvc']
        _g_sigma_upper = params['g_sigma_upper_hvc']
        print("")
        print("| ... extracting kinematically hvc components ... |")
        print("| vlos-lower: %1.f [km/s]" % _g_vlos_lower)
        print("| vlos-upper: %1.f [km/s]" % _g_vlos_upper)
        print("| vdisp-lower: %1.f [km/s]" % _g_sigma_lower)
        print("| vdisp-upper: %1.f [km/s]" % _g_sigma_upper)
        print("")
    else:
        _g_vlos_lower = params['vel_min']
        _g_vlos_upper = params['vel_max']
        _g_sigma_lower = params['g_sigma_lower']
        _g_sigma_upper = params['g_sigma_upper']





    i1 = params['_i0']
    j1 = params['_j0']

    nparams_step = 2*(3*max_ngauss+2) + (max_ngauss + 7)

    sn_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    x_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    std_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    p_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    bg_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    rms_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)


    for i in range(0, max_ngauss):
        for j in range(0, max_ngauss):

            sn_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i) & \
                                                    (_fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :] > 0), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0]
            sn_ng_opt_t[i, :, :] += sn_ng_opt_slice[i, j, :, :]
            sn_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    0.0, 0.0)])[0] # put zero to rms-error as it is not available
            sn_ng_opt_e_t[i, :, :] += sn_ng_opt_slice_e[i, j, :, :]

            x_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*i, :, :], 1E-7)])[0] # otherwise put 1E-7 used for summing up Xs below
            x_ng_opt_t[i, :, :] += x_ng_opt_slice[i, j, :, :]
            x_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 0 + 3*i, :, :], 1E-7)])[0] # otherwise put 1E-7 used for summing up Xs below
            x_ng_opt_e_t[i, :, :] += x_ng_opt_slice_e[i, j, :, :]

            std_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 3 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up stds below
            std_ng_opt_t[i, :, :] += std_ng_opt_slice[i, j, :, :]
            std_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 1 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up stds below
            std_ng_opt_e_t[i, :, :] += std_ng_opt_slice_e[i, j, :, :]

            p_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up ps below
            p_ng_opt_t[i, :, :] += p_ng_opt_slice[i, j, :, :]
            p_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 2 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up ps below
            p_ng_opt_e_t[i, :, :] += p_ng_opt_slice_e[i, j, :, :]

            bg_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 1, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up bgs below
            bg_ng_opt_t[i, :, :] += bg_ng_opt_slice[i, j, :, :]
            bg_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 1, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up bgs below
            bg_ng_opt_e_t[i, :, :] += bg_ng_opt_slice_e[i, j, :, :]

            rms_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0] # otherwise put 0.0
            rms_ng_opt_t[i, :, :] += rms_ng_opt_slice[i, j, :, :]
            rms_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    0.0, 0.0)])[0] # put zero to rms-error as it is not available
            rms_ng_opt_e_t[i, :, :] += rms_ng_opt_slice_e[i, j, :, :]


    sn_ng_opt = np.where(sn_ng_opt_t != 3*0.0, sn_ng_opt_t, -1E9)
    sn_ng_opt_e = np.where(sn_ng_opt_t != 3*0.0, sn_ng_opt_t, -1E9)

    x_ng_opt = np.where(x_ng_opt_t != 3*1E-7, x_ng_opt_t, -1E9)
    x_ng_opt_e = np.where(x_ng_opt_e_t != 3*1E-7, x_ng_opt_e_t, -1E9)

    std_ng_opt = np.where(std_ng_opt_t != 3*0.0, std_ng_opt_t, -1E9)
    std_ng_opt_e = np.where(std_ng_opt_e_t != 3*0.0, std_ng_opt_e_t, -1E9)

    p_ng_opt = np.where(p_ng_opt_t != 3*0.0, p_ng_opt_t, -1E9)
    p_ng_opt_e = np.where(p_ng_opt_e_t != 3*0.0, p_ng_opt_e_t, -1E9)

    bg_ng_opt = np.where(bg_ng_opt_t != 3*0.0, bg_ng_opt_t, -1E9)
    bg_ng_opt_e = np.where(bg_ng_opt_e_t != 3*0.0, bg_ng_opt_e_t, -1E9)

    rms_ng_opt = np.where(rms_ng_opt_t != 3*0.0, rms_ng_opt_t, -1E9)
    rms_ng_opt_e = np.where(rms_ng_opt_e_t != 3*0.0, rms_ng_opt_e_t, -1E9)


    i1 = params['_i0']
    j1 = params['_j0']


    if _kin_comp == 'sgfit' or _kin_comp == 'psgfit':
        _ng = 1 
    else:
        _ng = max_ngauss

    for i in range(0, _ng):



        if _kin_comp == 'psgfit': # if ng_opt == 0 <-- ngauss==1
            _filter = ( \
                    (ng_opt[:, :] == 0) & \
                    (sn_ng_opt[i, :, :] > peak_sn_limit) & \
                    (x_ng_opt[i,:, :] >= _g_vlos_lower) & \
                    (x_ng_opt[i,:, :] < _g_vlos_upper) & \
                    (std_ng_opt[i,:, :] >= _g_sigma_lower) & \
                    (std_ng_opt[i,:, :] < _g_sigma_upper))
        else:
            _filter = ( \
                    (ng_opt[:, :] >= i) & \
                    (sn_ng_opt[i, :, :] > peak_sn_limit) & \
                    (x_ng_opt[i,:, :] >= _g_vlos_lower) & \
                    (x_ng_opt[i,:, :] < _g_vlos_upper) & \
                    (std_ng_opt[i,:, :] >= _g_sigma_lower) & \
                    (std_ng_opt[i,:, :] < _g_sigma_upper))

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                        _filter, \
                                        np.sqrt(2*np.pi)*_fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] * \
                                        _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :], np.nan)])

        else: # if others
            _nparray_t = np.array([np.where( \
                                        _filter, \
                                        np.sqrt(2*np.pi)*std_ng_opt[i, :, :] * p_ng_opt[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=0)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.0.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()


        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                        _filter & \
                                        (_fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :] > 0.0) & \
                                        (_fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] > 0.0), \
                                        np.sqrt(2*np.pi) * \
                                        _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] * \
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :] * \
                                        ((_fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :])**2 + \
                                         (_fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 1 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :])**2)**0.5, \
                                        np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                        _filter & \
                                        (p_ng_opt[i, :, :] > 0.0) & \
                                        (std_ng_opt[i, :, :] > 0.0), \
                                        np.sqrt(2*np.pi) * \
                                        std_ng_opt[i, :, :] * \
                                        p_ng_opt_e[i, :, :] * \
                                        ((p_ng_opt_e[i, :, :] / p_ng_opt[i, :, :])**2 + \
                                         (std_ng_opt_e[i, :, :] / std_ng_opt[i, :, :])**2)**0.5, \
                                        np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=0)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.0.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            _fitsarray_gfit_results2[nparams_step*i + 2 + 3*i, :, :], np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            x_ng_opt[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=1)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.1.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()


        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 0 + 3*i, :, :], np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            x_ng_opt_e[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=1)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.1.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :], np.nan)])

        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            std_ng_opt[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=2)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.2.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 1 + 3*i, :, :], np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            std_ng_opt_e[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=2)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.2.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            _fitsarray_gfit_results2[nparams_step*i + 1 + 3*i, :, :], np.nan)]) #: this is wrong as bg is always at 1 

        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            bg_ng_opt[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=3)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.3.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()


        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 1, :, :], np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            bg_ng_opt_e[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=3)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.3.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            _fitsarray_gfit_results2[nparams_step*(i+1)-max_ngauss-7+i, :, :], np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            rms_ng_opt[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=4)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.4.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            0.0, np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            rms_ng_opt_e[i, :, :]  , np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=4)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.4.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :], np.nan)])

        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            p_ng_opt[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=5)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.5.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :], np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            p_ng_opt_e[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=5)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.5.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :] / \
                                            _fitsarray_gfit_results2[nparams_step*(i+1)-max_ngauss-7+i, :, :], np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            sn_ng_opt[i, :, :],  np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=6)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.6.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            0.0, np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            sn_ng_opt_e[i, :, :],  np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=6)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.6.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            ng_opt+1, np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            ng_opt+1, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=7)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.7.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            0.0, np.nan)])
        else: # if others
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            0.0, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=7)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.7.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

    print('[—> fits written ..', _nparray_t.shape)


def extract_maps_ngfit(_fitsarray_gfit_results2, params, _output_dir, _kin_comp, ng_opt, _hdu):

    max_ngauss = params['max_ngauss']
    nparams = (3*max_ngauss+2)
    peak_sn_limit = params['peak_sn_limit']

    if _kin_comp == 'ngfit':
        _g_vlos_lower = -1E10 # small value
        _g_vlos_upper = 1E10 # large value
        _g_sigma_lower = -1E10 # small value
        _g_sigma_upper = 1E10 # large value
        print("")
        print("| ... extracting all the Gaussian components ... |")
        print("| vlos-lower: %1.f [km/s]" % _g_vlos_lower)
        print("| vlos-upper: %1.f [km/s]" % _g_vlos_upper)
        print("| vdisp-lower: %1.f [km/s]" % _g_sigma_lower)
        print("| vdisp-upper: %1.f [km/s]" % _g_sigma_upper)
        print("")
    else:
        _g_vlos_lower = params['g_vlos_lower']
        _g_vlos_upper = params['g_vlos_upper']
        _g_sigma_lower = params['g_sigma_lower']
        _g_sigma_upper = params['g_sigma_upper']






    nparams_step = 2*(3*max_ngauss+2) + (max_ngauss + 7)

    sn_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    x_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    std_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    p_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    bg_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    rms_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)


    for i in range(0, max_ngauss):
        for j in range(0, max_ngauss):

            sn_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j >= i) & \
                                                    (_fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :] > 0), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0]
            sn_ng_opt_t[i, :, :] += sn_ng_opt_slice[i, j, :, :]
            sn_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    0.0, 0.0)])[0] # put zero to rms-error as it is not available
            sn_ng_opt_e_t[i, :, :] += sn_ng_opt_slice_e[i, j, :, :]

            x_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*i, :, :], 1E-7)])[0] # otherwise put 1E-7 used for summing up Xs below
            x_ng_opt_t[i, :, :] += x_ng_opt_slice[i, j, :, :]
            x_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 0 + 3*i, :, :], 1E-7)])[0] # otherwise put 1E-7 used for summing up Xs below
            x_ng_opt_e_t[i, :, :] += x_ng_opt_slice_e[i, j, :, :]

            std_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 3 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up stds below
            std_ng_opt_t[i, :, :] += std_ng_opt_slice[i, j, :, :]
            std_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 1 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up stds below
            std_ng_opt_e_t[i, :, :] += std_ng_opt_slice_e[i, j, :, :]

            p_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up ps below
            p_ng_opt_t[i, :, :] += p_ng_opt_slice[i, j, :, :]
            p_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 2 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up ps below
            p_ng_opt_e_t[i, :, :] += p_ng_opt_slice_e[i, j, :, :]

            bg_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*0 + 1, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up bgs below
            bg_ng_opt_t[i, :, :] += bg_ng_opt_slice[i, j, :, :]
            bg_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 1, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up bgs below
            bg_ng_opt_e_t[i, :, :] += bg_ng_opt_slice_e[i, j, :, :]

            rms_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0] # otherwise put 0.0
            rms_ng_opt_t[i, :, :] += rms_ng_opt_slice[i, j, :, :]
            rms_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    0.0, 0.0)])[0] # put zero to rms-error as it is not available
            rms_ng_opt_e_t[i, :, :] += rms_ng_opt_slice_e[i, j, :, :]


    sn_ng_opt = np.where(sn_ng_opt_t != 3*0.0, sn_ng_opt_t, -1E9)
    sn_ng_opt_e = np.where(sn_ng_opt_t != 3*0.0, sn_ng_opt_t, -1E9)

    x_ng_opt = np.where(x_ng_opt_t != 3*1E-7, x_ng_opt_t, -1E9)
    x_ng_opt_e = np.where(x_ng_opt_e_t != 3*1E-7, x_ng_opt_e_t, -1E9)

    std_ng_opt = np.where(std_ng_opt_t != 3*0.0, std_ng_opt_t, -1E9)
    std_ng_opt_e = np.where(std_ng_opt_e_t != 3*0.0, std_ng_opt_e_t, -1E9)

    p_ng_opt = np.where(p_ng_opt_t != 3*0.0, p_ng_opt_t, -1E9)
    p_ng_opt_e = np.where(p_ng_opt_e_t != 3*0.0, p_ng_opt_e_t, -1E9)

    bg_ng_opt = np.where(bg_ng_opt_t != 3*0.0, bg_ng_opt_t, -1E9)
    bg_ng_opt_e = np.where(bg_ng_opt_e_t != 3*0.0, bg_ng_opt_e_t, -1E9)

    rms_ng_opt = np.where(rms_ng_opt_t != 3*0.0, rms_ng_opt_t, -1E9)
    rms_ng_opt_e = np.where(rms_ng_opt_e_t != 3*0.0, rms_ng_opt_e_t, -1E9)



    i1 = params['_i0']
    j1 = params['_j0']


    _ng = max_ngauss

    for i in range(0, _ng):



        if _kin_comp == 'ngfit': #
            _filter = ( \
                    (ng_opt[:, :] >= i) & \
                    (sn_ng_opt[i, :, :] > peak_sn_limit) & \
                    (x_ng_opt[i,:, :] >= _g_vlos_lower) & \
                    (x_ng_opt[i,:, :] < _g_vlos_upper) & \
                    (std_ng_opt[i,:, :] >= _g_sigma_lower) & \
                    (std_ng_opt[i,:, :] < _g_sigma_upper))

        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        np.sqrt(2*np.pi)*_fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] * \
                                        _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=0)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.0.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()


        _nparray_t = np.array([np.where( \
                                        _filter & \
                                        (_fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :] > 0.0) & \
                                        (_fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] > 0.0), \
                                        np.sqrt(2*np.pi) * \
                                        _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] * \
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :] * \
                                        ((_fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :])**2 + \
                                         (_fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 1 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :])**2)**0.5, \
                                        np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=0)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.0.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=1)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.1.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()


        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 0 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=1)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.1.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=2)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.2.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 1 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=2)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.2.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        _fitsarray_gfit_results2[nparams_step*0 + 1, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=3)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.3.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 1, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=3)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.3.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        _fitsarray_gfit_results2[nparams_step*(i+1)-max_ngauss-7+i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=4)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.4.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        0.0, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=4)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.4.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=5)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.5.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=5)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.5.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()


        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :] / \
                                        _fitsarray_gfit_results2[nparams_step*(i+1)-max_ngauss-7+i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=6)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.6.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        0.0, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=6)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.6.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        ng_opt+1, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=7)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.7.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        0.0, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #update_header_cube_to_2d(_hdulist_nparray, _hdu)
        update_header_cube_to_2d(_hdulist_nparray, _hdu, _params=params, bunit_code=7)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.7.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

    print('[—> fits written ..', _nparray_t.shape)


def extract_maps_old(_fitsarray_gfit_results2, peak_sn_limit, params, ngauss, _output_dir, _sgfit):

    ngauss = 1 # for sgfit
    max_ngauss = params['max_ngauss']
    nparams_end = 2*(3*max_ngauss+2)
    sn = _fitsarray_gfit_results2[4, :, :] / _fitsarray_gfit_results2[nparams_end, :, :]
    sn_mask = np.where(sn > peak_sn_limit)

    print(peak_sn_limit)


    for i in range(0, ngauss):
        _nparray_t = np.array([np.where(sn > peak_sn_limit, np.sqrt(2*np.pi)*_fitsarray_gfit_results2[3, :, :]*_fitsarray_gfit_results2[4, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.0.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()
        _nparray_t = np.array([np.where(sn > peak_sn_limit, _fitsarray_gfit_results2[0, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.0.e.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where(sn > peak_sn_limit, _fitsarray_gfit_results2[2, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.1.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()
        _nparray_t = np.array([np.where(sn > peak_sn_limit, _fitsarray_gfit_results2[2+3*max_ngauss+2, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.1.e.fits' % (_output_dir, _sgfit, _sgfit,  max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where(sn > peak_sn_limit, _fitsarray_gfit_results2[3, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.2.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()
        _nparray_t = np.array([np.where(sn > peak_sn_limit, _fitsarray_gfit_results2[2+3*max_ngauss+3, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.2.e.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where(sn > peak_sn_limit, _fitsarray_gfit_results2[1, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.3.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()
        _nparray_t = np.array([np.where(sn > peak_sn_limit, _fitsarray_gfit_results2[2+3*max_ngauss+1, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.3.e.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where(sn > peak_sn_limit, _fitsarray_gfit_results2[nparams_end, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.4.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()
        _nparray_t = np.array([np.where(sn > peak_sn_limit, 0*_fitsarray_gfit_results2[nparams_end, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.4.e.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where(sn > peak_sn_limit, _fitsarray_gfit_results2[4, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.5.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()
        _nparray_t = np.array([np.where(sn > peak_sn_limit, 0*_fitsarray_gfit_results2[2+3*max_ngauss+4, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.5.e.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()

        _nparray_t = np.array([np.where(sn > peak_sn_limit, _fitsarray_gfit_results2[4, :, :]/_fitsarray_gfit_results2[nparams_end, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.6.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()
        _nparray_t = np.array([np.where(sn > peak_sn_limit, 0*_fitsarray_gfit_results2[4, :, :]/_fitsarray_gfit_results2[nparams_end, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.6.e.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()



    print("gogo")
    print(_nparray_t.shape)
    print("gogo")







def g1_opt_bf(_fitsarray_gfit_results2):
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    return g_opt


def g2_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, sn_pass_ng_opt, bf_limit):
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g1_sorted = g_num_sort[0, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]

    g1_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) > np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g_opt[g1_cond1] += 1
    g1_0 = np.array([np.where(g_opt > 0, g1_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g2_sorted = g_num_sort[1, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g2_sorted[np.newaxis], axis=0)[0]

    g2_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond2 = np.where((g_num_sort[0, :, :] > g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g_opt[g2_cond1] += 1
    g_opt[g2_cond2] += 1
    g2_0 = np.array([np.where(g_opt > 1, g2_sorted, 0)])
    
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g1_sorted = g_num_sort[0, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]

    g1_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond2 = np.where((g_num_sort[0, :, :] < g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g_opt[g1_cond1] += 1
    g_opt[g1_cond2] += 1
    g1_1 = np.array([np.where(g_opt > 1, g1_sorted, 0)])

    g2_opt = g1_0 + g1_1 \
            + g2_0
    return g2_opt



def g3_ms_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gg, gl, g3): # gg > gl

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    gl_sorted = g_num_sort[gl, :, :]
    if gl > gg: # as z is sorted so always zg > zl > z3 
        gl_cond1 = np.where((bevidences_sort[gg, :, :] - bevidences_sort[gl, :, :]) < np.log10(bf_limit))
    else:
        gl_cond1 = np.where((bevidences_sort[gl, :, :] - bevidences_sort[gg, :, :]) < np.log10(bf_limit))

    gl_cond2 = np.where(g_num_sort[gl, :, :] < g_num_sort[gg, :, :])
    gl_cond3 = np.where((bevidences_sort[gl, :, :] - bevidences_sort[g3, :, :]) > np.log(bf_limit))
    
    g_opt[gl_cond1] += 1
    g_opt[gl_cond2] += 1
    g_opt[gl_cond3] += 1

    gl_0 = np.array([np.where(g_opt > 2, gl_sorted, 0)])


    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g3_sorted = g_num_sort[g3, :, :]
    if gl > gg: # as z is sorted so always zg > zl > z3 
        g3_cond1 = np.where((bevidences_sort[gg, :, :] - bevidences_sort[gl, :, :]) < np.log(bf_limit))
    else:
        g3_cond1 = np.where((bevidences_sort[gl, :, :] - bevidences_sort[gg, :, :]) < np.log(bf_limit))

    g3_cond2 = np.where(g_num_sort[gl, :, :] < g_num_sort[gg, :, :])
    g3_cond3 = np.where((bevidences_sort[gl, :, :] - bevidences_sort[g3, :, :]) < np.log(bf_limit))
    g3_cond4 = np.where(g_num_sort[gl, :, :] > g_num_sort[g3, :, :])
    
    g_opt[g3_cond1] += 1
    g_opt[g3_cond2] += 1
    g_opt[g3_cond3] += 1
    g_opt[g3_cond4] += 1
    g3_0 = np.array([np.where(g_opt > 3, g3_sorted, 0)])
    
    
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    gl_sorted = g_num_sort[gl, :, :]
    if gl > gg: # as z is sorted so always zg > zl > z3 
        gl_cond1 = np.where((bevidences_sort[gg, :, :] - bevidences_sort[gl, :, :]) < np.log(bf_limit))
    else:
        gl_cond1 = np.where((bevidences_sort[gl, :, :] - bevidences_sort[gg, :, :]) < np.log(bf_limit))
    gl_cond2 = np.where(g_num_sort[gl, :, :] < g_num_sort[gg, :, :])
    gl_cond3 = np.where((bevidences_sort[gl, :, :] - bevidences_sort[g3, :, :]) < np.log(bf_limit))
    gl_cond4 = np.where(g_num_sort[gl, :, :] < g_num_sort[g3, :, :])
    
    g_opt[gl_cond1] += 1
    g_opt[gl_cond2] += 1
    g_opt[gl_cond3] += 1
    g_opt[gl_cond4] += 1
    gl_1 = np.array([np.where(g_opt > 3, gl_sorted, 0)])
    return gl_0, g3_0, gl_1







def math_opr1(_array_3d, g1, g2, opr):
    rel = { \
            '0': operator.gt, # >
            '1': operator.lt, # <
            '>=': operator.ge,
            '<=': operator.le,
            '==': operator.eq}
    return rel[opr](_array_3d[g1, :, :], _array_3d[g2, :, :])


def math_opr2(_array_3d, g1, g2, opr, bf_limit):
    rel = { \
            '0': operator.gt, # >
            '1': operator.lt, # <
            '>=': operator.ge,
            '<=': operator.le,
            '==': operator.eq}
    return rel[opr]( (_array_3d[g1, :, :] - _array_3d[g2, :, :]), np.log10(bf_limit) )




def g3_ms_bf_cond1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, \
                    g_sorted_n, \
                    n10, n11, opr1):



    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_sorted = g_num_sort[g_sorted_n, :, :]

    g_cond1_opr = math_opr2(bevidences_sort, n10, n11, opr1, bf_limit)
    g_cond1 = np.where(g_cond1_opr)

    g_opt[g_cond1] += 1
    g_opt_result_cond1 = np.array([np.where(g_opt > 0, g_sorted, 0)])

    return g_opt_result_cond1



def g3_ms_bf_cond3(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, \
                    g_sorted_n, \
                    n10, n11, opr1, \
                    n20, n21, opr2, \
                    n30, n31, opr3):




    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_sorted = g_num_sort[g_sorted_n, :, :]

    g_cond1_opr = math_opr2(bevidences_sort, n10, n11, opr1, bf_limit)
    g_cond1 = np.where(g_cond1_opr)

    g_cond2_opr = math_opr1(g_num_sort, n20, n21, opr2)
    g_cond2 = np.where(g_cond2_opr)

    g_cond3_opr = math_opr2(bevidences_sort, n30, n31, opr3, bf_limit)
    g_cond3 = np.where(g_cond3_opr)

    g_opt[g_cond1] += 1
    g_opt[g_cond2] += 1
    g_opt[g_cond3] += 1
    g_opt_result_cond3 = np.array([np.where(g_opt > 2, g_sorted, 0)])

    return g_opt_result_cond3


def g3_ms_bf_cond4(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, \
                    g_sorted_n, \
                    n10, n11, opr1, \
                    n20, n21, opr2, \
                    n30, n31, opr3, \
                    n40, n41, opr4):





    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_sorted = g_num_sort[g_sorted_n, :, :]

    g_cond1_opr = math_opr2(bevidences_sort, n10, n11, opr1, bf_limit)
    g_cond1 = np.where(g_cond1_opr)

    g_cond2_opr = math_opr1(g_num_sort, n20, n21, opr2)
    g_cond2 = np.where(g_cond2_opr)

    g_cond3_opr = math_opr2(bevidences_sort, n30, n31, opr3, bf_limit)
    g_cond3 = np.where(g_cond3_opr)

    g_cond4_opr = math_opr2(bevidences_sort, n40, n41, opr4, bf_limit)
    g_cond4 = np.where(g_cond4_opr)

    g_opt[g_cond1] += 1
    g_opt[g_cond2] += 1
    g_opt[g_cond3] += 1
    g_opt[g_cond4] += 1
    g_opt_result_cond4 = np.array([np.where(g_opt > 3, g_sorted, 0)])

    return g_opt_result_cond4



def g4_ms_bf_cond5(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, \
                    g_sorted_n, \
                    n10, n11, opr1, \
                    n20, n21, opr2, \
                    n30, n31, opr3, \
                    n40, n41, opr4, \
                    n50, n51, opr5):

 


    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_sorted = g_num_sort[g_sorted_n, :, :]

    g_cond1_opr = math_opr2(bevidences_sort, n10, n11, opr1, bf_limit)
    g_cond1 = np.where(g_cond1_opr)

    g_cond2_opr = math_opr1(g_num_sort, n20, n21, opr2)
    g_cond2 = np.where(g_cond2_opr)

    g_cond3_opr = math_opr2(bevidences_sort, n30, n31, opr3, bf_limit)
    g_cond3 = np.where(g_cond3_opr)

    g_cond4_opr = math_opr1(g_num_sort, n40, n41, opr4)
    g_cond4 = np.where(g_cond4_opr)

    g_cond5_opr = math_opr2(bevidences_sort, n50, n51, opr5, bf_limit)
    g_cond5 = np.where(g_cond5_opr)

    g_opt[g_cond1] += 1
    g_opt[g_cond2] += 1
    g_opt[g_cond3] += 1
    g_opt[g_cond4] += 1
    g_opt[g_cond5] += 1
    g_opt_result_cond5 = np.array([np.where(g_opt > 4, g_sorted, 0)])

    return g_opt_result_cond5


def g4_ms_bf_cond6(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, \
                    g_sorted_n, \
                    n10, n11, opr1, \
                    n20, n21, opr2, \
                    n30, n31, opr3, \
                    n40, n41, opr4, \
                    n50, n51, opr5, \
                    n60, n61, opr6):

 


    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_sorted = g_num_sort[g_sorted_n, :, :]

    g_cond1_opr = math_opr2(bevidences_sort, n10, n11, opr1, bf_limit)
    g_cond1 = np.where(g_cond1_opr)

    g_cond2_opr = math_opr1(g_num_sort, n20, n21, opr2)
    g_cond2 = np.where(g_cond2_opr)

    g_cond3_opr = math_opr2(bevidences_sort, n30, n31, opr3, bf_limit)
    g_cond3 = np.where(g_cond3_opr)

    g_cond4_opr = math_opr1(g_num_sort, n40, n41, opr4)
    g_cond4 = np.where(g_cond4_opr)

    g_cond5_opr = math_opr2(bevidences_sort, n50, n51, opr5, bf_limit)
    g_cond5 = np.where(g_cond5_opr)

    g_cond6_opr = math_opr1(g_num_sort, n60, n61, opr6)
    g_cond6 = np.where(g_cond6_opr)
    
    g_opt[g_cond1] += 1
    g_opt[g_cond2] += 1
    g_opt[g_cond3] += 1
    g_opt[g_cond4] += 1
    g_opt[g_cond5] += 1
    g_opt[g_cond6] += 1
    g_opt_result_cond6 = np.array([np.where(g_opt > 5, g_sorted, 0)])

    return g_opt_result_cond6



def g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, \
                    g_sorted_n, \
                    _cond, \
                    _cond_N, \
                    g123_sn_pass):

 



    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_sorted = g_num_sort[g_sorted_n, :, :]

    if _cond_N == 0: # single s1, s2, s3 ... :: s/n flag only,  no bf_limit
            g_cond_sn = np.where(g123_sn_pass) # g123_sn_pass --> g1, g2, g3, ... sn_pass
            g_opt[g_cond_sn] += 1
            _cond_N = 1 # update for np.where(g_opt > _cond_N-1) below : "g_opt > _cond_N-1"

    elif _cond_N == 1:
        for i in range(0, int(_cond_N/2 + 0.5)):
            g_cond_z_opr = math_opr2(bevidences_sort, _cond[i*2][0], _cond[i*2][1], _cond[i*2][2], bf_limit)

            g_cond_z = np.where(g_cond_z_opr & g123_sn_pass) # 1 only when satisfying the dual conditions = BF_LIMIT + SN_PASS
            g_opt[g_cond_z] += 1


    elif _cond_N > 1:
        for i in range(0, int(_cond_N/2 + 0.5)):
            g_cond_z_opr = math_opr2(bevidences_sort, _cond[i*2][0], _cond[i*2][1], _cond[i*2][2], bf_limit)
            g_cond_z = np.where(g_cond_z_opr & g123_sn_pass)
            g_opt[g_cond_z] += 1

        for i in range(0, int(_cond_N/2)):
            g_cond_n_opr = math_opr1(g_num_sort, _cond[i*2+1][0], _cond[i*2+1][1], _cond[i*2+1][2])
            g_cond_n = np.where(g_cond_n_opr & g123_sn_pass)
            g_opt[g_cond_n] += 1

    g_opt_result_cond_N = np.array([np.where(g_opt > _cond_N-1, g_sorted, -10)]) # -10 flag


    return g_opt_result_cond_N


def g3_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit):
    
    _cond_N = 1
    _cond1 = ([0, 1, '>'], )
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, 0, _cond1, _cond_N)

    _cond_N = 3
    _cond3 = ([0, 1, '<'], [0, 1, '>'], [1, 2, '>'])
    g2_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, 1, _cond3, _cond_N)

    _cond_N = 4
    _cond4 = ([0, 1, '<'], [0, 1, '>'], [1, 2, '<'], [1, 2, '>'])
    g3_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, 2, _cond4, _cond_N)
    
    _cond_N = 4
    _cond4 = ([0, 1, '<'], [0, 1, '>'], [1, 2, '<'], [1, 2, '<'])
    g2_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, 1, _cond4, _cond_N)
    
    _cond_N = 3
    _cond3 = ([0, 1, '<'], [0, 1, '<'], [0, 2, '>'])
    g1_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, 0, _cond3, _cond_N)


    _cond_N = 4
    _cond4 = ([0, 1, '<'], [0, 1, '<'], [0, 2, '<'], [0, 2, '<'])
    g3_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, 2, _cond4, _cond_N)
    
    _cond_N = 4
    _cond4 = ([0, 1, '<'], [0, 1, '<'], [0, 2, '<'], [0, 2, '<'])
    g1_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, 0, _cond4, _cond_N)

    g3_opt = g1_0 + g1_1 + g1_2 \
            + g2_0 + g2_1 \
            + g3_0 + g3_1

    return g3_opt







def g2_opt_bf_snp(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, sn_pass_ng_opt, g01, g02):

    g1_sorted = g_num_sort[g01, :, :]
    g1_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]
    g1_sn_flag = np.array([np.where(g1_sn_pass_ng_opt_value == (g1_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

    g2_sorted = g_num_sort[g02, :, :]
    g2_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g2_sorted[np.newaxis], axis=0)[0]
    g2_sn_flag = np.array([np.where(g2_sn_pass_ng_opt_value == (g2_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

    g1_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0)), 1, 0)])[0]
    g2_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1)), 1, 0)])[0]

    g0_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0)), 1, 0)])[0]
    g12_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1)), 1, 0)])[0]

    i1 = _params['_i0']
    j1 = _params['_j0']
    g12_sn_flag = g1_sn_flag + g2_sn_flag
    print("ng-12: ", "sn_pass: ", np.where(g12_sn_flag == 2))
    print("")
    print("ng-1: ", g1_sorted[j1, i1], "sn_flag: ", g1_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g1_sn_flag[j1, i1])
    print("ng-2: ", g2_sorted[j1, i1], "sn_flag: ", g2_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g2_sn_flag[j1, i1])

    _g2_opt_bf_snp_1  = g2_opt_bf_snp1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1_sn_pass, g01)
    _g2_opt_bf_snp_2  = g2_opt_bf_snp1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g2_sn_pass, g02)

    _g2_opt_bf_snp_0  = g2_opt_bf_snp0(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g0_sn_pass)
    _g2_opt_bf_snp_12  = g2_opt_bf_snp2(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g12_sn_pass, g01, g02)


    print(g12_sn_pass[j1, i1])
    print(_g2_opt_bf_snp_12[0, j1, i1])

    _g2_opt_t1 = np.array([np.where(_g2_opt_bf_snp_0 > -1, _g2_opt_bf_snp_0, -1)][0])
    _g2_opt_t2 = np.array([np.where(_g2_opt_bf_snp_1 > -1, _g2_opt_bf_snp_1, _g2_opt_t1)][0])
    _g2_opt_t3 = np.array([np.where(_g2_opt_bf_snp_2 > -1, _g2_opt_bf_snp_2, _g2_opt_t2)][0])
    _g2_opt_t4 = np.array([np.where(_g2_opt_bf_snp_12 > -1, _g2_opt_bf_snp_12, _g2_opt_t3)][0])

    print(_g2_opt_t4[0, j1, i1])
    return _g2_opt_t4



def g2_opt_bf_snp2(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g12_sn_pass, g01, g02): # g01 < g02 : g indices (0, 1, ...)
    g_sort_n = g01
    _cond_N = 1
    _cond1 = ([g01, g02, '0'], )
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12_sn_pass)


    g_sort_n = g02
    _cond_N = 2
    _cond2 = ([g01, g02, '1'], [g01, g02, '0'])
    g2_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond2, _cond_N, g12_sn_pass)


    g_sort_n = g01
    _cond_N = 2
    _cond2 = ([g01, g02, '1'], [g01, g02, '1'])
    g1_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond2, _cond_N, g12_sn_pass)

    g2_opt_snp2 = g1_0 + g1_1 \
            + g2_0 + 10*2 # -10 flag

    return g2_opt_snp2


def g2_opt_bf_snp1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1_sn_pass, g01): # g01 : g indices (0, 1, 2, ...)
    g_sort_n = g01
    _cond_N = 0
    _cond0 = ([g01, g01, '='], ) # dummy condition
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond0, _cond_N, g1_sn_pass)
    g2_opt_snp1 = g1_0 + 10*0

    return g2_opt_snp1




def g2_opt_bf_snp0(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g0_sn_pass):
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_cond_sn = np.where(g0_sn_pass) # s/n < sn_limit
    g_opt[g_cond_sn] += 1

    g0_0 = np.array([np.where(g_opt > 0, -10, -10)])
    g2_opt_snp0 = g0_0[0]

    return g2_opt_snp0




def g3_opt_bf_snp_org(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, sn_pass_ng_opt, g01, g02, g03):
    g1_sorted = g_num_sort[g01, :, :]
    g1_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]
    g1_sn_flag = np.array([np.where(g1_sn_pass_ng_opt_value == (g1_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

    g2_sorted = g_num_sort[g02, :, :]
    g2_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g2_sorted[np.newaxis], axis=0)[0]
    g2_sn_flag = np.array([np.where(g2_sn_pass_ng_opt_value == (g2_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

    g3_sorted = g_num_sort[g03, :, :]
    g3_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g3_sorted[np.newaxis], axis=0)[0]
    g3_sn_flag = np.array([np.where(g3_sn_pass_ng_opt_value == (g3_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

    g1_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 0)), 1, 0)])[0]
    g2_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1) & (g3_sn_flag == 0)), 1, 0)])[0]
    g3_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0) & (g3_sn_flag == 1)), 1, 0)])[0]

    g12_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1) & (g3_sn_flag == 0)), 1, 0)])[0]
    g13_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 1)), 1, 0)])[0]
    g23_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1) & (g3_sn_flag == 1)), 1, 0)])[0]

    g0_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0) & (g3_sn_flag == 0)), 1, 0)])[0]
    g123_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1) & (g3_sn_flag == 1)), 1, 0)])[0]

    i1 = _params['_i0']
    j1 = _params['_j0']
    g123_sn_flag = g1_sn_flag + g2_sn_flag + g3_sn_flag
    print("ng-23: ", "sn_pass: ", np.where(g123_sn_flag == 3))
    print("")
    print("ng-1: ", g1_sorted[j1, i1], "sn_flag: ", g1_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g1_sn_flag[j1, i1])
    print("ng-2: ", g2_sorted[j1, i1], "sn_flag: ", g2_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g2_sn_flag[j1, i1])
    print("ng-3: ", g3_sorted[j1, i1], "sn_flag: ", g3_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g3_sn_flag[j1, i1])

    _g3_opt_bf_snp_1  = g3_opt_bf_snp1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1_sn_pass, g01)
    _g3_opt_bf_snp_2  = g3_opt_bf_snp1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g2_sn_pass, g02)
    _g3_opt_bf_snp_3  = g3_opt_bf_snp1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g3_sn_pass, g03)

    _g3_opt_bf_snp_12  = g3_opt_bf_snp2(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g12_sn_pass, g01, g02)
    _g3_opt_bf_snp_13  = g3_opt_bf_snp2(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g13_sn_pass, g01, g03)
    _g3_opt_bf_snp_23  = g3_opt_bf_snp2(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g23_sn_pass, g02, g03)

    _g3_opt_bf_snp_0  = g3_opt_bf_snp0(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g0_sn_pass)
    _g3_opt_bf_snp_123 = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g123_sn_pass, g01, g02, g03) # g01 < g02 < g03 : g indices (0, 1, 2, ...)



    _g3_opt_t1 = np.array([np.where(_g3_opt_bf_snp_0 > -1, _g3_opt_bf_snp_0, -1)][0])
    _g3_opt_t2 = np.array([np.where(_g3_opt_bf_snp_1 > -1, _g3_opt_bf_snp_1, _g3_opt_t1)][0])
    _g3_opt_t3 = np.array([np.where(_g3_opt_bf_snp_2 > -1, _g3_opt_bf_snp_2, _g3_opt_t2)][0])
    _g3_opt_t4 = np.array([np.where(_g3_opt_bf_snp_3 > -1, _g3_opt_bf_snp_3, _g3_opt_t3)][0])
    _g3_opt_t5 = np.array([np.where(_g3_opt_bf_snp_12 > -1, _g3_opt_bf_snp_12, _g3_opt_t4)][0])
    _g3_opt_t6 = np.array([np.where(_g3_opt_bf_snp_13 > -1, _g3_opt_bf_snp_13, _g3_opt_t5)][0])
    _g3_opt_t7 = np.array([np.where(_g3_opt_bf_snp_23 > -1, _g3_opt_bf_snp_23, _g3_opt_t6)][0])
    _g3_opt_t8 = np.array([np.where(_g3_opt_bf_snp_123 > -1, _g3_opt_bf_snp_123, _g3_opt_t7)][0])

    return _g3_opt_t8


def g3_opt_bf_snp_bg(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, sn_pass_ng_opt, g01, g02, g03):

    g1_sorted = g_num_sort[g01, :, :]
    g1_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]
    g1_sn_flag = np.array([np.where(g1_sn_pass_ng_opt_value == (g1_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

    g2_sorted = g_num_sort[g02, :, :]
    g2_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g2_sorted[np.newaxis], axis=0)[0]
    g2_sn_flag = np.array([np.where(g2_sn_pass_ng_opt_value == (g2_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

    g3_sorted = g_num_sort[g03, :, :]
    g3_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g3_sorted[np.newaxis], axis=0)[0]
    g3_sn_flag = np.array([np.where(g3_sn_pass_ng_opt_value == (g3_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

    g1_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 0)), 1, 0)])[0]
    g2_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1) & (g3_sn_flag == 0)), 1, 0)])[0]
    g3_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0) & (g3_sn_flag == 1)), 1, 0)])[0]

    g12_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1) & (g3_sn_flag == 0)), 1, 0)])[0]
    g13_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 1)), 1, 0)])[0]
    g23_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1) & (g3_sn_flag == 1)), 1, 0)])[0]

    g0_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0) & (g3_sn_flag == 0)), 1, 0)])[0]
    g123_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1) & (g3_sn_flag == 1)), 1, 0)])[0]

    i1 = _params['_i0']
    j1 = _params['_j0']
    g123_sn_flag = g1_sn_flag + g2_sn_flag + g3_sn_flag
    print("ng-23: ", "sn_pass: ", np.where(g123_sn_flag == 3))
    print("")
    print("ng-1: ", g1_sorted[j1, i1], "sn_flag: ", g1_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g1_sn_flag[j1, i1])
    print("ng-2: ", g2_sorted[j1, i1], "sn_flag: ", g2_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g2_sn_flag[j1, i1])
    print("ng-3: ", g3_sorted[j1, i1], "sn_flag: ", g3_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g3_sn_flag[j1, i1])

    _g3_opt_bf_snp_1  = g3_opt_bf_snp1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1_sn_pass, g01)
    _g3_opt_bf_snp_2  = g3_opt_bf_snp1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g2_sn_pass, g02)
    _g3_opt_bf_snp_3  = g3_opt_bf_snp1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g3_sn_pass, g03)

    _g3_opt_bf_snp_12  = g3_opt_bf_snp2(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g12_sn_pass, g01, g02)
    _g3_opt_bf_snp_13  = g3_opt_bf_snp2(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g13_sn_pass, g01, g03)
    _g3_opt_bf_snp_23  = g3_opt_bf_snp2(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g23_sn_pass, g02, g03)

    _g3_opt_bf_snp_0  = g3_opt_bf_snp0(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g0_sn_pass)
    _g3_opt_bf_snp_123 = g3_opt_bf_snp3(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g123_sn_pass, g01, g02, g03)

    print(g23_sn_pass[j1, i1])
    print(_g3_opt_bf_snp_23[0, j1, i1])

    _g3_opt_t1 = np.array([np.where(_g3_opt_bf_snp_0 > -1, _g3_opt_bf_snp_0, -1)][0])
    _g3_opt_t2 = np.array([np.where(_g3_opt_bf_snp_1 > -1, _g3_opt_bf_snp_1, _g3_opt_t1)][0])
    _g3_opt_t3 = np.array([np.where(_g3_opt_bf_snp_2 > -1, _g3_opt_bf_snp_2, _g3_opt_t2)][0])
    _g3_opt_t4 = np.array([np.where(_g3_opt_bf_snp_3 > -1, _g3_opt_bf_snp_3, _g3_opt_t3)][0])
    _g3_opt_t5 = np.array([np.where(_g3_opt_bf_snp_12 > -1, _g3_opt_bf_snp_12, _g3_opt_t4)][0])
    _g3_opt_t6 = np.array([np.where(_g3_opt_bf_snp_13 > -1, _g3_opt_bf_snp_13, _g3_opt_t5)][0])
    _g3_opt_t7 = np.array([np.where(_g3_opt_bf_snp_23 > -1, _g3_opt_bf_snp_23, _g3_opt_t6)][0])
    print(_g3_opt_t7[0, j1, i1])
    print(_g3_opt_bf_snp_123[0, j1, i1])
    _g3_opt_t8 = np.array([np.where(_g3_opt_bf_snp_123 > -1, _g3_opt_bf_snp_123, _g3_opt_t7)][0])

    print(_g3_opt_t8[0, j1, i1])
    return _g3_opt_t8



def g3_opt_bf_snp0(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g0_sn_pass):
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_cond_sn = np.where(g0_sn_pass) # s/n < sn_limit
    g_opt[g_cond_sn] += 1

    g0_0 = np.array([np.where(g_opt > 0, -10, -10)])
    g3_opt_snp0 = g0_0[0]

    return g3_opt_snp0



def g3_opt_bf_snp1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1_sn_pass, g01): # g01 : g indices (0, 1, 2, ...)
    g_sort_n = g01
    _cond_N = 0
    _cond0 = ([g01, g01, '='], ) # dummy condition
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond0, _cond_N, g1_sn_pass)
    g3_opt_snp1 = g1_0 + 10*0

    return g3_opt_snp1



def g3_opt_bf_snp2(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g12_sn_pass, g01, g02): # g01 < g02 : g indices (0, 1, 2, ...)

    g_sort_n = g01
    _cond_N = 1
    _cond1 = ([g01, g02, '0'], )
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12_sn_pass)

    g_sort_n = g02
    _cond_N = 2
    _cond2 = ([g01, g02, '1'], [g01, g02, '0'])
    g2_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond2, _cond_N, g12_sn_pass)

    g_sort_n = g01
    _cond_N = 2
    _cond2 = ([g01, g02, '1'], [g01, g02, '1'])
    g1_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond2, _cond_N, g12_sn_pass)
    
    g3_opt_snp2 = g1_0 + g2_0 + g1_1 + 10*2

    return g3_opt_snp2




def g3_opt_bf_snp3(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g123_sn_pass, g01, g02, g03): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    g_sort_n = g01
    _cond_N = 1
    _cond1 = ([g01, g02, '0'], )
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g123_sn_pass)

    g_sort_n = g02
    _cond_N = 3
    _cond3 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '0'])
    g2_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond3, _cond_N, g123_sn_pass)

    g_sort_n = g03
    _cond_N = 4
    _cond4 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '1'], [g02, g03, '0'])
    g3_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond4, _cond_N, g123_sn_pass)
    
    g_sort_n = g02
    _cond_N = 4
    _cond4 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '1'], [g02, g03, '1'])
    g2_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond4, _cond_N, g123_sn_pass)
    
    g_sort_n = g01
    _cond_N = 3
    _cond3 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '0'])
    g1_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond3, _cond_N, g123_sn_pass)


    g_sort_n = g03
    _cond_N = 4
    _cond4 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '1'], [g01, g03, '0'])
    g3_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond4, _cond_N, g123_sn_pass)
    
    g_sort_n = g01
    _cond_N = 4
    _cond4 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '1'], [g01, g03, '1'])
    g1_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond4, _cond_N, g123_sn_pass)

    g3_opt_snp3 =  g1_0 + g1_1 + g1_2 \
            + g2_0 + g2_1 \
            + g3_0 + g3_1 + 10*6 # -10 flag see above

    return g3_opt_snp3




def snp0_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g0_sn_pass): #
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_cond_sn = np.where(g0_sn_pass) # s/n < sn_limit
    g_opt[g_cond_sn] += 1

    g0_0 = np.array([np.where(g_opt > 0, -10, -10)])
    g0_opt_snp0 = g0_0[0]

    return g0_opt_snp0




def snp1_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1_sn_pass, g01): # g01: g indices (0, 1, 2, ...)
    g_sort_n = g01
    _cond_N = 0
    _cond0 = ([g01, g01, '='], ) # dummy condition
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond0, _cond_N, g1_sn_pass)
    g1_opt_snp1 = g1_0 + 10*0

    return g1_opt_snp1





def snp2_tree_opt_bf_bg(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g12_sn_pass, g01, g02): # g01 < g02: g indices (0, 1, 2, ...)
    print("")
    print("")
    g1x_t = set_cond_tree(1, 1, 2, 1) # [1]
    g1x = cp.deepcopy(g1x_t)

    print("1, 1, 2, 1")
    print(g1x[0])
    print(g1x[1])
    print("")
    print("")

    n_G1x = 2 # from the condition matrix

    g_sort_n = g01 # for g01
    g1x_filtered = np.zeros((n_G1x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g1x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G1x):
        _cond_N = len(g1x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            if g1x_t[i][j][0] == 0:
                g1x[i][j][0] = g1x_t[i][j][0] + (g01 - 0)
            if g1x_t[i][j][0] == 1:
                g1x[i][j][0] = g1x_t[i][j][0] + (g02 - 1)

            if g1x_t[i][j][1] == 0:
                g1x[i][j][1] = g1x_t[i][j][1] + (g01 - 0)
            if g1x_t[i][j][1] == 1:
                g1x[i][j][1] = g1x_t[i][j][1] + (g02 - 1)

            _cond1_list.append(g1x[i][j])

        _cond1 = tuple(_cond1_list)
        g1x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12_sn_pass)
        g1x_filtered_sum[:, :] += g1x_filtered[i, :, :]

    print("after update")
    print(g1x[0])
    print(g1x[1])
    print("")
    print("")
    

    g2x_t = set_cond_tree(2, 2, 2, 0) # [0]
    g2x = cp.deepcopy(g2x_t)

    print("2, 2, 2, 0")
    print(g2x[0])
    print("")
    print("")

    n_G2x = 1 # from the condition matrix
    g_sort_n = g02 # for g02
    g2x_filtered = np.zeros((n_G2x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g2x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G2x):
        _cond_N = len(g2x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            if g2x_t[i][j][0] == 0:
                g2x[i][j][0] = g2x_t[i][j][0] + (g01 - 0)
            if g2x_t[i][j][0] == 1:
                g2x[i][j][0] = g2x_t[i][j][0] + (g02 - 1)

            if g2x_t[i][j][1] == 0:
                g2x[i][j][1] = g2x_t[i][j][1] + (g01 - 0)
            if g2x_t[i][j][1] == 1:
                g2x[i][j][1] = g2x_t[i][j][1] + (g02 - 1)

            _cond1_list.append(g2x[i][j])

        _cond1 = tuple(_cond1_list)
        g2x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12_sn_pass)
        g2x_filtered_sum[:, :] += g2x_filtered[i, :, :]
    

    print("after update")
    print(g2x[0])
    print("")
    print("")


    g2_opt_snp2 = g1x_filtered_sum + g2x_filtered_sum + 10*(n_G1x + n_G2x - 1)

    return g2_opt_snp2


def snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g12_sn_pass, g01, g02): # g01 < g02: g indices (0, 1, 2, ...)
    print("")
    print("")
    g1x_t = set_cond_tree(1, 1, 2, 1) # [1]
    g1x = cp.deepcopy(g1x_t)


    n_G1x = 2 # from the condition matrix

    g_sort_n = g01 # for g01
    g1x_filtered = np.zeros((n_G1x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g1x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G1x):
        _cond_N = len(g1x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            if g1x_t[i][j][0] == 0:
                g1x[i][j][0] = g1x_t[i][j][0] + (g01 - 0)
            if g1x_t[i][j][0] == 1:
                g1x[i][j][0] = g1x_t[i][j][0] + (g02 - 1)

            if g1x_t[i][j][1] == 0:
                g1x[i][j][1] = g1x_t[i][j][1] + (g01 - 0)
            if g1x_t[i][j][1] == 1:
                g1x[i][j][1] = g1x_t[i][j][1] + (g02 - 1)

            _cond1_list.append(g1x[i][j])

        _cond1 = tuple(_cond1_list)
        g1x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12_sn_pass)
        g1x_filtered_sum[:, :] += g1x_filtered[i, :, :]

    

    g2x_t = set_cond_tree(2, 2, 2, 0) # [0]
    g2x = cp.deepcopy(g2x_t)


    n_G2x = 1 # from the condition matrix
    g_sort_n = g02 # for g02
    g2x_filtered = np.zeros((n_G2x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g2x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G2x):
        _cond_N = len(g2x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            if g2x_t[i][j][0] == 0:
                g2x[i][j][0] = g2x_t[i][j][0] + (g01 - 0)
            if g2x_t[i][j][0] == 1:
                g2x[i][j][0] = g2x_t[i][j][0] + (g02 - 1)

            if g2x_t[i][j][1] == 0:
                g2x[i][j][1] = g2x_t[i][j][1] + (g01 - 0)
            if g2x_t[i][j][1] == 1:
                g2x[i][j][1] = g2x_t[i][j][1] + (g02 - 1)

            _cond1_list.append(g2x[i][j])

        _cond1 = tuple(_cond1_list)
        g2x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12_sn_pass)
        g2x_filtered_sum[:, :] += g2x_filtered[i, :, :]
    

    print("after update")
    print(g2x[0])
    print("")
    print("")


    g2_opt_snp2 = g1x_filtered_sum + g2x_filtered_sum + 10*(n_G1x + n_G2x - 1)

    return g2_opt_snp2



def snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g123_sn_pass, g01, g02, g03): # g01 < g02 < g03 : g indices (0, 1, 2, ...)
    print("")
    print("")
    g1x_t = set_cond_tree(1, 1, 3, 2) # [2]
    g1x = cp.deepcopy(g1x_t)

    print("1, 1, 3, 2")
    print(g1x[0])
    print(g1x[1])
    print(g1x[2])
    print("")
    print("")

    n_G1x = 3 # from the condition matrix

    g_sort_n = g01 # for g01
    g1x_filtered = np.zeros((n_G1x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g1x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G1x):
        _cond_N = len(g1x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            if g1x_t[i][j][0] == 0:
                g1x[i][j][0] = g1x_t[i][j][0] + (g01 - 0)
            if g1x_t[i][j][0] == 1:
                g1x[i][j][0] = g1x_t[i][j][0] + (g02 - 1)
            if g1x_t[i][j][0] == 2: # dummy but to make coding easier
                g1x[i][j][0] = g1x_t[i][j][0] + (g03 - 2)

            if g1x_t[i][j][1] == 0: # dummy but to make coding easier
                g1x[i][j][1] = g1x_t[i][j][1] + (g01 - 0)
            if g1x_t[i][j][1] == 1:
                g1x[i][j][1] = g1x_t[i][j][1] + (g02 - 1)
            if g1x_t[i][j][1] == 2:
                g1x[i][j][1] = g1x_t[i][j][1] + (g03 - 2)

            _cond1_list.append(g1x[i][j])

        _cond1 = tuple(_cond1_list)
        g1x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g123_sn_pass)
        g1x_filtered_sum[:, :] += g1x_filtered[i, :, :]

    print("after update")
    print(g1x[0])
    print(g1x[1])
    print(g1x[2])
    print("")
    print("")
    

    g2x_t = set_cond_tree(2, 2, 3, 1) # [1]
    g2x = cp.deepcopy(g2x_t)

    print("2, 2, 3, 1")
    print(g2x[0])
    print(g2x[1])
    print("")
    print("")



    n_G2x = 2 # from the condition matrix
    g_sort_n = g02 # for g02
    g2x_filtered = np.zeros((n_G2x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g2x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G2x):
        _cond_N = len(g2x[i])
        _cond1_list = []
        for j in range(0, _cond_N):

            if g2x_t[i][j][0] == 0:
                g2x[i][j][0] = g2x_t[i][j][0] + (g01 - 0)
            if g2x_t[i][j][0] == 1:
                g2x[i][j][0] = g2x_t[i][j][0] + (g02 - 1)
            if g2x_t[i][j][0] == 2: # dummy but to make coding easier
                g2x[i][j][0] = g2x_t[i][j][0] + (g03 - 2)

            if g2x_t[i][j][1] == 0: # dummy but to make coding easier
                g2x[i][j][1] = g2x_t[i][j][1] + (g01 - 0)
            if g2x_t[i][j][1] == 1:
                g2x[i][j][1] = g2x_t[i][j][1] + (g02 - 1)
            if g2x_t[i][j][1] == 2:
                g2x[i][j][1] = g2x_t[i][j][1] + (g03 - 2)

            _cond1_list.append(g2x[i][j])

        _cond1 = tuple(_cond1_list)
        g2x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g123_sn_pass)
        g2x_filtered_sum[:, :] += g2x_filtered[i, :, :]
    

    print("after update")
    print(g2x[0])
    print(g2x[1])
    print("")
    print("")

    g3x_t = set_cond_tree(3, 3, 3, 0) # [0]
    g3x = cp.deepcopy(g3x_t)

    print("3, 3, 3, 0")
    print(g3x[0])
    print(g3x[1])
    print("seheon")
    n_G3x = 2 # from the condition matrix
    g_sort_n = g03 # for g03
    g3x_filtered = np.zeros((n_G3x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g3x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G3x):
        _cond_N = len(g3x[i])
        _cond1_list = []
        for j in range(0, _cond_N):

            if g3x_t[i][j][0] == 0:
                g3x[i][j][0] = g3x_t[i][j][0] + (g01 - 0)
            if g3x_t[i][j][0] == 1:
                g3x[i][j][0] = g3x_t[i][j][0] + (g02 - 1)
            if g3x_t[i][j][0] == 2: # dummy but to make coding easier
                g3x[i][j][0] = g3x_t[i][j][0] + (g03 - 2)

            if g3x_t[i][j][1] == 0: # dummy but to make coding easier
                g3x[i][j][1] = g3x_t[i][j][1] + (g01 - 0)
            if g3x_t[i][j][1] == 1:
                g3x[i][j][1] = g3x_t[i][j][1] + (g02 - 1)
            if g3x_t[i][j][1] == 2:
                g3x[i][j][1] = g3x_t[i][j][1] + (g03 - 2)

            _cond1_list.append(g3x[i][j])

        _cond1 = tuple(_cond1_list)
        g3x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g123_sn_pass)
        g3x_filtered_sum[0, :, :] += g3x_filtered[i, :, :]
    

    print("after update")
    print(g3x[0])
    print(g3x[1])
    print("seheon")

    g3_opt_snp3 = g1x_filtered_sum + g2x_filtered_sum + g3x_filtered_sum + 10*(n_G1x + n_G2x + n_G3x - 1)

    return g3_opt_snp3



def snp4_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1234_sn_pass, g01, g02, g03, g04): # g01 < g02 < g03 : g indices (0, 1, 2, ...)
    print("")
    print("")
    g1x_t = set_cond_tree(1, 1, 4, 3) # [3]
    g1x = cp.deepcopy(g1x_t)

    print("1, 1, 4, 3")
    print(g1x[0])
    print(g1x[1])
    print(g1x[2])
    print(g1x[3])
    print("")
    print("")

    n_G1x = 4 # from the condition matrix

    g_sort_n = g01 # for g01
    g1x_filtered = np.zeros((n_G1x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g1x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G1x):
        _cond_N = len(g1x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            if g1x_t[i][j][0] == 0:
                g1x[i][j][0] = g1x_t[i][j][0] + (g01 - 0)
            if g1x_t[i][j][0] == 1:
                g1x[i][j][0] = g1x_t[i][j][0] + (g02 - 1)
            if g1x_t[i][j][0] == 2:
                g1x[i][j][0] = g1x_t[i][j][0] + (g03 - 2)
            if g1x_t[i][j][0] == 3:
                g1x[i][j][0] = g1x_t[i][j][0] + (g04 - 3)

            if g1x_t[i][j][1] == 0:
                g1x[i][j][1] = g1x_t[i][j][1] + (g01 - 0)
            if g1x_t[i][j][1] == 1:
                g1x[i][j][1] = g1x_t[i][j][1] + (g02 - 1)
            if g1x_t[i][j][1] == 2:
                g1x[i][j][1] = g1x_t[i][j][1] + (g03 - 2)
            if g1x_t[i][j][1] == 3:
                g1x[i][j][1] = g1x_t[i][j][1] + (g04 - 3)

            _cond1_list.append(g1x[i][j])

        _cond1 = tuple(_cond1_list)
        g1x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g1234_sn_pass)
        g1x_filtered_sum[:, :] += g1x_filtered[i, :, :]

    print("after update")
    print(g1x[0])
    print(g1x[1])
    print(g1x[2])
    print(g1x[3])
    print("")
    print("")
    

    g2x_t = set_cond_tree(2, 2, 4, 2) # [2]
    g2x = cp.deepcopy(g2x_t)

    print("2, 2, 4, 2")
    print(g2x[0])
    print(g2x[1])
    print(g2x[2])
    print("")
    print("")

    n_G2x = 3 # from the condition matrix
    g_sort_n = g02 # for g02
    g2x_filtered = np.zeros((n_G2x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g2x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G2x):
        _cond_N = len(g2x[i])
        _cond1_list = []
        for j in range(0, _cond_N):

            if g2x_t[i][j][0] == 0:
                g2x[i][j][0] = g2x_t[i][j][0] + (g01 - 0)
            if g2x_t[i][j][0] == 1:
                g2x[i][j][0] = g2x_t[i][j][0] + (g02 - 1)
            if g2x_t[i][j][0] == 2:
                g2x[i][j][0] = g2x_t[i][j][0] + (g03 - 2)
            if g2x_t[i][j][0] == 3:
                g2x[i][j][0] = g2x_t[i][j][0] + (g04 - 3)

            if g2x_t[i][j][1] == 0:
                g2x[i][j][1] = g2x_t[i][j][1] + (g01 - 0)
            if g2x_t[i][j][1] == 1:
                g2x[i][j][1] = g2x_t[i][j][1] + (g02 - 1)
            if g2x_t[i][j][1] == 2:
                g2x[i][j][1] = g2x_t[i][j][1] + (g03 - 2)
            if g2x_t[i][j][1] == 3:
                g2x[i][j][1] = g2x_t[i][j][1] + (g04 - 3)

            _cond1_list.append(g2x[i][j])

        _cond1 = tuple(_cond1_list)
        g2x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g1234_sn_pass)
        g2x_filtered_sum[:, :] += g2x_filtered[i, :, :]
    

    print("after update")
    print(g2x[0])
    print(g2x[1])
    print(g2x[2])
    print("")
    print("")

    g3x_t = set_cond_tree(3, 3, 4, 1) # [1]
    g3x = cp.deepcopy(g3x_t)

    print("3, 3, 4, 1")
    print(g3x[0])
    print(g3x[1])
    print(g3x[2])
    print(g3x[3])
    n_G3x = 4 # from the condition matrix
    g_sort_n = g03 # for g03
    g3x_filtered = np.zeros((n_G3x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g3x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G3x):
        _cond_N = len(g3x[i])
        _cond1_list = []
        for j in range(0, _cond_N):

            if g3x_t[i][j][0] == 0:
                g3x[i][j][0] = g3x_t[i][j][0] + (g01 - 0)
            if g3x_t[i][j][0] == 1:
                g3x[i][j][0] = g3x_t[i][j][0] + (g02 - 1)
            if g3x_t[i][j][0] == 2:
                g3x[i][j][0] = g3x_t[i][j][0] + (g03 - 2)
            if g3x_t[i][j][0] == 3:
                g3x[i][j][0] = g3x_t[i][j][0] + (g04 - 3)

            if g3x_t[i][j][1] == 0:
                g3x[i][j][1] = g3x_t[i][j][1] + (g01 - 0)
            if g3x_t[i][j][1] == 1:
                g3x[i][j][1] = g3x_t[i][j][1] + (g02 - 1)
            if g3x_t[i][j][1] == 2:
                g3x[i][j][1] = g3x_t[i][j][1] + (g03 - 2)
            if g3x_t[i][j][1] == 3:
                g3x[i][j][1] = g3x_t[i][j][1] + (g04 - 3)

            _cond1_list.append(g3x[i][j])

        _cond1 = tuple(_cond1_list)
        g3x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g1234_sn_pass)
        g3x_filtered_sum[0, :, :] += g3x_filtered[i, :, :]
    

    print("after update")
    print(g3x[0])
    print(g3x[1])
    print(g3x[2])
    print(g3x[3])


    g4x_t = set_cond_tree(4, 4, 4, 0) # [0]
    g4x = cp.deepcopy(g4x_t)

    print("4, 4, 4, 0")
    print(g4x[0])
    print(g4x[1])
    print(g4x[2])
    print(g4x[3])
    n_G4x = 4 # from the condition matrix
    g_sort_n = g04 # for g04
    g4x_filtered = np.zeros((n_G4x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g4x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G4x):
        _cond_N = len(g4x[i])
        _cond1_list = []
        for j in range(0, _cond_N):

            if g4x_t[i][j][0] == 0:
                g4x[i][j][0] = g4x_t[i][j][0] + (g01 - 0)
            if g4x_t[i][j][0] == 1:
                g4x[i][j][0] = g4x_t[i][j][0] + (g02 - 1)
            if g4x_t[i][j][0] == 2:
                g4x[i][j][0] = g4x_t[i][j][0] + (g03 - 2)
            if g4x_t[i][j][0] == 3:
                g4x[i][j][0] = g4x_t[i][j][0] + (g04 - 3)

            if g4x_t[i][j][1] == 0:
                g4x[i][j][1] = g4x_t[i][j][1] + (g01 - 0)
            if g4x_t[i][j][1] == 1:
                g4x[i][j][1] = g4x_t[i][j][1] + (g02 - 1)
            if g4x_t[i][j][1] == 2:
                g4x[i][j][1] = g4x_t[i][j][1] + (g03 - 2)
            if g4x_t[i][j][1] == 3:
                g4x[i][j][1] = g4x_t[i][j][1] + (g04 - 3)

            _cond1_list.append(g4x[i][j])

        _cond1 = tuple(_cond1_list)
        g4x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g1234_sn_pass)
        g4x_filtered_sum[0, :, :] += g4x_filtered[i, :, :]
    

    print("after update")
    print(g4x[0])
    print(g4x[1])
    print(g4x[2])
    print(g4x[3])

    g4_opt_snp4 = g1x_filtered_sum + g2x_filtered_sum + g3x_filtered_sum + g4x_filtered_sum + 10*(n_G1x + n_G2x + n_G3x + n_G4x - 1)

    return g4_opt_snp4


def snp_gx_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, g1234_sn_pass, gx_list, n_GNx_list): # g01 < g02 < g03 : g indices (0, 1, 2, ...)







    gNx_t = [0 for i in range(len(gx_list))]
    gNx =   [0 for i in range(len(gx_list))]

    gNx_filtered_sum = np.zeros((len(gx_list), _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    gNx_filtered_sum_all = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    n_GNx_sum = 0
    for g in range(0, len(gx_list)):
        n_GNx = n_GNx_list[g]
        n_GNx_sum += n_GNx
        g_sorted_n = gx_list[g] # for g0X

        gNx_t[g] = set_cond_tree(g+1, g+1, max_ngauss, max_ngauss-g-1) # [max_ngauss-g-1]
        gNx = cp.deepcopy(gNx_t)
    
        gNx_filtered = np.zeros((n_GNx, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
        gNx_filtered_sum_t = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

        for i in range(0, n_GNx):
            _cond_N = len(gNx[g][i])
            _cond1_list = []
            for j in range(0, _cond_N):
                for k in range(0, len(gx_list)):
                    if gNx_t[g][i][j][0] == k:
                        gNx[g][i][j][0] = gNx_t[g][i][j][0] + (gx_list[k] - k)
    
                    if gNx_t[g][i][j][1] == k:
                        gNx[g][i][j][1] = gNx_t[g][i][j][1] + (gx_list[k] - k)
    
                _cond1_list.append(gNx[g][i][j])


            _cond1 = tuple(_cond1_list)
            gNx_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sorted_n, _cond1, _cond_N, g1234_sn_pass)
            gNx_filtered_sum_t[:, :] += gNx_filtered[i, :, :]

        gNx_filtered_sum[g, :, :] = gNx_filtered_sum_t[:, :]

    for g in range(0, len(gx_list)):
        gNx_filtered_sum_all[:, :] += gNx_filtered_sum[g, :, :]

    gNx_filtered_sum_all[:, :] += 10*(n_GNx_sum - 1) # -10 flag

    return gNx_filtered_sum_all[:, :]


def snp_gx_tree_opt_bf1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, g1234_sn_pass, g01, g02, g03, g04): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    gx_list = [g01, g02, g03, g04]
    n_G1x = 4
    n_G2x = 3
    n_G3x = 4
    n_G4x = 4

    g1x_t = set_cond_tree(1, 1, 4, 3) # [n_gx-1]
    g1x = cp.deepcopy(g1x_t)

    n_G1x = 4 # from the condition matrix

    g_sort_n = g01 # for g01
    g1x_filtered = np.zeros((n_G1x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g1x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G1x):
        _cond_N = len(g1x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            for k in range(0, max_ngauss):
                if g1x_t[i][j][0] == k:
                    g1x[i][j][0] = g1x_t[i][j][0] + (gx_list[k] - k)

                if g1x_t[i][j][1] == k:
                    g1x[i][j][1] = g1x_t[i][j][1] + (gx_list[k] - k)

            _cond1_list.append(g1x[i][j])

        _cond1 = tuple(_cond1_list)
        g1x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g1234_sn_pass)
        g1x_filtered_sum[:, :] += g1x_filtered[i, :, :]


    g2x_t = set_cond_tree(2, 2, 4, 2) # [2]
    g2x = cp.deepcopy(g2x_t)

    n_G2x = 3 # from the condition matrix
    g_sort_n = g02 # for g02
    g2x_filtered = np.zeros((n_G2x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g2x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G2x):
        _cond_N = len(g2x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            for k in range(0, max_ngauss):
                if g2x_t[i][j][0] == k:
                    g2x[i][j][0] = g2x_t[i][j][0] + (gx_list[k] - k)

                if g2x_t[i][j][1] == k:
                    g2x[i][j][1] = g2x_t[i][j][1] + (gx_list[k] - k)

            _cond1_list.append(g2x[i][j])

        _cond1 = tuple(_cond1_list)
        g2x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g1234_sn_pass)
        g2x_filtered_sum[:, :] += g2x_filtered[i, :, :]
    
    g3x_t = set_cond_tree(3, 3, 4, 1) # [1]
    g3x = cp.deepcopy(g3x_t)

    n_G3x = 4 # from the condition matrix
    g_sort_n = g03 # for g03
    g3x_filtered = np.zeros((n_G3x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g3x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G3x):
        _cond_N = len(g3x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            for k in range(0, max_ngauss):
                if g3x_t[i][j][0] == k:
                    g3x[i][j][0] = g3x_t[i][j][0] + (gx_list[k] - k)

                if g3x_t[i][j][1] == k:
                    g3x[i][j][1] = g3x_t[i][j][1] + (gx_list[k] - k)

            _cond1_list.append(g3x[i][j])

        _cond1 = tuple(_cond1_list)
        g3x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g1234_sn_pass)
        g3x_filtered_sum[0, :, :] += g3x_filtered[i, :, :]
    
    g4x_t = set_cond_tree(4, 4, 4, 0) # [0]
    g4x = cp.deepcopy(g4x_t)

    n_G4x = 4 # from the condition matrix
    g_sort_n = g04 # for g04
    g4x_filtered = np.zeros((n_G4x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g4x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G4x):
        _cond_N = len(g4x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            for k in range(0, max_ngauss):
                if g4x_t[i][j][0] == k:
                    g4x[i][j][0] = g4x_t[i][j][0] + (gx_list[k] - k)

                if g4x_t[i][j][1] == k:
                    g4x[i][j][1] = g4x_t[i][j][1] + (gx_list[k] - k)

            _cond1_list.append(g4x[i][j])

        _cond1 = tuple(_cond1_list)
        g4x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g1234_sn_pass)
        g4x_filtered_sum[0, :, :] += g4x_filtered[i, :, :]
    
    g4_opt_snp4 = g1x_filtered_sum + g2x_filtered_sum + g3x_filtered_sum + g4x_filtered_sum + 10*(n_G1x + n_G2x + n_G3x + n_G4x - 1)

    return g4_opt_snp4


def snp5_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g12345_sn_pass, g01, g02, g03, g04, g05): # g01 < g02 < g03 : g indices (0, 1, 2, ...)
    print("")
    print("")
    g1x_t = set_cond_tree(1, 1, 5, 4) # [4]
    g1x = cp.deepcopy(g1x_t)

    print("1, 1, 5, 4")
    print(g1x[0])
    print(g1x[1])
    print(g1x[2])
    print(g1x[3])
    print(g1x[4])
    print("")
    print("")

    n_G1x = 5 # from the condition matrix

    g_sort_n = g01 # for g01
    g1x_filtered = np.zeros((n_G1x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g1x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G1x):
        _cond_N = len(g1x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            if g1x_t[i][j][0] == 0:
                g1x[i][j][0] = g1x_t[i][j][0] + (g01 - 0)
            if g1x_t[i][j][0] == 1:
                g1x[i][j][0] = g1x_t[i][j][0] + (g02 - 1)
            if g1x_t[i][j][0] == 2:
                g1x[i][j][0] = g1x_t[i][j][0] + (g03 - 2)
            if g1x_t[i][j][0] == 3:
                g1x[i][j][0] = g1x_t[i][j][0] + (g04 - 3)
            if g1x_t[i][j][0] == 4:
                g1x[i][j][0] = g1x_t[i][j][0] + (g05 - 4)

            if g1x_t[i][j][1] == 0:
                g1x[i][j][1] = g1x_t[i][j][1] + (g01 - 0)
            if g1x_t[i][j][1] == 1:
                g1x[i][j][1] = g1x_t[i][j][1] + (g02 - 1)
            if g1x_t[i][j][1] == 2:
                g1x[i][j][1] = g1x_t[i][j][1] + (g03 - 2)
            if g1x_t[i][j][1] == 3:
                g1x[i][j][1] = g1x_t[i][j][1] + (g04 - 3)
            if g1x_t[i][j][1] == 4:
                g1x[i][j][1] = g1x_t[i][j][1] + (g05 - 4)

            _cond1_list.append(g1x[i][j])

        _cond1 = tuple(_cond1_list)
        g1x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12345_sn_pass)
        g1x_filtered_sum[:, :] += g1x_filtered[i, :, :]

    print("after update")
    print(g1x[0])
    print(g1x[1])
    print(g1x[2])
    print(g1x[3])
    print(g1x[4])
    print("")
    print("")
    

    g2x_t = set_cond_tree(2, 2, 5, 3) # [3]
    g2x = cp.deepcopy(g2x_t)

    print("2, 2, 5, 3")
    print(g2x[0])
    print(g2x[1])
    print(g2x[2])
    print(g2x[3])
    print("")
    print("")

    n_G2x = 4 # from the condition matrix
    g_sort_n = g02 # for g02
    g2x_filtered = np.zeros((n_G2x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g2x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G2x):
        _cond_N = len(g2x[i])
        _cond1_list = []
        for j in range(0, _cond_N):

            if g2x_t[i][j][0] == 0:
                g2x[i][j][0] = g2x_t[i][j][0] + (g01 - 0)
            if g2x_t[i][j][0] == 1:
                g2x[i][j][0] = g2x_t[i][j][0] + (g02 - 1)
            if g2x_t[i][j][0] == 2:
                g2x[i][j][0] = g2x_t[i][j][0] + (g03 - 2)
            if g2x_t[i][j][0] == 3:
                g2x[i][j][0] = g2x_t[i][j][0] + (g04 - 3)
            if g2x_t[i][j][0] == 4:
                g2x[i][j][0] = g2x_t[i][j][0] + (g05 - 4)

            if g2x_t[i][j][1] == 0:
                g2x[i][j][1] = g2x_t[i][j][1] + (g01 - 0)
            if g2x_t[i][j][1] == 1:
                g2x[i][j][1] = g2x_t[i][j][1] + (g02 - 1)
            if g2x_t[i][j][1] == 2:
                g2x[i][j][1] = g2x_t[i][j][1] + (g03 - 2)
            if g2x_t[i][j][1] == 3:
                g2x[i][j][1] = g2x_t[i][j][1] + (g04 - 3)
            if g2x_t[i][j][1] == 4:
                g2x[i][j][1] = g2x_t[i][j][1] + (g05 - 4)

            _cond1_list.append(g2x[i][j])

        _cond1 = tuple(_cond1_list)
        g2x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12345_sn_pass)
        g2x_filtered_sum[:, :] += g2x_filtered[i, :, :]
    

    print("after update")
    print(g2x[0])
    print(g2x[1])
    print(g2x[2])
    print(g2x[3])
    print("")
    print("")

    g3x_t = set_cond_tree(3, 3, 5, 2) # [2]
    g3x = cp.deepcopy(g3x_t)

    print("3, 3, 5, 2")
    print(g3x[0])
    print(g3x[1])
    print(g3x[2])
    print(g3x[3])
    print(g3x[4])
    print(g3x[5])
    n_G3x = 6 # from the condition matrix
    g_sort_n = g03 # for g03
    g3x_filtered = np.zeros((n_G3x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g3x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G3x):
        _cond_N = len(g3x[i])
        _cond1_list = []
        for j in range(0, _cond_N):

            if g3x_t[i][j][0] == 0:
                g3x[i][j][0] = g3x_t[i][j][0] + (g01 - 0)
            if g3x_t[i][j][0] == 1:
                g3x[i][j][0] = g3x_t[i][j][0] + (g02 - 1)
            if g3x_t[i][j][0] == 2:
                g3x[i][j][0] = g3x_t[i][j][0] + (g03 - 2)
            if g3x_t[i][j][0] == 3:
                g3x[i][j][0] = g3x_t[i][j][0] + (g04 - 3)
            if g3x_t[i][j][0] == 4:
                g3x[i][j][0] = g3x_t[i][j][0] + (g05 - 4)

            if g3x_t[i][j][1] == 0:
                g3x[i][j][1] = g3x_t[i][j][1] + (g01 - 0)
            if g3x_t[i][j][1] == 1:
                g3x[i][j][1] = g3x_t[i][j][1] + (g02 - 1)
            if g3x_t[i][j][1] == 2:
                g3x[i][j][1] = g3x_t[i][j][1] + (g03 - 2)
            if g3x_t[i][j][1] == 3:
                g3x[i][j][1] = g3x_t[i][j][1] + (g04 - 3)
            if g3x_t[i][j][1] == 4:
                g3x[i][j][1] = g3x_t[i][j][1] + (g05 - 4)

            _cond1_list.append(g3x[i][j])

        _cond1 = tuple(_cond1_list)
        g3x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12345_sn_pass)
        g3x_filtered_sum[0, :, :] += g3x_filtered[i, :, :]
    

    print("after update")
    print(g3x[0])
    print(g3x[1])
    print(g3x[2])
    print(g3x[3])
    print(g3x[4])
    print(g3x[5])


    g4x_t = set_cond_tree(4, 4, 5, 1) # [1]
    g4x = cp.deepcopy(g4x_t)

    print("4, 4, 5, 1")
    print(g4x[0])
    print(g4x[1])
    print(g4x[2])
    print(g4x[3])
    print(g4x[4])
    print(g4x[5])
    print(g4x[6])
    print(g4x[7])
    n_G4x = 8 # from the condition matrix
    g_sort_n = g04 # for g04
    g4x_filtered = np.zeros((n_G4x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g4x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G4x):
        _cond_N = len(g4x[i])
        _cond1_list = []
        for j in range(0, _cond_N):

            if g4x_t[i][j][0] == 0:
                g4x[i][j][0] = g4x_t[i][j][0] + (g01 - 0)
            if g4x_t[i][j][0] == 1:
                g4x[i][j][0] = g4x_t[i][j][0] + (g02 - 1)
            if g4x_t[i][j][0] == 2:
                g4x[i][j][0] = g4x_t[i][j][0] + (g03 - 2)
            if g4x_t[i][j][0] == 3:
                g4x[i][j][0] = g4x_t[i][j][0] + (g04 - 3)
            if g4x_t[i][j][0] == 4:
                g4x[i][j][0] = g4x_t[i][j][0] + (g05 - 4)

            if g4x_t[i][j][1] == 0:
                g4x[i][j][1] = g4x_t[i][j][1] + (g01 - 0)
            if g4x_t[i][j][1] == 1:
                g4x[i][j][1] = g4x_t[i][j][1] + (g02 - 1)
            if g4x_t[i][j][1] == 2:
                g4x[i][j][1] = g4x_t[i][j][1] + (g03 - 2)
            if g4x_t[i][j][1] == 3:
                g4x[i][j][1] = g4x_t[i][j][1] + (g04 - 3)
            if g4x_t[i][j][1] == 4:
                g4x[i][j][1] = g4x_t[i][j][1] + (g05 - 4)

            _cond1_list.append(g4x[i][j])

        _cond1 = tuple(_cond1_list)
        g4x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12345_sn_pass)
        g4x_filtered_sum[0, :, :] += g4x_filtered[i, :, :]
    

    print("after update")
    print(g4x[0])
    print(g4x[1])
    print(g4x[2])
    print(g4x[3])
    print(g4x[4])
    print(g4x[5])
    print(g4x[6])
    print(g4x[7])


    g5x_t = set_cond_tree(5, 5, 5, 0) # [0]
    g5x = cp.deepcopy(g5x_t)

    print("5, 5, 5, 0")
    print(g5x[0])
    print(g5x[1])
    print(g5x[2])
    print(g5x[3])
    print(g5x[4])
    print(g5x[5])
    print(g5x[6])
    print(g5x[7])

    n_G5x = 8 # from the condition matrix
    g_sort_n = g05 # for g05
    g5x_filtered = np.zeros((n_G5x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g5x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G5x):
        _cond_N = len(g5x[i])
        _cond1_list = []
        for j in range(0, _cond_N):

            if g5x_t[i][j][0] == 0:
                g5x[i][j][0] = g5x_t[i][j][0] + (g01 - 0)
            if g5x_t[i][j][0] == 1:
                g5x[i][j][0] = g5x_t[i][j][0] + (g02 - 1)
            if g5x_t[i][j][0] == 2:
                g5x[i][j][0] = g5x_t[i][j][0] + (g03 - 2)
            if g5x_t[i][j][0] == 3:
                g5x[i][j][0] = g5x_t[i][j][0] + (g04 - 3)
            if g5x_t[i][j][0] == 4:
                g5x[i][j][0] = g5x_t[i][j][0] + (g05 - 4)

            if g5x_t[i][j][1] == 0:
                g5x[i][j][1] = g5x_t[i][j][1] + (g01 - 0)
            if g5x_t[i][j][1] == 1:
                g5x[i][j][1] = g5x_t[i][j][1] + (g02 - 1)
            if g5x_t[i][j][1] == 2:
                g5x[i][j][1] = g5x_t[i][j][1] + (g03 - 2)
            if g5x_t[i][j][1] == 3:
                g5x[i][j][1] = g5x_t[i][j][1] + (g04 - 3)
            if g5x_t[i][j][1] == 4:
                g5x[i][j][1] = g5x_t[i][j][1] + (g05 - 4)

            _cond1_list.append(g5x[i][j])

        _cond1 = tuple(_cond1_list)
        g5x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12345_sn_pass)
        g5x_filtered_sum[0, :, :] += g5x_filtered[i, :, :]
    

    print("after update")
    print(g5x[0])
    print(g5x[1])
    print(g5x[2])
    print(g5x[3])
    print(g5x[4])
    print(g5x[5])
    print(g5x[6])
    print(g5x[7])

    g5_opt_snp5 = g1x_filtered_sum + g2x_filtered_sum + g3x_filtered_sum + g4x_filtered_sum + g5x_filtered_sum + 10*(n_G1x + n_G2x + n_G3x + n_G4x + n_G5x - 1)

    return g5_opt_snp5




def g3_opt_bf_snp(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, sn_pass_ng_opt, g01, g02, g03):

    gx_sn_flag = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    for i in range(0, max_ngauss):
        gx_sorted = g_num_sort[i, :, :]
        gx_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, gx_sorted[np.newaxis], axis=0)[0]
        gx_sn_flag[i, :, :] = np.array([np.where(gx_sn_pass_ng_opt_value == (gx_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

        print(gx_sn_flag[i, :, :])
    gx_list = [g01, g02, g03]

    n_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            n_sn_pass += 1

    gx_sn_pass = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)
    gx_opt_bf_snp = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)
    gx_opt_bf_tx = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)

    flag_string = ''
    for k3 in range(0, max_ngauss):
        flag_string_seg = '(gx_sn_flag[%d, :, :] == 0)' % k3
    
        flag_string = flag_string + flag_string_seg
        if k3 < max_ngauss-1:
            flag_string = flag_string + ' & '

    gx_sn_pass[0, :, :] = np.array([np.where(eval(flag_string), 1, 0)])[0]

    n_sn_pass = 1 # count from 1
    flag_string = ''
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            TF_flag = list(j) # True False flag
            print(TF_flag)

            flag_string = ''
            flag_string = ''

            k3_s = 0
            k2 = 0 
            while True:
                for k3 in range(k3_s, max_ngauss):
                    if k3 == TF_flag[k2]:
                        flag_string_seg = '(gx_sn_flag[%d, :, :] == 1)' % k3
                        flag_string = flag_string + flag_string_seg
                        if k3 < max_ngauss-1:
                            flag_string = flag_string + ' & '
                        
                        if k2 < len(TF_flag)-1:
                            k2 += 1
                        break
                    else:
                        flag_string_seg = '(gx_sn_flag[%d, :, :] == 0)' % k3
                
                        flag_string = flag_string + flag_string_seg
                        if k3 < max_ngauss-1:
                            flag_string = flag_string + ' & '

                if k3 < max_ngauss-1:
                    k3_s = k3+1
                else:
                    break # exit the while loop


    
            print(flag_string)
            gx_sn_pass[n_sn_pass, :, :] = np.array([np.where(eval(flag_string), 1, 0)])[0]
            n_sn_pass += 1





    i1 = _params['_i0']
    j1 = _params['_j0']


    print("")




    gx_opt_bf_snp[0, :, :] = snp0_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[0, :, :])

    n_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            print(list(j)[0])
            if len(j) == 1:
                gx1 = list(j)[0]
                gx_opt_bf_snp[n_sn_pass, :, :] = snp1_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1)
            elif len(j) == 2:
                gx1 = list(j)[0]
                gx2 = list(j)[1]
                gx_opt_bf_snp[n_sn_pass, :, :] = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2)
            elif len(j) == 3:
                gx1 = list(j)[0]
                gx2 = list(j)[1]
                gx3 = list(j)[2]
                gx_opt_bf_snp[n_sn_pass, :, :] = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2, gx3)
            elif len(j) == 4:
                gx1 = list(j)[0]
                gx2 = list(j)[1]
                gx3 = list(j)[2]
                gx4 = list(j)[3]
                gx_opt_bf_snp[n_sn_pass, :, :] = snp4_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2, gx3, gx4)


            n_sn_pass += 1




    gx_opt_bf_tx[0, :, :] = np.array([np.where(gx_opt_bf_snp[0, :, :] > -1, gx_opt_bf_snp[0, :, :], -1)][0])

    n_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            gx_opt_bf_tx[n_sn_pass, :, :] = np.array([np.where(gx_opt_bf_snp[n_sn_pass, :, :] > -1, gx_opt_bf_snp[n_sn_pass, :, :], gx_opt_bf_tx[n_sn_pass-1, :, :])][0])
            n_sn_pass += 1

    print("optimal-ng:", gx_opt_bf_tx[n_sn_pass-1, j1, i1])
    return gx_opt_bf_tx[n_sn_pass-1, :, :]






def g4_opt_bf_snp(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, sn_pass_ng_opt, g01, g02, g03, g04):

    gx_sn_flag = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    for i in range(0, max_ngauss):
        gx_sorted = g_num_sort[i, :, :]
        gx_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, gx_sorted[np.newaxis], axis=0)[0]
        gx_sn_flag[i, :, :] = np.array([np.where(gx_sn_pass_ng_opt_value == (gx_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

        print(gx_sn_flag[i, :, :])
    gx_list = [g01, g02, g03, g04]

    n_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            n_sn_pass += 1

    gx_sn_pass = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)
    gx_opt_bf_snp = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)
    gx_opt_bf_tx = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)

    flag_string = ''
    for k3 in range(0, max_ngauss):
        flag_string_seg = '(gx_sn_flag[%d, :, :] == 0)' % k3
    
        flag_string = flag_string + flag_string_seg
        if k3 < max_ngauss-1:
            flag_string = flag_string + ' & '

    gx_sn_pass[0, :, :] = np.array([np.where(eval(flag_string), 1, 0)])[0]

    n_sn_pass = 1 # count from 1
    flag_string = ''
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            TF_flag = list(j) # True False flag
            print(TF_flag)

            flag_string = ''
            flag_string = ''

            k3_s = 0
            k2 = 0 
            while True:
                for k3 in range(k3_s, max_ngauss):
                    if k3 == TF_flag[k2]:
                        flag_string_seg = '(gx_sn_flag[%d, :, :] == 1)' % k3
                        flag_string = flag_string + flag_string_seg
                        if k3 < max_ngauss-1:
                            flag_string = flag_string + ' & '
                        
                        if k2 < len(TF_flag)-1:
                            k2 += 1
                        break
                    else:
                        flag_string_seg = '(gx_sn_flag[%d, :, :] == 0)' % k3
                
                        flag_string = flag_string + flag_string_seg
                        if k3 < max_ngauss-1:
                            flag_string = flag_string + ' & '

                if k3 < max_ngauss-1:
                    k3_s = k3+1
                else:
                    break # exit the while loop


    
            print(flag_string)
            gx_sn_pass[n_sn_pass, :, :] = np.array([np.where(eval(flag_string), 1, 0)])[0]
            n_sn_pass += 1





    i1 = _params['_i0']
    j1 = _params['_j0']
    g1234_sn_flag = gx_sn_pass[1, :, :] + gx_sn_pass[2, :, :] + gx_sn_pass[3, :, :] + gx_sn_pass[4, :, :]


    print("ng-23: ", "sn_pass: ", np.where(g1234_sn_flag == 4))
    print("")
    print("ng-1: ", gx_sorted[j1, i1], "sn_flag: ", gx_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", gx_sn_flag[0, j1, i1])
    print("ng-2: ", gx_sorted[j1, i1], "sn_flag: ", gx_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", gx_sn_flag[1, j1, i1])
    print("ng-3: ", gx_sorted[j1, i1], "sn_flag: ", gx_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", gx_sn_flag[2, j1, i1])
    print("ng-4: ", gx_sorted[j1, i1], "sn_flag: ", gx_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", gx_sn_flag[3, j1, i1])

    g1234_sn_pass = np.array([np.where(((gx_sn_flag[0, :, :]== 1) & (gx_sn_flag[1, :, :] == 1) & (gx_sn_flag[2, :, :] == 1) & (gx_sn_flag[3, :, :] == 1)), 1, 0)])[0]

    print("here")
    print(g1234_sn_pass)
    print("")
    print(gx_sn_pass[15, :, :])
    print("here")


    gx_opt_bf_snp[0, :, :] = snp0_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[0, :, :])

    n_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            print(list(j)[0])
            if len(j) == 1:
                gx1 = list(j)[0]
                gx_opt_bf_snp[n_sn_pass, :, :] = snp1_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1)
            elif len(j) == 2:
                gx1 = list(j)[0]
                gx2 = list(j)[1]
                gx_opt_bf_snp[n_sn_pass, :, :] = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2)
            elif len(j) == 3:
                gx1 = list(j)[0]
                gx2 = list(j)[1]
                gx3 = list(j)[2]
                gx_opt_bf_snp[n_sn_pass, :, :] = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2, gx3)
            elif len(j) == 4:
                gx1 = list(j)[0]
                gx2 = list(j)[1]
                gx3 = list(j)[2]
                gx4 = list(j)[3]
                gx_opt_bf_snp[n_sn_pass, :, :] = snp4_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2, gx3, gx4)


            n_sn_pass += 1




    gx_opt_bf_tx[0, :, :] = np.array([np.where(gx_opt_bf_snp[0, :, :] > -1, gx_opt_bf_snp[0, :, :], -1)][0])

    n_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            gx_opt_bf_tx[n_sn_pass, :, :] = np.array([np.where(gx_opt_bf_snp[n_sn_pass, :, :] > -1, gx_opt_bf_snp[n_sn_pass, :, :], gx_opt_bf_tx[n_sn_pass-1, :, :])][0])
            n_sn_pass += 1

    print("optimal-ng:", gx_opt_bf_tx[15, j1, i1])
    return gx_opt_bf_tx[15, :, :]



def gx_opt_bf_snp_org(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, sn_pass_ng_opt, gx_list):

    gx_sn_flag = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, max_ngauss):
        gx_sorted = g_num_sort[i, :, :]
        gx_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, gx_sorted[np.newaxis], axis=0)[0]
        gx_sn_flag[i, :, :] = np.array([np.where(gx_sn_pass_ng_opt_value == (gx_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0


    n_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            n_sn_pass += 1

    gx_sn_pass = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)
    gx_opt_bf_snp = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)
    gx_opt_bf_tx = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)

    flag_string = ''
    for k3 in range(0, max_ngauss):
        flag_string_seg = '(gx_sn_flag[%d, :, :] == 0)' % k3
    
        flag_string = flag_string + flag_string_seg
        if k3 < max_ngauss-1:
            flag_string = flag_string + ' & '

    n_sn_pass = 0
    gx_sn_pass[n_sn_pass, :, :] = np.array([np.where(eval(flag_string), 1, 0)])[0]

    n_sn_pass = 1 # count from 1
    flag_string = ''
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            TF_flag = list(j) # True False flag
            print(TF_flag)

            flag_string = ''
            flag_string = ''

            k3_s = 0
            k2 = 0 
            while True:
                for k3 in range(k3_s, max_ngauss):
                    if k3 == TF_flag[k2]:
                        flag_string_seg = '(gx_sn_flag[%d, :, :] == 1)' % k3
                        flag_string = flag_string + flag_string_seg
                        if k3 < max_ngauss-1:
                            flag_string = flag_string + ' & '
                        
                        if k2 < len(TF_flag)-1:
                            k2 += 1
                        break
                    else:
                        flag_string_seg = '(gx_sn_flag[%d, :, :] == 0)' % k3
                
                        flag_string = flag_string + flag_string_seg
                        if k3 < max_ngauss-1:
                            flag_string = flag_string + ' & '

                if k3 < max_ngauss-1:
                    k3_s = k3+1
                else:
                    break # exit the while loop


    
            gx_sn_pass[n_sn_pass, :, :] = np.array([np.where(eval(flag_string), 1, 0)])[0]
            n_sn_pass += 1

    i1 = _params['_i0']
    j1 = _params['_j0']
    n_sn_pass = 0
    gx_opt_bf_snp[n_sn_pass, :, :] = snp0_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :])

    n_sn_pass = 1 # count from 1


    n_GNx_list = [0 for i in range(0, 6)]
    n_GNx_list[1] = [1] # from the condition matrix
    n_GNx_list[2] = [2, 1] # from the condition matrix
    n_GNx_list[3] = [3, 2, 2] # from the condition matrix
    n_GNx_list[4] = [4, 3, 4, 4] # from the condition matrix
    n_GNx_list[5] = [5, 4, 6, 8, 8] # from the condition matrix
    print(n_GNx_list[1])
    print(n_GNx_list[2])
    print(n_GNx_list[3])
    print(n_GNx_list[4])
    print(n_GNx_list[5])

    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):

            if len(j) == 1:
                gx1 = list(j)[0]
                gx_opt_bf_snp[n_sn_pass, :, :] = snp1_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1)
            elif len(j) == 2:
                gx1 = list(j)[0]
                gx2 = list(j)[1]
                gx_opt_bf_snp[n_sn_pass, :, :] = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2)
                print("GGDDDDD")
                print("GGDDDDD")
                print("GGDDDDD")
                print("GGDDDDD")
            elif len(j) == 3:
                gx_list = list(j)
                n_GNx_list = [3, 2, 2] # from the condition matrix
                gx_opt_bf_snp[n_sn_pass, :, :] = snp_gx_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, gx_sn_pass[n_sn_pass, :, :], gx_list, n_GNx_list)
            elif len(j) == 4:
                gx_list = list(j)
                n_GNx_list = [4, 3, 4, 4] # from the condition matrix
                gx_opt_bf_snp[n_sn_pass, :, :] = snp_gx_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, gx_sn_pass[n_sn_pass, :, :], gx_list, n_GNx_list)
            elif len(j) == 5:
                gx1 = list(j)[0]
                gx2 = list(j)[1]
                gx3 = list(j)[2]
                gx4 = list(j)[3]
                gx5 = list(j)[4]
                gx_opt_bf_snp[n_sn_pass, :, :] = snp5_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2, gx3, gx4, gx5)
            elif len(j) == 6:
                gx1 = list(j)[0]
                gx2 = list(j)[1]
                gx3 = list(j)[2]
                gx4 = list(j)[3]
                gx5 = list(j)[4]
                gx6 = list(j)[5]
                gx_opt_bf_snp[n_sn_pass, :, :] = snp6_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2, gx3, gx4, gx5, gx6)
            elif len(j) == 7:
                gx2 = list(j)[1]
                gx3 = list(j)[2]
                gx4 = list(j)[3]
                gx5 = list(j)[4]
                gx6 = list(j)[5]
                gx7 = list(j)[6]
                gx_opt_bf_snp[n_sn_pass, :, :] = snp7_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2, gx3, gx4, gx5, gx6, gx7)

            n_sn_pass += 1


    n_sn_pass = 0
    gx_opt_bf_tx[n_sn_pass, :, :] = np.array([np.where(gx_opt_bf_snp[0, :, :] > -1, gx_opt_bf_snp[n_sn_pass, :, :], -1)][0])

    n_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            gx_opt_bf_tx[n_sn_pass, :, :] = np.array([np.where(gx_opt_bf_snp[n_sn_pass, :, :] > -1, gx_opt_bf_snp[n_sn_pass, :, :], gx_opt_bf_tx[n_sn_pass-1, :, :])][0])
            n_sn_pass += 1

    print("optimal-ng:", gx_opt_bf_tx[n_sn_pass-1, j1, i1])
    return gx_opt_bf_tx[n_sn_pass-1, :, :]



def find_gx_opt_bf_snp_dev(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, sn_pass_ng_opt, gx_list):

    gx_sn_flag = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, max_ngauss):
        gx_sorted = g_num_sort[i, :, :]
        gx_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, gx_sorted[np.newaxis], axis=0)[0]
        gx_sn_flag[i, :, :] = np.array([np.where(gx_sn_pass_ng_opt_value == (gx_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0


    n_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            n_sn_pass += 1

    gx_sn_pass = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)
    gx_opt_bf_snp = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)
    gx_opt_bf_tx = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)

    flag_string = ''
    for k3 in range(0, max_ngauss):
        flag_string_seg = '(gx_sn_flag[%d, :, :] == 0)' % k3
    
        flag_string = flag_string + flag_string_seg
        if k3 < max_ngauss-1:
            flag_string = flag_string + ' & '

    n_sn_pass = 0
    gx_sn_pass[n_sn_pass, :, :] = np.array([np.where(eval(flag_string), 1, 0)])[0]

    n_sn_pass = 1 # count from 1
    flag_string = ''
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            TF_flag = list(j) # True False flag
            print(TF_flag)

            flag_string = ''
            flag_string = ''

            k3_s = 0
            k2 = 0 
            while True:
                for k3 in range(k3_s, max_ngauss):
                    if k3 == TF_flag[k2]:
                        flag_string_seg = '(gx_sn_flag[%d, :, :] == 1)' % k3
                        flag_string = flag_string + flag_string_seg
                        if k3 < max_ngauss-1:
                            flag_string = flag_string + ' & '
                        
                        if k2 < len(TF_flag)-1:
                            k2 += 1
                        break
                    else:
                        flag_string_seg = '(gx_sn_flag[%d, :, :] == 0)' % k3
                
                        flag_string = flag_string + flag_string_seg
                        if k3 < max_ngauss-1:
                            flag_string = flag_string + ' & '

                if k3 < max_ngauss-1:
                    k3_s = k3+1
                else:
                    break # exit the while loop


    
            gx_sn_pass[n_sn_pass, :, :] = np.array([np.where(eval(flag_string), 1, 0)])[0]
            n_sn_pass += 1

    n_sn_pass = 0
    gx_opt_bf_snp[n_sn_pass, :, :] = snp0_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :])

    n_sn_pass = 1 # count from 1


    n_GNx_list = [0 for i in range(0, 6)]
    n_GNx_list[1] = [1] # from the condition matrix
    n_GNx_list[2] = [2, 1] # from the condition matrix
    n_GNx_list[3] = [3, 2, 2] # from the condition matrix
    n_GNx_list[4] = [4, 3, 4, 4] # from the condition matrix
    n_GNx_list[5] = [5, 4, 6, 8, 8] # from the condition matrix
    print(n_GNx_list[1])
    print(n_GNx_list[2])
    print(n_GNx_list[3])
    print(n_GNx_list[4])
    print(n_GNx_list[5])

    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            print(len(j), j, "seheon")

            gx_list_comb = list(j)
            GNx_index_comb = len(j)
            gx_opt_bf_snp[n_sn_pass, :, :] = snp_gx_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, gx_sn_pass[n_sn_pass, :, :], gx_list_comb, n_GNx_list[GNx_index_comb])




            n_sn_pass += 1


    n_sn_pass = 0
    gx_opt_bf_tx[n_sn_pass, :, :] = np.array([np.where(gx_opt_bf_snp[0, :, :] > -1, gx_opt_bf_snp[n_sn_pass, :, :], -1)][0])

    n_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            gx_opt_bf_tx[n_sn_pass, :, :] = np.array([np.where(gx_opt_bf_snp[n_sn_pass, :, :] > -1, gx_opt_bf_snp[n_sn_pass, :, :], gx_opt_bf_tx[n_sn_pass-1, :, :])][0])
            n_sn_pass += 1

    i1 = _params['_i0']
    j1 = _params['_j0']
    print("optimal-ng-seheon:", gx_opt_bf_tx[n_sn_pass-1, j1, i1])
    return gx_opt_bf_tx[n_sn_pass-1, :, :]



def find_gx_opt_bf_snp(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, sn_pass_ng_opt, gx_list):

    gx_sn_flag = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, max_ngauss):
        gx_sorted = g_num_sort[i, :, :]

        gx_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, gx_sorted[np.newaxis], axis=0)[0]

        gx_sn_flag[i, :, :] = np.array([np.where(gx_sn_pass_ng_opt_value == (gx_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0






    loop_index_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            loop_index_sn_pass += 1

    gx_sn_pass = np.zeros((loop_index_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)
    gx_opt_bf_snp = np.zeros((loop_index_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)
    gx_opt_bf_tx = np.zeros((loop_index_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)

    flag_string = ''
    for k3 in range(0, max_ngauss):
        flag_string_seg = '(gx_sn_flag[%d, :, :] == 0)' % k3
    
        flag_string = flag_string + flag_string_seg
        if k3 < max_ngauss-1:
            flag_string = flag_string + ' & '



    loop_index_sn_pass = 0
    gx_sn_pass[loop_index_sn_pass, :, :] = np.array([np.where(eval(flag_string), 1, 0)])[0]


    loop_index_sn_pass = 1 # count from 1
    flag_string = ''
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            TF_flag = list(j) # True False flag

            flag_string = ''
            flag_string = ''

            k3_s = 0
            k2 = 0 
            while True:
                for k3 in range(k3_s, max_ngauss):
                    if k3 == TF_flag[k2]:
                        flag_string_seg = '(gx_sn_flag[%d, :, :] == 1)' % k3
                        flag_string = flag_string + flag_string_seg
                        if k3 < max_ngauss-1:
                            flag_string = flag_string + ' & '
                        
                        if k2 < len(TF_flag)-1:
                            k2 += 1
                        break
                    else:
                        flag_string_seg = '(gx_sn_flag[%d, :, :] == 0)' % k3
                
                        flag_string = flag_string + flag_string_seg
                        if k3 < max_ngauss-1:
                            flag_string = flag_string + ' & '

                if k3 < max_ngauss-1:
                    k3_s = k3+1
                else:
                    break # exit the while loop


            gx_sn_pass[loop_index_sn_pass, :, :] = np.array([np.where(eval(flag_string), 1, 0)])[0]


            loop_index_sn_pass += 1


    loop_index_sn_pass = 0 # the 1st combination
    gx_opt_bf_snp[loop_index_sn_pass, :, :] = snp0_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[loop_index_sn_pass, :, :])

    loop_index_sn_pass = 1 # count from 1, for the other combinations.. 


    n_GNx_list = [[0 for _i1 in range(0)] for _j1 in range(max_ngauss+1)]

    for gx_1 in range(1, max_ngauss+1):
        n_GNx_list[gx_1].append(gx_1)
        for gx_2 in range(0, gx_1-1):
            n_GNx_list[gx_1].append((2**gx_2)*(gx_1-1-gx_2))


    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            gx_list_comb = list(j)
            GNx_index_comb = len(j)
            gx_opt_bf_snp[loop_index_sn_pass, :, :] = snp_gx_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, gx_sn_pass[loop_index_sn_pass, :, :], gx_list_comb, n_GNx_list[GNx_index_comb])

            loop_index_sn_pass += 1


    loop_index_sn_pass = 0 # null case : [] : no gaussians pass the s/n limit  --> put -1
    gx_opt_bf_tx[loop_index_sn_pass, :, :] = np.array([np.where(gx_opt_bf_snp[0, :, :] > -1, gx_opt_bf_snp[loop_index_sn_pass, :, :], -1)][0])



    loop_index_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            gx_opt_bf_tx[loop_index_sn_pass, :, :] = np.array([np.where(gx_opt_bf_snp[loop_index_sn_pass, :, :] > -1, gx_opt_bf_snp[loop_index_sn_pass, :, :], gx_opt_bf_tx[loop_index_sn_pass-1, :, :])][0])


            loop_index_sn_pass += 1



    gx_opt_bf_tx[loop_index_sn_pass-1, :, :] = np.array([np.where((gx_opt_bf_tx[loop_index_sn_pass-1, :, :] == -1) & (sn_pass_ng_opt[0, :, :] == 1.0) , 0, gx_opt_bf_tx[loop_index_sn_pass-1, :, :])][0])




    return gx_opt_bf_tx[loop_index_sn_pass-1, :, :]



def g4_opt_bf_snp_org(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, sn_pass_ng_opt, g01, g02, g03, g04):

    g1_sorted = g_num_sort[g01, :, :]
    g1_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]
    g1_sn_flag = np.array([np.where(g1_sn_pass_ng_opt_value == (g1_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

    g2_sorted = g_num_sort[g02, :, :]
    g2_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g2_sorted[np.newaxis], axis=0)[0]
    g2_sn_flag = np.array([np.where(g2_sn_pass_ng_opt_value == (g2_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

    g3_sorted = g_num_sort[g03, :, :]
    g3_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g3_sorted[np.newaxis], axis=0)[0]
    g3_sn_flag = np.array([np.where(g3_sn_pass_ng_opt_value == (g3_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

    g4_sorted = g_num_sort[g04, :, :]
    g4_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g4_sorted[np.newaxis], axis=0)[0]
    g4_sn_flag = np.array([np.where(g4_sn_pass_ng_opt_value == (g4_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

    g1_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 0) & (g4_sn_flag == 0)), 1, 0)])[0]
    g2_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1) & (g3_sn_flag == 0) & (g4_sn_flag == 0)), 1, 0)])[0]
    g3_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0) & (g3_sn_flag == 1) & (g4_sn_flag == 0)), 1, 0)])[0]
    g4_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0) & (g3_sn_flag == 0) & (g4_sn_flag == 1)), 1, 0)])[0]

    g12_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1) & (g3_sn_flag == 0) & (g4_sn_flag == 0)), 1, 0)])[0]
    g13_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 1) & (g4_sn_flag == 0)), 1, 0)])[0]
    g14_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 0) & (g4_sn_flag == 1)), 1, 0)])[0]
    g23_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1) & (g3_sn_flag == 1) & (g4_sn_flag == 0)), 1, 0)])[0]
    g24_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1) & (g3_sn_flag == 0) & (g4_sn_flag == 1)), 1, 0)])[0]
    g34_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0) & (g3_sn_flag == 1) & (g4_sn_flag == 1)), 1, 0)])[0]

    g123_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1) & (g3_sn_flag == 1) & (g4_sn_flag == 0)), 1, 0)])[0]
    g124_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1) & (g3_sn_flag == 0) & (g4_sn_flag == 1)), 1, 0)])[0]
    g134_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 1) & (g4_sn_flag == 1)), 1, 0)])[0]
    g234_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1) & (g3_sn_flag == 1) & (g4_sn_flag == 1)), 1, 0)])[0]

    g0_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0) & (g3_sn_flag == 0) & (g4_sn_flag == 0)), 1, 0)])[0]
    g1234_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1) & (g3_sn_flag == 1) & (g4_sn_flag == 1)), 1, 0)])[0]

    i1 = _params['_i0']
    j1 = _params['_j0']
    g1234_sn_flag = g1_sn_flag + g2_sn_flag + g3_sn_flag + g4_sn_flag
    print("ng-23: ", "sn_pass: ", np.where(g1234_sn_flag == 4))
    print("")
    print("ng-1: ", g1_sorted[j1, i1], "sn_flag: ", g1_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g1_sn_flag[j1, i1])
    print("ng-2: ", g2_sorted[j1, i1], "sn_flag: ", g2_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g2_sn_flag[j1, i1])
    print("ng-3: ", g3_sorted[j1, i1], "sn_flag: ", g3_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g3_sn_flag[j1, i1])
    print("ng-4: ", g4_sorted[j1, i1], "sn_flag: ", g4_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g4_sn_flag[j1, i1])

    _g4_opt_bf_snp_1 = snp1_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1_sn_pass, g01)
    _g4_opt_bf_snp_2 = snp1_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g2_sn_pass, g02)
    _g4_opt_bf_snp_3 = snp1_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g3_sn_pass, g03)
    _g4_opt_bf_snp_4 = snp1_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g4_sn_pass, g04)

    _g4_opt_bf_snp_12 = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g12_sn_pass, g01, g02)
    _g4_opt_bf_snp_13 = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g13_sn_pass, g01, g03)
    _g4_opt_bf_snp_14 = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g14_sn_pass, g01, g04)
    _g4_opt_bf_snp_23 = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g23_sn_pass, g02, g03)
    _g4_opt_bf_snp_24 = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g24_sn_pass, g02, g04)
    _g4_opt_bf_snp_34 = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g34_sn_pass, g03, g04)

    _g4_opt_bf_snp_123 = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g123_sn_pass, g01, g02, g03)
    _g4_opt_bf_snp_124 = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g124_sn_pass, g01, g02, g04)
    _g4_opt_bf_snp_134 = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g134_sn_pass, g01, g03, g04)
    _g4_opt_bf_snp_234 = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g234_sn_pass, g02, g03, g04)

    _g4_opt_bf_snp_0 = snp0_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g0_sn_pass)
    _g4_opt_bf_snp_1234 = snp4_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1234_sn_pass, g01, g02, g03, g04)

    print(g23_sn_pass[j1, i1])
    print(_g4_opt_bf_snp_23[0, j1, i1])

    _g4_opt_t1 = np.array([np.where(_g4_opt_bf_snp_0 > -1, _g4_opt_bf_snp_0, -1)][0])

    _g4_opt_t2 = np.array([np.where(_g4_opt_bf_snp_1 > -1, _g4_opt_bf_snp_1, _g4_opt_t1)][0])
    _g4_opt_t3 = np.array([np.where(_g4_opt_bf_snp_2 > -1, _g4_opt_bf_snp_2, _g4_opt_t2)][0])
    _g4_opt_t4 = np.array([np.where(_g4_opt_bf_snp_3 > -1, _g4_opt_bf_snp_3, _g4_opt_t3)][0])
    _g4_opt_t5 = np.array([np.where(_g4_opt_bf_snp_4 > -1, _g4_opt_bf_snp_4, _g4_opt_t4)][0])

    _g4_opt_t6 = np.array([np.where(_g4_opt_bf_snp_12 > -1, _g4_opt_bf_snp_12, _g4_opt_t5)][0])
    _g4_opt_t7 = np.array([np.where(_g4_opt_bf_snp_13 > -1, _g4_opt_bf_snp_13, _g4_opt_t6)][0])
    _g4_opt_t8 = np.array([np.where(_g4_opt_bf_snp_14 > -1, _g4_opt_bf_snp_14, _g4_opt_t7)][0])
    _g4_opt_t9 = np.array([np.where(_g4_opt_bf_snp_23 > -1, _g4_opt_bf_snp_23, _g4_opt_t8)][0])
    _g4_opt_t10 = np.array([np.where(_g4_opt_bf_snp_24 > -1, _g4_opt_bf_snp_24, _g4_opt_t9)][0])
    _g4_opt_t11 = np.array([np.where(_g4_opt_bf_snp_34 > -1, _g4_opt_bf_snp_34, _g4_opt_t10)][0])

    _g4_opt_t12 = np.array([np.where(_g4_opt_bf_snp_123 > -1, _g4_opt_bf_snp_123, _g4_opt_t11)][0])
    _g4_opt_t13 = np.array([np.where(_g4_opt_bf_snp_124 > -1, _g4_opt_bf_snp_124, _g4_opt_t12)][0])
    _g4_opt_t14 = np.array([np.where(_g4_opt_bf_snp_134 > -1, _g4_opt_bf_snp_134, _g4_opt_t13)][0])
    _g4_opt_t15 = np.array([np.where(_g4_opt_bf_snp_234 > -1, _g4_opt_bf_snp_234, _g4_opt_t14)][0])

    _g4_opt_t16 = np.array([np.where(_g4_opt_bf_snp_1234 > -1, _g4_opt_bf_snp_1234, _g4_opt_t15)][0])


    print("optimal-ng:", _g4_opt_t16[0, j1, i1])
    return _g4_opt_t16


def g5_opt_bf_snp(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, sn_pass_ng_opt, g01, g02, g03, g04, g05):

    g1_sorted = g_num_sort[g01, :, :]
    g1_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]
    g1_sn_flag = np.array([np.where(g1_sn_pass_ng_opt_value == (g1_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

    g2_sorted = g_num_sort[g02, :, :]
    g2_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g2_sorted[np.newaxis], axis=0)[0]
    g2_sn_flag = np.array([np.where(g2_sn_pass_ng_opt_value == (g2_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

    g3_sorted = g_num_sort[g03, :, :]
    g3_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g3_sorted[np.newaxis], axis=0)[0]
    g3_sn_flag = np.array([np.where(g3_sn_pass_ng_opt_value == (g3_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

    g4_sorted = g_num_sort[g04, :, :]
    g4_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g4_sorted[np.newaxis], axis=0)[0]
    g4_sn_flag = np.array([np.where(g4_sn_pass_ng_opt_value == (g4_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

    g5_sorted = g_num_sort[g05, :, :]
    g5_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g5_sorted[np.newaxis], axis=0)[0]
    g5_sn_flag = np.array([np.where(g5_sn_pass_ng_opt_value == (g5_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

    g1_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 0) & (g4_sn_flag == 0) & (g5_sn_flag == 0)), 1, 0)])[0]
    g2_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1) & (g3_sn_flag == 0) & (g4_sn_flag == 0) & (g5_sn_flag == 0)), 1, 0)])[0]
    g3_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0) & (g3_sn_flag == 1) & (g4_sn_flag == 0) & (g5_sn_flag == 0)), 1, 0)])[0]
    g4_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0) & (g3_sn_flag == 0) & (g4_sn_flag == 1) & (g5_sn_flag == 0)), 1, 0)])[0]
    g5_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0) & (g3_sn_flag == 0) & (g4_sn_flag == 0) & (g5_sn_flag == 1)), 1, 0)])[0]

    g12_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1) & (g3_sn_flag == 0) & (g4_sn_flag == 0) & (g5_sn_flag == 0)), 1, 0)])[0]
    g13_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 1) & (g4_sn_flag == 0) & (g5_sn_flag == 0)), 1, 0)])[0]
    g14_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 0) & (g4_sn_flag == 1) & (g5_sn_flag == 0)), 1, 0)])[0]
    g15_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 0) & (g4_sn_flag == 0) & (g5_sn_flag == 1)), 1, 0)])[0]
    g23_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1) & (g3_sn_flag == 1) & (g4_sn_flag == 0) & (g5_sn_flag == 0)), 1, 0)])[0]
    g24_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1) & (g3_sn_flag == 0) & (g4_sn_flag == 1) & (g5_sn_flag == 0)), 1, 0)])[0]
    g25_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1) & (g3_sn_flag == 0) & (g4_sn_flag == 0) & (g5_sn_flag == 1)), 1, 0)])[0]
    g34_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0) & (g3_sn_flag == 1) & (g4_sn_flag == 1) & (g5_sn_flag == 0)), 1, 0)])[0]
    g35_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0) & (g3_sn_flag == 1) & (g4_sn_flag == 0) & (g5_sn_flag == 1)), 1, 0)])[0]
    g45_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0) & (g3_sn_flag == 0) & (g4_sn_flag == 1) & (g5_sn_flag == 1)), 1, 0)])[0]

    g123_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1) & (g3_sn_flag == 1) & (g4_sn_flag == 0) & (g5_sn_flag == 0)), 1, 0)])[0]
    g124_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1) & (g3_sn_flag == 0) & (g4_sn_flag == 1) & (g5_sn_flag == 1)), 1, 0)])[0]
    g125_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1) & (g3_sn_flag == 0) & (g4_sn_flag == 1) & (g5_sn_flag == 1)), 1, 0)])[0]
    g134_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 1) & (g4_sn_flag == 1) & (g5_sn_flag == 0)), 1, 0)])[0]
    g135_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 1) & (g4_sn_flag == 0) & (g5_sn_flag == 1)), 1, 0)])[0]
    g145_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 0) & (g4_sn_flag == 1) & (g5_sn_flag == 1)), 1, 0)])[0]
    g234_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1) & (g3_sn_flag == 1) & (g4_sn_flag == 1) & (g5_sn_flag == 0)), 1, 0)])[0]
    g235_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1) & (g3_sn_flag == 1) & (g4_sn_flag == 0) & (g5_sn_flag == 1)), 1, 0)])[0]
    g245_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1) & (g3_sn_flag == 0) & (g4_sn_flag == 1) & (g5_sn_flag == 1)), 1, 0)])[0]
    g345_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0) & (g3_sn_flag == 1) & (g4_sn_flag == 1) & (g5_sn_flag == 1)), 1, 0)])[0]

    g1234_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1) & (g3_sn_flag == 1) & (g4_sn_flag == 1) & (g5_sn_flag == 0)), 1, 0)])[0]
    g1235_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1) & (g3_sn_flag == 1) & (g4_sn_flag == 0) & (g5_sn_flag == 1)), 1, 0)])[0]
    g1245_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1) & (g3_sn_flag == 0) & (g4_sn_flag == 1) & (g5_sn_flag == 1)), 1, 0)])[0]
    g1345_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 1) & (g4_sn_flag == 1) & (g5_sn_flag == 1)), 1, 0)])[0]
    g2345_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1) & (g3_sn_flag == 1) & (g4_sn_flag == 1) & (g5_sn_flag == 1)), 1, 0)])[0]

    g0_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0) & (g3_sn_flag == 0) & (g4_sn_flag == 0) & (g5_sn_flag == 0)), 1, 0)])[0]
    g12345_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1) & (g3_sn_flag == 1) & (g4_sn_flag == 1) & (g5_sn_flag == 1)), 1, 0)])[0]

    i1 = _params['_i0']
    j1 = _params['_j0']

    g12345_sn_flag = g1_sn_flag + g2_sn_flag + g3_sn_flag + g4_sn_flag + g5_sn_flag
    print("ng-23: ", "sn_pass: ", np.where(g12345_sn_flag == 5))
    print("")
    print("ng-1: ", g1_sorted[j1, i1], "sn_flag: ", g1_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g1_sn_flag[j1, i1])
    print("ng-2: ", g2_sorted[j1, i1], "sn_flag: ", g2_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g2_sn_flag[j1, i1])
    print("ng-3: ", g3_sorted[j1, i1], "sn_flag: ", g3_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g3_sn_flag[j1, i1])
    print("ng-4: ", g4_sorted[j1, i1], "sn_flag: ", g4_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g4_sn_flag[j1, i1])
    print("ng-5: ", g5_sorted[j1, i1], "sn_flag: ", g5_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g5_sn_flag[j1, i1])

    _g5_opt_bf_snp_1 = snp1_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1_sn_pass, g01)
    _g5_opt_bf_snp_2 = snp1_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g2_sn_pass, g02)
    _g5_opt_bf_snp_3 = snp1_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g3_sn_pass, g03)
    _g5_opt_bf_snp_4 = snp1_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g4_sn_pass, g04)
    _g5_opt_bf_snp_5 = snp1_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g5_sn_pass, g05)

    _g5_opt_bf_snp_12 = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g12_sn_pass, g01, g02)
    _g5_opt_bf_snp_13 = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g13_sn_pass, g01, g03)
    _g5_opt_bf_snp_14 = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g14_sn_pass, g01, g04)
    _g5_opt_bf_snp_15 = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g15_sn_pass, g01, g05)
    _g5_opt_bf_snp_23 = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g23_sn_pass, g02, g03)
    _g5_opt_bf_snp_24 = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g24_sn_pass, g02, g04)
    _g5_opt_bf_snp_25 = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g25_sn_pass, g02, g05)
    _g5_opt_bf_snp_34 = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g34_sn_pass, g03, g04)
    _g5_opt_bf_snp_35 = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g35_sn_pass, g03, g05)
    _g5_opt_bf_snp_45 = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g45_sn_pass, g04, g05)

    _g5_opt_bf_snp_123 = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g123_sn_pass, g01, g02, g03)
    _g5_opt_bf_snp_124 = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g124_sn_pass, g01, g02, g04)
    _g5_opt_bf_snp_125 = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g125_sn_pass, g01, g02, g05)
    _g5_opt_bf_snp_134 = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g134_sn_pass, g01, g03, g04)
    _g5_opt_bf_snp_135 = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g135_sn_pass, g01, g03, g05)
    _g5_opt_bf_snp_145 = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g145_sn_pass, g01, g04, g05)
    _g5_opt_bf_snp_234 = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g234_sn_pass, g02, g03, g04)
    _g5_opt_bf_snp_235 = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g235_sn_pass, g02, g03, g05)
    _g5_opt_bf_snp_245 = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g245_sn_pass, g02, g04, g05)
    _g5_opt_bf_snp_345 = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g345_sn_pass, g03, g04, g05)


    _g5_opt_bf_snp_1234 = snp4_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1234_sn_pass, g01, g02, g03, g04)
    _g5_opt_bf_snp_1235 = snp4_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1235_sn_pass, g01, g02, g03, g05)
    _g5_opt_bf_snp_1245 = snp4_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1245_sn_pass, g01, g02, g04, g05)
    _g5_opt_bf_snp_1345 = snp4_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1345_sn_pass, g01, g03, g04, g05)
    _g5_opt_bf_snp_2345 = snp4_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g2345_sn_pass, g02, g03, g04, g05)

    _g5_opt_bf_snp_0 = snp0_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g0_sn_pass)
    _g5_opt_bf_snp_12345 = snp5_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g12345_sn_pass, g01, g02, g03, g04, g05)


    print(g23_sn_pass[j1, i1])
    print(_g5_opt_bf_snp_23[0, j1, i1])

    _g5_opt_t1 = np.array([np.where(_g5_opt_bf_snp_0 > -1, _g5_opt_bf_snp_0, -1)][0])

    _g5_opt_t2 = np.array([np.where(_g5_opt_bf_snp_1 > -1, _g5_opt_bf_snp_1, _g5_opt_t1)][0])
    _g5_opt_t3 = np.array([np.where(_g5_opt_bf_snp_2 > -1, _g5_opt_bf_snp_2, _g5_opt_t2)][0])
    _g5_opt_t4 = np.array([np.where(_g5_opt_bf_snp_3 > -1, _g5_opt_bf_snp_3, _g5_opt_t3)][0])
    _g5_opt_t5 = np.array([np.where(_g5_opt_bf_snp_4 > -1, _g5_opt_bf_snp_4, _g5_opt_t4)][0])
    _g5_opt_t6 = np.array([np.where(_g5_opt_bf_snp_5 > -1, _g5_opt_bf_snp_5, _g5_opt_t5)][0])

    _g5_opt_t7 = np.array( [np.where(_g5_opt_bf_snp_12 > -1, _g5_opt_bf_snp_12, _g5_opt_t6)][0])
    _g5_opt_t8 = np.array( [np.where(_g5_opt_bf_snp_13 > -1, _g5_opt_bf_snp_13, _g5_opt_t7)][0])
    _g5_opt_t9 = np.array( [np.where(_g5_opt_bf_snp_14 > -1, _g5_opt_bf_snp_14, _g5_opt_t8)][0])
    _g5_opt_t10 = np.array([np.where(_g5_opt_bf_snp_15 > -1, _g5_opt_bf_snp_15, _g5_opt_t9)][0])
    _g5_opt_t11 = np.array([np.where(_g5_opt_bf_snp_23 > -1, _g5_opt_bf_snp_23, _g5_opt_t10)][0])
    _g5_opt_t12 = np.array([np.where(_g5_opt_bf_snp_24 > -1, _g5_opt_bf_snp_24, _g5_opt_t11)][0])
    _g5_opt_t13 = np.array([np.where(_g5_opt_bf_snp_25 > -1, _g5_opt_bf_snp_25, _g5_opt_t12)][0])
    _g5_opt_t14 = np.array([np.where(_g5_opt_bf_snp_34 > -1, _g5_opt_bf_snp_34, _g5_opt_t13)][0])
    _g5_opt_t15 = np.array([np.where(_g5_opt_bf_snp_35 > -1, _g5_opt_bf_snp_35, _g5_opt_t14)][0])
    _g5_opt_t16 = np.array([np.where(_g5_opt_bf_snp_45 > -1, _g5_opt_bf_snp_45, _g5_opt_t15)][0])

    _g5_opt_t17 = np.array([np.where(_g5_opt_bf_snp_123 > -1, _g5_opt_bf_snp_123, _g5_opt_t16)][0])
    _g5_opt_t18 = np.array([np.where(_g5_opt_bf_snp_124 > -1, _g5_opt_bf_snp_124, _g5_opt_t17)][0])
    _g5_opt_t19 = np.array([np.where(_g5_opt_bf_snp_125 > -1, _g5_opt_bf_snp_125, _g5_opt_t18)][0])
    _g5_opt_t20 = np.array([np.where(_g5_opt_bf_snp_134 > -1, _g5_opt_bf_snp_134, _g5_opt_t19)][0])
    _g5_opt_t21 = np.array([np.where(_g5_opt_bf_snp_135 > -1, _g5_opt_bf_snp_135, _g5_opt_t20)][0])
    _g5_opt_t22 = np.array([np.where(_g5_opt_bf_snp_145 > -1, _g5_opt_bf_snp_145, _g5_opt_t21)][0])
    _g5_opt_t23 = np.array([np.where(_g5_opt_bf_snp_234 > -1, _g5_opt_bf_snp_234, _g5_opt_t22)][0])
    _g5_opt_t24 = np.array([np.where(_g5_opt_bf_snp_235 > -1, _g5_opt_bf_snp_235, _g5_opt_t23)][0])
    _g5_opt_t25 = np.array([np.where(_g5_opt_bf_snp_245 > -1, _g5_opt_bf_snp_245, _g5_opt_t24)][0])
    _g5_opt_t26 = np.array([np.where(_g5_opt_bf_snp_345 > -1, _g5_opt_bf_snp_345, _g5_opt_t25)][0])

    _g5_opt_t27 = np.array([np.where(_g5_opt_bf_snp_1234 > -1, _g5_opt_bf_snp_1234, _g5_opt_t26)][0])
    _g5_opt_t28 = np.array([np.where(_g5_opt_bf_snp_1235 > -1, _g5_opt_bf_snp_1235, _g5_opt_t27)][0])
    _g5_opt_t29 = np.array([np.where(_g5_opt_bf_snp_1245 > -1, _g5_opt_bf_snp_1245, _g5_opt_t28)][0])
    _g5_opt_t30 = np.array([np.where(_g5_opt_bf_snp_1345 > -1, _g5_opt_bf_snp_1345, _g5_opt_t29)][0])
    _g5_opt_t31 = np.array([np.where(_g5_opt_bf_snp_2345 > -1, _g5_opt_bf_snp_2345, _g5_opt_t30)][0])

    _g5_opt_t32 = np.array([np.where(_g5_opt_bf_snp_12345 > -1, _g5_opt_bf_snp_12345, _g5_opt_t31)][0])

    print("optimal-ng:", _g5_opt_t32[0, j1, i1])
    return _g5_opt_t32





def g4_opt_bf_snp0(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g0_sn_pass):
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_cond_sn = np.where(g0_sn_pass) # s/n < sn_limit
    g_opt[g_cond_sn] += 1

    g0_0 = np.array([np.where(g_opt > 0, -10, -10)])
    g4_opt_snp0 = g0_0[0]

    return g4_opt_snp0




def g4_opt_bf_snp1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1_sn_pass, g01): # g01 : g indices (0, 1, 2, ...)
    g_sort_n = g01
    _cond_N = 0
    _cond0 = ([g01, g01, '='], ) # dummy condition
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond0, _cond_N, g1_sn_pass)
    g4_opt_snp1 = g1_0 + 10*0

    return g4_opt_snp1




def g4_opt_bf_snp2(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g12_sn_pass, g01, g02): # g01 < g02 : g indices (0, 1, 2, ...)

    g_sort_n = g01
    _cond_N = 1
    _cond1 = ([g01, g02, '0'], )
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12_sn_pass)

    g_sort_n = g02
    _cond_N = 2
    _cond2 = ([g01, g02, '1'], [g01, g02, '0'])
    g2_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond2, _cond_N, g12_sn_pass)

    g_sort_n = g01
    _cond_N = 2
    _cond2 = ([g01, g02, '1'], [g01, g02, '1'])
    g1_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond2, _cond_N, g12_sn_pass)
    
    g4_opt_snp2 = g1_0 + g2_0 + g1_1 + 10*2

    return g4_opt_snp2






def set_cond_child(seed_cond0, _Gn, _ng):

    seed_cond1 = [[[0 for i in range(3)] for j in range(0)] for k in range(2)]

    for i in range(0, 2):
        seed_cond1[i] = cp.deepcopy(seed_cond0)

    e1 = _Gn-1
    e2 = _ng-1
    e3 = '0'
    cond_t1 = [e1, e2, e3]
    seed_cond1[0].append(cond_t1)

    e1 = _Gn-1
    e2 = _ng-1
    e3 = '1'
    cond_t1 = [e1, e2, e3]
    seed_cond1[1].append(cond_t1)
    seed_cond1[1].append(cond_t1)


    return seed_cond1



def set_cond_child_org(seed_cond0, _Gn, _ng):

    seed_cond1 = [[[0 for i in range(3)] for j in range(0)] for k in range(2)]

    for i in range(0, 2):
        seed_cond1[i] = cp.deepcopy(seed_cond0)

    e1 = _Gn-1
    e2 = _ng-1
    e3 = '0'
    cond_t1 = [e1, e2, e3]
    seed_cond1[0].append(cond_t1)

    e1 = _Gn-1
    e2 = _ng-1
    e3 = '1'
    cond_t1 = [e1, e2, e3]
    seed_cond1[1].append(cond_t1)
    seed_cond1[1].append(cond_t1)


    return seed_cond1




def set_cond_tree_old(n_snp, _Gn, _max_ngauss, _gt_conds_index): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    if n_snp == 1: # check here ...
        n_core_seeds = 1
    elif n_snp == 2: # check here ...
        n_core_seeds = 1
    else:
        n_core_seeds = 2**(n_snp-2) # check here...

    seed_cond0 = [[[0 for i in range(3)] for j in range(0)] for k in range(n_core_seeds)]
    seed_cond1 = [[[0 for i in range(3)] for j in range(0)] for k in range(2)]

    _gt = [0 for i in range(_max_ngauss)]
    _gt_conds = [[0 for i in range(0)] for j in range(_max_ngauss-_Gn+1)]

    if n_snp == 1:
        _gt_conds[0].append(seed_cond0[0])

    if n_snp == 2:
        for i in range(0, 1):
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 1, '0'])

        _gt_conds[0].append(seed_cond0[0])

    if n_snp == 3:


        for i in range(0, 1):
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 1, '0'])
            seed_cond0[i].append([1, 2, '1'])
            seed_cond0[i].append([1, 2, '0'])

        for i in range(1, 2):
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 2, '1'])
            seed_cond0[i].append([0, 2, '0'])

        _gt_conds[0].append(seed_cond0[0])
        _gt_conds[0].append(seed_cond0[1])

    if n_snp == 4:
        for i in range(0, 1):
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 1, '0'])
            seed_cond0[i].append([1, 2, '1'])
            seed_cond0[i].append([1, 2, '0'])
            seed_cond0[i].append([2, 3, '1'])
            seed_cond0[i].append([2, 3, '0'])
    
        for i in range(1, 2):
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([1, 2, '1'])
            seed_cond0[i].append([1, 2, '0'])
            seed_cond0[i].append([1, 3, '1'])
            seed_cond0[i].append([1, 3, '0'])
    
        for i in range(2, 3):
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 2, '1'])
            seed_cond0[i].append([0, 2, '0'])
            seed_cond0[i].append([2, 3, '1'])
            seed_cond0[i].append([2, 3, '0'])
    
        for i in range(3, 4):
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 2, '1'])
            seed_cond0[i].append([0, 2, '0'])
            seed_cond0[i].append([0, 3, '1'])
            seed_cond0[i].append([0, 3, '0'])
    
    
        _gt_conds[0].append(seed_cond0[0])
        _gt_conds[0].append(seed_cond0[1])
        _gt_conds[0].append(seed_cond0[2])
        _gt_conds[0].append(seed_cond0[3])

    if n_snp == 5:
        for i in range(0, 1):
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 1, '0'])
            seed_cond0[i].append([1, 2, '1'])
            seed_cond0[i].append([1, 2, '0'])
            seed_cond0[i].append([2, 3, '1'])
            seed_cond0[i].append([2, 3, '0'])
            seed_cond0[i].append([2, 3, '0'])
            seed_cond0[i].append([2, 3, '0'])

        for i in range(1, 2):
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([1, 2, '1'])
            seed_cond0[i].append([1, 2, '0'])
            seed_cond0[i].append([1, 3, '1'])
            seed_cond0[i].append([1, 3, '0'])
            seed_cond0[i].append([1, 3, '0'])
            seed_cond0[i].append([1, 3, '0'])
   
        for i in range(2, 3):
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 2, '1'])
            seed_cond0[i].append([0, 2, '0'])
            seed_cond0[i].append([2, 3, '1'])
            seed_cond0[i].append([2, 3, '0'])
            seed_cond0[i].append([2, 3, '0'])
            seed_cond0[i].append([2, 3, '0'])
    
        for i in range(3, 4):
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 2, '1'])
            seed_cond0[i].append([0, 2, '0'])
            seed_cond0[i].append([0, 3, '1'])
            seed_cond0[i].append([0, 3, '0'])
            seed_cond0[i].append([0, 3, '0'])
            seed_cond0[i].append([0, 3, '0'])

        for i in range(4, 5):
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 2, '1'])
            seed_cond0[i].append([0, 2, '0'])
            seed_cond0[i].append([0, 3, '1'])
            seed_cond0[i].append([0, 3, '0'])
            seed_cond0[i].append([0, 3, '0'])
            seed_cond0[i].append([0, 3, '0'])

        for i in range(5, 6):
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 2, '1'])
            seed_cond0[i].append([0, 2, '0'])
            seed_cond0[i].append([0, 3, '1'])
            seed_cond0[i].append([0, 3, '0'])
            seed_cond0[i].append([0, 3, '0'])
            seed_cond0[i].append([0, 3, '0'])

        for i in range(6, 7):
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 2, '1'])
            seed_cond0[i].append([0, 2, '0'])
            seed_cond0[i].append([0, 3, '1'])
            seed_cond0[i].append([0, 3, '0'])
            seed_cond0[i].append([0, 3, '0'])
            seed_cond0[i].append([0, 3, '0'])

        for i in range(7, 8):
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 2, '1'])
            seed_cond0[i].append([0, 2, '0'])
            seed_cond0[i].append([0, 3, '1'])
            seed_cond0[i].append([0, 3, '0'])
            seed_cond0[i].append([0, 3, '0'])
            seed_cond0[i].append([0, 3, '0'])



    
    




    
    
        _gt_conds[0].append(seed_cond0[0])
        _gt_conds[0].append(seed_cond0[1])
        _gt_conds[0].append(seed_cond0[2])
        _gt_conds[0].append(seed_cond0[3])
        _gt_conds[0].append(seed_cond0[4])
        _gt_conds[0].append(seed_cond0[5])
        _gt_conds[0].append(seed_cond0[6])
        _gt_conds[0].append(seed_cond0[7])




    for i in range(0, n_core_seeds): 
        _gt[0] = set_cond_child(seed_cond0[i], _Gn, _Gn+1)

        for j1 in range(1, _max_ngauss-_Gn): # 
            _gt[j1] = set_cond_child(_gt[j1-1][1], _Gn, _Gn+1+j1)


        for j2 in range(1, _max_ngauss-_Gn+1):
            for k1 in range(0, j2-1):
                _gt_conds[j2].append(_gt[k1][0])

            _gt_conds[j2].append(_gt[j2-1][0])
            _gt_conds[j2].append(_gt[j2-1][1])




    return _gt_conds[_gt_conds_index]









def set_cond_tree(n_snp, _Gn, _max_ngauss, _gt_conds_index): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    if n_snp == 1: # (n-0)
        n_core_seeds = 1
    elif n_snp == 2: # 2**0 x (n-1)
        n_core_seeds = 1
    else: # 2**(n-2) x (n-(n-1))
        n_core_seeds = 2**(n_snp-2)

    seed_cond0 = [[[0 for i in range(3)] for j in range(0)] for k in range(n_core_seeds)]
    base_cond0 = [[[0 for i in range(3)] for j in range(0)] for k in range(n_core_seeds)]

    _gt = [0 for i in range(_max_ngauss)]
    _gt_conds = [[0 for i in range(0)] for j in range(_max_ngauss-_Gn+1)]

    if n_snp == 1:
        _gt_conds[0].append(seed_cond0[0])

    if n_snp == 2:
        for i in range(0, 1):
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 1, '0'])

        _gt_conds[0].append(seed_cond0[0])


    base_cond0[0] = [[0, 1, '1'], [0, 1, '0']]


    if n_snp >= 3:
        for n_snp_i in range(3, n_snp+1):
            base_cond0_i = 0
            for n0 in range(0, 2**(n_snp_i-2)):
    
                if n0 % 2 == 0: # even 
                    seed_cond0[n0] = cp.deepcopy(base_cond0[base_cond0_i])
            
                    _add_inc1 = cp.deepcopy(seed_cond0[n0])
                    delt_cond1 = _add_inc1[-1][1] - _add_inc1[-1][0]
    
                    _add_inc1[-2][0] += delt_cond1
                    _add_inc1[-2][1] = n_snp_i-1
    
                    _add_inc1[-1][0] += delt_cond1
                    _add_inc1[-1][1] = n_snp_i-1
    
                    seed_cond0[n0].append(_add_inc1[-2])
                    seed_cond0[n0].append(_add_inc1[-1])
    
                    seed_cond0[n0+1] = cp.deepcopy(base_cond0[base_cond0_i])
                    seed_cond0[n0+1][-1][2] = '1'
            
                    _add_inc2 = cp.deepcopy(seed_cond0[base_cond0_i])
                    delt_cond2 = _add_inc2[-1][1] - _add_inc2[-1][0]
    
                    _add_inc2[-2][0] = _add_inc1[-2][0] - delt_cond2
                    _add_inc2[-2][1] = n_snp_i-1
    
                    _add_inc2[-1][0] = _add_inc1[-1][0] - delt_cond2
                    _add_inc2[-1][1] = n_snp_i-1
    
                    seed_cond0[n0+1].append(_add_inc2[-2])
                    seed_cond0[n0+1].append(_add_inc2[-1])
    
                    base_cond0_i += 1
    
            for n0 in range(0, 2**(n_snp_i-2)):
                base_cond0[n0] = cp.deepcopy(seed_cond0[n0])
    
    
        for n_snp_i in range(0, 2**(n_snp-2)): # 3:2, 4:4, 5:8, 6:16 ...
            _gt_conds[0].append(seed_cond0[n_snp_i])



    for i in range(0, n_core_seeds): 
        _gt[0] = set_cond_child(seed_cond0[i], _Gn, _Gn+1)

        for j1 in range(1, _max_ngauss-_Gn): # 
            _gt[j1] = set_cond_child(_gt[j1-1][1], _Gn, _Gn+1+j1)


        for j2 in range(1, _max_ngauss-_Gn+1):
            for k1 in range(0, j2-1):
                _gt_conds[j2].append(_gt[k1][0])

            _gt_conds[j2].append(_gt[j2-1][0])
            _gt_conds[j2].append(_gt[j2-1][1])




    return _gt_conds[_gt_conds_index]




def set_cond_tree_dev(n_snp, _Gn, _max_ngauss, _gt_conds_index): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    if n_snp == 1: # (n-0)
        n_core_seeds = 1
    elif n_snp == 2: # 2**0 x (n-1)
        n_core_seeds = 1
    else: # 2**(n-2) x (n-(n-1))
        n_core_seeds = 2**(n_snp-2)

    seed_cond0 = [[[0 for i in range(3)] for j in range(0)] for k in range(n_core_seeds)]
    base_cond0 = [[[0 for i in range(3)] for j in range(0)] for k in range(n_core_seeds)]

    _gt = [0 for i in range(_max_ngauss)]
    _gt_conds = [[0 for i in range(0)] for j in range(_max_ngauss-_Gn+1)]

    if n_snp == 1:
        _gt_conds[0].append(seed_cond0[0])

    if n_snp == 2:
        for i in range(0, 1):
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 1, '0'])

        _gt_conds[0].append(seed_cond0[0])


    base_cond0[0] = [[0, 1, '1'], [0, 1, '0']]


    if n_snp >= 3:
        for n_snp_i in range(3, n_snp+1):
            base_cond0_i = 0
            for n0 in range(0, 2**(n_snp_i-2)):
    
                if n0 % 2 == 0: # even 
                    seed_cond0[n0] = cp.deepcopy(base_cond0[base_cond0_i])
            
                    _add_inc1 = cp.deepcopy(seed_cond0[n0])
                    delt_cond1 = _add_inc1[-1][1] - _add_inc1[-1][0]
    
                    _add_inc1[-2][0] += delt_cond1
                    _add_inc1[-2][1] = n_snp_i-1
    
                    _add_inc1[-1][0] += delt_cond1
                    _add_inc1[-1][1] = n_snp_i-1
    
                    seed_cond0[n0].append(_add_inc1[-2])
                    seed_cond0[n0].append(_add_inc1[-1])
    
                    print(seed_cond0[n0])
    
                    seed_cond0[n0+1] = cp.deepcopy(base_cond0[base_cond0_i])
                    seed_cond0[n0+1][-1][2] = '1'
            
                    _add_inc2 = cp.deepcopy(seed_cond0[base_cond0_i])
                    delt_cond2 = _add_inc2[-1][1] - _add_inc2[-1][0]
    
                    _add_inc2[-2][0] = _add_inc1[-2][0] - delt_cond2
                    _add_inc2[-2][1] = n_snp_i-1
    
    
                    _add_inc2[-1][0] = _add_inc1[-1][0] - delt_cond2
                    _add_inc2[-1][1] = n_snp_i-1
    
                    seed_cond0[n0+1].append(_add_inc2[-2])
                    seed_cond0[n0+1].append(_add_inc2[-1])
    
                    print(seed_cond0[n0+1])
    
                    base_cond0_i += 1
    
            print("")
            for n0 in range(0, 2**(n_snp_i-2)):
                base_cond0[n0] = cp.deepcopy(seed_cond0[n0])
    
    
        for n_snp_i in range(0, 2**(n_snp-2)): # 3:2, 4:4, 5:8, 6:16 ...
            _gt_conds[0].append(seed_cond0[n_snp_i])



    for i in range(0, n_core_seeds): 
        _gt[0] = set_cond_child(seed_cond0[i], _Gn, _Gn+1)

        for j1 in range(1, _max_ngauss-_Gn): # 
            _gt[j1] = set_cond_child(_gt[j1-1][1], _Gn, _Gn+1+j1)


        for j2 in range(1, _max_ngauss-_Gn+1):
            for k1 in range(0, j2-1):
                _gt_conds[j2].append(_gt[k1][0])

            _gt_conds[j2].append(_gt[j2-1][0])
            _gt_conds[j2].append(_gt[j2-1][1])




    return _gt_conds[_gt_conds_index]


def set_cond_tree_bg(n_snp, _Gn, _max_ngauss): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    if _Gn == 1: # check here ...
        n_core_seeds = 2**(_Gn-1)
    else:
        n_core_seeds = 2**(_Gn-2)

    seed_cond0 = [[[0 for i in range(3)] for j in range(0)] for k in range(n_core_seeds)]
    seed_cond1 = [[[0 for i in range(3)] for j in range(0)] for k in range(2)]

    _gt = [0 for i in range(_max_ngauss)]
    _gt_conds = [[0 for i in range(0)] for j in range(_max_ngauss-_Gn+1)]

    if n_snp == 1:
        _gt_conds[0].append(seed_cond0[0])

    if n_snp == 2:
        for i in range(0, 1):
            seed_cond0[i].append([1, 2, '1'])
            seed_cond0[i].append([1, 2, '0'])

        _gt_conds[0].append(seed_cond0[0])

    if n_snp == 3:
        for i in range(0, 1):
            seed_cond0[i].append([1, 2, '1'])
            seed_cond0[i].append([1, 2, '0'])
            seed_cond0[i].append([2, 3, '1'])
            seed_cond0[i].append([2, 3, '0'])

        for i in range(1, 2):
            seed_cond0[i].append([1, 2, '1'])
            seed_cond0[i].append([1, 2, '1'])
            seed_cond0[i].append([1, 3, '1'])
            seed_cond0[i].append([1, 3, '0'])

        _gt_conds[0].append(seed_cond0[0])
        _gt_conds[0].append(seed_cond0[1])

    if n_snp == 4:
        for i in range(0, 1):
            seed_cond0[i].append([1, 2, '1'])
            seed_cond0[i].append([1, 2, '0'])
            seed_cond0[i].append([2, 3, '1'])
            seed_cond0[i].append([2, 3, '0'])
            seed_cond0[i].append([3, 4, '1'])
            seed_cond0[i].append([3, 4, '0'])
    
        for i in range(1, 2):
            seed_cond0[i].append([1, 2, '1'])
            seed_cond0[i].append([1, 2, '1'])
            seed_cond0[i].append([2, 3, '1'])
            seed_cond0[i].append([2, 3, '0'])
            seed_cond0[i].append([2, 4, '1'])
            seed_cond0[i].append([2, 4, '0'])
    
        for i in range(2, 3):
            seed_cond0[i].append([1, 2, '1'])
            seed_cond0[i].append([1, 2, '1'])
            seed_cond0[i].append([1, 3, '1'])
            seed_cond0[i].append([1, 3, '0'])
            seed_cond0[i].append([3, 4, '1'])
            seed_cond0[i].append([3, 4, '0'])
    
        for i in range(3, 4):
            seed_cond0[i].append([1, 2, '1'])
            seed_cond0[i].append([1, 2, '1'])
            seed_cond0[i].append([1, 3, '1'])
            seed_cond0[i].append([1, 3, '0'])
            seed_cond0[i].append([1, 4, '1'])
            seed_cond0[i].append([1, 4, '0'])
    
    
        _gt_conds[0].append(seed_cond0[0])
        _gt_conds[0].append(seed_cond0[1])
        _gt_conds[0].append(seed_cond0[2])
        _gt_conds[0].append(seed_cond0[3])


    for i in range(0, n_core_seeds): 
        _gt[0] = set_cond_child(seed_cond0[i], _Gn, _Gn+1)

        for j1 in range(1, _max_ngauss-_Gn): # 
            _gt[j1] = set_cond_child(_gt[j1-1][1], _Gn, _Gn+1+j1)


        for j2 in range(1, _max_ngauss-_Gn+1):
            for k1 in range(0, j2-1):
                _gt_conds[j2].append(_gt[k1][0])

            _gt_conds[j2].append(_gt[j2-1][0])
            _gt_conds[j2].append(_gt[j2-1][1])
            print("")

    for i in range(0, _max_ngauss-_Gn+1):
        for j in range(0, n_core_seeds*(i+1)):
            print(_gt_conds[i][j])
        print("")

    return _gt_conds



def set_cond_tree_1(_Gn, _max_ngauss): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    n_core_seeds = 2**(_Gn-1)

    seed_cond0 = [[[0 for i in range(3)] for j in range(0)] for k in range(n_core_seeds)]
    seed_cond1 = [[[0 for i in range(3)] for j in range(0)] for k in range(2)]

    _gt = [0 for i in range(_max_ngauss)]
    _gt_conds = [[0 for i in range(0)] for j in range(_max_ngauss-_Gn+1)]


    _gt_conds[0].append(seed_cond0[0])

    for i in range(0, n_core_seeds): 
        _gt[0] = set_cond_child(seed_cond0[i], _Gn, _Gn+1)

        for j1 in range(1, _max_ngauss-_Gn): # 
            _gt[j1] = set_cond_child(_gt[j1-1][1], _Gn, _Gn+1+j1)


        for j2 in range(1, _max_ngauss-_Gn+1):
            for k1 in range(0, j2-1):
                _gt_conds[j2].append(_gt[k1][0])

            _gt_conds[j2].append(_gt[j2-1][0])
            _gt_conds[j2].append(_gt[j2-1][1])
            print("")


    for i in range(0, _max_ngauss-_Gn+1):
        for j in range(0, 2**(_Gn-1)*(i+1)):
            print(_gt_conds[i][j])
        print("")

    return _gt_conds



def set_cond_tree_2(_Gn, _max_ngauss): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    n_core_seeds = 2**(_Gn-2)

    seed_cond0 = [[[0 for i in range(3)] for j in range(0)] for k in range(n_core_seeds)]
    seed_cond1 = [[[0 for i in range(3)] for j in range(0)] for k in range(2)]

    _gt = [0 for i in range(_max_ngauss)]
    _gt_conds = [[0 for i in range(0)] for j in range(_max_ngauss-_Gn+1)]

    for i in range(0, 1):
        seed_cond0[i].append([1, 2, '1'])
        seed_cond0[i].append([1, 2, '0'])
    _gt_conds[0].append(seed_cond0[0])

    for i in range(0, n_core_seeds): 
        _gt[0] = set_cond_child(seed_cond0[i], _Gn, _Gn+1)

        for j1 in range(1, _max_ngauss-_Gn): # 
            _gt[j1] = set_cond_child(_gt[j1-1][1], _Gn, _Gn+1+j1)


        for j2 in range(1, _max_ngauss-_Gn+1):
            for k1 in range(0, j2-1):
                _gt_conds[j2].append(_gt[k1][0])

            _gt_conds[j2].append(_gt[j2-1][0])
            _gt_conds[j2].append(_gt[j2-1][1])
            print("")


    for i in range(0, _max_ngauss-_Gn+1):
        for j in range(0, 2**(_Gn-2)*(i+1)):
            print(_gt_conds[i][j])
        print("")

    return _gt_conds



def set_cond_tree_3(_Gn, _max_ngauss): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    n_core_seeds = 2**(_Gn-2)

    seed_cond0 = [[[0 for i in range(3)] for j in range(0)] for k in range(n_core_seeds)]
    seed_cond1 = [[[0 for i in range(3)] for j in range(0)] for k in range(2)]

    _gt = [0 for i in range(_max_ngauss)]
    _gt_conds = [[0 for i in range(0)] for j in range(_max_ngauss-_Gn+1)]

    for i in range(0, 1):
        seed_cond0[i].append([1, 2, '1'])
        seed_cond0[i].append([1, 2, '0'])
        seed_cond0[i].append([2, 3, '1'])
        seed_cond0[i].append([2, 3, '0'])

    for i in range(1, 2):
        seed_cond0[i].append([1, 2, '1'])
        seed_cond0[i].append([1, 2, '1'])
        seed_cond0[i].append([1, 3, '1'])
        seed_cond0[i].append([1, 3, '0'])


    _gt_conds[0].append(seed_cond0[0])
    _gt_conds[0].append(seed_cond0[1])



    for i in range(0, n_core_seeds): 
        _gt[0] = set_cond_child(seed_cond0[i], _Gn, _Gn+1)

        for j1 in range(1, _max_ngauss-_Gn): # 
            _gt[j1] = set_cond_child(_gt[j1-1][1], _Gn, _Gn+1+j1)


        for j2 in range(1, _max_ngauss-_Gn+1):
            for k1 in range(0, j2-1):
                _gt_conds[j2].append(_gt[k1][0])

            _gt_conds[j2].append(_gt[j2-1][0])
            _gt_conds[j2].append(_gt[j2-1][1])
            print("")




    for i in range(0, _max_ngauss-_Gn+1):
        for j in range(0, 2**(_Gn-2)*(i+1)):
            print(_gt_conds[i][j])
        print("")


    return _gt_conds



def set_cond_tree_4(_Gn, _max_ngauss): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    n_core_seeds = 2**(_Gn-2)

    seed_cond0 = [[[0 for i in range(3)] for j in range(0)] for k in range(n_core_seeds)]
    seed_cond1 = [[[0 for i in range(3)] for j in range(0)] for k in range(2)]

    _gt = [0 for i in range(_max_ngauss)]
    _gt_conds = [[0 for i in range(0)] for j in range(_max_ngauss-_Gn+1)]

    for i in range(0, 1):
        seed_cond0[i].append([1, 2, '1'])
        seed_cond0[i].append([1, 2, '0'])
        seed_cond0[i].append([2, 3, '1'])
        seed_cond0[i].append([2, 3, '0'])
        seed_cond0[i].append([3, 4, '1'])
        seed_cond0[i].append([3, 4, '0'])

    for i in range(1, 2):
        seed_cond0[i].append([1, 2, '1'])
        seed_cond0[i].append([1, 2, '1'])
        seed_cond0[i].append([2, 3, '1'])
        seed_cond0[i].append([2, 3, '0'])
        seed_cond0[i].append([2, 4, '1'])
        seed_cond0[i].append([2, 4, '0'])

    for i in range(2, 3):
        seed_cond0[i].append([1, 2, '1'])
        seed_cond0[i].append([1, 2, '1'])
        seed_cond0[i].append([1, 3, '1'])
        seed_cond0[i].append([1, 3, '0'])
        seed_cond0[i].append([3, 4, '1'])
        seed_cond0[i].append([3, 4, '0'])

    for i in range(3, 4):
        seed_cond0[i].append([1, 2, '1'])
        seed_cond0[i].append([1, 2, '1'])
        seed_cond0[i].append([1, 3, '1'])
        seed_cond0[i].append([1, 3, '0'])
        seed_cond0[i].append([1, 4, '1'])
        seed_cond0[i].append([1, 4, '0'])


    _gt_conds[0].append(seed_cond0[0])
    _gt_conds[0].append(seed_cond0[1])
    _gt_conds[0].append(seed_cond0[2])
    _gt_conds[0].append(seed_cond0[3])



    for i in range(0, n_core_seeds): 
        _gt[0] = set_cond_child(seed_cond0[i], _Gn, _Gn+1)

        for j1 in range(1, _max_ngauss-_Gn): # 
            _gt[j1] = set_cond_child(_gt[j1-1][1], _Gn, _Gn+1+j1)


        for j2 in range(1, _max_ngauss-_Gn+1):
            for k1 in range(0, j2-1):
                _gt_conds[j2].append(_gt[k1][0])

            _gt_conds[j2].append(_gt[j2-1][0])
            _gt_conds[j2].append(_gt[j2-1][1])
            print("")

    for i in range(0, _max_ngauss-_Gn+1):
        for j in range(0, 2**(_Gn-2)*(i+1)):
            print(_gt_conds[i][j])
        print("")


    return _gt_conds




def g4_opt_bf_snp3(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g123_sn_pass, g01, g02, g03): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    g_sort_n = g01
    _cond_N = 1
    _cond1 = ([g01, g02, '0'], )
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g123_sn_pass)

    g_sort_n = g02
    _cond_N = 3
    _cond3 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '0'])
    g2_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond3, _cond_N, g123_sn_pass)

    g_sort_n = g03
    _cond_N = 4
    _cond4 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '1'], [g02, g03, '0'])
    g3_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond4, _cond_N, g123_sn_pass)

    
    g_sort_n = g02
    _cond_N = 4
    _cond4 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '1'], [g02, g03, '1'])
    g2_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond4, _cond_N, g123_sn_pass)

    
    g_sort_n = g01
    _cond_N = 3
    _cond3 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '0'])
    g1_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond3, _cond_N, g123_sn_pass)



    g_sort_n = g03
    _cond_N = 4
    _cond4 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '1'], [g01, g03, '1'])
    g3_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond4, _cond_N, g123_sn_pass)

    
    g_sort_n = g01
    _cond_N = 4
    _cond4 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '1'], [g01, g03, '1'])
    g1_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond4, _cond_N, g123_sn_pass)



    g4_opt_snp3 =  g1_0 + g1_1 + g1_2 \
            + g2_0 + g2_1 \
            + g3_0 + g3_1 + 10*6 # -10 flag see above

    return g4_opt_snp3



def g4_opt_bf_snp4(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1234_sn_pass, g01, g02, g03, g04): # g01 < g02 < g03 < g04 : g indices (0, 1, 2, ...)

    g_sort_n = g01
    _cond_N = 1
    _cond1 = ([g01, g02, '0'], )
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g1234_sn_pass)

    g_sort_n = g02
    _cond_N = 3
    _cond3 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '0'])
    g2_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond3, _cond_N, g1234_sn_pass)

    g_sort_n = g03
    _cond_N = 5
    _cond5 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '1'], [g02, g03, '0'], [g03, g04, '0'])
    g3_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond5, _cond_N, g1234_sn_pass)
  
    g_sort_n = g04
    _cond_N = 6
    _cond6 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '1'], [g02, g03, '0'], [g03, g04, '1'], [g03, g04, '0'])
    g4_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    g_sort_n = g03
    _cond_N = 6
    _cond6 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '1'], [g02, g03, '0'], [g03, g04, '1'], [g03, g04, '1'])
    g3_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    g_sort_n = g02
    _cond_N = 5
    _cond5 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '1'], [g02, g03, '1'], [g02, g04, '0'])
    g2_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond5, _cond_N, g1234_sn_pass)

    g_sort_n = g04
    _cond_N = 6
    _cond6 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '1'], [g02, g03, '1'], [g02, g04, '1'], [g02, g04, '0'])
    g4_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    g_sort_n = g02
    _cond_N = 6
    _cond6 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '1'], [g02, g03, '1'], [g02, g04, '1'], [g02, g04, '1'])
    g2_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    g_sort_n = g01
    _cond_N = 3
    _cond3 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '0'])
    g1_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond3, _cond_N, g1234_sn_pass)

    g_sort_n = g03
    _cond_N = 5
    _cond5 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '1'], [g01, g03, '0'], [g03, g04, '0'])
    g3_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond5, _cond_N, g1234_sn_pass)

    g_sort_n = g04
    _cond_N = 6
    _cond6 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '1'], [g01, g03, '0'], [g03, g04, '1'], [g03, g04, '0'])
    g4_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    g_sort_n = g03
    _cond_N = 6
    _cond6 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '1'], [g01, g03, '0'], [g03, g04, '1'], [g03, g04, '1'])
    g3_3 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    g_sort_n = g01
    _cond_N = 5
    _cond5 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '1'], [g01, g03, '1'], [g01, g04, '0'])
    g1_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond5, _cond_N, g1234_sn_pass)


    g_sort_n = g04
    _cond_N = 6
    _cond6 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '1'], [g01, g03, '1'], [g01, g04, '1'], [g01, g04, '0'])
    g4_3 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    g_sort_n = g01
    _cond_N = 6
    _cond6 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '1'], [g01, g03, '1'], [g01, g04, '1'], [g01, g04, '1'])
    g1_3 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    g4_opt_snp4 = g1_0 + g1_1 + g1_2 + g1_3 \
            + g2_0 + g2_1 + g2_2 \
            + g3_0 + g3_1 + g3_2 + g3_3 \
            + g4_0 + g4_1 + g4_2 + g4_3 + 10*14 # -10 flag

    return g4_opt_snp4



def g5_opt_bf_snp5(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1234_sn_pass, g01, g02, g03, g05): # g01 < g02 < g03 < g04 < g05: g indices (0, 1, 2, ...)

    g_sort_n = g01
    _cond_N = 1
    _cond1 = ([g01, g02, '>'], )
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g1234_sn_pass)

    g_sort_n = g02
    _cond_N = 3
    _cond3 = ([g01, g02, '<'], [g01, g02, '>'], [g02, g03, '>'])
    g2_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond3, _cond_N, g1234_sn_pass)

    g_sort_n = g03
    _cond_N = 5
    _cond5 = ([g01, g02, '<'], [g01, g02, '>'], [g02, g03, '<'], [g02, g03, '>'], [g03, g04, '>'])
    g3_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond5, _cond_N, g1234_sn_pass)
  
    g_sort_n = g04
    _cond_N = 6
    _cond6 = ([g01, g02, '<'], [g01, g02, '>'], [g02, g03, '<'], [g02, g03, '>'], [g03, g04, '<'], [g03, g04, '>'])
    g4_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    g_sort_n = g03
    _cond_N = 6
    _cond6 = ([g01, g02, '<'], [g01, g02, '>'], [g02, g03, '<'], [g02, g03, '>'], [g03, g04, '<'], [g03, g04, '<'])
    g3_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    g_sort_n = g02
    _cond_N = 5
    _cond5 = ([g01, g02, '<'], [g01, g02, '>'], [g02, g03, '<'], [g02, g03, '<'], [g02, g04, '>'])
    g2_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond5, _cond_N, g1234_sn_pass)

    g_sort_n = g04
    _cond_N = 6
    _cond6 = ([g01, g02, '<'], [g01, g02, '>'], [g02, g03, '<'], [g02, g03, '<'], [g02, g04, '<'], [g02, g04, '>'])
    g4_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    g_sort_n = g02
    _cond_N = 6
    _cond6 = ([g01, g02, '<'], [g01, g02, '>'], [g02, g03, '<'], [g02, g03, '<'], [g02, g04, '<'], [g02, g04, '<'])
    g2_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    g_sort_n = g01
    _cond_N = 3
    _cond3 = ([g01, g02, '<'], [g01, g02, '<'], [g01, g03, '>'])
    g1_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond3, _cond_N, g1234_sn_pass)

    g_sort_n = g03
    _cond_N = 5
    _cond5 = ([g01, g02, '<'], [g01, g02, '<'], [g01, g03, '<'], [g01, g03, '>'], [g03, g04, '>'])
    g3_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond5, _cond_N, g1234_sn_pass)

    g_sort_n = g04
    _cond_N = 6
    _cond6 = ([g01, g02, '<'], [g01, g02, '<'], [g01, g03, '<'], [g01, g03, '>'], [g03, g04, '<'], [g03, g04, '>'])
    g4_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    g_sort_n = g03
    _cond_N = 6
    _cond6 = ([g01, g02, '<'], [g01, g02, '<'], [g01, g03, '<'], [g01, g03, '>'], [g03, g04, '<'], [g03, g04, '<'])
    g3_3 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    g_sort_n = g01
    _cond_N = 5
    _cond5 = ([g01, g02, '<'], [g01, g02, '<'], [g01, g03, '<'], [g01, g03, '<'], [g01, g04, '>'])
    g1_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond5, _cond_N, g1234_sn_pass)


    g_sort_n = g04
    _cond_N = 6
    _cond6 = ([g01, g02, '<'], [g01, g02, '<'], [g01, g03, '<'], [g01, g03, '<'], [g01, g04, '<'], [g01, g04, '>'])
    g4_3 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    g_sort_n = g01
    _cond_N = 6
    _cond6 = ([g01, g02, '<'], [g01, g02, '<'], [g01, g03, '<'], [g01, g03, '<'], [g01, g04, '<'], [g01, g04, '<'])
    g1_3 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    g4_opt_snp4 = g1_0 + g1_1 + g1_2 + g1_3 \
            + g2_0 + g2_1 + g2_2 \
            + g3_0 + g3_1 + g3_2 + g3_3 \
            + g4_0 + g4_1 + g4_2 + g4_3 + 10*14 # -10 flag

    return g4_opt_snp4



def g5_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, sn_pass_ng_opt, bf_limit):
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g1_sorted = g_num_sort[0, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]

    g1_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) > np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g_opt[g1_cond1] += 1
    g1_0 = np.array([np.where(g_opt > 0, g1_sorted, 0)])
    
    
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g2_sorted = g_num_sort[1, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g2_sorted[np.newaxis], axis=0)[0]

    g2_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond2 = np.where((g_num_sort[0, :, :] > g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond3 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[2, :, :]) > np.log(bf_limit)) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    
    g_opt[g2_cond1] += 1
    g_opt[g2_cond2] += 1
    g_opt[g2_cond3] += 1
    g2_0 = np.array([np.where(g_opt > 2, g2_sorted, 0)])
    
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g3_sorted = g_num_sort[2, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g3_sorted[np.newaxis], axis=0)[0]

    g3_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond2 = np.where((g_num_sort[0, :, :] > g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond3 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond4 = np.where((g_num_sort[1, :, :] > g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond5 = np.where(((bevidences_sort[2, :, :] - bevidences_sort[3, :, :]) > np.log(bf_limit)) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    
    g_opt[g3_cond1] += 1
    g_opt[g3_cond2] += 1
    g_opt[g3_cond3] += 1
    g_opt[g3_cond4] += 1
    g_opt[g3_cond5] += 1
    g3_0 = np.array([np.where(g_opt > 4, g3_sorted, 0)])
    
    
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g4_sorted = g_num_sort[3, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g4_sorted[np.newaxis], axis=0)[0]

    g4_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond2 = np.where((g_num_sort[0, :, :] > g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond3 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond4 = np.where((g_num_sort[1, :, :] > g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond5 = np.where(((bevidences_sort[2, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond6 = np.where((g_num_sort[2, :, :] > g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond7 = np.where(((bevidences_sort[3, :, :] - bevidences_sort[4, :, :]) > np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    
    g_opt[g4_cond1] += 1
    g_opt[g4_cond2] += 1
    g_opt[g4_cond3] += 1
    g_opt[g4_cond4] += 1
    g_opt[g4_cond5] += 1
    g_opt[g4_cond6] += 1
    g_opt[g4_cond7] += 1
    g4_0 = np.array([np.where(g_opt > 6, g4_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g5_sorted = g_num_sort[4, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g5_sorted[np.newaxis], axis=0)[0]

    g5_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond2 = np.where((g_num_sort[0, :, :] > g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond3 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond4 = np.where((g_num_sort[1, :, :] > g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond5 = np.where(((bevidences_sort[2, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond6 = np.where((g_num_sort[2, :, :] > g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond7 = np.where(((bevidences_sort[3, :, :] - bevidences_sort[4, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond8 = np.where((g_num_sort[3, :, :] > g_num_sort[4, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
   
    g_opt[g5_cond1] += 1
    g_opt[g5_cond2] += 1
    g_opt[g5_cond3] += 1
    g_opt[g5_cond4] += 1
    g_opt[g5_cond5] += 1
    g_opt[g5_cond6] += 1
    g_opt[g5_cond7] += 1
    g_opt[g5_cond8] += 1
    g5_0 = np.array([np.where(g_opt > 7, g5_sorted, 0)])


    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g4_sorted = g_num_sort[3, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g4_sorted[np.newaxis], axis=0)[0]

    g4_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond2 = np.where((g_num_sort[0, :, :] > g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond3 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond4 = np.where((g_num_sort[1, :, :] > g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond5 = np.where(((bevidences_sort[2, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond6 = np.where((g_num_sort[2, :, :] > g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond7 = np.where(((bevidences_sort[3, :, :] - bevidences_sort[4, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond8 = np.where((g_num_sort[3, :, :] < g_num_sort[4, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
   
    g_opt[g4_cond1] += 1
    g_opt[g4_cond2] += 1
    g_opt[g4_cond3] += 1
    g_opt[g4_cond4] += 1
    g_opt[g4_cond5] += 1
    g_opt[g4_cond6] += 1
    g_opt[g4_cond7] += 1
    g_opt[g4_cond8] += 1
    g4_1 = np.array([np.where(g_opt > 7, g4_sorted, 0)])


    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g3_sorted = g_num_sort[2, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g3_sorted[np.newaxis], axis=0)[0]

    g3_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond2 = np.where((g_num_sort[0, :, :] > g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond3 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond4 = np.where((g_num_sort[1, :, :] > g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond5 = np.where(((bevidences_sort[2, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond6 = np.where((g_num_sort[2, :, :] < g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond7 = np.where(((bevidences_sort[2, :, :] - bevidences_sort[4, :, :]) > np.log(bf_limit)) & (sn_pass_ng_opt_value == (g3_sorted+1)))
   
    g_opt[g3_cond1] += 1
    g_opt[g3_cond2] += 1
    g_opt[g3_cond3] += 1
    g_opt[g3_cond4] += 1
    g_opt[g3_cond5] += 1
    g_opt[g3_cond6] += 1
    g_opt[g3_cond7] += 1
    g3_1 = np.array([np.where(g_opt > 6, g3_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g5_sorted = g_num_sort[4, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g5_sorted[np.newaxis], axis=0)[0]

    g5_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond2 = np.where((g_num_sort[0, :, :] > g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond3 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond4 = np.where((g_num_sort[1, :, :] > g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond5 = np.where(((bevidences_sort[2, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond6 = np.where((g_num_sort[2, :, :] < g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond7 = np.where(((bevidences_sort[2, :, :] - bevidences_sort[4, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond8 = np.where((g_num_sort[2, :, :] > g_num_sort[4, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
   
    g_opt[g5_cond1] += 1
    g_opt[g5_cond2] += 1
    g_opt[g5_cond3] += 1
    g_opt[g5_cond4] += 1
    g_opt[g5_cond5] += 1
    g_opt[g5_cond6] += 1
    g_opt[g5_cond7] += 1
    g_opt[g5_cond8] += 1
    g5_1 = np.array([np.where(g_opt > 7, g5_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g3_sorted = g_num_sort[2, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g3_sorted[np.newaxis], axis=0)[0]

    g3_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond2 = np.where((g_num_sort[0, :, :] > g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond3 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond4 = np.where((g_num_sort[1, :, :] > g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond5 = np.where(((bevidences_sort[2, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond6 = np.where((g_num_sort[2, :, :] < g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond7 = np.where(((bevidences_sort[2, :, :] - bevidences_sort[4, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond8 = np.where((g_num_sort[2, :, :] < g_num_sort[4, :, :]) & (sn_pass_ng_opt_value == (g3_sorted+1)))
   
    g_opt[g3_cond1] += 1
    g_opt[g3_cond2] += 1
    g_opt[g3_cond3] += 1
    g_opt[g3_cond4] += 1
    g_opt[g3_cond5] += 1
    g_opt[g3_cond6] += 1
    g_opt[g3_cond7] += 1
    g_opt[g3_cond8] += 1
    g3_2 = np.array([np.where(g_opt > 7, g3_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g2_sorted = g_num_sort[1, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g2_sorted[np.newaxis], axis=0)[0]

    g2_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond2 = np.where((g_num_sort[0, :, :] > g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond3 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond4 = np.where((g_num_sort[1, :, :] < g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond5 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[3, :, :]) > np.log(bf_limit)) & (sn_pass_ng_opt_value == (g2_sorted+1)))

    g_opt[g2_cond1] += 1
    g_opt[g2_cond2] += 1
    g_opt[g2_cond3] += 1
    g_opt[g2_cond4] += 1
    g_opt[g2_cond5] += 1
    g2_1 = np.array([np.where(g_opt > 4, g2_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g4_sorted = g_num_sort[3, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g4_sorted[np.newaxis], axis=0)[0]

    g4_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond2 = np.where((g_num_sort[0, :, :] > g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond3 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond4 = np.where((g_num_sort[1, :, :] < g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond5 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond6 = np.where((g_num_sort[1, :, :] > g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond7 = np.where(((bevidences_sort[3, :, :] - bevidences_sort[4, :, :]) > np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))

    g_opt[g4_cond1] += 1
    g_opt[g4_cond2] += 1
    g_opt[g4_cond3] += 1
    g_opt[g4_cond4] += 1
    g_opt[g4_cond5] += 1
    g_opt[g4_cond6] += 1
    g_opt[g4_cond7] += 1
    g4_2 = np.array([np.where(g_opt > 6, g4_sorted, 0)])


    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g5_sorted = g_num_sort[4, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g5_sorted[np.newaxis], axis=0)[0]

    g5_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond2 = np.where((g_num_sort[0, :, :] > g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond3 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond4 = np.where((g_num_sort[1, :, :] < g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond5 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond6 = np.where((g_num_sort[1, :, :] > g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond7 = np.where(((bevidences_sort[3, :, :] - bevidences_sort[4, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond8 = np.where((g_num_sort[3, :, :] > g_num_sort[4, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))

    g_opt[g5_cond1] += 1
    g_opt[g5_cond2] += 1
    g_opt[g5_cond3] += 1
    g_opt[g5_cond4] += 1
    g_opt[g5_cond5] += 1
    g_opt[g5_cond6] += 1
    g_opt[g5_cond7] += 1
    g_opt[g5_cond8] += 1
    g5_2 = np.array([np.where(g_opt > 7, g5_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g4_sorted = g_num_sort[3, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g4_sorted[np.newaxis], axis=0)[0]

    g4_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond2 = np.where((g_num_sort[0, :, :] > g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond3 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond4 = np.where((g_num_sort[1, :, :] < g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond5 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond6 = np.where((g_num_sort[1, :, :] > g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond7 = np.where(((bevidences_sort[3, :, :] - bevidences_sort[4, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond8 = np.where((g_num_sort[3, :, :] < g_num_sort[4, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))

    g_opt[g4_cond1] += 1
    g_opt[g4_cond2] += 1
    g_opt[g4_cond3] += 1
    g_opt[g4_cond4] += 1
    g_opt[g4_cond5] += 1
    g_opt[g4_cond6] += 1
    g_opt[g4_cond7] += 1
    g_opt[g4_cond8] += 1
    g4_3 = np.array([np.where(g_opt > 7, g4_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g2_sorted = g_num_sort[1, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g2_sorted[np.newaxis], axis=0)[0]

    g2_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond2 = np.where((g_num_sort[0, :, :] > g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond3 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond4 = np.where((g_num_sort[1, :, :] < g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond5 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond6 = np.where((g_num_sort[1, :, :] < g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond7 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[4, :, :]) > np.log(bf_limit)) & (sn_pass_ng_opt_value == (g2_sorted+1)))

    g_opt[g2_cond1] += 1
    g_opt[g2_cond2] += 1
    g_opt[g2_cond3] += 1
    g_opt[g2_cond4] += 1
    g_opt[g2_cond5] += 1
    g_opt[g2_cond6] += 1
    g_opt[g2_cond7] += 1
    g2_2 = np.array([np.where(g_opt > 6, g2_sorted, 0)])


    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g5_sorted = g_num_sort[4, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g5_sorted[np.newaxis], axis=0)[0]

    g5_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond2 = np.where((g_num_sort[0, :, :] > g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond3 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond4 = np.where((g_num_sort[1, :, :] < g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond5 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond6 = np.where((g_num_sort[1, :, :] < g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond7 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[4, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond8 = np.where((g_num_sort[1, :, :] > g_num_sort[4, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))

    g_opt[g5_cond1] += 1
    g_opt[g5_cond2] += 1
    g_opt[g5_cond3] += 1
    g_opt[g5_cond4] += 1
    g_opt[g5_cond5] += 1
    g_opt[g5_cond6] += 1
    g_opt[g5_cond7] += 1
    g_opt[g5_cond8] += 1
    g5_3 = np.array([np.where(g_opt > 7, g5_sorted, 0)])


    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g2_sorted = g_num_sort[1, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g2_sorted[np.newaxis], axis=0)[0]

    g2_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond2 = np.where((g_num_sort[0, :, :] > g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond3 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond4 = np.where((g_num_sort[1, :, :] < g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond5 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond6 = np.where((g_num_sort[1, :, :] < g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond7 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[4, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond8 = np.where((g_num_sort[1, :, :] < g_num_sort[4, :, :]) & (sn_pass_ng_opt_value == (g2_sorted+1)))

    g_opt[g2_cond1] += 1
    g_opt[g2_cond2] += 1
    g_opt[g2_cond3] += 1
    g_opt[g2_cond4] += 1
    g_opt[g2_cond5] += 1
    g_opt[g2_cond6] += 1
    g_opt[g2_cond7] += 1
    g_opt[g2_cond8] += 1
    g2_3 = np.array([np.where(g_opt > 7, g2_sorted, 0)])


    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g1_sorted = g_num_sort[0, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]

    g1_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond2 = np.where((g_num_sort[0, :, :] < g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond3 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[2, :, :]) > np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1)))

    g_opt[g1_cond1] += 1
    g_opt[g1_cond2] += 1
    g_opt[g1_cond3] += 1
    g1_1 = np.array([np.where(g_opt > 2, g1_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g3_sorted = g_num_sort[2, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g3_sorted[np.newaxis], axis=0)[0]

    g3_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond2 = np.where((g_num_sort[0, :, :] < g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond3 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond4 = np.where((g_num_sort[0, :, :] > g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond5 = np.where(((bevidences_sort[2, :, :] - bevidences_sort[3, :, :]) > np.log(bf_limit)) & (sn_pass_ng_opt_value == (g3_sorted+1)))

    g_opt[g3_cond1] += 1
    g_opt[g3_cond2] += 1
    g_opt[g3_cond3] += 1
    g_opt[g3_cond4] += 1
    g_opt[g3_cond5] += 1
    g3_3 = np.array([np.where(g_opt > 4, g3_sorted, 0)])


    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g4_sorted = g_num_sort[3, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g4_sorted[np.newaxis], axis=0)[0]

    g4_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond2 = np.where((g_num_sort[0, :, :] < g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond3 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond4 = np.where((g_num_sort[0, :, :] > g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond5 = np.where(((bevidences_sort[2, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond6 = np.where((g_num_sort[2, :, :] > g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond7 = np.where(((bevidences_sort[3, :, :] - bevidences_sort[4, :, :]) > np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))

    g_opt[g4_cond1] += 1
    g_opt[g4_cond2] += 1
    g_opt[g4_cond3] += 1
    g_opt[g4_cond4] += 1
    g_opt[g4_cond5] += 1
    g_opt[g4_cond6] += 1
    g_opt[g4_cond7] += 1
    g4_4 = np.array([np.where(g_opt > 6, g4_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g5_sorted = g_num_sort[4, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g5_sorted[np.newaxis], axis=0)[0]

    g5_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond2 = np.where((g_num_sort[0, :, :] < g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond3 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond4 = np.where((g_num_sort[0, :, :] > g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond5 = np.where(((bevidences_sort[2, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond6 = np.where((g_num_sort[2, :, :] > g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond7 = np.where(((bevidences_sort[3, :, :] - bevidences_sort[4, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond8 = np.where((g_num_sort[3, :, :] > g_num_sort[4, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))

    g_opt[g5_cond1] += 1
    g_opt[g5_cond2] += 1
    g_opt[g5_cond3] += 1
    g_opt[g5_cond4] += 1
    g_opt[g5_cond5] += 1
    g_opt[g5_cond6] += 1
    g_opt[g5_cond7] += 1
    g_opt[g5_cond8] += 1
    g5_4 = np.array([np.where(g_opt > 7, g5_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g4_sorted = g_num_sort[4, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g4_sorted[np.newaxis], axis=0)[0]

    g4_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond2 = np.where((g_num_sort[0, :, :] < g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond3 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond4 = np.where((g_num_sort[0, :, :] > g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond5 = np.where(((bevidences_sort[2, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond6 = np.where((g_num_sort[2, :, :] > g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond7 = np.where(((bevidences_sort[3, :, :] - bevidences_sort[4, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond8 = np.where((g_num_sort[3, :, :] < g_num_sort[4, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))

    g_opt[g4_cond1] += 1
    g_opt[g4_cond2] += 1
    g_opt[g4_cond3] += 1
    g_opt[g4_cond4] += 1
    g_opt[g4_cond5] += 1
    g_opt[g4_cond6] += 1
    g_opt[g4_cond7] += 1
    g_opt[g4_cond8] += 1
    g4_5 = np.array([np.where(g_opt > 7, g4_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g3_sorted = g_num_sort[2, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g3_sorted[np.newaxis], axis=0)[0]

    g3_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g3_cond2 = np.where((g_num_sort[0, :, :] < g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g3_cond3 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g3_cond4 = np.where((g_num_sort[0, :, :] > g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g3_cond5 = np.where(((bevidences_sort[2, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g3_cond6 = np.where((g_num_sort[2, :, :] < g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g3_cond7 = np.where(((bevidences_sort[2, :, :] - bevidences_sort[4, :, :]) > np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))

    g_opt[g3_cond1] += 1
    g_opt[g3_cond2] += 1
    g_opt[g3_cond3] += 1
    g_opt[g3_cond4] += 1
    g_opt[g3_cond5] += 1
    g_opt[g3_cond6] += 1
    g_opt[g3_cond7] += 1
    g3_4 = np.array([np.where(g_opt > 6, g3_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g5_sorted = g_num_sort[4, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g5_sorted[np.newaxis], axis=0)[0]

    g5_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond2 = np.where((g_num_sort[0, :, :] < g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond3 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond4 = np.where((g_num_sort[0, :, :] > g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond5 = np.where(((bevidences_sort[2, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond6 = np.where((g_num_sort[2, :, :] < g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond7 = np.where(((bevidences_sort[2, :, :] - bevidences_sort[4, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond8 = np.where((g_num_sort[2, :, :] > g_num_sort[4, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))

    g_opt[g5_cond1] += 1
    g_opt[g5_cond2] += 1
    g_opt[g5_cond3] += 1
    g_opt[g5_cond4] += 1
    g_opt[g5_cond5] += 1
    g_opt[g5_cond6] += 1
    g_opt[g5_cond7] += 1
    g_opt[g5_cond8] += 1
    g5_5 = np.array([np.where(g_opt > 7, g5_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g3_sorted = g_num_sort[2, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g3_sorted[np.newaxis], axis=0)[0]

    g3_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond2 = np.where((g_num_sort[0, :, :] < g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond3 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond4 = np.where((g_num_sort[0, :, :] > g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond5 = np.where(((bevidences_sort[2, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond6 = np.where((g_num_sort[2, :, :] < g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond7 = np.where(((bevidences_sort[2, :, :] - bevidences_sort[4, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g3_sorted+1)))
    g3_cond8 = np.where((g_num_sort[2, :, :] < g_num_sort[4, :, :]) & (sn_pass_ng_opt_value == (g3_sorted+1)))

    g_opt[g3_cond1] += 1
    g_opt[g3_cond2] += 1
    g_opt[g3_cond3] += 1
    g_opt[g3_cond4] += 1
    g_opt[g3_cond5] += 1
    g_opt[g3_cond6] += 1
    g_opt[g3_cond7] += 1
    g_opt[g3_cond8] += 1
    g3_5 = np.array([np.where(g_opt > 7, g3_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g1_sorted = g_num_sort[0, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]

    g1_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond2 = np.where((g_num_sort[0, :, :] < g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond3 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond4 = np.where((g_num_sort[0, :, :] < g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond5 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[3, :, :]) > np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1)))

    g_opt[g1_cond1] += 1
    g_opt[g1_cond2] += 1
    g_opt[g1_cond3] += 1
    g_opt[g1_cond4] += 1
    g_opt[g1_cond5] += 1
    g1_2 = np.array([np.where(g_opt > 4, g1_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g4_sorted = g_num_sort[3, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g4_sorted[np.newaxis], axis=0)[0]

    g4_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond2 = np.where((g_num_sort[0, :, :] < g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond3 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond4 = np.where((g_num_sort[0, :, :] < g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond5 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond6 = np.where((g_num_sort[0, :, :] > g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond7 = np.where(((bevidences_sort[3, :, :] - bevidences_sort[4, :, :]) > np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))

    g_opt[g4_cond1] += 1
    g_opt[g4_cond2] += 1
    g_opt[g4_cond3] += 1
    g_opt[g4_cond4] += 1
    g_opt[g4_cond5] += 1
    g_opt[g4_cond6] += 1
    g_opt[g4_cond7] += 1
    g4_6 = np.array([np.where(g_opt > 6, g4_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g5_sorted = g_num_sort[4, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g5_sorted[np.newaxis], axis=0)[0]

    g5_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond2 = np.where((g_num_sort[0, :, :] < g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond3 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond4 = np.where((g_num_sort[0, :, :] < g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond5 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond6 = np.where((g_num_sort[0, :, :] > g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond7 = np.where(((bevidences_sort[3, :, :] - bevidences_sort[4, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond8 = np.where((g_num_sort[3, :, :] > g_num_sort[4, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))

    g_opt[g5_cond1] += 1
    g_opt[g5_cond2] += 1
    g_opt[g5_cond3] += 1
    g_opt[g5_cond4] += 1
    g_opt[g5_cond5] += 1
    g_opt[g5_cond6] += 1
    g_opt[g5_cond7] += 1
    g_opt[g5_cond8] += 1
    g5_6 = np.array([np.where(g_opt > 7, g5_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g4_sorted = g_num_sort[3, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g4_sorted[np.newaxis], axis=0)[0]

    g4_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond2 = np.where((g_num_sort[0, :, :] < g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond3 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond4 = np.where((g_num_sort[0, :, :] < g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond5 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond6 = np.where((g_num_sort[0, :, :] > g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond7 = np.where(((bevidences_sort[3, :, :] - bevidences_sort[4, :, :]) < np.log(bf_limit))& (sn_pass_ng_opt_value == (g4_sorted+1)))
    g4_cond8 = np.where((g_num_sort[3, :, :] < g_num_sort[4, :, :]) & (sn_pass_ng_opt_value == (g4_sorted+1)))

    g_opt[g4_cond1] += 1
    g_opt[g4_cond2] += 1
    g_opt[g4_cond3] += 1
    g_opt[g4_cond4] += 1
    g_opt[g4_cond5] += 1
    g_opt[g4_cond6] += 1
    g_opt[g4_cond7] += 1
    g_opt[g4_cond8] += 1
    g4_7 = np.array([np.where(g_opt > 7, g4_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g1_sorted = g_num_sort[0, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]

    g1_cond1 = np.where((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1))
    g1_cond2 = np.where((g_num_sort[0, :, :] < g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond3 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond4 = np.where((g_num_sort[0, :, :] < g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond5 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond6 = np.where((g_num_sort[0, :, :] < g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond7 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[4, :, :]) > np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1)))

    g_opt[g1_cond1] += 1
    g_opt[g1_cond2] += 1
    g_opt[g1_cond3] += 1
    g_opt[g1_cond4] += 1
    g_opt[g1_cond5] += 1
    g_opt[g1_cond6] += 1
    g_opt[g1_cond7] += 1
    g1_3 = np.array([np.where(g_opt > 6, g1_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g5_sorted = g_num_sort[4, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g5_sorted[np.newaxis], axis=0)[0]

    g5_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond2 = np.where((g_num_sort[0, :, :] < g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond3 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond4 = np.where((g_num_sort[0, :, :] < g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond5 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond6 = np.where((g_num_sort[0, :, :] < g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond7 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[4, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g5_sorted+1)))
    g5_cond8 = np.where((g_num_sort[0, :, :] > g_num_sort[4, :, :]) & (sn_pass_ng_opt_value == (g5_sorted+1)))

    g_opt[g5_cond1] += 1
    g_opt[g5_cond2] += 1
    g_opt[g5_cond3] += 1
    g_opt[g5_cond4] += 1
    g_opt[g5_cond5] += 1
    g_opt[g5_cond6] += 1
    g_opt[g5_cond7] += 1
    g_opt[g5_cond8] += 1
    g5_7 = np.array([np.where(g_opt > 7, g5_sorted, 0)])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    g1_sorted = g_num_sort[0, :, :]

    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]

    g1_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond2 = np.where((g_num_sort[0, :, :] < g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond3 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond4 = np.where((g_num_sort[0, :, :] < g_num_sort[2, :, :]) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond5 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[3, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond6 = np.where((g_num_sort[0, :, :] < g_num_sort[3, :, :]) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond7 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[4, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond8 = np.where((g_num_sort[0, :, :] < g_num_sort[4, :, :]) & (sn_pass_ng_opt_value == (g1_sorted+1)))

    g_opt[g1_cond1] += 1
    g_opt[g1_cond2] += 1
    g_opt[g1_cond3] += 1
    g_opt[g1_cond4] += 1
    g_opt[g1_cond5] += 1
    g_opt[g1_cond6] += 1
    g_opt[g1_cond7] += 1
    g_opt[g1_cond8] += 1
    g1_4 = np.array([np.where(g_opt > 7, g1_sorted, 0)])


    g5_opt = g1_0 + g1_1 + g1_2 + g1_3 + g1_4 \
            + g2_0 + g2_1 + g2_2 + g2_3 \
            + g3_0 + g3_1 + g3_2 + g3_3 + g3_4 + g3_5 \
            + g4_0 + g4_1 + g4_2 + g4_3 + g4_4 + g4_5 + g4_6 + g4_7 \
            + g5_0 + g5_1 + g5_2 + g5_3 + g5_4 + g5_5 + g5_6 + g5_7

    return g5_opt



def generate_gfit_indices(n_gauss, gauss_param):




    nparams_eachmodel = 2*(3*n_gauss+2) + n_gauss + 7
    baygaud_gfit_indices = []

    if gauss_param == '_int':
        start_index = 4
    elif gauss_param == '_vdisp':
        start_index = 3
    elif gauss_param == '_vlos':
        start_index = 2


    for i in range(1, n_gauss + 1):
        indices = np.array([start_index + j * 3 for j in range(i)])
        baygaud_gfit_indices.append(indices)
        start_index += nparams_eachmodel

    return np.array(baygaud_gfit_indices)


def sort_gaussian_parameters1(params):
    sorted_params = np.empty_like(params)

    for model_idx in range(6):  # 가우스 모델 6까지
        start_idx = model_idx * 53
        
        gaussian_indices = np.arange(start_idx + 2, start_idx + 2 + 3*(model_idx + 1), 3)  # 각 가우스 함수의 첫번째 파라미터 인덱스

        sorted_gaussian_indices = np.argsort(params[gaussian_indices])




        for i, gaussian_idx in enumerate(gaussian_indices):
            if gaussian_idx in sorted_gaussian_indices:
                sorted_params[start_idx + i*9] = params[gaussian_idx]
                sorted_params[start_idx + i*9 + 1] = params[gaussian_idx + 1]
                sorted_params[start_idx + i*9 + 2] = params[gaussian_idx + 2]
                sorted_params[start_idx + i*9 + 3] = params[gaussian_idx + 3]
                sorted_params[start_idx + i*9 + 4] = params[gaussian_idx + 4]
                sorted_params[start_idx + i*9 + 5] = params[gaussian_idx + 5]
                sorted_params[start_idx + i*9 + 6] = params[gaussian_idx + 6]
                sorted_params[start_idx + i*9 + 7] = params[gaussian_idx + 7]
                sorted_params[start_idx + i*9 + 8] = params[gaussian_idx + 8]
            else:
                sorted_params[start_idx + i*9:start_idx + (i+1)*9] = params[start_idx + i*9:start_idx + (i+1)*9]

        sorted_params[start_idx:start_idx+2] = params[start_idx:start_idx+2]
        sorted_params[start_idx+53:start_idx+159] = params[start_idx+53:start_idx+159]

    return sorted_params



def sort_gaussians_wrt_gparam_serialized(_fitsarray_gfit_results2, n_gauss, gauss_param):

    if gauss_param == '_int':
        sort_wrt = 2
    elif gauss_param == '_vdisp':
        sort_wrt = 1
    elif gauss_param == '_vlos':
        sort_wrt = 0

    _fitsarray_gfit_results2_sorted = _fitsarray_gfit_results2
    nparams_eachmodel = 2*(3*n_gauss+2) + n_gauss + 7

    naxis1 = _fitsarray_gfit_results2.shape[2] # (gfit_params, naxis2, naxis1)
    naxis2 = _fitsarray_gfit_results2.shape[1] # (gfit_params, naxis2, naxis1)

    for _x in range(naxis1):
        for _y in range(naxis2):

            for model_idx in range(n_gauss):  # 가우스 모델 n_gauss까지
                start_idx = model_idx * nparams_eachmodel
                
                gfit_params_list = []
                gfit_params_e_list = []
                
                for gaussian_idx in range(model_idx + 1):
                    gaussian_first_idx = start_idx + 2 + gaussian_idx * 3
                    gaussian_last_idx = gaussian_first_idx + 3

                    gaussian_e_first_idx = start_idx + 4 + (model_idx + 1 + gaussian_idx) * 3
                    gaussian_e_last_idx = gaussian_e_first_idx + 3
                    
                    gfit_params_list.append(_fitsarray_gfit_results2[gaussian_first_idx:gaussian_last_idx, _y, _x])
                    gfit_params_e_list.append(_fitsarray_gfit_results2[gaussian_e_first_idx:gaussian_e_last_idx, _y, _x])


                gfit_params_t1 = np.concatenate(gfit_params_list)
                gfit_params_t2 = gfit_params_t1.reshape(-1, 3)
                gfit_params = np.vstack(gfit_params_t2)

                gfit_params_e_t1 = np.concatenate(gfit_params_e_list)
                gfit_params_e_t2 = gfit_params_e_t1.reshape(-1, 3)
                gfit_params_e = np.vstack(gfit_params_e_t2)

                gfit_params_sorted_indices = np.argsort(-gfit_params[:, sort_wrt])
                gfit_params_sorted = gfit_params[gfit_params_sorted_indices]
                gfit_params_e_sorted = gfit_params_e[gfit_params_sorted_indices]


                cur_model_gfit_params_first_idx = start_idx + 2 + 0 * 3
                cur_model_gfit_params_last_idx = start_idx + 2 + 0 * 3 + 3*(model_idx+1)

                cur_model_gfit_params_e_first_idx = start_idx + 4 + (model_idx + 1 + 0) * 3
                cur_model_gfit_params_e_end_idx = start_idx + 4 + (model_idx + 1 + 0) * 3 + 3*(model_idx+1)
                
                _fitsarray_gfit_results2_sorted[cur_model_gfit_params_first_idx:cur_model_gfit_params_last_idx, _y, _x] = gfit_params_sorted.flatten()
                _fitsarray_gfit_results2_sorted[cur_model_gfit_params_e_first_idx:cur_model_gfit_params_e_end_idx, _y, _x] = gfit_params_e_sorted.flatten()


    return _fitsarray_gfit_results2_sorted





# mode: 0=vlos, 1=vdisp, 2=peak_amp, 3=integrated_int(sqrt(2*pi)*sigma*amp)
@njit(parallel=True, cache=True)
def _sort_blocks_numba(blk_p, blk_e, mode):
    """
    Sort a single model block in-place style (returns new arrays).
    blk_p, blk_e: shape (3*k, ny, nx) for k Gaussians (each Gaussian has 3 params)
    mode: which key to sort on (descending)
    """
    ny = blk_p.shape[1]
    nx = blk_p.shape[2]
    k  = blk_p.shape[0] // 3

    out_p = np.empty_like(blk_p)
    out_e = np.empty_like(blk_e)

    sqrt2pi = np.sqrt(2.0 * np.pi)

    # Parallelize over rows; inner loop over columns
    for y in prange(ny):
        for x in range(nx):
            # Build sort keys per Gaussian for this pixel
            keys = np.empty(k, dtype=blk_p.dtype)
            if mode == 3:
                # integrated intensity = sqrt(2*pi) * vdisp * peak_amp
                for g in range(k):
                    vdisp = blk_p[3 * g + 1, y, x]
                    amp   = blk_p[3 * g + 2, y, x]
                    keys[g] = sqrt2pi * vdisp * amp
            else:
                for g in range(k):
                    keys[g] = blk_p[3 * g + mode, y, x]

            # Descending order
            order = np.argsort(-keys)

            # Reorder params and errors by the sorted indices
            for r in range(k):
                src = order[r]
                dst_base = 3 * r
                src_base = 3 * src

                # params
                out_p[dst_base + 0, y, x] = blk_p[src_base + 0, y, x]
                out_p[dst_base + 1, y, x] = blk_p[src_base + 1, y, x]
                out_p[dst_base + 2, y, x] = blk_p[src_base + 2, y, x]
                # errors
                out_e[dst_base + 0, y, x] = blk_e[src_base + 0, y, x]
                out_e[dst_base + 1, y, x] = blk_e[src_base + 1, y, x]
                out_e[dst_base + 2, y, x] = blk_e[src_base + 2, y, x]

    return out_p, out_e


def sort_gaussians_wrt_gparam_numba(_fitsarray_gfit_results2, n_gauss, gauss_param):
    """
    Numba-accelerated sorter with the same output meaning as the vectorized version.

    Inputs
    ------
    _fitsarray_gfit_results2 : np.ndarray, shape (P, ny, nx)
        Packed parameter cube across models (1..n_gauss). Layout must match the
        original function:
          - For model with k components (k = 1..n_gauss):
              params block: [start+2 : start+2+3*k)    -> (vlos, vdisp, peak) per Gaussian
              errs   block: [start+4+3*k : start+4+6*k)
          - nparams_eachmodel = 2*(3*n_gauss + 2) + n_gauss + 7
    n_gauss : int
        Maximum number of Gaussians.
    gauss_param : str
        One of '_peak_amp', '_vdisp', '_vlos', '_integrated_int'.

    Returns
    -------
    np.ndarray
        Sorted copy of _fitsarray_gfit_results2.
    """
    if gauss_param == '_peak_amp':
        mode = 2
    elif gauss_param == '_vdisp':
        mode = 1
    elif gauss_param == '_vlos':
        mode = 0
    elif gauss_param == '_integrated_int':
        mode = 3
    else:
        raise ValueError(f"Unknown gauss_param: {gauss_param}")

    arr_in  = _fitsarray_gfit_results2
    arr_out = np.copy(arr_in)

    ny = arr_in.shape[1]
    nx = arr_in.shape[2]

    nparams_eachmodel = 2 * (3 * n_gauss + 2) + n_gauss + 7

    for model_idx in range(n_gauss):
        k = model_idx + 1
        start = model_idx * nparams_eachmodel

        # Slice contiguously for this model block
        pblk = arr_in[start + 2 : start + 2 + 3 * k, :, :]
        eblk = arr_in[start + 4 + 3 * k : start + 4 + 6 * k, :, :]

        # Make sure blocks are contiguous for Numba speed
        pblk_c = np.ascontiguousarray(pblk)
        eblk_c = np.ascontiguousarray(eblk)

        # Sort this model block
        p_sorted, e_sorted = _sort_blocks_numba(pblk_c, eblk_c, mode)

        # Write back into the output array
        arr_out[start + 2 : start + 2 + 3 * k, :, :] = p_sorted
        arr_out[start + 4 + 3 * k : start + 4 + 6 * k, :, :] = e_sorted

    return arr_out





def sort_gaussians_wrt_gparam_vectorized(_fitsarray_gfit_results2, n_gauss, gauss_param):
    if gauss_param == '_peak_amp':
        sort_wrt = 2
    elif gauss_param == '_vdisp':
        sort_wrt = 1
    elif gauss_param == '_vlos':
        sort_wrt = 0
    elif gauss_param == '_integrated_int':
        sort_wrt = 0


    _fitsarray_gfit_results2_sorted = np.copy(_fitsarray_gfit_results2)

    nparams_eachmodel = 2 * (3 * n_gauss + 2) + n_gauss + 7
    naxis1 = _fitsarray_gfit_results2.shape[2]
    naxis2 = _fitsarray_gfit_results2.shape[1]

    for model_idx in range(n_gauss):
        all_params = np.empty((naxis2, naxis1, model_idx+1, 3))
        all_params_e = np.empty((naxis2, naxis1, model_idx+1, 3))

        all_params_integrated_intensity = np.empty((naxis2, naxis1, model_idx+1, 3))
        all_params_e_integrated_intensity = np.empty((naxis2, naxis1, model_idx+1, 3))

        start_idx = model_idx * nparams_eachmodel
        for gaussian_idx in range(model_idx + 1):
            gaussian_first_idx = start_idx + 2 + gaussian_idx * 3
            gaussian_last_idx = gaussian_first_idx + 3
            gaussian_e_first_idx = start_idx + 4 + (model_idx + 1 + gaussian_idx) * 3
            gaussian_e_last_idx = gaussian_e_first_idx + 3

            all_params[:, :, gaussian_idx, :] = _fitsarray_gfit_results2[gaussian_first_idx:gaussian_last_idx].transpose((1, 2, 0))
            all_params_e[:, :, gaussian_idx, :] = _fitsarray_gfit_results2[gaussian_e_first_idx:gaussian_e_last_idx].transpose((1, 2, 0))

            if gauss_param == '_integrated_int':
                all_params_integrated_intensity[:, :, gaussian_idx, :] = _fitsarray_gfit_results2[gaussian_first_idx:gaussian_last_idx].transpose((1, 2, 0))
                all_params_e_integrated_intensity[:, :, gaussian_idx, :] = _fitsarray_gfit_results2[gaussian_e_first_idx:gaussian_e_last_idx].transpose((1, 2, 0))

                vdisp_cur_gauss = all_params_integrated_intensity[:, :, gaussian_idx, 1]
                peak_amp_cur_gauss = all_params_integrated_intensity[:, :, gaussian_idx, 2]

                all_params_integrated_intensity[:, :, gaussian_idx, 0] = np.sqrt(2*np.pi) * vdisp_cur_gauss * peak_amp_cur_gauss
            

        if gauss_param == '_integrated_int':
            sorted_indices_integrated_int = np.argsort(-all_params_integrated_intensity[:, :, :, 0], axis=2) # descending order

            all_params_sorted_integrated_int = np.take_along_axis(all_params, sorted_indices_integrated_int[..., None], axis=2)
            all_params_e_sorted_integrated_int = np.take_along_axis(all_params_e, sorted_indices_integrated_int[..., None], axis=2)

            flattened_params = all_params_sorted_integrated_int.reshape(naxis2, naxis1, -1)
            flattened_params_e = all_params_e_sorted_integrated_int.reshape(naxis2, naxis1, -1)

        else:
            sorted_indices = np.argsort(-all_params[:, :, :, sort_wrt], axis=2) # descending order
            all_params_sorted = np.take_along_axis(all_params, sorted_indices[..., None], axis=2)
            all_params_e_sorted = np.take_along_axis(all_params_e, sorted_indices[..., None], axis=2)

            flattened_params = all_params_sorted.reshape(naxis2, naxis1, -1)
            flattened_params_e = all_params_e_sorted.reshape(naxis2, naxis1, -1)



        start_idx = model_idx * nparams_eachmodel

        cur_model_gfit_params_first_idx = start_idx + 2 + 0 * 3
        cur_model_gfit_params_last_idx = start_idx + 2 + 0 * 3 + 3*(model_idx+1)

        cur_model_gfit_params_e_first_idx = start_idx + 4 + (model_idx + 1 + 0) * 3
        cur_model_gfit_params_e_end_idx = start_idx + 4 + (model_idx + 1 + 0) * 3 + 3*(model_idx+1)

        _fitsarray_gfit_results2_sorted[cur_model_gfit_params_first_idx:cur_model_gfit_params_last_idx, :, :] = flattened_params.transpose(2, 0, 1)
        _fitsarray_gfit_results2_sorted[cur_model_gfit_params_e_first_idx:cur_model_gfit_params_e_end_idx, :, :] = flattened_params_e.transpose(2, 0, 1)

    return _fitsarray_gfit_results2_sorted




@njit(parallel=True, cache=True)
def _fill_sn_slice_numba_kernel(arr, out_slice, max_ngauss, nparams_step, lower, upper):
    """
    Numba kernel that fills out_slice with the same logic as the original:
    out_slice[i, j, y, x] = num/denom if (j>=i) and (denom>0) and (lower <= sigma < upper), else 0.
    """
    ny = arr.shape[1]
    nx = arr.shape[2]
    for i in prange(max_ngauss):
        for j in range(max_ngauss):
            # If j < i, fill zeros directly
            if j < i:
                for y in range(ny):
                    for x in range(nx):
                        out_slice[i, j, y, x] = 0.0
                continue

            idx_denom = nparams_step * (j + 1) - max_ngauss - 7 + j
            idx_sigma = nparams_step * j + 3 + 3 * i
            idx_num   = nparams_step * j + 4 + 3 * i

            for y in range(ny):
                for x in range(nx):
                    den = arr[idx_denom, y, x]
                    sig = arr[idx_sigma, y, x]
                    if (den > 0.0) and (sig >= lower) and (sig < upper):
                        out_slice[i, j, y, x] = arr[idx_num, y, x] / den
                    else:
                        out_slice[i, j, y, x] = 0.0

def compute_sn_ng_opt_slice_numba(sn_ng_opt_slice,
                                  _fitsarray_gfit_results2,
                                  max_ngauss,
                                  nparams_step,
                                  _params,
                                  verbose: bool = False,
                                  i1: int | None = None,
                                  j1: int | None = None):
    """
    Numba-accelerated version with a verbose toggle.
    The heavy math runs in a JIT kernel; prints are done in Python when verbose=True.
    """
    arr = _fitsarray_gfit_results2
    ny, nx = arr.shape[1], arr.shape[2]

    # Resolve (j1, i1) for debug print
    if j1 is None:
        j1 = int(_params.get('_j0', 0))
    if i1 is None:
        i1 = int(_params.get('_i0', 0))
    j1 = max(0, min(j1, ny - 1))
    i1 = max(0, min(i1, nx - 1))

    # Compute with Numba kernel
    _fill_sn_slice_numba_kernel(
        arr,
        sn_ng_opt_slice,
        max_ngauss,
        nparams_step,
        _params['g_sigma_lower'],
        _params['g_sigma_upper'],
    )

    # Match the original print behavior if requested
    if verbose:
        for i in range(max_ngauss):
            for j in range(max_ngauss):
                if j >= i:
                    print(i, j,
                          arr[nparams_step*j + 2 + 3*i, j1, i1],
                          arr[nparams_step*j + 4 + 3*i, j1, i1],
                          arr[nparams_step*(j+1)-max_ngauss-7+j, j1, i1])
            print("")




from numba import njit, prange

@njit(parallel=True, cache=True)
def _fill_sn_slice_sorted_wrt_amp_kernel(arr, out_slice, max_ngauss, nparams_step, lower, upper):
    """
    Numba kernel for the same formula:
        out[i, j, y, x] = num/denom if (j>=i) and (den>0) and (lower<=sigma<upper), else 0
    arr: (P, ny, nx), out_slice: (max_ngauss, max_ngauss, ny, nx)
    """
    ny = arr.shape[1]
    nx = arr.shape[2]
    for i in prange(max_ngauss):
        for j in range(max_ngauss):
            if j < i:
                for y in range(ny):
                    for x in range(nx):
                        out_slice[i, j, y, x] = 0.0
                continue

            idx_denom = nparams_step * (j + 1) - max_ngauss - 7 + j
            idx_sigma = nparams_step * j + 3 + 3 * i
            idx_num   = nparams_step * j + 4 + 3 * i

            for y in range(ny):
                for x in range(nx):
                    den = arr[idx_denom, y, x]
                    sig = arr[idx_sigma, y, x]
                    if (den > 0.0) and (sig >= lower) and (sig < upper):
                        out_slice[i, j, y, x] = arr[idx_num, y, x] / den
                    else:
                        out_slice[i, j, y, x] = 0.0

def compute_sn_ng_opt_slice_sorted_wrt_amp_numba(
    sn_ng_opt_slice_gparam_sorted_wrt_amp: np.ndarray,
    _fitsarray_gfit_results2_sorted_wrt_amp: np.ndarray,
    max_ngauss: int,
    nparams_step: int,
    _params: dict,
    verbose: bool = False,
    i1: int | None = None,   # x index to print
    j1: int | None = None,   # y index to print
):
    """
    Numba-accelerated version with a verbose toggle.
    """
    arr = _fitsarray_gfit_results2_sorted_wrt_amp
    ny, nx = arr.shape[1], arr.shape[2]

    # Resolve (j1, i1) for debug prints
    if j1 is None:
        j1 = int(_params.get('_j0', 0))
    if i1 is None:
        i1 = int(_params.get('_i0', 0))
    j1 = max(0, min(j1, ny - 1))
    i1 = max(0, min(i1, nx - 1))

    _fill_sn_slice_sorted_wrt_amp_kernel(
        arr,
        sn_ng_opt_slice_gparam_sorted_wrt_amp,
        max_ngauss,
        nparams_step,
        _params['g_sigma_lower'],
        _params['g_sigma_upper'],
    )

    if verbose:
        for i in range(max_ngauss):
            for j in range(max_ngauss):
                if j >= i:
                    print(i, j,
                          arr[nparams_step * j + 2 + 3 * i, j1, i1],
                          arr[nparams_step * j + 4 + 3 * i, j1, i1],
                          arr[nparams_step * (j + 1) - max_ngauss - 7 + j, j1, i1])
            print("")





import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True)
def _fill_sn_slice_sorted_wrt_vdisp_kernel(arr, out_slice, max_ngauss, nparams_step, lower, upper):
    """
    Numba kernel:
      out[i, j, y, x] = num/den if (j>=i) and (den>0) and (lower<=sigma<upper), else 0
    arr: (P, ny, nx), out_slice: (max_ngauss, max_ngauss, ny, nx)
    """
    ny = arr.shape[1]
    nx = arr.shape[2]
    for i in prange(max_ngauss):
        for j in range(max_ngauss):
            if j < i:
                for y in range(ny):
                    for x in range(nx):
                        out_slice[i, j, y, x] = 0.0
                continue

            idx_denom = nparams_step * (j + 1) - max_ngauss - 7 + j
            idx_sigma = nparams_step * j + 3 + 3 * i
            idx_num   = nparams_step * j + 4 + 3 * i

            for y in range(ny):
                for x in range(nx):
                    den = arr[idx_denom, y, x]
                    sig = arr[idx_sigma, y, x]
                    if (den > 0.0) and (sig >= lower) and (sig < upper):
                        out_slice[i, j, y, x] = arr[idx_num, y, x] / den
                    else:
                        out_slice[i, j, y, x] = 0.0

def compute_sn_ng_opt_slice_sorted_wrt_vdisp_numba(
    sn_ng_opt_slice_gparam_sorted_wrt_vdisp: np.ndarray,
    _fitsarray_gfit_results2_sorted_wrt_vdisp: np.ndarray,
    max_ngauss: int,
    nparams_step: int,
    _params: dict,
    verbose: bool = False,
    i1: int | None = None,   # x index to print
    j1: int | None = None,   # y index to print
):
    """
    Numba-accelerated version (same logic). Prints like the original when verbose=True.
    """
    arr = _fitsarray_gfit_results2_sorted_wrt_vdisp
    ny, nx = arr.shape[1], arr.shape[2]

    # Resolve (j1, i1) for debug prints
    if j1 is None:
        j1 = int(_params.get('_j0', 0))
    if i1 is None:
        i1 = int(_params.get('_i0', 0))
    j1 = max(0, min(j1, ny - 1))
    i1 = max(0, min(i1, nx - 1))

    _fill_sn_slice_sorted_wrt_vdisp_kernel(
        arr,
        sn_ng_opt_slice_gparam_sorted_wrt_vdisp,
        max_ngauss,
        nparams_step,
        _params['g_sigma_lower'],
        _params['g_sigma_upper'],
    )

    if verbose:
        for i in range(max_ngauss):
            for j in range(max_ngauss):
                if j >= i:
                    print(i, j,
                          arr[nparams_step * j + 2 + 3 * i, j1, i1],
                          arr[nparams_step * j + 4 + 3 * i, j1, i1],
                          arr[nparams_step * (j + 1) - max_ngauss - 7 + j, j1, i1])
            print("")





import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True)
def _fill_sn_slice_sorted_wrt_vlos_kernel(arr, out_slice, max_ngauss, nparams_step, lower, upper):
    """
    Numba kernel:
      out[i, j, y, x] = num/den if (j>=i) and (den>0) and (lower<=sigma<upper), else 0
    arr: (P, ny, nx), out_slice: (max_ngauss, max_ngauss, ny, nx)
    """
    ny = arr.shape[1]
    nx = arr.shape[2]
    for i in prange(max_ngauss):
        for j in range(max_ngauss):
            if j < i:
                for y in range(ny):
                    for x in range(nx):
                        out_slice[i, j, y, x] = 0.0
                continue

            idx_denom = nparams_step * (j + 1) - max_ngauss - 7 + j
            idx_sigma = nparams_step * j + 3 + 3 * i
            idx_num   = nparams_step * j + 4 + 3 * i

            for y in range(ny):
                for x in range(nx):
                    den = arr[idx_denom, y, x]
                    sig = arr[idx_sigma, y, x]
                    if (den > 0.0) and (sig >= lower) and (sig < upper):
                        out_slice[i, j, y, x] = arr[idx_num, y, x] / den
                    else:
                        out_slice[i, j, y, x] = 0.0

def compute_sn_ng_opt_slice_sorted_wrt_vlos_numba(
    sn_ng_opt_slice_gparam_sorted_wrt_vlos: np.ndarray,
    _fitsarray_gfit_results2_sorted_wrt_vlos: np.ndarray,
    max_ngauss: int,
    nparams_step: int,
    _params: dict,
    verbose: bool = False,
    i1: int | None = None,   # x index to print
    j1: int | None = None,   # y index to print
):
    """
    Numba-accelerated version (same logic). Prints like the original when verbose=True.
    """
    arr = _fitsarray_gfit_results2_sorted_wrt_vlos
    ny, nx = arr.shape[1], arr.shape[2]

    # Resolve (j1, i1) for debug prints
    if j1 is None:
        j1 = int(_params.get('_j0', 0))
    if i1 is None:
        i1 = int(_params.get('_i0', 0))
    j1 = max(0, min(j1, ny - 1))
    i1 = max(0, min(i1, nx - 1))

    _fill_sn_slice_sorted_wrt_vlos_kernel(
        arr,
        sn_ng_opt_slice_gparam_sorted_wrt_vlos,
        max_ngauss,
        nparams_step,
        _params['g_sigma_lower'],
        _params['g_sigma_upper'],
    )

    if verbose:
        for i in range(max_ngauss):
            for j in range(max_ngauss):
                if j >= i:
                    print(i, j,
                          arr[nparams_step * j + 2 + 3 * i, j1, i1],
                          arr[nparams_step * j + 4 + 3 * i, j1, i1],
                          arr[nparams_step * (j + 1) - max_ngauss - 7 + j, j1, i1])
            print("")



import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True)
def _fill_sn_slice_sorted_wrt_integrated_int_kernel(arr, out_slice, max_ngauss, nparams_step, lower, upper):
    """
    Numba kernel:
      out[i, j, y, x] = num/den if (j>=i) and (den>0) and (lower<=sigma<upper), else 0
    arr: (P, ny, nx), out_slice: (max_ngauss, max_ngauss, ny, nx)
    """
    ny = arr.shape[1]
    nx = arr.shape[2]
    for i in prange(max_ngauss):
        for j in range(max_ngauss):
            if j < i:
                for y in range(ny):
                    for x in range(nx):
                        out_slice[i, j, y, x] = 0.0
                continue

            idx_denom = nparams_step * (j + 1) - max_ngauss - 7 + j
            idx_sigma = nparams_step * j + 3 + 3 * i
            idx_num   = nparams_step * j + 4 + 3 * i

            for y in range(ny):
                for x in range(nx):
                    den = arr[idx_denom, y, x]
                    sig = arr[idx_sigma, y, x]
                    if (den > 0.0) and (sig >= lower) and (sig < upper):
                        out_slice[i, j, y, x] = arr[idx_num, y, x] / den
                    else:
                        out_slice[i, j, y, x] = 0.0

def compute_sn_ng_opt_slice_sorted_wrt_integrated_int_numba(
    sn_ng_opt_slice_gparam_sorted_wrt_integrated_int: np.ndarray,
    _fitsarray_gfit_results2_sorted_wrt_integrated_int: np.ndarray,
    max_ngauss: int,
    nparams_step: int,
    _params: dict,
    verbose: bool = False,
    i1: int | None = None,   # x index to print
    j1: int | None = None,   # y index to print
):
    """
    Numba-accelerated version (same logic). Prints like the original when verbose=True.
    """
    arr = _fitsarray_gfit_results2_sorted_wrt_integrated_int
    ny, nx = arr.shape[1], arr.shape[2]

    # Resolve (j1, i1) for debug prints
    if j1 is None:
        j1 = int(_params.get('_j0', 0))
    if i1 is None:
        i1 = int(_params.get('_i0', 0))
    j1 = max(0, min(j1, ny - 1))
    i1 = max(0, min(i1, nx - 1))

    _fill_sn_slice_sorted_wrt_integrated_int_kernel(
        arr,
        sn_ng_opt_slice_gparam_sorted_wrt_integrated_int,
        max_ngauss,
        nparams_step,
        _params['g_sigma_lower'],
        _params['g_sigma_upper'],
    )

    if verbose:
        for i in range(max_ngauss):
            for j in range(max_ngauss):
                if j >= i:
                    print(i, j,
                          arr[nparams_step * j + 2 + 3 * i, j1, i1],
                          arr[nparams_step * j + 4 + 3 * i, j1, i1],
                          arr[nparams_step * (j + 1) - max_ngauss - 7 + j, j1, i1])
            print("")




def main():
    np.seterr(divide='ignore', invalid='ignore')

    _time_start = datetime.now()

    if len(sys.argv) == 3:
        configfile = sys.argv[1]
        _params=read_configfile(configfile)
        _classified_index = int(sys.argv[2])

        print("")
        print(" ____________________________________________")
        print("[____________________________________________]")
        print("")
        print(" :: Running baygaud_classify.py with %s ::" % configfile)
        print("")
        print("")

        _dir_baygaud_combined = _params['wdir'] + '/' + _params['_combdir'] + ".%d" % _classified_index
        if os.path.exists(_dir_baygaud_combined):
            print("")
            print(" ____________________________________________")
            print("[____________________________________________]")
            print("")
            print(" %s directory already exists." % _dir_baygaud_combined)
            print(" Try to use another output-index...")
            print("")
            print("")
            sys.exit()

    else: 
        print("")
        print(" ____________________________________________")
        print("[____________________________________________]")
        print("")
        print(" :: baygaud_classify.py usage ::")
        print("")
        print(" Usage: Running baygaud_classify.py with baygaud_params.yaml and output-index")
        print(" > python3 baygaud_classify.py [ARG1: _baygaud_params.yaml] [ARG2: output-index, 1, 2, ...]")
        print(" e.g.,")
        print(" > python3 baygaud_classify.py _baygaud_params.ngc2403.yaml 1")
        print("")
        sys.exit()


    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> read fits header ...]")
    print("")
    print("")
    with fits.open(_params['wdir'] + '/' + _params['input_datacube'], 'update') as hdu:
        naxis1 = hdu[0].header['NAXIS1']
        naxis2 = hdu[0].header['NAXIS2']
        naxis3 = hdu[0].header['NAXIS3']
        cdelt3 = abs(hdu[0].header['CDELT3'])

    max_ngauss = _params['max_ngauss']
    outputdir_segs = _params['wdir'] + '/' + _params['_segdir']

    cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s) # in km/s

    _x = np.linspace(0, 1, naxis3, dtype=np.float32)
    _vel_min = cube.spectral_axis.min().value
    _vel_max = cube.spectral_axis.max().value
    _params['vel_min'] = _vel_min   
    _params['vel_max'] = _vel_max


    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> load baygaud-segs from the output dir ...]")
    print("")
    print("")
    _list_segs_bf = [_file for _file in os.listdir(outputdir_segs) if _file.startswith("G%02d" % _params['max_ngauss'])]

    _list_segs_bf.sort(key = lambda x: x.split('.x')[1], reverse=False) # reverse with x pixels
    

    print(_list_segs_bf)
    print()
    print()

    nparams = 2*(3*max_ngauss+2) + max_ngauss + 7
    gfit_results = np.full((naxis1, naxis2, max_ngauss, nparams), fill_value=-1E9, dtype=float)
    
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> load gfit_results from the baygaud-segs output ...]")
    print("")
    print("")
    for _segs_bf in _list_segs_bf:

        _slab_t = np.load('%s/%s' % (outputdir_segs, _segs_bf))
        nx = _slab_t.shape[0]
        ny = _slab_t.shape[1]
        x0 = int(_slab_t[0, 0, 0, nparams-2])
        y0 = int(_slab_t[0, 0, 0, nparams-1])
        x1 = int(_slab_t[0, nx-1, 0, nparams-2]) + 1
        y1 = int(_slab_t[0, ny-1, 0, nparams-1]) + 1

        gfit_results[x0:x1, y0:y1, :, :] = _slab_t
        

    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> make baygaud-classified output directories ...]")
    print("")
    print("")
    make_dirs("%s/%s.%d" % (_params['wdir'], _params['_combdir'], _classified_index))
    _dir_baygaud_combined = _params['wdir'] + '/' + _params['_combdir'] + ".%d" % _classified_index

    if len(sys.argv) == 3:
        shutil.copyfile(configfile, "%s/%s" % (_dir_baygaud_combined, configfile))
    else:
        print("_baygaud.yaml file is not present...")


    make_dirs("%s/sgfit" % _dir_baygaud_combined)
    make_dirs("%s/psgfit" % _dir_baygaud_combined)
    make_dirs("%s/ngfit" % _dir_baygaud_combined)

    if _params['_cool_extraction'] == 'Y':
        make_dirs("%s/cool" % _dir_baygaud_combined)

    if _params['_warm_extraction'] == 'Y':
        make_dirs("%s/warm" % _dir_baygaud_combined)

    if _params['_hot_extraction'] == 'Y':
        make_dirs("%s/hot" % _dir_baygaud_combined)

    if _params['_hvc_extraction'] == 'Y':
        make_dirs("%s/hvc" % _dir_baygaud_combined)


    if _params['_sort_gauss_wrt_integrated_int'] == 'Y':
        make_dirs("%s/ngfit_wrt_integrated_int" % _dir_baygaud_combined)

    if _params['_sort_gauss_wrt_vdisp'] == 'Y':
        make_dirs("%s/ngfit_wrt_vdisp" % _dir_baygaud_combined)

    if _params['_sort_gauss_wrt_peak_amp'] == 'Y':
        make_dirs("%s/ngfit_wrt_peak_amp" % _dir_baygaud_combined)

    if _params['_sort_gauss_wrt_vlos'] == 'Y':
        make_dirs("%s/ngfit_wrt_vlos" % _dir_baygaud_combined)


    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> declare _fitsarray_gfit_results1 : 4d numpy array ...]")
    print("")
    print("")
    _fitsarray_gfit_results1 = np.transpose(gfit_results, axes=[2, 3, 1, 0]) # 4d array

    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> declare _fitsarray_gfit_results2 : 3d numpy array ...]")
    print("")
    print("")

    _fitsarray_gfit_results2 = np.concatenate(_fitsarray_gfit_results1 , axis=0)

    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> sort _fitsarray_gfit_results2 w.r.t. 'Int', 'VDISP, 'VLOS' and 'Peak_amp' ...]")
    print("")
    print("")
    _fitsarray_gfit_results2_sorted_wrt_amp = sort_gaussians_wrt_gparam_numba(_fitsarray_gfit_results2, max_ngauss, '_peak_amp')
    _fitsarray_gfit_results2_sorted_wrt_vdisp = sort_gaussians_wrt_gparam_numba(_fitsarray_gfit_results2, max_ngauss, '_vdisp')
    _fitsarray_gfit_results2_sorted_wrt_vlos = sort_gaussians_wrt_gparam_numba(_fitsarray_gfit_results2, max_ngauss, '_vlos')
    _fitsarray_gfit_results2_sorted_wrt_integrated_int = sort_gaussians_wrt_gparam_numba(_fitsarray_gfit_results2, max_ngauss, '_integrated_int')



    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> declare bevidences : 3d numpy array ...]")
    print("")
    print("")
    bevidences = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bevidences_gparam_sorted_wrt_amp = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_amp.shape[1], _fitsarray_gfit_results2_sorted_wrt_amp.shape[2]), dtype=float)
    bevidences_gparam_sorted_wrt_vdisp = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_vdisp.shape[1], _fitsarray_gfit_results2_sorted_wrt_vdisp.shape[2]), dtype=float)
    bevidences_gparam_sorted_wrt_vlos = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_vlos.shape[1], _fitsarray_gfit_results2_sorted_wrt_vlos.shape[2]), dtype=float)
    bevidences_gparam_sorted_wrt_integrated_int = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_integrated_int.shape[1], _fitsarray_gfit_results2_sorted_wrt_integrated_int.shape[2]), dtype=float)


    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> declare g_num_sort : 3d numpy array ...]")
    print("")
    print("")
    g_num_sort = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_num_sort_gparam_sorted_wrt_amp = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_amp.shape[1], _fitsarray_gfit_results2_sorted_wrt_amp.shape[2]), dtype=float)
    g_num_sort_gparam_sorted_wrt_vdisp = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_vdisp.shape[1], _fitsarray_gfit_results2_sorted_wrt_vdisp.shape[2]), dtype=float)
    g_num_sort_gparam_sorted_wrt_vlos = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_vlos.shape[1], _fitsarray_gfit_results2_sorted_wrt_vlos.shape[2]), dtype=float)
    g_num_sort_gparam_sorted_wrt_integrated_int = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_integrated_int.shape[1], _fitsarray_gfit_results2_sorted_wrt_integrated_int.shape[2]), dtype=float)




    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> declare numpy arrays for S/N slices: 3d numpy arrays ...]")
    print("")
    print("")
    peak_sn_pass_for_ng_opt = _params['peak_sn_pass_for_ng_opt']
    nparams_step = 2*(3*max_ngauss+2) + (max_ngauss + 7)

    sn_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    #sn_ng_opt = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_pass_ng_opt = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    #sn_pass_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    #x_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    #x_ng_opt = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    sn_ng_opt_slice_gparam_sorted_wrt_amp = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2_sorted_wrt_amp.shape[1], _fitsarray_gfit_results2_sorted_wrt_amp.shape[2]), dtype=float)
    #sn_ng_opt_gparam_sorted_wrt_amp = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_amp.shape[1], _fitsarray_gfit_results2_sorted_wrt_amp.shape[2]), dtype=float)
    sn_pass_ng_opt_gparam_sorted_wrt_amp = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_amp.shape[1], _fitsarray_gfit_results2_sorted_wrt_amp.shape[2]), dtype=float)
    #sn_pass_ng_opt_t_gparam_sorted_wrt_amp = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_amp.shape[1], _fitsarray_gfit_results2_sorted_wrt_amp.shape[2]), dtype=float)
    #x_ng_opt_slice_gparam_sorted_wrt_amp = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2_sorted_wrt_amp.shape[1], _fitsarray_gfit_results2_sorted_wrt_amp.shape[2]), dtype=float)
    #x_ng_opt_gparam_sorted_wrt_amp = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_amp.shape[1], _fitsarray_gfit_results2_sorted_wrt_amp.shape[2]), dtype=float)

    sn_ng_opt_slice_gparam_sorted_wrt_vdisp = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2_sorted_wrt_vdisp.shape[1], _fitsarray_gfit_results2_sorted_wrt_vdisp.shape[2]), dtype=float)
    #sn_ng_opt_gparam_sorted_wrt_vdisp = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_vdisp.shape[1], _fitsarray_gfit_results2_sorted_wrt_vdisp.shape[2]), dtype=float)
    sn_pass_ng_opt_gparam_sorted_wrt_vdisp = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_vdisp.shape[1], _fitsarray_gfit_results2_sorted_wrt_vdisp.shape[2]), dtype=float)
    #sn_pass_ng_opt_t_gparam_sorted_wrt_vdisp = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_vdisp.shape[1], _fitsarray_gfit_results2_sorted_wrt_vdisp.shape[2]), dtype=float)
    #x_ng_opt_slice_gparam_sorted_wrt_vdisp = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2_sorted_wrt_vdisp.shape[1], _fitsarray_gfit_results2_sorted_wrt_vdisp.shape[2]), dtype=float)
    #x_ng_opt_gparam_sorted_wrt_vdisp = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_vdisp.shape[1], _fitsarray_gfit_results2_sorted_wrt_vdisp.shape[2]), dtype=float)

    sn_ng_opt_slice_gparam_sorted_wrt_vlos = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2_sorted_wrt_vlos.shape[1], _fitsarray_gfit_results2_sorted_wrt_vlos.shape[2]), dtype=float)
    #sn_ng_opt_gparam_sorted_wrt_vlos = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_vlos.shape[1], _fitsarray_gfit_results2_sorted_wrt_vlos.shape[2]), dtype=float)
    sn_pass_ng_opt_gparam_sorted_wrt_vlos = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_vlos.shape[1], _fitsarray_gfit_results2_sorted_wrt_vlos.shape[2]), dtype=float)
    #sn_pass_ng_opt_t_gparam_sorted_wrt_vlos = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_vlos.shape[1], _fitsarray_gfit_results2_sorted_wrt_vlos.shape[2]), dtype=float)
    #x_ng_opt_slice_gparam_sorted_wrt_vlos = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2_sorted_wrt_vlos.shape[1], _fitsarray_gfit_results2_sorted_wrt_vlos.shape[2]), dtype=float)
    #x_ng_opt_gparam_sorted_wrt_vlos = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_vlos.shape[1], _fitsarray_gfit_results2_sorted_wrt_vlos.shape[2]), dtype=float)

    sn_ng_opt_slice_gparam_sorted_wrt_integrated_int = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2_sorted_wrt_integrated_int.shape[1], _fitsarray_gfit_results2_sorted_wrt_integrated_int.shape[2]), dtype=float)
    #sn_ng_opt_gparam_sorted_wrt_integrated_int = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_integrated_int.shape[1], _fitsarray_gfit_results2_sorted_wrt_integrated_int.shape[2]), dtype=float)
    sn_pass_ng_opt_gparam_sorted_wrt_integrated_int = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_integrated_int.shape[1], _fitsarray_gfit_results2_sorted_wrt_integrated_int.shape[2]), dtype=float)
    #sn_pass_ng_opt_t_gparam_sorted_wrt_integrated_int = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_integrated_int.shape[1], _fitsarray_gfit_results2_sorted_wrt_integrated_int.shape[2]), dtype=float)
    #x_ng_opt_slice_gparam_sorted_wrt_integrated_int = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2_sorted_wrt_integrated_int.shape[1], _fitsarray_gfit_results2_sorted_wrt_integrated_int.shape[2]), dtype=float)
    #x_ng_opt_gparam_sorted_wrt_integrated_int = np.zeros((max_ngauss, _fitsarray_gfit_results2_sorted_wrt_integrated_int.shape[1], _fitsarray_gfit_results2_sorted_wrt_integrated_int.shape[2]), dtype=float)


    i1 = _params['_i0']
    j1 = _params['_j0']
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> check _fitsarray_gfit_results2 for (x:%d, y:%d)...]" % (i1, j1))
    print("")
    print("")

    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> extract sn_ng_opt_slice from _fitsarray_Gfit_results2 array[params, y, x] ...]")
    print("")
    print("")

    # -----------------------------------------
    #for i in range(0, max_ngauss):
    #    for j in range(0, max_ngauss):
    #        sn_ng_opt_slice[i, j, :, :] = np.array([np.where( \
    #                                                (j >= i) & \
    #                                                (_fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :] > 0) & \
    #                                                (_fitsarray_gfit_results2[nparams_step*j + 3 + 3*i, :, :] >= _params['g_sigma_lower']) & \
    #                                                (_fitsarray_gfit_results2[nparams_step*j + 3 + 3*i, :, :] < _params['g_sigma_upper']), \
    #                                                _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0]
    #
    #        if j >= i:
    #            print(i, j, _fitsarray_gfit_results2[nparams_step*j + 2 + 3*i, j1, i1], _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, j1, i1], _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, j1, i1])
    #    print("")
    # -----------------------------------------
    compute_sn_ng_opt_slice_numba(sn_ng_opt_slice, _fitsarray_gfit_results2,
                              max_ngauss, nparams_step, _params,
                              verbose=False)   # or True



    # -----------------------------------------
    #for i in range(0, max_ngauss):
    #    for j in range(0, max_ngauss):
    #        sn_ng_opt_slice_gparam_sorted_wrt_amp[i, j, :, :] = np.array([np.where( \
    #                                                (j >= i) & \
    #                                                (_fitsarray_gfit_results2_sorted_wrt_amp[nparams_step*(j+1)-max_ngauss-7+j, :, :] > 0) & \
    #                                                (_fitsarray_gfit_results2_sorted_wrt_amp[nparams_step*j + 3 + 3*i, :, :] >= _params['g_sigma_lower']) & \
    #                                                (_fitsarray_gfit_results2_sorted_wrt_amp[nparams_step*j + 3 + 3*i, :, :] < _params['g_sigma_upper']), \
    #                                                _fitsarray_gfit_results2_sorted_wrt_amp[nparams_step*j + 4 + 3*i, :, :] / _fitsarray_gfit_results2_sorted_wrt_amp[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0]
    #
    #        if j >= i:
    #            print(i, j, _fitsarray_gfit_results2_sorted_wrt_amp[nparams_step*j + 2 + 3*i, j1, i1], _fitsarray_gfit_results2_sorted_wrt_amp[nparams_step*j + 4 + 3*i, j1, i1], _fitsarray_gfit_results2_sorted_wrt_amp[nparams_step*(j+1)-max_ngauss-7+j, j1, i1])
    #    print("")
    # -----------------------------------------
    compute_sn_ng_opt_slice_sorted_wrt_amp_numba(
        sn_ng_opt_slice_gparam_sorted_wrt_amp,
        _fitsarray_gfit_results2_sorted_wrt_amp,
        max_ngauss,
        nparams_step,
        _params,
        False, # print check pixels (j1, i1)
        i1,   # x index to print
        j1   # y index to print
    )
        



    # -----------------------------------------
    #for i in range(0, max_ngauss):
    #    for j in range(0, max_ngauss):
    #        sn_ng_opt_slice_gparam_sorted_wrt_vdisp[i, j, :, :] = np.array([np.where( \
    #                                                (j >= i) & \
    #                                                (_fitsarray_gfit_results2_sorted_wrt_vdisp[nparams_step*(j+1)-max_ngauss-7+j, :, :] > 0) & \
    #                                                (_fitsarray_gfit_results2_sorted_wrt_vdisp[nparams_step*j + 3 + 3*i, :, :] >= _params['g_sigma_lower']) & \
    #                                                (_fitsarray_gfit_results2_sorted_wrt_vdisp[nparams_step*j + 3 + 3*i, :, :] < _params['g_sigma_upper']), \
    #                                                _fitsarray_gfit_results2_sorted_wrt_vdisp[nparams_step*j + 4 + 3*i, :, :] / _fitsarray_gfit_results2_sorted_wrt_vdisp[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0]
    #
    #        if j >= i:
    #            print(i, j, _fitsarray_gfit_results2_sorted_wrt_vdisp[nparams_step*j + 2 + 3*i, j1, i1], _fitsarray_gfit_results2_sorted_wrt_vdisp[nparams_step*j + 4 + 3*i, j1, i1], _fitsarray_gfit_results2_sorted_wrt_vdisp[nparams_step*(j+1)-max_ngauss-7+j, j1, i1])
    #    print("")
    # -----------------------------------------
    compute_sn_ng_opt_slice_sorted_wrt_vdisp_numba(
        sn_ng_opt_slice_gparam_sorted_wrt_vdisp,
        _fitsarray_gfit_results2_sorted_wrt_vdisp,
        max_ngauss,
        nparams_step,
        _params,
        False, # print check pixels (j1, i1)
        i1,   # x index to print
        j1,   # y index to print
    )




    # -----------------------------------------
    #for i in range(0, max_ngauss):
    #    for j in range(0, max_ngauss):
    #        sn_ng_opt_slice_gparam_sorted_wrt_vlos[i, j, :, :] = np.array([np.where( \
    #                                                (j >= i) & \
    #                                                (_fitsarray_gfit_results2_sorted_wrt_vlos[nparams_step*(j+1)-max_ngauss-7+j, :, :] > 0) & \
    #                                                (_fitsarray_gfit_results2_sorted_wrt_vlos[nparams_step*j + 3 + 3*i, :, :] >= _params['g_sigma_lower']) & \
    #                                                (_fitsarray_gfit_results2_sorted_wrt_vlos[nparams_step*j + 3 + 3*i, :, :] < _params['g_sigma_upper']), \
    #                                                _fitsarray_gfit_results2_sorted_wrt_vlos[nparams_step*j + 4 + 3*i, :, :] / _fitsarray_gfit_results2_sorted_wrt_vlos[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0]
    #
    #        if j >= i:
    #            print(i, j, _fitsarray_gfit_results2_sorted_wrt_vlos[nparams_step*j + 2 + 3*i, j1, i1], _fitsarray_gfit_results2_sorted_wrt_vlos[nparams_step*j + 4 + 3*i, j1, i1], _fitsarray_gfit_results2_sorted_wrt_vlos[nparams_step*(j+1)-max_ngauss-7+j, j1, i1])
    #    print("")
    # -----------------------------------------
    compute_sn_ng_opt_slice_sorted_wrt_vlos_numba(
        sn_ng_opt_slice_gparam_sorted_wrt_vlos,
        _fitsarray_gfit_results2_sorted_wrt_vlos,
        max_ngauss,
        nparams_step,
        _params,
        False, # print check pixels (j1, i1)
        i1,   # x index to print
        j1,   # y index to print
    )


    # -----------------------------------------
    #for i in range(0, max_ngauss):
    #    for j in range(0, max_ngauss):
    #        sn_ng_opt_slice_gparam_sorted_wrt_integrated_int[i, j, :, :] = np.array([np.where( \
    #                                                (j >= i) & \
    #                                                (_fitsarray_gfit_results2_sorted_wrt_integrated_int[nparams_step*(j+1)-max_ngauss-7+j, :, :] > 0) & \
    #                                                (_fitsarray_gfit_results2_sorted_wrt_integrated_int[nparams_step*j + 3 + 3*i, :, :] >= _params['g_sigma_lower']) & \
    #                                                (_fitsarray_gfit_results2_sorted_wrt_integrated_int[nparams_step*j + 3 + 3*i, :, :] < _params['g_sigma_upper']), \
    #                                                _fitsarray_gfit_results2_sorted_wrt_integrated_int[nparams_step*j + 4 + 3*i, :, :] / _fitsarray_gfit_results2_sorted_wrt_integrated_int[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0]
    #
    #        if j >= i:
    #            print(i, j, _fitsarray_gfit_results2_sorted_wrt_integrated_int[nparams_step*j + 2 + 3*i, j1, i1], _fitsarray_gfit_results2_sorted_wrt_integrated_int[nparams_step*j + 4 + 3*i, j1, i1], _fitsarray_gfit_results2_sorted_wrt_integrated_int[nparams_step*(j+1)-max_ngauss-7+j, j1, i1])
    #    print("")
    # -----------------------------------------
    compute_sn_ng_opt_slice_sorted_wrt_integrated_int_numba(
        sn_ng_opt_slice_gparam_sorted_wrt_integrated_int,
        _fitsarray_gfit_results2_sorted_wrt_integrated_int,
        max_ngauss,
        nparams_step,
        _params,
        False, # print check pixels (j1, i1)
        i1,   # x index to print
        j1,   # y index to print
    )

    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> check sn_ng_opt_slice ...]")
    print("")
    print("")


    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> check sn_pass_ng_opt from sn_ng_opt_slice ...]")
    print("")
    print("")
    for i in range(0, max_ngauss):
        for j in range(0, max_ngauss):
            sn_pass_ng_opt[i, :, :] += np.array([np.where( \
                                            (sn_ng_opt_slice[j, i, :, :] > peak_sn_pass_for_ng_opt), \
                                            1, 0)][0])

    for i in range(0, max_ngauss):
        for j in range(0, max_ngauss):
            sn_pass_ng_opt_gparam_sorted_wrt_amp[i, :, :] += np.array([np.where( \
                                            (sn_ng_opt_slice_gparam_sorted_wrt_amp[j, i, :, :] > peak_sn_pass_for_ng_opt), \
                                            1, 0)][0])
            sn_pass_ng_opt_gparam_sorted_wrt_vdisp[i, :, :] += np.array([np.where( \
                                            (sn_ng_opt_slice_gparam_sorted_wrt_vdisp[j, i, :, :] > peak_sn_pass_for_ng_opt), \
                                            1, 0)][0])
            sn_pass_ng_opt_gparam_sorted_wrt_vlos[i, :, :] += np.array([np.where( \
                                            (sn_ng_opt_slice_gparam_sorted_wrt_vlos[j, i, :, :] > peak_sn_pass_for_ng_opt), \
                                            1, 0)][0])
            sn_pass_ng_opt_gparam_sorted_wrt_integrated_int[i, :, :] += np.array([np.where( \
                                            (sn_ng_opt_slice_gparam_sorted_wrt_integrated_int[j, i, :, :] > peak_sn_pass_for_ng_opt), \
                                            1, 0)][0])



    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> extract log-Z from _fitsarray_gfit_results2 array ...]")
    print("[--> to bevidences array ...]")
    print("")
    print("")
    nparams_step = 2*(3*max_ngauss+2) + max_ngauss + 7
    for i in range(max_ngauss):
        bevidences[i, :, :] = _fitsarray_gfit_results2[nparams_step*(i+1)-7, :, :] # corresponing log-Z

    for i in range(max_ngauss):
        bevidences_gparam_sorted_wrt_amp[i, :, :] = _fitsarray_gfit_results2_sorted_wrt_amp[nparams_step*(i+1)-7, :, :] # corresponing log-Z
        bevidences_gparam_sorted_wrt_vdisp[i, :, :] = _fitsarray_gfit_results2_sorted_wrt_vdisp[nparams_step*(i+1)-7, :, :] # corresponing log-Z
        bevidences_gparam_sorted_wrt_vlos[i, :, :] = _fitsarray_gfit_results2_sorted_wrt_vlos[nparams_step*(i+1)-7, :, :] # corresponing log-Z
        bevidences_gparam_sorted_wrt_integrated_int[i, :, :] = _fitsarray_gfit_results2_sorted_wrt_integrated_int[nparams_step*(i+1)-7, :, :] # corresponing log-Z

    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> sort the coupled (max_ngauss, log-Z) with log-Z ...]")
    print("[--> in descending order : max(log-Z) first ...]")
    print("")
    print("")
    g_num_sort = bevidences.argsort(axis=0)[::-1] # descening order : arg
    bevidences_sort = np.sort(bevidences, axis=0)[::-1] # descening order : log-Z

    g_num_sort_gparam_sorted_wrt_amp = bevidences_gparam_sorted_wrt_amp.argsort(axis=0)[::-1] # descening order : arg
    bevidences_sort_gparam_sorted_wrt_amp = np.sort(bevidences_gparam_sorted_wrt_amp, axis=0)[::-1] # descening order : log-Z
    g_num_sort_gparam_sorted_wrt_vdisp = bevidences_gparam_sorted_wrt_vdisp.argsort(axis=0)[::-1] # descening order : arg
    bevidences_sort_gparam_sorted_wrt_vdisp = np.sort(bevidences_gparam_sorted_wrt_vdisp, axis=0)[::-1] # descening order : log-Z
    g_num_sort_gparam_sorted_wrt_vlos = bevidences_gparam_sorted_wrt_vlos.argsort(axis=0)[::-1] # descening order : arg
    bevidences_sort_gparam_sorted_wrt_vlos = np.sort(bevidences_gparam_sorted_wrt_vlos, axis=0)[::-1] # descening order : log-Z
    g_num_sort_gparam_sorted_wrt_integrated_int = bevidences_gparam_sorted_wrt_integrated_int.argsort(axis=0)[::-1] # descening order : arg
    bevidences_sort_gparam_sorted_wrt_integrated_int = np.sort(bevidences_gparam_sorted_wrt_integrated_int, axis=0)[::-1] # descening order : log-Z




    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> derive the optimal number of Gaussian components ...]")
    print("[--> given the sn_pass + bayes factor limit ...]")
    print("[--> opt_ngmap_gmax_ng array : optimal n-gauss array ...]")
    print("[--> max_ngauss: %d ...]" % max_ngauss)
    print("")
    print("")
    bf_limit = int(_params['bayes_factor_limit'])

    if max_ngauss == 1:
        print(bf_limit)
        print(g_num_sort)
        print(bevidences_sort)
        opt_ngmap_gmax_ng = g1_opt_bf(_fitsarray_gfit_results2)
        print() 
        print(bf_limit)
        print(g_num_sort_gparam_sorted_wrt_amp)
        print(bevidences_sort_gparam_sorted_wrt_amp)
        opt_ngmap_gmax_ng_gparam_sorted_wrt_amp = g1_opt_bf(_fitsarray_gfit_results2_sorted_wrt_amp)
        print() 
        print(g_num_sort_gparam_sorted_wrt_vdisp)
        print(bevidences_sort_gparam_sorted_wrt_vdisp)
        opt_ngmap_gmax_ng_gparam_sorted_wrt_vdisp = g1_opt_bf(_fitsarray_gfit_results2_sorted_wrt_vdisp)
        print() 
        print(g_num_sort_gparam_sorted_wrt_vlos)
        print(bevidences_sort_gparam_sorted_wrt_vlos)
        opt_ngmap_gmax_ng_gparam_sorted_wrt_vlos = g1_opt_bf(_fitsarray_gfit_results2_sorted_wrt_vlos)
        print() 
        print(g_num_sort_gparam_sorted_wrt_integrated_int)
        print(bevidences_sort_gparam_sorted_wrt_integrated_int)
        opt_ngmap_gmax_ng_gparam_sorted_wrt_integrated_int = g1_opt_bf(_fitsarray_gfit_results2_sorted_wrt_integrated_int)
        print() 


    elif max_ngauss > 1:
        gx_list = [] # [0, 1, 2, ...] == [1, 2, 3, ...]
        for i in range(0, max_ngauss):
            gx_list.append(i)
        opt_ngmap_gmax_ng = find_gx_opt_bf_snp(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, sn_pass_ng_opt, gx_list)

        gx_list_gparam_sorted_wrt_amp = [] # [0, 1, 2, ...] == [1, 2, 3, ...]
        for i in range(0, max_ngauss):
            gx_list_gparam_sorted_wrt_amp.append(i)
        opt_ngmap_gmax_ng_gparam_sorted_wrt_amp = find_gx_opt_bf_snp(_fitsarray_gfit_results2_sorted_wrt_amp, bevidences_sort_gparam_sorted_wrt_amp, g_num_sort_gparam_sorted_wrt_amp, bf_limit, max_ngauss, sn_pass_ng_opt_gparam_sorted_wrt_amp, gx_list_gparam_sorted_wrt_amp)
        gx_list_gparam_sorted_wrt_vdisp = [] # [0, 1, 2, ...] == [1, 2, 3, ...]
        for i in range(0, max_ngauss):
            gx_list_gparam_sorted_wrt_vdisp.append(i)
        opt_ngmap_gmax_ng_gparam_sorted_wrt_vdisp = find_gx_opt_bf_snp(_fitsarray_gfit_results2_sorted_wrt_vdisp, bevidences_sort_gparam_sorted_wrt_vdisp, g_num_sort_gparam_sorted_wrt_vdisp, bf_limit, max_ngauss, sn_pass_ng_opt_gparam_sorted_wrt_vdisp, gx_list_gparam_sorted_wrt_vdisp)
        gx_list_gparam_sorted_wrt_vlos = [] # [0, 1, 2, ...] == [1, 2, 3, ...]
        for i in range(0, max_ngauss):
            gx_list_gparam_sorted_wrt_vlos.append(i)
        opt_ngmap_gmax_ng_gparam_sorted_wrt_vlos = find_gx_opt_bf_snp(_fitsarray_gfit_results2_sorted_wrt_vlos, bevidences_sort_gparam_sorted_wrt_vlos, g_num_sort_gparam_sorted_wrt_vlos, bf_limit, max_ngauss, sn_pass_ng_opt_gparam_sorted_wrt_vlos, gx_list_gparam_sorted_wrt_vlos)
        gx_list_gparam_sorted_wrt_integrated_int = [] # [0, 1, 2, ...] == [1, 2, 3, ...]
        for i in range(0, max_ngauss):
            gx_list_gparam_sorted_wrt_integrated_int.append(i)
        opt_ngmap_gmax_ng_gparam_sorted_wrt_integrated_int = find_gx_opt_bf_snp(_fitsarray_gfit_results2_sorted_wrt_integrated_int, bevidences_sort_gparam_sorted_wrt_integrated_int, g_num_sort_gparam_sorted_wrt_integrated_int, bf_limit, max_ngauss, sn_pass_ng_opt_gparam_sorted_wrt_integrated_int, gx_list_gparam_sorted_wrt_integrated_int)



# ---- CHECK PROFILES
#    print(" ____________________________________________")
#    print("[____________________________________________]")
#    print("[--> (%d, %d) -- optimal ng: %d ...]" % (i1, j1, opt_ngmap_gmax_ng[j1, i1]))
#    print("[--> (%d, %d) -- optimal ng: %d ...]" % (i1, j1, opt_ngmap_gmax_ng_gparam_sorted_wrt_amp[j1, i1]))
#    print("[--> (%d, %d) -- optimal ng: %d ...]" % (i1, j1, opt_ngmap_gmax_ng_gparam_sorted_wrt_vdisp[j1, i1]))
#    print("[--> (%d, %d) -- optimal ng: %d ...]" % (i1, j1, opt_ngmap_gmax_ng_gparam_sorted_wrt_vlos[j1, i1]))
#    print("[--> (%d, %d) -- optimal ng: %d ...]" % (i1, j1, opt_ngmap_gmax_ng_gparam_sorted_wrt_integrated_int[j1, i1]))
#    print("")
#    print("")
#
#    i1 = _params['_i0']
#    j1 = _params['_j0']



    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> extract :: single Gaussian component:: given the optimal n-gauss map ...]")
    print("")
    print("")
    extract_maps(_fitsarray_gfit_results2, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='sgfit', ng_opt=opt_ngmap_gmax_ng, _hdu=hdu)

    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> extract :: perfect single Gaussian component:: given the optimal n-gauss map ...]")
    print("")
    print("")
    extract_maps(_fitsarray_gfit_results2, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='psgfit', ng_opt=opt_ngmap_gmax_ng, _hdu=hdu)

    if _params['_cool_extraction'] == 'Y':
        print(" ____________________________________________")
        print("[____________________________________________]")
        print("[--> extract :: kinematically cool Gaussian component:: given the optimal n-gauss map ...]")
        print("")
        print("")
        extract_maps(_fitsarray_gfit_results2, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='cool', ng_opt=opt_ngmap_gmax_ng, _hdu=hdu)

    if _params['_warm_extraction'] == 'Y':
        print(" ____________________________________________")
        print("[____________________________________________]")
        print("[--> extract :: kinematically warm Gaussian component:: given the optimal n-gauss map ...]")
        print("")
        print("")
        extract_maps(_fitsarray_gfit_results2, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='warm', ng_opt=opt_ngmap_gmax_ng, _hdu=hdu)


    if _params['_hot_extraction'] == 'Y':
        print(" ____________________________________________")
        print("[____________________________________________]")
        print("[--> extract :: kinematically hot Gaussian component:: given the optimal n-gauss map ...]")
        print("")
        print("")
        extract_maps(_fitsarray_gfit_results2, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='hot', ng_opt=opt_ngmap_gmax_ng, _hdu=hdu)

    if _params['_hvc_extraction'] == 'Y':
        print(" ____________________________________________")
        print("[____________________________________________]")
        print("[--> extract :: hvc Gaussian component:: given the optimal n-gauss map ...]")
        print("")
        print("")
        extract_maps(_fitsarray_gfit_results2, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='hvc', ng_opt=opt_ngmap_gmax_ng, _hdu=hdu)

    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> extract :: all the Gaussian components:: given max_ngauss ...]")
    print("")
    print("")
    extract_maps(_fitsarray_gfit_results2, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='ngfit', ng_opt=opt_ngmap_gmax_ng, _hdu=hdu)


    if _params['_sort_gauss_wrt_peak_amp'] == 'Y':
        print(" ____________________________________________")
        print("[____________________________________________]")
        print("[--> extract :: sort all the Gaussian components :: w.r.t. peak-flux :: given max_ngauss ...]")
        print("")
        print("")
        extract_maps(_fitsarray_gfit_results2_sorted_wrt_amp, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='ngfit_wrt_peak_amp', ng_opt=opt_ngmap_gmax_ng_gparam_sorted_wrt_amp, _hdu=hdu)

    if _params['_sort_gauss_wrt_vdisp'] == 'Y':
        print(" ____________________________________________")
        print("[____________________________________________]")
        print("[--> extract :: sort all the Gaussian components :: w.r.t. vdisp :: given max_ngauss ...]")
        print("")
        print("")
        extract_maps(_fitsarray_gfit_results2_sorted_wrt_vdisp, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='ngfit_wrt_vdisp', ng_opt=opt_ngmap_gmax_ng_gparam_sorted_wrt_vdisp, _hdu=hdu)

    if _params['_sort_gauss_wrt_vlos'] == 'Y':
        print(" ____________________________________________")
        print("[____________________________________________]")
        print("[--> extract :: sort all the Gaussian components :: w.r.t. vlos :: given max_ngauss ...]")
        print("")
        print("")
        extract_maps(_fitsarray_gfit_results2_sorted_wrt_vlos, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='ngfit_wrt_vlos', ng_opt=opt_ngmap_gmax_ng_gparam_sorted_wrt_vlos, _hdu=hdu)

    if _params['_sort_gauss_wrt_integrated_int'] == 'Y':
        print(" ____________________________________________")
        print("[____________________________________________]")
        print("[--> extract :: sort all the Gaussian components :: w.r.t. integrated int. :: given max_ngauss ...]")
        print("")
        print("")
        extract_maps(_fitsarray_gfit_results2_sorted_wrt_integrated_int, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='ngfit_wrt_integrated_int', ng_opt=opt_ngmap_gmax_ng_gparam_sorted_wrt_integrated_int, _hdu=hdu)



    if _params['_bulk_extraction'] == 'Y':
        print(" ____________________________________________")
        print("[____________________________________________]")
        print("[--> extract ::bulk motions:: given the optimal n-gauss map ...]")
        print("")
        print("")

        if not os.path.exists(_params['_bulk_model_dir'] + _params['_bulk_ref_vf']):
            print(" _________________________________________________________")
            print("[ BULK MOTION REFERENCE VELOCITY FIELD MAP IS NOT PRESENT ]")
            print(" _________________________________________________________")
            print(" _________________________________________________________")
            print("[ %s is not present in %s ]" % (_params['_bulk_ref_vf'], _params['_bulk_model_dir']))
            print(" _________________________________________________________")
            print("[--> exit now ...]")
            print("")
            print("")

        if not os.path.exists(_params['_bulk_model_dir'] + _params['_bulk_delv_limit']):
            print(" _______________________________________________")
            print("[ BULK MOTION VELOCITY LIMIT MAP IS NOT PRESENT ]")
            print(" _______________________________________________")
            print("[ %s is not present in %s ]" % (_params['_bulk_delv_limit'], _params['_bulk_model_dir']))
            print(" ______________________________________________________")
            print("[--> exit now ...]")
            print("")
            print("")


        if not os.path.exists("%s/bulk" % _dir_baygaud_combined):
            make_dirs("%s/bulk" % _dir_baygaud_combined)

        if not os.path.exists("%s/non_bulk" % _dir_baygaud_combined):
            make_dirs("%s/non_bulk" % _dir_baygaud_combined)

        bulk_ref_vf = fitsio.read(_params['_bulk_model_dir'] + '/' + _params['_bulk_ref_vf'])
        bulk_delv_limit = fitsio.read(_params['_bulk_model_dir'] + '/' + _params['_bulk_delv_limit']) * _params['_bulk_delv_limit_factor']

        extract_maps_bulk(_fitsarray_gfit_results2, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='bulk', \
                            ng_opt=opt_ngmap_gmax_ng, _bulk_ref_vf=bulk_ref_vf, _bulk_delv_limit=bulk_delv_limit, _hdu=hdu)


    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> extract :: hvc component:: given the optimal n-gauss map ...]")
    print("")
    print("")

    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> save the full baygaud_gfit_results in fits format...]")
    print("")
    print("")
    #_hdu_nparray_gfit_results = fits.PrimaryHDU(_fitsarray_gfit_results2)
    #_hdulist_nparray_gfit_result = fits.HDUList([_hdu_nparray_gfit_results])
    #_hdulist_nparray_gfit_result.writeto('%s/baygaud_gfit_results.fits' % _dir_baygaud_combined, overwrite=True)
    #_hdulist_nparray_gfit_result.close()

    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> save the full baygaud_gfit_results in binary format...]")
    with open('%s/baygaud_gfit_results.npy' % _dir_baygaud_combined,'wb') as _f:
        np.save(_f, _fitsarray_gfit_results2)

    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> _fitsarray_gfit_results2-shape: ", _fitsarray_gfit_results2.shape)
    print("[--> baygaud classification completed: %d Gaussians...]" % max_ngauss)
    print("")
    print("")
    print("[--> duration: ", datetime.now() - _time_start)
    print("")


if __name__ == '__main__':
    main()

