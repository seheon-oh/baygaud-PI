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

#|-----------------------------------------|
import time, sys, os
from datetime import datetime
import shutil

#|-----------------------------------------|
import numpy as np
from numpy import linalg, array, sum, log, exp, pi, std, diag, concatenate
import gc
import operator
import copy as cp

import itertools as itt

#........................................ 
# import make_dirs
from _dirs_files import make_dirs

#........................................ 
# import for dynesty lines
import json
import sys
import scipy.stats, scipy
#import pymultinest
import matplotlib.pyplot as plt

import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from dynesty import NestedSampler

#........................................ 
# import for pybaygaud
import fitsio
import astropy.units as u
from astropy.io import fits
from spectral_cube import SpectralCube
from _baygaud_params import _params


#........................................ 
# import _fits_io
from _fits_io import update_header_cube_to_2d

#........................................ 
# global parameters
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


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def find_nearest_values_along_axis(nparray, nparray_ref, axis):
    argmin_index = np.abs(nparray - nparray_ref).argmin(axis=axis)
    _shape = nparray.shape
    index = list(np.ix_(*[np.arange(i) for i in _shape]))
    index[axis] = np.expand_dims(argmin_index, axis=axis)

    return np.squeeze(nparray[index])
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def find_nearest_index_along_axis(nparray, nparray_ref, axis):
    argmin_index = np.abs(nparray - nparray_ref).argmin(axis=axis)
    _shape = nparray.shape
    index = list(np.ix_(*[np.arange(i) for i in _shape]))
    index[axis] = np.expand_dims(argmin_index, axis=axis)

    return index[axis][0]
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def extract_maps_bulk(_fitsarray_gfit_results2, params, _output_dir, _kin_comp, ng_opt, _bulk_ref_vf, _bulk_delv_limit, _hdu):

    max_ngauss = params['max_ngauss']
    nparams = (3*max_ngauss+2)
    peak_sn_limit = params['peak_sn_limit']

    if _kin_comp == 'bulk':
        _vlos_lower = params['vlos_lower']
        _vlos_upper = params['vlos_upper']
        _vdisp_lower = params['vdisp_lower']
        _vdisp_upper = params['vdisp_upper']


    # 0. A = peak * sqrt(2pi) * std
    # xxx.0.fits
    # xxx.0.e.fits
    #----------------------------------
    # 1. x0 : central velocity in km/s
    # xxx.1.fits
    # xxx.1.e.fits
    #----------------------------------
    # 2. std : velocity dispersion in km/s
    # xxx.2.fits
    # xxx.2.e.fits
    #----------------------------------
    # 3. bg : background in Jy
    # xxx.3.fits
    # xxx.3.e.fits
    #----------------------------------
    # 4. rms : rms in Jy
    # xxx.4.fits
    # xxx.4.e.fits
    #----------------------------------
    # 5. peak_flux in Jy
    # xxx.5.fits
    # xxx.5.e.fits
    #----------------------------------
    # 6. peak s/n
    # xxx.6.fits
    # xxx.6.e.fits
    #----------------------------------
    # 7. N-gauss
    # xxx.7.fits
    # xxx.7.e.fits

    # sigma-flux : alternate to rms

    #print(_fitsarray_gfit_results2[3:nparams:3, 410, 410])
    #print('%f lower:%f upper:%f' % (_fitsarray_gfit_results2[3, 1, 1], _vdisp_lower, _vdisp_upper))

    # ---------------------------------------------------------------
    # number of parameters per step
    nparams_step = 2*(3*max_ngauss+2) + (max_ngauss + 7)


    # --------------------------------------------------------------- 
    # arrays for slices
    # _______________________________________________________________ 
    # s/n ng_opt array
    sn_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    # ---------------------------------------------------------------
    # x ng_opt array
    x_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # _______________________________________________________________

    # ---------------------------------------------------------------
    # std ng_opt array
    std_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # _______________________________________________________________

    # ---------------------------------------------------------------
    # p ng_opt array
    p_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # _______________________________________________________________

    # ---------------------------------------------------------------
    # bg ng_opt array
    bg_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    # ---------------------------------------------------------------
    # rms ng_opt array
    rms_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # _______________________________________________________________


    # ----------------------------------------------------------------------------------- #
    # 1. extract the optimal gaussian components given ng_opt[:, :]
    # --> sn_ng_opt[0, :, :], sn_ng_opt[1, :, :], sn_ng_opt[2, :, :], ...
    # ___________________________________________________________________________________ #
    for i in range(0, max_ngauss):
        for j in range(0, max_ngauss):
            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-1. S/N slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... peak flux:4 / rms:edge-max_ngauss-7+j
            #print("here", _fitsarray_gfit_results2[ nparams_step*(j+1)-max_ngauss-7+j, 460, 556]) 
            sn_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i) & \
                                                    (_fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :] > 0), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0]
            sn_ng_opt_t[i, :, :] += sn_ng_opt_slice[i, j, :, :]
            # -----------------------------------------------------------------------------------
            # .... 2 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            sn_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    0.0, 0.0)])[0] # put zero to rms-error as it is not available
            sn_ng_opt_e_t[i, :, :] += sn_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________

            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-2. x slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... 2 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            x_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*i, :, :], 1E-7)])[0] # otherwise put 1E-7 used for summing up Xs below
            x_ng_opt_t[i, :, :] += x_ng_opt_slice[i, j, :, :]
            #---------------------------------------------------
            # .... 3*(j+1) + 2 (sig, bg errors) + 0 (x error) + 3*i
            x_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 0 + 3*i, :, :], 1E-7)])[0] # otherwise put 1E-7 used for summing up Xs below
            x_ng_opt_e_t[i, :, :] += x_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________

            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-3. std slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... 3 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            std_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 3 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up stds below
            std_ng_opt_t[i, :, :] += std_ng_opt_slice[i, j, :, :]
            #---------------------------------------------------
            # .... 3*(j+1) + 2 (sig, bg errors) + 1 (std error) + 3*i
            std_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 1 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up stds below
            std_ng_opt_e_t[i, :, :] += std_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________

            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-4. peak flux slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... 4 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            p_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up ps below
            p_ng_opt_t[i, :, :] += p_ng_opt_slice[i, j, :, :]
            #---------------------------------------------------
            # .... 3*(j+1) + 2 (sig, bg errors) + 2 (peak flux error) + 3*i
            p_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 2 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up ps below
            p_ng_opt_e_t[i, :, :] += p_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________

            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-5. bg slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... 4 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            bg_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 1, :, :], 0.0)])[0] # otherwise put 1E-7 used for summing up bgs below
            bg_ng_opt_t[i, :, :] += bg_ng_opt_slice[i, j, :, :]
            #---------------------------------------------------
            # .... 3*(j+1) + 2 (0:sig-error, 1:bg-error)
            bg_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 1, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up bgs below
            bg_ng_opt_e_t[i, :, :] += bg_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________

            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-6. rms slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... 4 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            rms_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0] # otherwise put 0.0
            rms_ng_opt_t[i, :, :] += rms_ng_opt_slice[i, j, :, :]
            #---------------------------------------------------
            # .... 3*(j+1) + 2 (0:sig-error, 1:bg-error)
            rms_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    0.0, 0.0)])[0] # put zero to rms-error as it is not available
            rms_ng_opt_e_t[i, :, :] += rms_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________


    # ----------------------------------------------------------------------------------- #
    # 2. replace blank elements with a blank value of -1E9
    # ___________________________________________________________________________________ #
    #
    # 1. sn
    sn_ng_opt = np.where(sn_ng_opt_t != 3*0.0, sn_ng_opt_t, -1E9)
    sn_ng_opt_e = np.where(sn_ng_opt_t != 3*0.0, sn_ng_opt_t, -1E9)

    # 2. x : 1E-7 blank as there could be x with 0 km/s 
    x_ng_opt = np.where(x_ng_opt_t != 3*1E-7, x_ng_opt_t, -1E9)
    x_ng_opt_e = np.where(x_ng_opt_e_t != 3*1E-7, x_ng_opt_e_t, -1E9)

    # 3. std
    std_ng_opt = np.where(std_ng_opt_t != 3*0.0, std_ng_opt_t, -1E9)
    std_ng_opt_e = np.where(std_ng_opt_e_t != 3*0.0, std_ng_opt_e_t, -1E9)

    # 4. peak
    p_ng_opt = np.where(p_ng_opt_t != 3*0.0, p_ng_opt_t, -1E9)
    p_ng_opt_e = np.where(p_ng_opt_e_t != 3*0.0, p_ng_opt_e_t, -1E9)

    # 5. bg
    bg_ng_opt = np.where(bg_ng_opt_t != 3*0.0, bg_ng_opt_t, -1E9)
    bg_ng_opt_e = np.where(bg_ng_opt_e_t != 3*0.0, bg_ng_opt_e_t, -1E9)

    # 6. rms
    rms_ng_opt = np.where(rms_ng_opt_t != 3*0.0, rms_ng_opt_t, -1E9)
    rms_ng_opt_e = np.where(rms_ng_opt_e_t != 3*0.0, rms_ng_opt_e_t, -1E9)
    #_______________________________________________________
    #print(np.where(x_ng_opt == 1E-7*3))

    if _kin_comp == 'sgfit' or _kin_comp == 'psgfit':
        _ng = 1 
    else:
        _ng = max_ngauss

    # ----------------------------------------------------------------------------------- #
    # 3. find the bulk motion index of sn_ng_opt, x_ng_opt, and std_ng_opt
    # ___________________________________________________________________________________ #
    #
    # [[0, 1, 2, 0, 2, ...
    #   0, 0, 2, 0, 1, ...
    #   0, 1, 0, 1, 2, ...
    #   0, 1, 2, 0, 1, ...
    # ]]
    #
    bk_index = find_nearest_index_along_axis(x_ng_opt, _bulk_ref_vf, axis=0)
    # ___________________________________________________________________________________ #

    #print(_fitsarray_gfit_results2[:, 410, 410])
    #print("******")
    #print(x_ng_opt[0, 460, 556])
    #print(x_ng_opt[1, 460, 556])
    #print(x_ng_opt[2, 460, 556])
    #print(x_ng_opt_e[0, 460, 556])
    #print(x_ng_opt_e[1, 460, 556])
    #print(x_ng_opt_e[2, 460, 556])

    #print(bk_index[500, 556])
    #print(ng_opt[500, 556])
    #print("******")

    #print("s"*20)
    #print(np.where(bk_index>0))
    #print(bk_index.shape)


    # ----------------------------------------------------------------------------------- #
    # 4. extract the bulk gaussian components given bk_index[:, ;]
    # --> sn_ng_opt_bulk[0, :, :]
    # --> x_ng_opt_bulk[1, :, :]
    # --> std_ng_opt_bulk[2, :, :], ...
    # --> THESE ARE CANDIDATES YET WHICH ARE LASTLY FILETERED GIVEN THE BULK LIMITS BELOW
    # ___________________________________________________________________________________ #

    _ax1 = np.arange(x_ng_opt.shape[1])[:, None]
    _ax2 = np.arange(x_ng_opt.shape[2])[None, :]
    # 1. sn
    sn_ng_opt_bulk = sn_ng_opt[bk_index, _ax1, _ax2]
    # 2. x
    x_ng_opt_bulk = x_ng_opt[bk_index, _ax1, _ax2]
    x_ng_opt_bulk_e = x_ng_opt_e[bk_index, _ax1, _ax2]
    # 3. std
    std_ng_opt_bulk = std_ng_opt[bk_index, _ax1, _ax2]
    std_ng_opt_bulk_e = std_ng_opt_e[bk_index, _ax1, _ax2]
    # 4. p
    p_ng_opt_bulk = p_ng_opt[bk_index, _ax1, _ax2]
    p_ng_opt_bulk_e = p_ng_opt_e[bk_index, _ax1, _ax2]
    # 5. bg
    bg_ng_opt_bulk = bg_ng_opt[bk_index, _ax1, _ax2]
    bg_ng_opt_bulk_e = bg_ng_opt_e[bk_index, _ax1, _ax2]
    # 6. rms
    rms_ng_opt_bulk = rms_ng_opt[bk_index, _ax1, _ax2]
    rms_ng_opt_bulk_e = rms_ng_opt_e[bk_index, _ax1, _ax2]
    # ___________________________________________________________________________________ #


    # ----------------------------------------------------------------------------------- #
    # CHECK POINT
    # ___________________________________________________________________________________ ###
    i1 = _params['_i0']
    j1 = _params['_j0']

    print("sn:", sn_ng_opt_bulk[j1, i1])
    print("x:", x_ng_opt_bulk[j1, i1])
    print("ref_x:", _bulk_ref_vf[j1, i1])
    print("delv:", _bulk_delv_limit[j1, i1])
    print("std:", std_ng_opt_bulk[j1, i1])

    print("vdisp_lower:", _vdisp_lower)
    print("vdisp_upper:", _vdisp_upper)
    print("vlos_lower:", _vlos_lower)
    print("vlos_upper:", _vlos_upper)
    
    # ----------------------------------------------------------------------------------- #



    # ----------------------------------------------------------------------------------- #
    # filter for bulk motions
    #
    _filter_bulk = ( \
            # 1. S/N limit: > peak_sn_limit
            (sn_ng_opt_bulk[:, :] > peak_sn_limit) & \
            # 2. VLOS limit: 
            (x_ng_opt_bulk[:, :] >= _bulk_ref_vf[:, :] - _bulk_delv_limit[:, :]) & \
            (x_ng_opt_bulk[:, :] < _bulk_ref_vf[:, :] + _bulk_delv_limit[:, :]) & \
            # 3. VDISP limit:
            (std_ng_opt_bulk[:, :] >= _vdisp_lower) & \
            (std_ng_opt_bulk[:, :] < _vdisp_upper))
        #print("filter_bulk", np.where(_filter_bulk == True))


    # ----------------------------------------------------------------------------------- #
    # 0-0. A = peak * sqrt(2pi) * std
    # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> A: integrated intensity : sqrt(2PI)*VDISP*peak_flux
                                    np.sqrt(2*np.pi)* std_ng_opt_bulk * p_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.0.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()

    # ----------------------------------
    # 0-1. A-error 
    # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
    _nparray_t = np.array([np.where( \
                                    (_filter_bulk) & \
                                    (p_ng_opt_bulk > 0.0) & \
                                    (std_ng_opt_bulk > 0.0), \
                                    # --> A: integrated intensity : sqrt(2PI)*VDISP*peak_flux-error
                                    np.sqrt(2*np.pi) * \
                                    # peak flux
                                    p_ng_opt_bulk * \
                                    # std
                                    std_ng_opt_bulk * \
                                    # sqrt( (pe/p)**2 + (stde/std)**2) 
                                    ((p_ng_opt_bulk_e/p_ng_opt_bulk)**2 + \
                                    (std_ng_opt_bulk_e/std_ng_opt_bulk)**2)**0.5, \
                                   np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.0.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    # ___________________________________________________________________________________ #


    # ----------------------------------------------------------------------------------- #
    # 1-0. x : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> x
                                    x_ng_opt_bulk, np.nan)])

    _nparray_bulk_x_extracted = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> x
                                    x_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.1.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    #----------------------------------
    # 1-1. x-error : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> x-error
                                    x_ng_opt_bulk_e, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.1.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    # ___________________________________________________________________________________ #


    # ----------------------------------------------------------------------------------- #
    # 2-0. std : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> std
                                    std_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.2.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    #----------------------------------
    # 2-1. std-error : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> std-error
                                    std_ng_opt_bulk_e, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.2.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    # ___________________________________________________________________________________ #

    # ----------------------------------------------------------------------------------- #
    # 3-0. bg : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> bg
                                    bg_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.3.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    #----------------------------------
    # 3-1. bg-error : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> bg-error
                                    bg_ng_opt_bulk_e, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.3.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    # ___________________________________________________________________________________ #


    # ----------------------------------------------------------------------------------- #
    # 4-0. rms : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> rms
                                    rms_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.4.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    #----------------------------------
    # 4-1. rms-error : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> rms-error
                                    0.0, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.4.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    # ___________________________________________________________________________________ #


    # ----------------------------------------------------------------------------------- #
    # 5-0. peak flux : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> peak flux
                                    p_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.5.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    #----------------------------------
    # 5-1. peak-flux-error : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> std-error
                                    p_ng_opt_bulk_e, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.5.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    # ___________________________________________________________________________________ #

    # ----------------------------------------------------------------------------------- #
    # 6-0. peak s/n : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> peak s/n
                                    p_ng_opt_bulk / rms_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.6.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    #----------------------------------
    # 6-1. peak s/n-error : put zero as rms-e is zero
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> peak s/n error : put zero as rms-e is zero
                                    0.0, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.6.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    # ___________________________________________________________________________________ #

    # ----------------------------------------------------------------------------------- #
    # 7-0. optimal N-gauss
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> N-gauss
                                    ng_opt+1, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.7.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    #----------------------------------
    # 7-1. optimal N-gauss error : put zero as it is not available 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> N-gauss error : put zero 
                                    0.0, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.7.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()




    for i in range(0, _ng):
        #print(ng_opt.shape)
        #print(x_ng_opt[0, 534, 486], std_ng_opt[0, 534, 486], sn_ng_opt[0, 534, 486], ng_opt[534, 486])
        #print(x_ng_opt[1, 534, 486], std_ng_opt[1, 534, 486], sn_ng_opt[1, 534, 486], ng_opt[534, 486])
        #print(x_ng_opt[2, 534, 486], std_ng_opt[2, 534, 486], sn_ng_opt[2, 534, 486], ng_opt[534, 486])

        #print(x_ng_opt[0, 474, 489], std_ng_opt[0, 474, 489], sn_ng_opt[0, 474, 489], ng_opt[474, 489])
        #print(x_ng_opt[1, 474, 489], std_ng_opt[1, 474, 489], sn_ng_opt[1, 474, 489], ng_opt[474, 489])
        #print(x_ng_opt[2, 474, 489], std_ng_opt[2, 474, 489], sn_ng_opt[2, 474, 489], ng_opt[474, 489])

        #----------------------------------------
        # ng_opt[0, y, x]
        # _fitsarray_gfit_results2[params, y, x]
        # sn_ng_opt[max_ngauss, y, x]
        # x_ng_opt[max_ngauss, y, x]
        # std_ng_opt[max_ngauss, y, x]
        #________________________________________

        # ----------------------------------------------------------------------------------- #
        # filter for non-bulk motions
        #
        _filter_non_bulk = ( \
            # 1. ng_opt > i : if the current ngauss index is smaller than or equals the optimal number of gaussians
            (ng_opt[:, :] >= i) & \
            # 2. S/N limit: > peak_sn_limit
            (sn_ng_opt[i, :, :] > peak_sn_limit) & \
            # 3. VLOS limit: VLOS LIMIT + EXCLUDING THE BULK motions extracted
            (x_ng_opt[i,:, :] >= _vlos_lower) & \
            (x_ng_opt[i,:, :] < _vlos_upper) & \
            (np.absolute(x_ng_opt[i,:, :] - _nparray_bulk_x_extracted[0, :, :]) > 0.1) & \
            # 4. VDISP limit:
            (std_ng_opt[i,:, :] >= _vdisp_lower) & \
            (std_ng_opt[i,:, :] < _vdisp_upper))
        #print("filter-", i, np.where(_filter == True))

        #----------------------------------
        # 0-0. A = peak * sqrt(2pi) * std
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        # --> A: integrated intensity : sqrt(2PI)*VDISP*peak_flux
                                        np.sqrt(2*np.pi)*_fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] * \
                                        _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.0.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 0-1. A-error 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk & \
                                        (_fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :] > 0.0) & \
                                        (_fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] > 0.0), \
                                        # --> A: integrated intensity : sqrt(2PI)*VDISP*peak_flux-error
                                        np.sqrt(2*np.pi) * \
                                        # std
                                        _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] * \
                                        # peak flux-error       
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :] * \
                                        #_fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :] * \
                                        # sqrt( (pe/p)**2 + (stde/std)**2) 
                                        ((_fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :])**2 + \
                                         (_fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 1 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :])**2)**0.5, \
                                        np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.0.e.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 1-0. x : 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        # --> x: 
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.1.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # bulk stop
        #print("CHECK HERE FOR BULK MAPS")
        #return _nparray_t[0], 0.1*_nparray_t[0]
        #----------------------------------

        #----------------------------------
        # 1-1. x-error : 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        # --> x-error: 
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 0 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.1.e.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 2-0. std : 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        # --> std: 
                                        _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.2.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 2-1. std-error : 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        # --> std-error: 
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 1 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.2.e.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 3-0. bg : 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        # --> bg: 
                                        _fitsarray_gfit_results2[nparams_step*i + 1 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.3.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 3-1. bg-error : 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        # --> bg-error: 
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 1 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.3.e.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 4-0. rms : 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        # --> rms: 
                                        _fitsarray_gfit_results2[nparams_step*(i+1)-max_ngauss-7+i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.4.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 4-1. rms-error : 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        # --> rms-error: put zero
                                        0.0, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.4.e.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 5-0. peak flux : 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        # --> peak flux: 
                                        _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.5.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 5-1. peak flux-error : 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        # --> peak flux-error: 
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.5.e.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()


        #----------------------------------
        # 6-0. peak sn :
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        # --> peak flux / rms: 
                                        _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :] / \
                                        _fitsarray_gfit_results2[nparams_step*(i+1)-max_ngauss-7+i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.6.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 6-1. peak sn-error :
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        # --> s/n error: put zero
                                        0.0, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.6.e.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 7-0. optimal N-gauss 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        # --> opt N-gauss
                                        ng_opt+1, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.7.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 7-1. optimal N-gauss error : put zero as it is not available 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter_non_bulk, \
                                        # --> opt N-gauss error: put zero
                                        0.0, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/non_bulk/non_bulk.G%d_%d.7.e.fits' % (_output_dir, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

    print('[—> fits written ..', _nparray_t.shape)




#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def extract_maps_bulk_org(_fitsarray_gfit_results2, params, _output_dir, _kin_comp, ng_opt, _bulk_ref_vf, _bulk_delv_limit):

    max_ngauss = params['max_ngauss']
    nparams = (3*max_ngauss+2)
    peak_sn_limit = params['peak_sn_limit']

    #---------------------------------------------------
    if _kin_comp == 'sgfit':
        _vlos_lower = params['vlos_lower']
        _vlos_upper = params['vlos_upper']
        _vdisp_lower = params['vdisp_lower']
        _vdisp_upper = params['vdisp_upper']
        print("")
        print("| ... extracting sgfit results ... |")
        print("| vlos-lower: %1.f" % _vlos_lower)
        print("| vlos-upper: %1.f" % _vlos_upper)
        print("| vdisp-lower: %1.f" % _vdisp_lower)
        print("| vdisp-upper: %1.f" % _vdisp_upper)
        print("")
    #---------------------------------------------------
    elif _kin_comp == 'psgfit':
        _vlos_lower = params['vlos_lower']
        _vlos_upper = params['vlos_upper']
        _vdisp_lower = params['vdisp_lower']
        _vdisp_upper = params['vdisp_upper']
        print("")
        print("| ... extracting sgfit results ... |")
        print("| vlos-lower: %1.f" % _vlos_lower)
        print("| vlos-upper: %1.f" % _vlos_upper)
        print("| vdisp-lower: %1.f" % _vdisp_lower)
        print("| vdisp-upper: %1.f" % _vdisp_upper)
        print("")
    #---------------------------------------------------
    elif _kin_comp == 'cool':
        _vlos_lower = params['vlos_lower_cool']
        _vlos_upper = params['vlos_upper_cool']
        _vdisp_lower = params['vdisp_lower_cool']
        _vdisp_upper = params['vdisp_upper_cool']
        print("")
        print("| ... extracting kinematically cool results ... |")
        print("| vlos-lower: %1.f" % _vlos_lower)
        print("| vlos-upper: %1.f" % _vlos_upper)
        print("| vdisp-lower: %1.f" % _vdisp_lower)
        print("| vdisp-upper: %1.f" % _vdisp_upper)
        print("")
    #---------------------------------------------------
    elif _kin_comp == 'warm':
        _vlos_lower = params['vlos_lower_warm']
        _vlos_upper = params['vlos_upper_warm']
        _vdisp_lower = params['vdisp_lower_warm']
        _vdisp_upper = params['vdisp_upper_warm']
        print("")
        print("| ... extracting kinematically warm results ... |")
        print("| vlos-lower: %1.f" % _vlos_lower)
        print("| vlos-upper: %1.f" % _vlos_upper)
        print("| vdisp-lower: %1.f" % _vdisp_lower)
        print("| vdisp-upper: %1.f" % _vdisp_upper)
        print("")
    #---------------------------------------------------
    elif _kin_comp == 'hot':
        _vlos_lower = params['vlos_lower_hot']
        _vlos_upper = params['vlos_upper_hot']
        _vdisp_lower = params['vdisp_lower_hot']
        _vdisp_upper = params['vdisp_upper_hot']
        print("")
        print("| ... extracting kinematically hot results ... |")
        print("| vlos-lower: %1.f" % _vlos_lower)
        print("| vlos-upper: %1.f" % _vlos_upper)
        print("| vdisp-lower: %1.f" % _vdisp_lower)
        print("| vdisp-upper: %1.f" % _vdisp_upper)
        print("")
    #---------------------------------------------------
    else: # including bulk
        _vlos_lower = params['vlos_lower']
        _vlos_upper = params['vlos_upper']
        _vdisp_lower = params['vdisp_lower']
        _vdisp_upper = params['vdisp_upper']



    # 0. A = peak * sqrt(2pi) * std
    # xxx.0.fits
    # xxx.0.e.fits
    #----------------------------------
    # 1. x0 : central velocity in km/s
    # xxx.1.fits
    # xxx.1.e.fits
    #----------------------------------
    # 2. std : velocity dispersion in km/s
    # xxx.2.fits
    # xxx.2.e.fits
    #----------------------------------
    # 3. bg : background in Jy
    # xxx.3.fits
    # xxx.3.e.fits
    #----------------------------------
    # 4. rms : rms in Jy
    # xxx.4.fits
    # xxx.4.e.fits
    #----------------------------------
    # 5. peak_flux in Jy
    # xxx.5.fits
    # xxx.5.e.fits
    #----------------------------------
    # 6. peak s/n
    # xxx.6.fits
    # xxx.6.e.fits
    #----------------------------------
    # 7. N-gauss
    # xxx.7.fits
    # xxx.7.e.fits

    # sigma-flux : alternate to rms

    #print(_fitsarray_gfit_results2[3:nparams:3, 410, 410])
    #print('%f lower:%f upper:%f' % (_fitsarray_gfit_results2[3, 1, 1], _vdisp_lower, _vdisp_upper))

    # ---------------------------------------------------------------
    # number of parameters per step
    nparams_step = 2*(3*max_ngauss+2) + (max_ngauss + 7)


    # --------------------------------------------------------------- 
    # arrays for slices
    # _______________________________________________________________ 
    # s/n ng_opt array
    sn_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    # ---------------------------------------------------------------
    # x ng_opt array
    x_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # _______________________________________________________________

    # ---------------------------------------------------------------
    # std ng_opt array
    std_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # _______________________________________________________________

    # ---------------------------------------------------------------
    # p ng_opt array
    p_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # _______________________________________________________________

    # ---------------------------------------------------------------
    # bg ng_opt array
    bg_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    # ---------------------------------------------------------------
    # rms ng_opt array
    rms_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # _______________________________________________________________


    # ----------------------------------------------------------------------------------- #
    # 1. extract the optimal gaussian components given ng_opt[:, :]
    # --> sn_ng_opt[0, :, :], sn_ng_opt[1, :, :], sn_ng_opt[2, :, :], ...
    # ___________________________________________________________________________________ #
    for i in range(0, max_ngauss):
        for j in range(0, max_ngauss):
            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-1. S/N slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... peak flux:4 / rms:edge-max_ngauss-7+j
            #print("here", _fitsarray_gfit_results2[ nparams_step*(j+1)-max_ngauss-7+j, 460, 556]) 
            sn_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i) & \
                                                    (_fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :] > 0), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0]
            sn_ng_opt_t[i, :, :] += sn_ng_opt_slice[i, j, :, :]
            # -----------------------------------------------------------------------------------
            # .... 2 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            sn_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    0.0, 0.0)])[0] # put zero to rms-error as it is not available
            sn_ng_opt_e_t[i, :, :] += sn_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________

            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-2. x slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... 2 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            x_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*i, :, :], 1E-7)])[0] # otherwise put 1E-7 used for summing up Xs below
            x_ng_opt_t[i, :, :] += x_ng_opt_slice[i, j, :, :]
            #---------------------------------------------------
            # .... 3*(j+1) + 2 (sig, bg errors) + 0 (x error) + 3*i
            x_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 0 + 3*i, :, :], 1E-7)])[0] # otherwise put 1E-7 used for summing up Xs below
            x_ng_opt_e_t[i, :, :] += x_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________

            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-3. std slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... 3 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            std_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 3 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up stds below
            std_ng_opt_t[i, :, :] += std_ng_opt_slice[i, j, :, :]
            #---------------------------------------------------
            # .... 3*(j+1) + 2 (sig, bg errors) + 1 (std error) + 3*i
            std_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 1 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up stds below
            std_ng_opt_e_t[i, :, :] += std_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________

            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-4. peak flux slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... 4 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            p_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up ps below
            p_ng_opt_t[i, :, :] += p_ng_opt_slice[i, j, :, :]
            #---------------------------------------------------
            # .... 3*(j+1) + 2 (sig, bg errors) + 2 (peak flux error) + 3*i
            p_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 2 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up ps below
            p_ng_opt_e_t[i, :, :] += p_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________

            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-5. bg slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... 4 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            bg_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 1, :, :], 0.0)])[0] # otherwise put 1E-7 used for summing up bgs below
            bg_ng_opt_t[i, :, :] += bg_ng_opt_slice[i, j, :, :]
            #---------------------------------------------------
            # .... 3*(j+1) + 2 (0:sig-error, 1:bg-error)
            bg_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 1, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up bgs below
            bg_ng_opt_e_t[i, :, :] += bg_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________

            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-6. rms slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... 4 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            rms_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0] # otherwise put 0.0
            rms_ng_opt_t[i, :, :] += rms_ng_opt_slice[i, j, :, :]
            #---------------------------------------------------
            # .... 3*(j+1) + 2 (0:sig-error, 1:bg-error)
            rms_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    0.0, 0.0)])[0] # put zero to rms-error as it is not available
            rms_ng_opt_e_t[i, :, :] += rms_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________


    # ----------------------------------------------------------------------------------- #
    # 2. replace blank elements with a blank value of -1E9
    # ___________________________________________________________________________________ #
    #
    # 1. sn
    sn_ng_opt = np.where(sn_ng_opt_t != 3*0.0, sn_ng_opt_t, -1E9)
    sn_ng_opt_e = np.where(sn_ng_opt_t != 3*0.0, sn_ng_opt_t, -1E9)

    # 2. x : 1E-7 blank as there could be x with 0 km/s 
    x_ng_opt = np.where(x_ng_opt_t != 3*1E-7, x_ng_opt_t, -1E9)
    x_ng_opt_e = np.where(x_ng_opt_e_t != 3*1E-7, x_ng_opt_e_t, -1E9)

    # 3. std
    std_ng_opt = np.where(std_ng_opt_t != 3*0.0, std_ng_opt_t, -1E9)
    std_ng_opt_e = np.where(std_ng_opt_e_t != 3*0.0, std_ng_opt_e_t, -1E9)

    # 4. peak
    p_ng_opt = np.where(p_ng_opt_t != 3*0.0, p_ng_opt_t, -1E9)
    p_ng_opt_e = np.where(p_ng_opt_e_t != 3*0.0, p_ng_opt_e_t, -1E9)

    # 5. bg
    bg_ng_opt = np.where(bg_ng_opt_t != 3*0.0, bg_ng_opt_t, -1E9)
    bg_ng_opt_e = np.where(bg_ng_opt_e_t != 3*0.0, bg_ng_opt_e_t, -1E9)

    # 6. rms
    rms_ng_opt = np.where(rms_ng_opt_t != 3*0.0, rms_ng_opt_t, -1E9)
    rms_ng_opt_e = np.where(rms_ng_opt_e_t != 3*0.0, rms_ng_opt_e_t, -1E9)
    #_______________________________________________________
    #print(np.where(x_ng_opt == 1E-7*3))

    if _kin_comp == 'sgfit' or _kin_comp == 'psgfit':
        _ng = 1 
    else:
        _ng = max_ngauss

    # ----------------------------------------------------------------------------------- #
    # 3. find the bulk motion index of sn_ng_opt, x_ng_opt, and std_ng_opt
    # ___________________________________________________________________________________ #
    #
    # [[0, 1, 2, 0, 2, ...
    #   0, 0, 2, 0, 1, ...
    #   0, 1, 0, 1, 2, ...
    #   0, 1, 2, 0, 1, ...
    # ]]
    #
    bk_index = find_nearest_index_along_axis(x_ng_opt, _bulk_ref_vf, axis=0)
    # ___________________________________________________________________________________ #

    #print(_fitsarray_gfit_results2[:, 410, 410])
    #print("******")
    #print(x_ng_opt[0, 460, 556])
    #print(x_ng_opt[1, 460, 556])
    #print(x_ng_opt[2, 460, 556])
    #print(x_ng_opt_e[0, 460, 556])
    #print(x_ng_opt_e[1, 460, 556])
    #print(x_ng_opt_e[2, 460, 556])

    #print(bk_index[500, 556])
    #print(ng_opt[500, 556])
    #print("******")

    #print("s"*20)
    #print(np.where(bk_index>0))
    #print(bk_index.shape)


    # ----------------------------------------------------------------------------------- #
    # 4. extract the bulk gaussian components given bk_index[:, ;]
    # --> sn_ng_opt_bulk[0, :, :]
    # --> x_ng_opt_bulk[1, :, :]
    # --> std_ng_opt_bulk[2, :, :], ...
    # ___________________________________________________________________________________ #

    _ax1 = np.arange(x_ng_opt.shape[1])[:, None]
    _ax2 = np.arange(x_ng_opt.shape[2])[None, :]
    # 1. sn
    sn_ng_opt_bulk = sn_ng_opt[bk_index, _ax1, _ax2]
    # 2. x
    x_ng_opt_bulk = x_ng_opt[bk_index, _ax1, _ax2]
    x_ng_opt_bulk_e = x_ng_opt_e[bk_index, _ax1, _ax2]
    # 3. std
    std_ng_opt_bulk = std_ng_opt[bk_index, _ax1, _ax2]
    std_ng_opt_bulk_e = std_ng_opt_e[bk_index, _ax1, _ax2]
    # 4. p
    p_ng_opt_bulk = p_ng_opt[bk_index, _ax1, _ax2]
    p_ng_opt_bulk_e = p_ng_opt_e[bk_index, _ax1, _ax2]
    # 5. bg
    bg_ng_opt_bulk = bg_ng_opt[bk_index, _ax1, _ax2]
    bg_ng_opt_bulk_e = bg_ng_opt_e[bk_index, _ax1, _ax2]
    # 6. rms
    rms_ng_opt_bulk = rms_ng_opt[bk_index, _ax1, _ax2]
    rms_ng_opt_bulk_e = rms_ng_opt_e[bk_index, _ax1, _ax2]
    # ___________________________________________________________________________________ #

    #i1 = 156
    #j1 = 200
    i1 = _params['_i0']
    j1 = _params['_j0']

    print("sn:", sn_ng_opt_bulk[j1, i1])
    print("x:", x_ng_opt_bulk[j1, i1])
    print("ref_x:", _bulk_ref_vf[j1, i1])
    print("delv:", _bulk_delv_limit[j1, i1])
    print("std:", std_ng_opt_bulk[j1, i1])

    print("vdisp_lower:", _vdisp_lower)
    print("vdisp_upper:", _vdisp_upper)
    print("vlos_lower:", _vlos_lower)
    print("vlos_upper:", _vlos_upper)

    _filter_bulk = ( \
            # 1. S/N limit: > peak_sn_limit
            (sn_ng_opt_bulk[:, :] > peak_sn_limit) & \
            # 2. VLOS limit: 
            (x_ng_opt_bulk[:, :] >= _bulk_ref_vf[:, :] - _bulk_delv_limit[:, :]) & \
            (x_ng_opt_bulk[:, :] < _bulk_ref_vf[:, :] + _bulk_delv_limit[:, :]) & \
            # 3. VDISP limit:
            (std_ng_opt_bulk[:, :] >= _vdisp_lower) & \
            (std_ng_opt_bulk[:, :] < _vdisp_upper))
        #print("filter_bulk", np.where(_filter_bulk == True))


    # ----------------------------------------------------------------------------------- #
    # 0-0. A = peak * sqrt(2pi) * std
    # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> A: integrated intensity : sqrt(2PI)*VDISP*peak_flux
                                    np.sqrt(2*np.pi)* std_ng_opt_bulk * p_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.0.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    # ----------------------------------
    # 0-1. A-error 
    # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
    _nparray_t = np.array([np.where( \
                                    (_filter_bulk) & \
                                    (p_ng_opt_bulk > 0.0) & \
                                    (std_ng_opt_bulk > 0.0), \
                                    # --> A: integrated intensity : sqrt(2PI)*VDISP*peak_flux-error
                                    np.sqrt(2*np.pi) * \
                                    # peak flux
                                    p_ng_opt_bulk * \
                                    # std
                                    std_ng_opt_bulk * \
                                    # sqrt( (pe/p)**2 + (stde/std)**2) 
                                    ((p_ng_opt_bulk_e/p_ng_opt_bulk)**2 + \
                                    (std_ng_opt_bulk_e/std_ng_opt_bulk)**2)**0.5, \
                                   np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.0.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    # ___________________________________________________________________________________ #


    # ----------------------------------------------------------------------------------- #
    # 1-0. x : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> x
                                    x_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.1.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    #----------------------------------
    # 1-1. x-error : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> x-error
                                    x_ng_opt_bulk_e, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.1.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    # ___________________________________________________________________________________ #


    # ----------------------------------------------------------------------------------- #
    # 2-0. std : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> std
                                    std_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.2.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    #----------------------------------
    # 2-1. std-error : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> std-error
                                    std_ng_opt_bulk_e, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.2.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    # ___________________________________________________________________________________ #

    # ----------------------------------------------------------------------------------- #
    # 3-0. bg : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> bg
                                    bg_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.3.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    #----------------------------------
    # 3-1. bg-error : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> bg-error
                                    bg_ng_opt_bulk_e, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.3.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    # ___________________________________________________________________________________ #


    # ----------------------------------------------------------------------------------- #
    # 4-0. rms : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> rms
                                    rms_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.4.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    #----------------------------------
    # 4-1. rms-error : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> rms-error
                                    0.0, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.4.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    # ___________________________________________________________________________________ #


    # ----------------------------------------------------------------------------------- #
    # 5-0. peak flux : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> peak flux
                                    p_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.5.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    #----------------------------------
    # 5-1. peak-flux-error : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> std-error
                                    p_ng_opt_bulk_e, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.5.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    # ___________________________________________________________________________________ #

    # ----------------------------------------------------------------------------------- #
    # 6-0. peak s/n : 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> peak s/n
                                    p_ng_opt_bulk / rms_ng_opt_bulk, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.6.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    #----------------------------------
    # 6-1. peak s/n-error : put zero as rms-e is zero
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> peak s/n error : put zero as rms-e is zero
                                    0.0, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.6.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    # ___________________________________________________________________________________ #

    # ----------------------------------------------------------------------------------- #
    # 7-0. optimal N-gauss
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> N-gauss
                                    ng_opt+1, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.7.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()
    #----------------------------------
    # 7-1. optimal N-gauss error : put zero as it is not available 
    _nparray_t = np.array([np.where( \
                                    _filter_bulk, \
                                    # --> N-gauss error : put zero 
                                    0.0, np.nan)])

    _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
    _hdulist_nparray = fits.HDUList([_hdu_nparray])
    #----------------------------------
    # update header based on the input cube's header info
    update_header_cube_to_2d(_hdulist_nparray, _hdu)
    #----------------------------------
    # CHECK header cards currently included
    #print(_hdulist_nparray[0].header.cards)
    _hdulist_nparray.writeto('%s/%s/%s.G%d.7.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss), overwrite=True)
    _hdulist_nparray.close()

#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def extract_maps(_fitsarray_gfit_results2, params, _output_dir, _kin_comp, ng_opt, _hdu):

    max_ngauss = params['max_ngauss']
    nparams = (3*max_ngauss+2)
    peak_sn_limit = params['peak_sn_limit']

    #---------------------------------------------------
    if _kin_comp == 'sgfit':
        _vlos_lower = params['vel_min']
        _vlos_upper = params['vel_max']
        _vdisp_lower = params['vdisp_lower']
        _vdisp_upper = params['vdisp_upper']
        print("")
        print("| ... extracting sgfit results ... |")
        print("| vlos-lower: %1.f [km/s]" % _vlos_lower)
        print("| vlos-upper: %1.f [km/s]" % _vlos_upper)
        print("| vdisp-lower: %1.f [km/s]" % _vdisp_lower)
        print("| vdisp-upper: %1.f [km/s]" % _vdisp_upper)
        print("")
    #---------------------------------------------------
    elif _kin_comp == 'psgfit':
        _vlos_lower = params['vel_min']
        _vlos_upper = params['vel_max']
        _vdisp_lower = params['vdisp_lower']
        _vdisp_upper = params['vdisp_upper']
        print("")
        print("| ... extracting psgfit results ... |")
        print("| vlos-lower: %1.f [km/s]" % _vlos_lower)
        print("| vlos-upper: %1.f [km/s]" % _vlos_upper)
        print("| vdisp-lower: %1.f [km/s]" % _vdisp_lower)
        print("| vdisp-upper: %1.f [km/s]" % _vdisp_upper)
        print("")
    #---------------------------------------------------
    elif _kin_comp == 'cool':
        _vlos_lower = params['vlos_lower_cool']
        _vlos_upper = params['vlos_upper_cool']
        _vdisp_lower = params['vdisp_lower_cool']
        _vdisp_upper = params['vdisp_upper_cool']
        print("")
        print("| ... extracting kinematically cool components ... |")
        print("| vlos-lower: %1.f [km/s]" % _vlos_lower)
        print("| vlos-upper: %1.f [km/s]" % _vlos_upper)
        print("| vdisp-lower: %1.f [km/s]" % _vdisp_lower)
        print("| vdisp-upper: %1.f [km/s]" % _vdisp_upper)
        print("")
    #---------------------------------------------------
    elif _kin_comp == 'warm':
        _vlos_lower = params['vlos_lower_warm']
        _vlos_upper = params['vlos_upper_warm']
        _vdisp_lower = params['vdisp_lower_warm']
        _vdisp_upper = params['vdisp_upper_warm']
        print("")
        print("| ... extracting kinematically warm components ... |")
        print("| vlos-lower: %1.f [km/s]" % _vlos_lower)
        print("| vlos-upper: %1.f [km/s]" % _vlos_upper)
        print("| vdisp-lower: %1.f [km/s]" % _vdisp_lower)
        print("| vdisp-upper: %1.f [km/s]" % _vdisp_upper)
        print("")
    #---------------------------------------------------
    elif _kin_comp == 'hot':
        _vlos_lower = params['vlos_lower_hot']
        _vlos_upper = params['vlos_upper_hot']
        _vdisp_lower = params['vdisp_lower_hot']
        _vdisp_upper = params['vdisp_upper_hot']
        print("")
        print("| ... extracting kinematically hot components ... |")
        print("| vlos-lower: %1.f [km/s]" % _vlos_lower)
        print("| vlos-upper: %1.f [km/s]" % _vlos_upper)
        print("| vdisp-lower: %1.f [km/s]" % _vdisp_lower)
        print("| vdisp-upper: %1.f [km/s]" % _vdisp_upper)
        print("")
    #---------------------------------------------------
    else:
        _vlos_lower = params['vel_min']
        _vlos_upper = params['vel_max']
        _vdisp_lower = params['vdisp_lower']
        _vdisp_upper = params['vdisp_upper']



    # 0. A = peak * sqrt(2pi) * std
    # xxx.0.fits
    # xxx.0.e.fits
    #----------------------------------
    # 1. x0 : central velocity in km/s
    # xxx.1.fits
    # xxx.1.e.fits
    #----------------------------------
    # 2. std : velocity dispersion in km/s
    # xxx.2.fits
    # xxx.2.e.fits
    #----------------------------------
    # 3. bg : background in Jy
    # xxx.3.fits
    # xxx.3.e.fits
    #----------------------------------
    # 4. rms : rms in Jy
    # xxx.4.fits
    # xxx.4.e.fits
    #----------------------------------
    # 5. peak_flux in Jy
    # xxx.5.fits
    # xxx.5.e.fits
    #----------------------------------
    # 6. peak s/n
    # xxx.6.fits
    # xxx.6.e.fits
    #----------------------------------
    # 7. N-gauss
    # xxx.7.fits
    # xxx.7.e.fits

    # sigma-flux : alternate to rms

    i1 = _params['_i0']
    j1 = _params['_j0']
    print("baygaud fitting results: profile (%d, %d)" % (i1, j1))
    print(_fitsarray_gfit_results2[:, j1, i1])
    #print('%f lower:%f upper:%f' % (_fitsarray_gfit_results2[3, 1, 1], _vdisp_lower, _vdisp_upper))

    #----------------------------------
    # given ng_opt --> make a filter : obsolete
    # ng_filter = lambda _ng, _i : _i if _i <= _ng else 0.0
    nparams_step = 2*(3*max_ngauss+2) + (max_ngauss + 7)

    # --------------------------------------------------------------- 
    # arrays for slices
    # _______________________________________________________________ 
    # s/n ng_opt array
    sn_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    # ---------------------------------------------------------------
    # x ng_opt array
    x_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # _______________________________________________________________

    # ---------------------------------------------------------------
    # std ng_opt array
    std_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # _______________________________________________________________

    # ---------------------------------------------------------------
    # p ng_opt array
    p_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # _______________________________________________________________

    # ---------------------------------------------------------------
    # bg ng_opt array
    bg_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    # ---------------------------------------------------------------
    # rms ng_opt array
    rms_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # _______________________________________________________________


    # ----------------------------------------------------------------------------------- #
    # 1. extract the optimal gaussian components given ng_opt[:, :]
    # --> sn_ng_opt[0, :, :], sn_ng_opt[1, :, :], sn_ng_opt[2, :, :], ...
	# ..> the resulting arrays (sn_ng_opt, x_ng_opt etc.) are synchronous to ng_opt[:, :] array
	# ..> for example, sn_ng_opt array has non-zero values up to ng_opt from the 0th index, sn_ng_opt[ ng_opt[:,:], :, :]
	# --> so these resulting arrays should be used together with ng_opt array
    # ___________________________________________________________________________________ #
    for i in range(0, max_ngauss):
        for j in range(0, max_ngauss):
            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-1. S/N slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... peak flux:4 / rms:edge-max_ngauss-7+j
            # print("RMS:", _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, 425, 425])

            sn_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i) & \
                                                    (_fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :] > 0), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0]
            sn_ng_opt_t[i, :, :] += sn_ng_opt_slice[i, j, :, :]
            # -----------------------------------------------------------------------------------
            # .... 2 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            sn_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    0.0, 0.0)])[0] # put zero to rms-error as it is not available
            sn_ng_opt_e_t[i, :, :] += sn_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________

            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-2. x slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... 2 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
			# x blank value is 1E-7 here to avoid any 0 km/s value in case
            x_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*i, :, :], 1E-7)])[0] # otherwise put 1E-7 used for summing up Xs below
            x_ng_opt_t[i, :, :] += x_ng_opt_slice[i, j, :, :]
            #---------------------------------------------------
            # .... 3*(j+1) + 2 (sig, bg errors) + 0 (x error) + 3*i
            x_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 0 + 3*i, :, :], 1E-7)])[0] # otherwise put 1E-7 used for summing up Xs below
            x_ng_opt_e_t[i, :, :] += x_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________

            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-3. std slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... 3 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            std_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 3 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up stds below
            std_ng_opt_t[i, :, :] += std_ng_opt_slice[i, j, :, :]
            #---------------------------------------------------
            # .... 3*(j+1) + 2 (sig, bg errors) + 1 (std error) + 3*i
            std_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 1 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up stds below
            std_ng_opt_e_t[i, :, :] += std_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________

            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-4. peak flux slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... 4 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            p_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up ps below
            p_ng_opt_t[i, :, :] += p_ng_opt_slice[i, j, :, :]
            #---------------------------------------------------
            # .... 3*(j+1) + 2 (sig, bg errors) + 2 (peak flux error) + 3*i
            p_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 2 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up ps below
            p_ng_opt_e_t[i, :, :] += p_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________

            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-5. bg slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... 4 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            bg_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 1, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up bgs below
            bg_ng_opt_t[i, :, :] += bg_ng_opt_slice[i, j, :, :]
            #---------------------------------------------------
            # .... 3*(j+1) + 2 (0:sig-error, 1:bg-error)
            bg_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 1, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up bgs below
            bg_ng_opt_e_t[i, :, :] += bg_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________

            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-6. rms slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... 4 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            rms_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0] # otherwise put 0.0
            rms_ng_opt_t[i, :, :] += rms_ng_opt_slice[i, j, :, :]
            #---------------------------------------------------
            # .... 3*(j+1) + 2 (0:sig-error, 1:bg-error)
            rms_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j == ng_opt) & \
                                                    (j >= i), \
                                                    0.0, 0.0)])[0] # put zero to rms-error as it is not available
            rms_ng_opt_e_t[i, :, :] += rms_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________


    # ----------------------------------------------------------------------------------- #
    # 2. replace blank elements with a blank value of -1E9
    # ___________________________________________________________________________________ #
    #
    # 1. sn
    sn_ng_opt = np.where(sn_ng_opt_t != 3*0.0, sn_ng_opt_t, -1E9)
    sn_ng_opt_e = np.where(sn_ng_opt_t != 3*0.0, sn_ng_opt_t, -1E9)

    # 2. x : 1E-7 blank as there could be x with 0 km/s 
    x_ng_opt = np.where(x_ng_opt_t != 3*1E-7, x_ng_opt_t, -1E9)
    x_ng_opt_e = np.where(x_ng_opt_e_t != 3*1E-7, x_ng_opt_e_t, -1E9)

    # 3. std
    std_ng_opt = np.where(std_ng_opt_t != 3*0.0, std_ng_opt_t, -1E9)
    std_ng_opt_e = np.where(std_ng_opt_e_t != 3*0.0, std_ng_opt_e_t, -1E9)

    # 4. peak
    p_ng_opt = np.where(p_ng_opt_t != 3*0.0, p_ng_opt_t, -1E9)
    p_ng_opt_e = np.where(p_ng_opt_e_t != 3*0.0, p_ng_opt_e_t, -1E9)

    # 5. bg
    bg_ng_opt = np.where(bg_ng_opt_t != 3*0.0, bg_ng_opt_t, -1E9)
    bg_ng_opt_e = np.where(bg_ng_opt_e_t != 3*0.0, bg_ng_opt_e_t, -1E9)

    # 6. rms
    rms_ng_opt = np.where(rms_ng_opt_t != 3*0.0, rms_ng_opt_t, -1E9)
    rms_ng_opt_e = np.where(rms_ng_opt_e_t != 3*0.0, rms_ng_opt_e_t, -1E9)
    #_______________________________________________________
    #print(np.where(x_ng_opt == 1E-7*3))


    #i1 = 600
    #j1 = 468
    i1 = _params['_i0']
    j1 = _params['_j0']
    #print("")
    #print("sn:", sn_ng_opt[0, j1, i1], "x:", x_ng_opt[0, j1, i1], "std:", std_ng_opt[0, j1, i1], "ng:", ng_opt[j1, i1])
    #print("sn:", sn_ng_opt[1, j1, i1], "x:", x_ng_opt[1, j1, i1], "std:", std_ng_opt[1, j1, i1], "ng:", ng_opt[j1, i1])
    #print("sn:", sn_ng_opt[2, j1, i1], "x:", x_ng_opt[2, j1, i1], "std:", std_ng_opt[2, j1, i1], "ng:", ng_opt[j1, i1])
    #print("")


    if _kin_comp == 'sgfit' or _kin_comp == 'psgfit':
        _ng = 1 
    else:
        _ng = max_ngauss

    for i in range(0, _ng):
        #print(ng_opt.shape)
        #print(x_ng_opt[0, 534, 486], std_ng_opt[0, 534, 486], sn_ng_opt[0, 534, 486], ng_opt[534, 486])
        #print(x_ng_opt[1, 534, 486], std_ng_opt[1, 534, 486], sn_ng_opt[1, 534, 486], ng_opt[534, 486])
        #print(x_ng_opt[2, 534, 486], std_ng_opt[2, 534, 486], sn_ng_opt[2, 534, 486], ng_opt[534, 486])

        #print(x_ng_opt[0, 474, 489], std_ng_opt[0, 474, 489], sn_ng_opt[0, 474, 489], ng_opt[474, 489])
        #print(x_ng_opt[1, 474, 489], std_ng_opt[1, 474, 489], sn_ng_opt[1, 474, 489], ng_opt[474, 489])
        #print(x_ng_opt[2, 474, 489], std_ng_opt[2, 474, 489], sn_ng_opt[2, 474, 489], ng_opt[474, 489])

        #----------------------------------------
        # ng_opt[0, y, x]
        # _fitsarray_gfit_results2[params, y, x]
        # sn_ng_opt[max_ngauss, y, x]
        # x_ng_opt[max_ngauss, y, x]
        # std_ng_opt[max_ngauss, y, x]
        #________________________________________

        #----------------------------------------
        if _kin_comp == 'psgfit': # if ng_opt == 0 <-- ngauss==1
            _filter = ( \
                    # 1. ng_opt > i : if the current ngauss index is smaller than or equals to the optimal number of gaussians
                    (ng_opt[:, :] == 0) & \
                    # 2. S/N limit: > peak_sn_limit
                    (sn_ng_opt[i, :, :] > peak_sn_limit) & \
                    # 3. VLOS limit: 
                    (x_ng_opt[i,:, :] >= _vlos_lower) & \
                    (x_ng_opt[i,:, :] < _vlos_upper) & \
                    # 4. VDISP limit:
                    (std_ng_opt[i,:, :] >= _vdisp_lower) & \
                    (std_ng_opt[i,:, :] < _vdisp_upper))
        #print("filter-", i, np.where(_filter == True))
        #----------------------------------------
        else:
            _filter = ( \
                    # 1. ng_opt >= i : if the current ngauss index is smaller than or equals to the optimal number of gaussians
                    (ng_opt[:, :] >= i) & \
                    # 2. S/N limit: > peak_sn_limit
                    (sn_ng_opt[i, :, :] > peak_sn_limit) & \
                    # 3. VLOS limit: 
                    (x_ng_opt[i,:, :] >= _vlos_lower) & \
                    (x_ng_opt[i,:, :] < _vlos_upper) & \
                    # 4. VDISP limit:
                    (std_ng_opt[i,:, :] >= _vdisp_lower) & \
                    (std_ng_opt[i,:, :] < _vdisp_upper))
        #print("filter-", i, np.where(_filter == True))

    # ___________________________________________________________________________________ #
    # ___________________________________________________________________________________ #
        if _kin_comp == 'sgfit': # if sgfit
            #----------------------------------
            # 0-0. A = peak * sqrt(2pi) * std
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                        _filter, \
                                        # --> A: integrated intensity : sqrt(2PI)*VDISP*peak_flux
                                        np.sqrt(2*np.pi)*_fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] * \
                                        _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :], np.nan)])

        else: # if others
            #----------------------------------
            # 0-0. A = peak * sqrt(2pi) * std
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                        _filter, \
                                        # --> A: integrated intensity : sqrt(2PI)*VDISP*peak_flux
                                        np.sqrt(2*np.pi)*std_ng_opt[i, :, :] * p_ng_opt[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.0.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()


        if _kin_comp == 'sgfit': # if sgfit
            #----------------------------------
            # 0-1. A-error 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                        _filter & \
                                        (_fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :] > 0.0) & \
                                        (_fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] > 0.0), \
                                        # --> A: integrated intensity : sqrt(2PI)*VDISP*peak_flux-error
                                        np.sqrt(2*np.pi) * \
                                        # std
                                        _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] * \
                                        # peak flux-error       
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :] * \
                                        #_fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :] * \
                                        # sqrt( (pe/p)**2 + (stde/std)**2) 
                                        ((_fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :])**2 + \
                                         (_fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 1 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :])**2)**0.5, \
                                        np.nan)])
        else: # if others
            #----------------------------------
            # 0-1. A-error 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                        _filter & \
                                        (p_ng_opt[i, :, :] > 0.0) & \
                                        (std_ng_opt[i, :, :] > 0.0), \
                                        # --> A: integrated intensity : sqrt(2PI)*VDISP*peak_flux-error
                                        np.sqrt(2*np.pi) * \
                                        # std
                                        std_ng_opt[i, :, :] * \
                                        # peak flux-error       
                                        p_ng_opt_e[i, :, :] * \
                                        #_fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :] * \
                                        # sqrt( (pe/p)**2 + (stde/std)**2) 
                                        ((p_ng_opt_e[i, :, :] / p_ng_opt[i, :, :])**2 + \
                                         (std_ng_opt_e[i, :, :] / std_ng_opt[i, :, :])**2)**0.5, \
                                        np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.0.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

    # ___________________________________________________________________________________ #
    # ___________________________________________________________________________________ #
        if _kin_comp == 'sgfit': # if sgfit
            #----------------------------------
            # 1-0. x : 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> x: 
                                            _fitsarray_gfit_results2[nparams_step*i + 2 + 3*i, :, :], np.nan)])
        else: # if others
            #----------------------------------
            # 1-0. x : 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> x: 
                                            x_ng_opt[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.1.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # bulk stop
        #print("CHECK HERE FOR BULK MAPS")
        #return _nparray_t[0], 0.1*_nparray_t[0]
        #----------------------------------

        if _kin_comp == 'sgfit': # if sgfit
            #----------------------------------
            # 1-1. x-error : 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> x-error: 
                                            _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 0 + 3*i, :, :], np.nan)])
        else: # if others
            #----------------------------------
            # 1-1. x-error : 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> x-error: 
                                            x_ng_opt_e[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.1.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            #----------------------------------
            # 2-0. std : 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> std: 
                                            _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :], np.nan)])

        else: # if others
            #----------------------------------
            # 2-0. std : 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> std: 
                                            std_ng_opt[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.2.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            #----------------------------------
            # 2-1. std-error : 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> std-error: 
                                            _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 1 + 3*i, :, :], np.nan)])
        else: # if others
            #----------------------------------
            # 2-1. std-error : 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> std-error: 
                                            std_ng_opt_e[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.2.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            #----------------------------------
            # 3-0. bg : 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> bg: 
                                            _fitsarray_gfit_results2[nparams_step*i + 1 + 3*i, :, :], np.nan)]) #: this is wrong as bg is always at 1 
                                            #_fitsarray_gfit_results2[nparams_step*0 + 1 + 3*0, :, :], np.nan)])
        # in case if you want to check _fitsarray_gfit_results2 structure
        #print(_fitsarray_gfit_results2[:, 500, 500])
        #sys.exit()

        else: # if others
            #----------------------------------
            # 3-0. bg : 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> bg: 
                                            #_fitsarray_gfit_results2[nparams_step*i + 1 + 3*i, :, :], np.nan)]) #: this is wrong as bg is always at 1 
                                            bg_ng_opt[i, :, :], np.nan)])
                                            #_fitsarray_gfit_results2[nparams_step*0 + 1 + 3*0, :, :], np.nan)])
                                            #_fitsarray_gfit_results2[nparams_step*0 + 1 + 3*0, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.3.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()


        if _kin_comp == 'sgfit': # if sgfit
            #----------------------------------
            # 3-1. bg-error : 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> bg-error: 
                                            _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 1, :, :], np.nan)])
        else: # if others
            #----------------------------------
            # 3-1. bg-error : 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> bg-error: 
                                            bg_ng_opt_e[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.3.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            #----------------------------------
            # 4-0. rms : the bg rms for the case with N(i) gaussian fitting
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> rms: 
                                            _fitsarray_gfit_results2[nparams_step*(i+1)-max_ngauss-7+i, :, :], np.nan)])
        else: # if others
            #----------------------------------
            # 4-0. rms : the bg rms for the case with N(i) gaussian fitting
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> rms: 
                                            rms_ng_opt[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.4.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            #----------------------------------
            # 4-1. rms-error : NOT AVAILABLE
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> rms-error: put zero
                                            0.0, np.nan)])
        else: # if others
            #----------------------------------
            # 4-1. rms-error : NOT AVAILABLE
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> rms-error: put zero
                                            rms_ng_opt_e[i, :, :]  , np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.4.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            #----------------------------------
            # 5-0. peak flux : 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> peak flux: 
                                            _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :], np.nan)])

        else: # if others
            #----------------------------------
            # 5-0. peak flux : 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> peak flux: 
                                            p_ng_opt[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.5.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            #----------------------------------
            # 5-1. peak flux-error : 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> peak flux-error: 
                                            _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :], np.nan)])
        else: # if others
            #----------------------------------
            # 5-1. peak flux-error : 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> peak flux-error: 
                                            p_ng_opt_e[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.5.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            #----------------------------------
            # 6-0. peak sn :
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> peak flux / rms: 
                                            _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :] / \
                                            _fitsarray_gfit_results2[nparams_step*(i+1)-max_ngauss-7+i, :, :], np.nan)])
        else: # if others
            #----------------------------------
            # 6-0. peak sn :
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> peak sn
                                            sn_ng_opt[i, :, :],  np.nan)])
                                            #p_ng_opt[i, :, :] / \
                                            #rms_ng_opt[i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.6.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            #----------------------------------
            # 6-1. peak sn-error :
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> s/n error: put zero
                                            0.0, np.nan)])
        else: # if others
            #----------------------------------
            # 6-1. peak sn-error :
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> s/n error: put zero
                                            sn_ng_opt_e[i, :, :],  np.nan)])
                                            #0.0, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.6.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            #----------------------------------
            # 7-0. optimal N-gauss 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> opt N-gauss
                                            ng_opt+1, np.nan)])
        else: # if others
            #----------------------------------
            # 7-0. optimal N-gauss 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> opt N-gauss
                                            ng_opt+1, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.7.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        if _kin_comp == 'sgfit': # if sgfit
            #----------------------------------
            # 7-1. optimal N-gauss error : put zero as it is not available 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> opt N-gauss error: put zero
                                            0.0, np.nan)])
        else: # if others
            #----------------------------------
            # 7-1. optimal N-gauss error : put zero as it is not available 
            # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            _nparray_t = np.array([np.where( \
                                            _filter, \
                                            # --> opt N-gauss error: put zero
                                            0.0, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.7.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

    print('[—> fits written ..', _nparray_t.shape)
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def extract_maps_ngfit(_fitsarray_gfit_results2, params, _output_dir, _kin_comp, ng_opt, _hdu):

    max_ngauss = params['max_ngauss']
    nparams = (3*max_ngauss+2)
    peak_sn_limit = params['peak_sn_limit']

    if _kin_comp == 'ngfit':
        _vlos_lower = -1E10 # small value
        _vlos_upper = 1E10 # large value
        _vdisp_lower = -1E10 # small value
        _vdisp_upper = 1E10 # large value
        print("")
        print("| ... extracting all the Gaussian components ... |")
        print("| vlos-lower: %1.f [km/s]" % _vlos_lower)
        print("| vlos-upper: %1.f [km/s]" % _vlos_upper)
        print("| vdisp-lower: %1.f [km/s]" % _vdisp_lower)
        print("| vdisp-upper: %1.f [km/s]" % _vdisp_upper)
        print("")
    #---------------------------------------------------
    else:
        _vlos_lower = params['vlos_lower']
        _vlos_upper = params['vlos_upper']
        _vdisp_lower = params['vdisp_lower']
        _vdisp_upper = params['vdisp_upper']



    # 0. A = peak * sqrt(2pi) * std
    # xxx.0.fits
    # xxx.0.e.fits
    #----------------------------------
    # 1. x0 : central velocity in km/s
    # xxx.1.fits
    # xxx.1.e.fits
    #----------------------------------
    # 2. std : velocity dispersion in km/s
    # xxx.2.fits
    # xxx.2.e.fits
    #----------------------------------
    # 3. bg : background in Jy
    # xxx.3.fits
    # xxx.3.e.fits
    #----------------------------------
    # 4. rms : rms in Jy
    # xxx.4.fits
    # xxx.4.e.fits
    #----------------------------------
    # 5. peak_flux in Jy
    # xxx.5.fits
    # xxx.5.e.fits
    #----------------------------------
    # 6. peak s/n
    # xxx.6.fits
    # xxx.6.e.fits
    #----------------------------------
    # 7. N-gauss
    # xxx.7.fits
    # xxx.7.e.fits

    # sigma-flux : alternate to rms

    #print(_fitsarray_gfit_results2[3:nparams:3, 500, 500])
    #print('%f lower:%f upper:%f' % (_fitsarray_gfit_results2[3, 1, 1], _vdisp_lower, _vdisp_upper))

    #----------------------------------
    # given ng_opt --> make a filter : obsolete
    # ng_filter = lambda _ng, _i : _i if _i <= _ng else 0.0
    nparams_step = 2*(3*max_ngauss+2) + (max_ngauss + 7)

    # --------------------------------------------------------------- 
    # arrays for slices
    # _______________________________________________________________ 
    # s/n ng_opt array
    sn_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    # ---------------------------------------------------------------
    # x ng_opt array
    x_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # _______________________________________________________________

    # ---------------------------------------------------------------
    # std ng_opt array
    std_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    std_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # _______________________________________________________________

    # ---------------------------------------------------------------
    # p ng_opt array
    p_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    p_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # _______________________________________________________________

    # ---------------------------------------------------------------
    # bg ng_opt array
    bg_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    bg_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    # ---------------------------------------------------------------
    # rms ng_opt array
    rms_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_slice_e = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    rms_ng_opt_e_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # _______________________________________________________________


    # ----------------------------------------------------------------------------------- #
    # 1. extract the optimal gaussian components given ng_opt[:, :]
    # --> sn_ng_opt[0, :, :], sn_ng_opt[1, :, :], sn_ng_opt[2, :, :], ...
    # ___________________________________________________________________________________ #
    for i in range(0, max_ngauss):
        for j in range(0, max_ngauss):
            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-1. S/N slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... peak flux:4 / rms:edge-max_ngauss-7+j
            # print("RMS:", _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, 425, 425])

            sn_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j >= i) & \
                                                    (_fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :] > 0), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0]
            sn_ng_opt_t[i, :, :] += sn_ng_opt_slice[i, j, :, :]
            # -----------------------------------------------------------------------------------
            # .... 2 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            sn_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    0.0, 0.0)])[0] # put zero to rms-error as it is not available
            sn_ng_opt_e_t[i, :, :] += sn_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________

            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-2. x slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... 2 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            x_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*i, :, :], 1E-7)])[0] # otherwise put 1E-7 used for summing up Xs below
            x_ng_opt_t[i, :, :] += x_ng_opt_slice[i, j, :, :]
            #---------------------------------------------------
            # .... 3*(j+1) + 2 (sig, bg errors) + 0 (x error) + 3*i
            x_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 0 + 3*i, :, :], 1E-7)])[0] # otherwise put 1E-7 used for summing up Xs below
            x_ng_opt_e_t[i, :, :] += x_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________

            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-3. std slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... 3 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            std_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 3 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up stds below
            std_ng_opt_t[i, :, :] += std_ng_opt_slice[i, j, :, :]
            #---------------------------------------------------
            # .... 3*(j+1) + 2 (sig, bg errors) + 1 (std error) + 3*i
            std_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 1 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up stds below
            std_ng_opt_e_t[i, :, :] += std_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________

            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-4. peak flux slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... 4 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            p_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up ps below
            p_ng_opt_t[i, :, :] += p_ng_opt_slice[i, j, :, :]
            #---------------------------------------------------
            # .... 3*(j+1) + 2 (sig, bg errors) + 2 (peak flux error) + 3*i
            p_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 2 + 2 + 3*i, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up ps below
            p_ng_opt_e_t[i, :, :] += p_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________

            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-5. bg slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... 4 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            bg_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*0 + 1, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up bgs below
            bg_ng_opt_t[i, :, :] += bg_ng_opt_slice[i, j, :, :]
            #---------------------------------------------------
            # .... 3*(j+1) + 2 (0:sig-error, 1:bg-error)
            bg_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 2 + 3*(j+1) + 1, :, :], 0.0)])[0] # otherwise put 0.0 used for summing up bgs below
            bg_ng_opt_e_t[i, :, :] += bg_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________

            # ___________________________________________________________________________________
            # -----------------------------------------------------------------------------------
            # 1-6. rms slice with ng : sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
            # .... 4 (0:sig, 1:bg, 2:x, 3:std, 4:peak) + 3*i
            rms_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0] # otherwise put 0.0
            rms_ng_opt_t[i, :, :] += rms_ng_opt_slice[i, j, :, :]
            #---------------------------------------------------
            # .... 3*(j+1) + 2 (0:sig-error, 1:bg-error)
            rms_ng_opt_slice_e[i, j, :, :] = np.array([np.where( \
                                                    (j >= i), \
                                                    0.0, 0.0)])[0] # put zero to rms-error as it is not available
            rms_ng_opt_e_t[i, :, :] += rms_ng_opt_slice_e[i, j, :, :]
            # ___________________________________________________________________________________


    # ----------------------------------------------------------------------------------- #
    # 2. replace blank elements with a blank value of -1E9
    # ___________________________________________________________________________________ #
    #
    # 1. sn
    sn_ng_opt = np.where(sn_ng_opt_t != 3*0.0, sn_ng_opt_t, -1E9)
    sn_ng_opt_e = np.where(sn_ng_opt_t != 3*0.0, sn_ng_opt_t, -1E9)

    # 2. x : 1E-7 blank as there could be x with 0 km/s 
    x_ng_opt = np.where(x_ng_opt_t != 3*1E-7, x_ng_opt_t, -1E9)
    x_ng_opt_e = np.where(x_ng_opt_e_t != 3*1E-7, x_ng_opt_e_t, -1E9)

    # 3. std
    std_ng_opt = np.where(std_ng_opt_t != 3*0.0, std_ng_opt_t, -1E9)
    std_ng_opt_e = np.where(std_ng_opt_e_t != 3*0.0, std_ng_opt_e_t, -1E9)

    # 4. peak
    p_ng_opt = np.where(p_ng_opt_t != 3*0.0, p_ng_opt_t, -1E9)
    p_ng_opt_e = np.where(p_ng_opt_e_t != 3*0.0, p_ng_opt_e_t, -1E9)

    # 5. bg
    bg_ng_opt = np.where(bg_ng_opt_t != 3*0.0, bg_ng_opt_t, -1E9)
    bg_ng_opt_e = np.where(bg_ng_opt_e_t != 3*0.0, bg_ng_opt_e_t, -1E9)

    # 6. rms
    rms_ng_opt = np.where(rms_ng_opt_t != 3*0.0, rms_ng_opt_t, -1E9)
    rms_ng_opt_e = np.where(rms_ng_opt_e_t != 3*0.0, rms_ng_opt_e_t, -1E9)
    #_______________________________________________________
    #print(np.where(x_ng_opt == 1E-7*3))



    #i1 = 505
    #j1 = 512
    i1 = _params['_i0']
    j1 = _params['_j0']
    #print("")
    #print("sn:", sn_ng_opt[0, j1, i1], "x:", x_ng_opt[0, j1, i1], "std:", std_ng_opt[0, j1, i1], "bg:", bg_ng_opt[0, j1, i1],"ng:", ng_opt[j1, i1])
    #print("sn:", sn_ng_opt[1, j1, i1], "x:", x_ng_opt[1, j1, i1], "std:", std_ng_opt[1, j1, i1], "bg:", bg_ng_opt[1, j1, i1],"ng:", ng_opt[j1, i1])
    #print("sn:", sn_ng_opt[2, j1, i1], "x:", x_ng_opt[2, j1, i1], "std:", std_ng_opt[2, j1, i1], "bg:", bg_ng_opt[2, j1, i1],"ng:", ng_opt[j1, i1])
    #print("sn:", sn_ng_opt[3, j1, i1], "x:", x_ng_opt[3, j1, i1], "std:", std_ng_opt[3, j1, i1], "bg:", bg_ng_opt[3, j1, i1],"ng:", ng_opt[j1, i1])
    #print("")


    _ng = max_ngauss

    for i in range(0, _ng):
        #print(ng_opt.shape)
        #print(x_ng_opt[0, 534, 486], std_ng_opt[0, 534, 486], sn_ng_opt[0, 534, 486], ng_opt[534, 486])
        #print(x_ng_opt[1, 534, 486], std_ng_opt[1, 534, 486], sn_ng_opt[1, 534, 486], ng_opt[534, 486])
        #print(x_ng_opt[2, 534, 486], std_ng_opt[2, 534, 486], sn_ng_opt[2, 534, 486], ng_opt[534, 486])

        #print(x_ng_opt[0, 474, 489], std_ng_opt[0, 474, 489], sn_ng_opt[0, 474, 489], ng_opt[474, 489])
        #print(x_ng_opt[1, 474, 489], std_ng_opt[1, 474, 489], sn_ng_opt[1, 474, 489], ng_opt[474, 489])
        #print(x_ng_opt[2, 474, 489], std_ng_opt[2, 474, 489], sn_ng_opt[2, 474, 489], ng_opt[474, 489])

        #----------------------------------------
        # ng_opt[0, y, x]
        # _fitsarray_gfit_results2[params, y, x]
        # sn_ng_opt[max_ngauss, y, x]
        # x_ng_opt[max_ngauss, y, x]
        # std_ng_opt[max_ngauss, y, x]
        #________________________________________

        #----------------------------------------
        if _kin_comp == 'ngfit': #
            _filter = ( \
                    # 1. ng_opt > i : if the current ngauss index is smaller than or equals the optimal number of gaussians
                    (ng_opt[:, :] >= i) & \
                    # 2. S/N limit: > peak_sn_limit
                    (sn_ng_opt[i, :, :] > peak_sn_limit) & \
                    # 3. VLOS limit: 
                    (x_ng_opt[i,:, :] >= _vlos_lower) & \
                    (x_ng_opt[i,:, :] < _vlos_upper) & \
                    # 4. VDISP limit:
                    (std_ng_opt[i,:, :] >= _vdisp_lower) & \
                    (std_ng_opt[i,:, :] < _vdisp_upper))
        #print("filter-", i, np.where(_filter == True))
        #----------------------------------------

        #----------------------------------
        # 0-0. A = peak * sqrt(2pi) * std
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        # --> A: integrated intensity : sqrt(2PI)*VDISP*peak_flux
                                        np.sqrt(2*np.pi)*_fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] * \
                                        _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.0.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()


        #----------------------------------
        # 0-1. A-error 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter & \
                                        (_fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :] > 0.0) & \
                                        (_fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] > 0.0), \
                                        # --> A: integrated intensity : sqrt(2PI)*VDISP*peak_flux-error
                                        np.sqrt(2*np.pi) * \
                                        # std
                                        _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :] * \
                                        # peak flux-error       
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :] * \
                                        #_fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :] * \
                                        # sqrt( (pe/p)**2 + (stde/std)**2) 
                                        ((_fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :])**2 + \
                                         (_fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 1 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :])**2)**0.5, \
                                        np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.0.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 1-0. x : 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        # --> x: 
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.1.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # bulk stop
        #print("CHECK HERE FOR BULK MAPS")
        #return _nparray_t[0], 0.1*_nparray_t[0]
        #----------------------------------

        #----------------------------------
        # 1-1. x-error : 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        # --> x-error: 
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 0 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.1.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 2-0. std : 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        # --> std: 
                                        _fitsarray_gfit_results2[nparams_step*i + 3 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.2.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 2-1. std-error : 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        # --> std-error: 
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 1 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.2.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 3-0. bg : 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        # --> bg: 
                                        _fitsarray_gfit_results2[nparams_step*0 + 1, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.3.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 3-1. bg-error : 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        # --> bg-error: 
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 1, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.3.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 4-0. rms : 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        # --> rms: 
                                        _fitsarray_gfit_results2[nparams_step*(i+1)-max_ngauss-7+i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.4.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 4-1. rms-error : 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        # --> rms-error: put zero
                                        0.0, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.4.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 5-0. peak flux : 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        # --> peak flux: 
                                        _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.5.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 5-1. peak flux-error : 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        # --> peak flux-error: 
                                        _fitsarray_gfit_results2[nparams_step*i + 2 + 3*(i+1) + 2 + 2 + 3*i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.5.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()


        #----------------------------------
        # 6-0. peak sn :
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        # --> peak flux / rms: 
                                        _fitsarray_gfit_results2[nparams_step*i + 4 + 3*i, :, :] / \
                                        _fitsarray_gfit_results2[nparams_step*(i+1)-max_ngauss-7+i, :, :], np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.6.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 6-1. peak sn-error :
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        # --> s/n error: put zero
                                        0.0, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.6.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 7-0. optimal N-gauss 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        # --> opt N-gauss
                                        ng_opt+1, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.7.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 7-1. optimal N-gauss error : put zero as it is not available 
        # sig(0) - bg(1) - x(2) - std(3) - peak_flux(4) - ....
        _nparray_t = np.array([np.where( \
                                        _filter, \
                                        # --> opt N-gauss error: put zero
                                        0.0, np.nan)])

        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.7.e.fits' % (_output_dir, _kin_comp, _kin_comp, max_ngauss, i+1), overwrite=True)
        _hdulist_nparray.close()

    print('[—> fits written ..', _nparray_t.shape)
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def extract_maps_old(_fitsarray_gfit_results2, peak_sn_limit, params, ngauss, _output_dir, _sgfit):

    ngauss = 1 # for sgfit
    max_ngauss = params['max_ngauss']
    nparams_end = 2*(3*max_ngauss+2)
    #sn = _fitsarray_gfit_results2[4:nparams_end:3, :, :] / _fitsarray_gfit_results2[nparams_end, :, :]
    sn = _fitsarray_gfit_results2[4, :, :] / _fitsarray_gfit_results2[nparams_end, :, :]
    sn_mask = np.where(sn > peak_sn_limit)

    print(peak_sn_limit)

    # 0. A = peak * sqrt(2pi) * std
    # xxx.0.fits
    # xxx.0.e.fits
    #----------------------------------
    # 1. x0 : central velocity in km/s
    # xxx.1.fits
    # xxx.1.e.fits
    #----------------------------------
    # 2. std : velocity dispersion in km/s
    # xxx.2.fits
    # xxx.2.e.fits
    #----------------------------------
    # 3. bg : background in Jy
    # xxx.3.fits
    # xxx.3.e.fits
    #----------------------------------
    # 4. rms : rms in Jy
    # xxx.4.fits
    # xxx.4.e.fits
    #----------------------------------
    # 5. peak_flux in Jy
    # xxx.5.fits
    # xxx.5.e.fits
    #----------------------------------
    # 6. peak s/n
    # xxx.6.fits
    # xxx.6.e.fits
    #----------------------------------
    # 7. N-gauss
    # xxx.7.fits
    # xxx.7.e.fits

    # sigma-flux : alternate to rms
    for i in range(0, ngauss):
        #----------------------------------
        # 0. A = peak * sqrt(2pi) * std
        _nparray_t = np.array([np.where(sn > peak_sn_limit, np.sqrt(2*np.pi)*_fitsarray_gfit_results2[3, :, :]*_fitsarray_gfit_results2[4, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.0.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()
        # 0. A-error
        _nparray_t = np.array([np.where(sn > peak_sn_limit, _fitsarray_gfit_results2[0, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.0.e.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 1. x0 : central velocity in km/s
        _nparray_t = np.array([np.where(sn > peak_sn_limit, _fitsarray_gfit_results2[2, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.1.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()
        # 1. x0-error in km/s
        _nparray_t = np.array([np.where(sn > peak_sn_limit, _fitsarray_gfit_results2[2+3*max_ngauss+2, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.1.e.fits' % (_output_dir, _sgfit, _sgfit,  max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 2. std : velocity dispersion in km/s
        _nparray_t = np.array([np.where(sn > peak_sn_limit, _fitsarray_gfit_results2[3, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.2.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()
        # 2. std-error in km/s
        _nparray_t = np.array([np.where(sn > peak_sn_limit, _fitsarray_gfit_results2[2+3*max_ngauss+3, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.2.e.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 3. bg : background in Jy
        _nparray_t = np.array([np.where(sn > peak_sn_limit, _fitsarray_gfit_results2[1, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.3.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()
        # 3. bg-error : background in Jy
        _nparray_t = np.array([np.where(sn > peak_sn_limit, _fitsarray_gfit_results2[2+3*max_ngauss+1, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.3.e.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 4. rms : rms in Jy
        _nparray_t = np.array([np.where(sn > peak_sn_limit, _fitsarray_gfit_results2[nparams_end, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.4.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()
        # 4. rms-error :  not available; put 0
        _nparray_t = np.array([np.where(sn > peak_sn_limit, 0*_fitsarray_gfit_results2[nparams_end, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.4.e.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 5. peak_flux in Jy
        _nparray_t = np.array([np.where(sn > peak_sn_limit, _fitsarray_gfit_results2[4, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.5.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()
        # 5. peak_flux error in Jy
        _nparray_t = np.array([np.where(sn > peak_sn_limit, 0*_fitsarray_gfit_results2[2+3*max_ngauss+4, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.5.e.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()

        #----------------------------------
        # 6. peak s/n
        _nparray_t = np.array([np.where(sn > peak_sn_limit, _fitsarray_gfit_results2[4, :, :]/_fitsarray_gfit_results2[nparams_end, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.6.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()
        # 6. peak-error s/n : not available; put 0
        _nparray_t = np.array([np.where(sn > peak_sn_limit, 0*_fitsarray_gfit_results2[4, :, :]/_fitsarray_gfit_results2[nparams_end, :, :], np.nan)])
        _hdu_nparray = fits.PrimaryHDU(_nparray_t[0])
        _hdulist_nparray = fits.HDUList([_hdu_nparray])
        #----------------------------------
        # update header based on the input cube's header info
        update_header_cube_to_2d(_hdulist_nparray, _hdu)
        #----------------------------------
        # CHECK header cards currently included
        #print(_hdulist_nparray[0].header.cards)
        _hdulist_nparray.writeto('%s/%s/%s.G%d_%d.6.e.fits' % (_output_dir, _sgfit, _sgfit, max_ngauss, i), overwrite=True)
        _hdulist_nparray.close()



    print("gogo")
    print(_nparray_t.shape)
    print("gogo")

#    sn = (_fitsarray_gfit_results2[4:nparams_end:3, :, :] / _fitsarray_gfit_results2[]) 
#
#    for i in range(0, ngauss): 
#        sn_mask = np.where()
#        sn_mask = np.where((_fitsarray_gfit_results2[0, :, :] - bevidences_sort[1, :, :]) > np.log(bf_limit))
#
#    _fitsarray_gfit_results2[nparams_step*(i+1)-7, :, :] 
#
#    _fitsarray_gfit_results2[
#
#    #------------------------------------------------------#
    # G1-0
    #------------------------------------------------------#
#-- END OF SUB-ROUTINE____________________________________________________________#






#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g1_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, sn_pass_ng_opt, bf_limit):
    #------------------------------------------------------#
    # G1-0
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 > bf_limit): --> g1
    #g1_cond1 = np.where((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) > np.log(bf_limit))
    g_opt += 0
    #print(g_opt)
    return g_opt
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g2_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, sn_pass_ng_opt, bf_limit):
    #------------------------------------------------------#
    # G1-0
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 > bf_limit): --> g1

    g1_sorted = g_num_sort[0, :, :]

    # take values of sn_pass_ng_opt given g1_sorted indices
    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]

    g1_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) > np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g_opt[g1_cond1] += 1
    g1_0 = np.array([np.where(g_opt > 0, g1_sorted, 0)])
    #print(g1_0)

    #------------------------------------------------------#
    # G2-0
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 > g2): --> g2

    g2_sorted = g_num_sort[1, :, :]

    # take values of sn_pass_ng_opt given g2_sorted indices
    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g2_sorted[np.newaxis], axis=0)[0]

    g2_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond2 = np.where((g_num_sort[0, :, :] > g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g_opt[g2_cond1] += 1
    g_opt[g2_cond2] += 1
    g2_0 = np.array([np.where(g_opt > 1, g2_sorted, 0)])
    #print(g2_0)
    
    #------------------------------------------------------#
    # G1-1
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 < g2): --> g1

    g1_sorted = g_num_sort[0, :, :]

    # take values of sn_pass_ng_opt given g1_sorted indices
    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]

    g1_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond2 = np.where((g_num_sort[0, :, :] < g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g_opt[g1_cond1] += 1
    g_opt[g1_cond2] += 1
    g1_1 = np.array([np.where(g_opt > 1, g1_sorted, 0)])
    #print(g1_1)

    g2_opt = g1_0 + g1_1 \
            + g2_0
    return g2_opt
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g3_ms_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gg, gl, g3): # gg > gl

    #------------------------------------------------------#
    # gl - 0
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (zg/zl < bf_limit) & (gg > gl) & (zl/z3 > bf_limit): --> gn2
    #
    #------------------------------------------------------#
    # pre version
    #g2_sorted = g_num_sort[1, :, :]
    #g2_cond1 = np.where((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log10(bf_limit))
    #g2_cond2 = np.where(g_num_sort[0, :, :] > g_num_sort[1, :, :])
    #g2_cond3 = np.where((bevidences_sort[1, :, :] - bevidences_sort[2, :, :]) > np.log(bf_limit))

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


    #------------------------------------------------------#
    # g3
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (zg/zl < bf_limit) & (gg > gl) & (zl/z3 < bf_limit) & (gl > g3): --> g3
    #------------------------------------------------------#
    # pre version
    #g3_sorted = g_num_sort[2, :, :]
    #g3_cond1 = np.where((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit))
    #g3_cond2 = np.where(g_num_sort[0, :, :] > g_num_sort[1, :, :])
    #g3_cond3 = np.where((bevidences_sort[1, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit))
    #g3_cond4 = np.where(g_num_sort[1, :, :] > g_num_sort[2, :, :])

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
    #print(g3_0)
    
    
    #------------------------------------------------------#
    # gl - 1
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (zg/zl < bf_limit) & (gg > gl) & (zl/z3 < bf_limit) & (gl < g3): --> gl
    #------------------------------------------------------#
    # pre version
    #g2_sorted = g_num_sort[1, :, :]
    #g2_cond1 = np.where((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit))
    #g2_cond2 = np.where(g_num_sort[0, :, :] > g_num_sort[1, :, :])
    #g2_cond3 = np.where((bevidences_sort[1, :, :] - bevidences_sort[2, :, :]) < np.log(bf_limit))
    #g2_cond4 = np.where(g_num_sort[1, :, :] < g_num_sort[2, :, :])

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
    #print(gl_1)
    return gl_0, g3_0, gl_1
#-- END OF SUB-ROUTINE____________________________________________________________#







#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def math_opr1(_array_3d, g1, g2, opr):
    rel = { \
            '0': operator.gt, # >
            '1': operator.lt, # <
            '>=': operator.ge,
            '<=': operator.le,
            '==': operator.eq}
    return rel[opr](_array_3d[g1, :, :], _array_3d[g2, :, :])
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def math_opr2(_array_3d, g1, g2, opr, bf_limit):
    rel = { \
            '0': operator.gt, # >
            '1': operator.lt, # <
            '>=': operator.ge,
            '<=': operator.le,
            '==': operator.eq}
    return rel[opr]( (_array_3d[g1, :, :] - _array_3d[g2, :, :]), np.log10(bf_limit) )
#-- END OF SUB-ROUTINE____________________________________________________________#




#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g3_ms_bf_cond1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, \
                    g_sorted_n, \
                    n10, n11, opr1):
    #------------------------------------------------------#
    # g_opt cond1

    # if (z1/z2 > bf_limit): --> g1

    # e.g., g1 : 1-0
    # cond1: n10, n11, opr1 = (0, 1, >)

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_sorted = g_num_sort[g_sorted_n, :, :]

    # cond1
    g_cond1_opr = math_opr2(bevidences_sort, n10, n11, opr1, bf_limit)
    g_cond1 = np.where(g_cond1_opr)

    g_opt[g_cond1] += 1
    g_opt_result_cond1 = np.array([np.where(g_opt > 0, g_sorted, 0)])

    return g_opt_result_cond1
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g3_ms_bf_cond3(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, \
                    g_sorted_n, \
                    n10, n11, opr1, \
                    n20, n21, opr2, \
                    n30, n31, opr3):

    #------------------------------------------------------#
    # g_opt cond3

    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 > bf_limit): --> g2 : 2-0
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 > bf_limit): --> g1 : 1-1

    # e.g., g2 : 2-0
    # cond1: n10, n11, opr1 = (0, 1, <)
    # cond2: n20, n21, opr2 = (0, 1, >)
    # cond3: n30, n31, opr3 = (1, 2, >)

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_sorted = g_num_sort[g_sorted_n, :, :]

    # cond1
    g_cond1_opr = math_opr2(bevidences_sort, n10, n11, opr1, bf_limit)
    g_cond1 = np.where(g_cond1_opr)

    # cond2
    g_cond2_opr = math_opr1(g_num_sort, n20, n21, opr2)
    g_cond2 = np.where(g_cond2_opr)

    # cond3
    g_cond3_opr = math_opr2(bevidences_sort, n30, n31, opr3, bf_limit)
    g_cond3 = np.where(g_cond3_opr)

    g_opt[g_cond1] += 1
    g_opt[g_cond2] += 1
    g_opt[g_cond3] += 1
    g_opt_result_cond3 = np.array([np.where(g_opt > 2, g_sorted, 0)])

    return g_opt_result_cond3
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g3_ms_bf_cond4(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, \
                    g_sorted_n, \
                    n10, n11, opr1, \
                    n20, n21, opr2, \
                    n30, n31, opr3, \
                    n40, n41, opr4):

    #------------------------------------------------------#
    # g_opt cond4

    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 > g3): --> g3 : 3-0
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 > g3): --> g3 : 3-1

    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 < g3): --> g2 : 2-1
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 < g3): --> g1 : 1-2

    # e.g., g3 : 3-0
    # cond1: n10, n11, opr1 = (0, 1, <)
    # cond2: n20, n21, opr2 = (0, 1, >)
    # cond3: n30, n31, opr3 = (1, 2, <)
    # cond4: n40, n41, opr4 = (1, 2, >)

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_sorted = g_num_sort[g_sorted_n, :, :]

    # cond1
    g_cond1_opr = math_opr2(bevidences_sort, n10, n11, opr1, bf_limit)
    g_cond1 = np.where(g_cond1_opr)

    # cond2
    g_cond2_opr = math_opr1(g_num_sort, n20, n21, opr2)
    g_cond2 = np.where(g_cond2_opr)

    # cond3
    g_cond3_opr = math_opr2(bevidences_sort, n30, n31, opr3, bf_limit)
    g_cond3 = np.where(g_cond3_opr)

    # cond4
    g_cond4_opr = math_opr2(bevidences_sort, n40, n41, opr4, bf_limit)
    g_cond4 = np.where(g_cond4_opr)

    g_opt[g_cond1] += 1
    g_opt[g_cond2] += 1
    g_opt[g_cond3] += 1
    g_opt[g_cond4] += 1
    g_opt_result_cond4 = np.array([np.where(g_opt > 3, g_sorted, 0)])

    return g_opt_result_cond4
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g4_ms_bf_cond5(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, \
                    g_sorted_n, \
                    n10, n11, opr1, \
                    n20, n21, opr2, \
                    n30, n31, opr3, \
                    n40, n41, opr4, \
                    n50, n51, opr5):

    #------------------------------------------------------#
    # g_opt cond5
 
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 > g3) & (z3/z4 > bf_limit): --> g3 : 3-0
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 < g3) & (z2/z4 > bf_limit): --> g2 : 2-1
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 > g3) & (z3/z4 > bf_limit): --> g3 : 3-2
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 < g3) & (z1/z4 > bf_limit): --> g1 : 1-2

    # cond1: n10, n11, opr1 = (0, 1, <)
    # cond2: n20, n21, opr2 = (0, 1, >)
    # cond3: n30, n31, opr3 = (1, 2, <)
    # cond4: n40, n41, opr4 = (1, 2, >)
    # cond5: n50, n51, opr5 = (2, 3, >)

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_sorted = g_num_sort[g_sorted_n, :, :]

    # cond1
    g_cond1_opr = math_opr2(bevidences_sort, n10, n11, opr1, bf_limit)
    g_cond1 = np.where(g_cond1_opr)

    # cond2
    g_cond2_opr = math_opr1(g_num_sort, n20, n21, opr2)
    g_cond2 = np.where(g_cond2_opr)

    # cond3
    g_cond3_opr = math_opr2(bevidences_sort, n30, n31, opr3, bf_limit)
    g_cond3 = np.where(g_cond3_opr)

    # cond4
    g_cond4_opr = math_opr1(g_num_sort, n40, n41, opr4)
    g_cond4 = np.where(g_cond4_opr)

    # cond5
    g_cond5_opr = math_opr2(bevidences_sort, n50, n51, opr5, bf_limit)
    g_cond5 = np.where(g_cond5_opr)

    g_opt[g_cond1] += 1
    g_opt[g_cond2] += 1
    g_opt[g_cond3] += 1
    g_opt[g_cond4] += 1
    g_opt[g_cond5] += 1
    g_opt_result_cond5 = np.array([np.where(g_opt > 4, g_sorted, 0)])

    return g_opt_result_cond5
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g4_ms_bf_cond6(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, \
                    g_sorted_n, \
                    n10, n11, opr1, \
                    n20, n21, opr2, \
                    n30, n31, opr3, \
                    n40, n41, opr4, \
                    n50, n51, opr5, \
                    n60, n61, opr6):

    #------------------------------------------------------#
    # g_opt cond6
 
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 > g3) & (z3/z4 < bf_limit) & (g3 > g4): --> g4 : 4-0
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 < g3) & (z2/z4 < bf_limit) & (g2 > g4): --> g4 : 4-1
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 > g3) & (z3/z4 < bf_limit) & (g3 > g4): --> g4 : 4-2
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 < g3) & (z1/z4 < bf_limit) & (g1 > g4): --> g4 : 4-3

    # cond1: n10, n11, opr1 = (0, 1, <)
    # cond2: n20, n21, opr2 = (0, 1, >)
    # cond3: n30, n31, opr3 = (1, 2, <)
    # cond4: n40, n41, opr4 = (1, 2, >)
    # cond5: n50, n51, opr5 = (2, 3, <)
    # cond6: n60, n61, opr6 = (2, 3, >)

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_sorted = g_num_sort[g_sorted_n, :, :]

    # cond1
    g_cond1_opr = math_opr2(bevidences_sort, n10, n11, opr1, bf_limit)
    g_cond1 = np.where(g_cond1_opr)

    # cond2
    g_cond2_opr = math_opr1(g_num_sort, n20, n21, opr2)
    g_cond2 = np.where(g_cond2_opr)

    # cond3
    g_cond3_opr = math_opr2(bevidences_sort, n30, n31, opr3, bf_limit)
    g_cond3 = np.where(g_cond3_opr)

    # cond4
    g_cond4_opr = math_opr1(g_num_sort, n40, n41, opr4)
    g_cond4 = np.where(g_cond4_opr)

    # cond5
    g_cond5_opr = math_opr2(bevidences_sort, n50, n51, opr5, bf_limit)
    g_cond5 = np.where(g_cond5_opr)

    # cond6
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
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, \
                    g_sorted_n, \
                    _cond, \
                    _cond_N, \
                    g123_sn_pass):

    #------------------------------------------------------#
    # For example
    # g_opt cond6
 
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 > g3) & (z3/z4 < bf_limit) & (g3 > g4): --> g4 : 4-0
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 < g3) & (z2/z4 < bf_limit) & (g2 > g4): --> g4 : 4-1
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 > g3) & (z3/z4 < bf_limit) & (g3 > g4): --> g4 : 4-2
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 < g3) & (z1/z4 < bf_limit) & (g1 > g4): --> g4 : 4-3

    # cond1: n10, n11, opr1 = (0, 1, <)
    # cond2: n20, n21, opr2 = (0, 1, >)
    # cond3: n30, n31, opr3 = (1, 2, <)
    # cond4: n40, n41, opr4 = (1, 2, >)
    # cond5: n50, n51, opr5 = (2, 3, <)
    # cond6: n60, n61, opr6 = (2, 3, >)

    #_cond = ([1, 2, '>'], \
    #         [2, 1, '<'], \
    #         [2, 1, '<'], \
    #         [2, 1, '<'], \
    #         [2, 1, '<'], \
    #         [2, 1, '<'])

    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_sorted = g_num_sort[g_sorted_n, :, :]

    if _cond_N == 0: # single s1, s2, s3 ... :: s/n flag only,  no bf_limit
            g_cond_sn = np.where(g123_sn_pass) # g123_sn_pass --> g1, g2, g3, ... sn_pass
            g_opt[g_cond_sn] += 1
            _cond_N = 1 # update for np.where(g_opt > _cond_N-1) below : "g_opt > _cond_N-1"

    elif _cond_N == 1:
        for i in range(0, int(_cond_N/2 + 0.5)):
            # conditions for evidences z
            # e.g., 
            #------------------------------------------------------#
            # _cond[i*2][0] = 2 --> g2 model with the lowest log-z if max_ngauss=3 (i.e., out of 0, 1, 2 gaussian models) : 2 is not the gaussian number but the model number
            # _cond[i*2][1] = 1 --> g1 model with the next lowest log-z : 1 is not the gaussian number but the model number
            # _cond[i*2][2] = '0' --> '>'
            #------------------------------------------------------#
            # check bf_limit condition --> True:pass, False:non-pass
            g_cond_z_opr = math_opr2(bevidences_sort, _cond[i*2][0], _cond[i*2][1], _cond[i*2][2], bf_limit)

            g_cond_z = np.where(g_cond_z_opr & g123_sn_pass) # 1 only when satisfying the dual conditions = BF_LIMIT + SN_PASS
            g_opt[g_cond_z] += 1

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    # CHECK POINT
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
            #print("SEHEON", "g_sorted:", g_sorted[468, 600], "g_sorted_n:", g_sorted_n, "g_num_sort:", g_num_sort[:, 468, 600], g_cond_z_opr[468, 600], "sn_pass:", g123_sn_pass[468, 600], "g_opt:", g_opt[468, 600])

    elif _cond_N > 1:
        for i in range(0, int(_cond_N/2 + 0.5)):
            # conditions for evidences z
            g_cond_z_opr = math_opr2(bevidences_sort, _cond[i*2][0], _cond[i*2][1], _cond[i*2][2], bf_limit)
            g_cond_z = np.where(g_cond_z_opr & g123_sn_pass)
            g_opt[g_cond_z] += 1

        for i in range(0, int(_cond_N/2)):
            # conditions for g number
            g_cond_n_opr = math_opr1(g_num_sort, _cond[i*2+1][0], _cond[i*2+1][1], _cond[i*2+1][2])
            g_cond_n = np.where(g_cond_n_opr & g123_sn_pass)
            g_opt[g_cond_n] += 1

    g_opt_result_cond_N = np.array([np.where(g_opt > _cond_N-1, g_sorted, -10)]) # -10 flag

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    # CHECK POINT
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    #print("468, 600", g123_sn_pass[468, 600], _cond_N, g_sorted[468, 600], g_opt_result_cond_N[0, 468, 600])

    return g_opt_result_cond_N
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g3_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit):
#________________________________________________________________________#
    #------------------------------------------------------#
    # G1-0
    #------------------------------------------------------#
    
    _cond_N = 1
    _cond1 = ([0, 1, '>'], )
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, 0, _cond1, _cond_N)

    #------------------------------------------------------#
    # G2-0
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 > bf_limit): --> g2
    _cond_N = 3
    _cond3 = ([0, 1, '<'], [0, 1, '>'], [1, 2, '>'])
    g2_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, 1, _cond3, _cond_N)

    #------------------------------------------------------#
    # G3-0
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 > g3): --> g3
    _cond_N = 4
    _cond4 = ([0, 1, '<'], [0, 1, '>'], [1, 2, '<'], [1, 2, '>'])
    g3_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, 2, _cond4, _cond_N)
    
    #------------------------------------------------------#
    # G2-1
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 < g3): --> g2
    _cond_N = 4
    _cond4 = ([0, 1, '<'], [0, 1, '>'], [1, 2, '<'], [1, 2, '<'])
    g2_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, 1, _cond4, _cond_N)
    
    #------------------------------------------------------#
    # G1-1
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 > bf_limit): --> g1
    _cond_N = 3
    _cond3 = ([0, 1, '<'], [0, 1, '<'], [0, 2, '>'])
    g1_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, 0, _cond3, _cond_N)


    #------------------------------------------------------#
    # G3-1
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 > g3): --> g3
    _cond_N = 4
    _cond4 = ([0, 1, '<'], [0, 1, '<'], [0, 2, '<'], [0, 2, '<'])
    g3_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, 2, _cond4, _cond_N)
    
    #------------------------------------------------------#
    # G1-2
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 < g3): --> g1
    _cond_N = 4
    _cond4 = ([0, 1, '<'], [0, 1, '<'], [0, 2, '<'], [0, 2, '<'])
    g1_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, 0, _cond4, _cond_N)

    #print(g1_2)
    g3_opt = g1_0 + g1_1 + g1_2 \
            + g2_0 + g2_1 \
            + g3_0 + g3_1
    #print(g3_opt[0])

    return g3_opt
#-- END OF SUB-ROUTINE____________________________________________________________#







#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g2_opt_bf_snp(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, sn_pass_ng_opt, g01, g02):

    #------------------------------------------------------#
    # 1. sn_pass flag for g1 
    # g1 number of z1
    g1_sorted = g_num_sort[g01, :, :]
    # sn flag of z1
    # take values of sn_pass_ng_opt given g1_sorted indices
    g1_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]
    # g1_sn_pass[j, i] = (0 or 1)
    g1_sn_flag = np.array([np.where(g1_sn_pass_ng_opt_value == (g1_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0
    #------------------------------------------------------#

    #------------------------------------------------------#
    # 2. sn_pass flag for g2 
    # g2 number of z2
    g2_sorted = g_num_sort[g02, :, :]
    # sn flag of z2
    # take values of sn_pass_ng_opt given g2_sorted indices
    g2_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g2_sorted[np.newaxis], axis=0)[0]
    # g2_sn_pass[j, i] = (0 or 1)
    g2_sn_flag = np.array([np.where(g2_sn_pass_ng_opt_value == (g2_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0
    #------------------------------------------------------#

    g1_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0)), 1, 0)])[0]
    g2_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1)), 1, 0)])[0]

    g0_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0)), 1, 0)])[0]
    g12_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1)), 1, 0)])[0]

    #i1 = 90
    #j1 = 213
    i1 = _params['_i0']
    j1 = _params['_j0']
    g12_sn_flag = g1_sn_flag + g2_sn_flag
    print("ng-12: ", "sn_pass: ", np.where(g12_sn_flag == 2))
    print("")
    print("ng-1: ", g1_sorted[j1, i1], "sn_flag: ", g1_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g1_sn_flag[j1, i1])
    print("ng-2: ", g2_sorted[j1, i1], "sn_flag: ", g2_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g2_sn_flag[j1, i1])

#
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
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g2_opt_bf_snp2(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g12_sn_pass, g01, g02): # g01 < g02 : g indices (0, 1, ...)
    #------------------------------------------------------#
    # G1-0
    #------------------------------------------------------#
    # if (z1/z2 > bf_limit): --> g1
    g_sort_n = g01
    _cond_N = 1
    _cond1 = ([g01, g02, '0'], )
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12_sn_pass)


    #------------------------------------------------------#
    # G2-0
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2): --> g2
    g_sort_n = g02
    _cond_N = 2
    _cond2 = ([g01, g02, '1'], [g01, g02, '0'])
    g2_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond2, _cond_N, g12_sn_pass)


    #------------------------------------------------------#
    # G1-1
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2): --> g1
    g_sort_n = g01
    _cond_N = 2
    _cond2 = ([g01, g02, '1'], [g01, g02, '1'])
    g1_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond2, _cond_N, g12_sn_pass)

    g2_opt_snp2 = g1_0 + g1_1 \
            + g2_0 + 10*2 # -10 flag

    return g2_opt_snp2
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g2_opt_bf_snp1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1_sn_pass, g01): # g01 : g indices (0, 1, 2, ...)
#________________________________________________________________________#
    # --> g01
    g_sort_n = g01
    _cond_N = 0
    _cond0 = ([g01, g01, '='], ) # dummy condition
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond0, _cond_N, g1_sn_pass)
    g2_opt_snp1 = g1_0 + 10*0
    #print(g2_opt_snp1[0])

    return g2_opt_snp1
#-- END OF SUB-ROUTINE____________________________________________________________#




#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g2_opt_bf_snp0(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g0_sn_pass):
#________________________________________________________________________#
    #------------------------------------------------------#
    #------------------------------------------------------#
    # --> blank = -1
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_cond_sn = np.where(g0_sn_pass) # s/n < sn_limit
    g_opt[g_cond_sn] += 1

    g0_0 = np.array([np.where(g_opt > 0, -10, -10)])
    g2_opt_snp0 = g0_0[0]
    #print(g2_opt_snp0[0])

    return g2_opt_snp0
#-- END OF SUB-ROUTINE____________________________________________________________#




#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g3_opt_bf_snp_org(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, sn_pass_ng_opt, g01, g02, g03):
    #------------------------------------------------------#
    # 1. sn_pass flag for g1 
    # g1 number of z1
    g1_sorted = g_num_sort[g01, :, :]
    # sn flag of z1
    # take values of sn_pass_ng_opt given g1_sorted indices
    g1_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]
    # g1_sn_pass[j, i] = (0 or 1)
    g1_sn_flag = np.array([np.where(g1_sn_pass_ng_opt_value == (g1_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0
    #------------------------------------------------------#

    #------------------------------------------------------#
    # 2. sn_pass flag for g2 
    # g2 number of z2
    g2_sorted = g_num_sort[g02, :, :]
    # sn flag of z2
    # take values of sn_pass_ng_opt given g2_sorted indices
    g2_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g2_sorted[np.newaxis], axis=0)[0]
    # g2_sn_pass[j, i] = (0 or 1)
    g2_sn_flag = np.array([np.where(g2_sn_pass_ng_opt_value == (g2_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0
    #------------------------------------------------------#

    #------------------------------------------------------#
    # 3. sn_pass flag for g3 
    # g3 number of z3
    g3_sorted = g_num_sort[g03, :, :]
    # sn flag of z3
    # take values of sn_pass_ng_opt given g3_sorted indices
    g3_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g3_sorted[np.newaxis], axis=0)[0]
    # g3_sn_pass[j, i] = (0 or 1)
    g3_sn_flag = np.array([np.where(g3_sn_pass_ng_opt_value == (g3_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0
    #------------------------------------------------------#

    g1_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 0)), 1, 0)])[0]
    g2_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1) & (g3_sn_flag == 0)), 1, 0)])[0]
    g3_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0) & (g3_sn_flag == 1)), 1, 0)])[0]

    g12_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1) & (g3_sn_flag == 0)), 1, 0)])[0]
    g13_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 1)), 1, 0)])[0]
    g23_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1) & (g3_sn_flag == 1)), 1, 0)])[0]

    g0_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0) & (g3_sn_flag == 0)), 1, 0)])[0]
    g123_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1) & (g3_sn_flag == 1)), 1, 0)])[0]

    #i1 = 90
    #j1 = 213
    i1 = _params['_i0']
    j1 = _params['_j0']
    g123_sn_flag = g1_sn_flag + g2_sn_flag + g3_sn_flag
    print("ng-23: ", "sn_pass: ", np.where(g123_sn_flag == 3))
    print("")
    print("ng-1: ", g1_sorted[j1, i1], "sn_flag: ", g1_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g1_sn_flag[j1, i1])
    print("ng-2: ", g2_sorted[j1, i1], "sn_flag: ", g2_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g2_sn_flag[j1, i1])
    print("ng-3: ", g3_sorted[j1, i1], "sn_flag: ", g3_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g3_sn_flag[j1, i1])

# ---------------------------
    _g3_opt_bf_snp_1  = g3_opt_bf_snp1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1_sn_pass, g01)
    _g3_opt_bf_snp_2  = g3_opt_bf_snp1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g2_sn_pass, g02)
    _g3_opt_bf_snp_3  = g3_opt_bf_snp1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g3_sn_pass, g03)

    _g3_opt_bf_snp_12  = g3_opt_bf_snp2(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g12_sn_pass, g01, g02)
    _g3_opt_bf_snp_13  = g3_opt_bf_snp2(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g13_sn_pass, g01, g03)
    _g3_opt_bf_snp_23  = g3_opt_bf_snp2(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g23_sn_pass, g02, g03)

    _g3_opt_bf_snp_0  = g3_opt_bf_snp0(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g0_sn_pass)
    _g3_opt_bf_snp_123 = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g123_sn_pass, g01, g02, g03) # g01 < g02 < g03 : g indices (0, 1, 2, ...)

# ---------------------------

    #print(g23_sn_pass[j1, i1])
    #print(_g3_opt_bf_snp_23[0, j1, i1])

    _g3_opt_t1 = np.array([np.where(_g3_opt_bf_snp_0 > -1, _g3_opt_bf_snp_0, -1)][0])
    _g3_opt_t2 = np.array([np.where(_g3_opt_bf_snp_1 > -1, _g3_opt_bf_snp_1, _g3_opt_t1)][0])
    _g3_opt_t3 = np.array([np.where(_g3_opt_bf_snp_2 > -1, _g3_opt_bf_snp_2, _g3_opt_t2)][0])
    _g3_opt_t4 = np.array([np.where(_g3_opt_bf_snp_3 > -1, _g3_opt_bf_snp_3, _g3_opt_t3)][0])
    _g3_opt_t5 = np.array([np.where(_g3_opt_bf_snp_12 > -1, _g3_opt_bf_snp_12, _g3_opt_t4)][0])
    _g3_opt_t6 = np.array([np.where(_g3_opt_bf_snp_13 > -1, _g3_opt_bf_snp_13, _g3_opt_t5)][0])
    _g3_opt_t7 = np.array([np.where(_g3_opt_bf_snp_23 > -1, _g3_opt_bf_snp_23, _g3_opt_t6)][0])
    _g3_opt_t8 = np.array([np.where(_g3_opt_bf_snp_123 > -1, _g3_opt_bf_snp_123, _g3_opt_t7)][0])
    #print(_g3_opt_t7[0, j1, i1])
    #print(_g3_opt_bf_snp_123[0, j1, i1])
    #print(_g3_opt_t8[0, j1, i1])

    return _g3_opt_t8
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g3_opt_bf_snp_bg(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, sn_pass_ng_opt, g01, g02, g03):

    #------------------------------------------------------#
    # 1. sn_pass flag for g1 
    # g1 number of z1
    g1_sorted = g_num_sort[g01, :, :]
    # sn flag of z1
    # take values of sn_pass_ng_opt given g1_sorted indices
    g1_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]
    # g1_sn_pass[j, i] = (0 or 1)
    g1_sn_flag = np.array([np.where(g1_sn_pass_ng_opt_value == (g1_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0
    #------------------------------------------------------#

    #------------------------------------------------------#
    # 2. sn_pass flag for g2 
    # g2 number of z2
    g2_sorted = g_num_sort[g02, :, :]
    # sn flag of z2
    # take values of sn_pass_ng_opt given g2_sorted indices
    g2_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g2_sorted[np.newaxis], axis=0)[0]
    # g2_sn_pass[j, i] = (0 or 1)
    g2_sn_flag = np.array([np.where(g2_sn_pass_ng_opt_value == (g2_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0
    #------------------------------------------------------#

    #------------------------------------------------------#
    # 3. sn_pass flag for g3 
    # g3 number of z3
    g3_sorted = g_num_sort[g03, :, :]
    # sn flag of z3
    # take values of sn_pass_ng_opt given g3_sorted indices
    g3_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g3_sorted[np.newaxis], axis=0)[0]
    # g3_sn_pass[j, i] = (0 or 1)
    g3_sn_flag = np.array([np.where(g3_sn_pass_ng_opt_value == (g3_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0
    #------------------------------------------------------#

    g1_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 0)), 1, 0)])[0]
    g2_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1) & (g3_sn_flag == 0)), 1, 0)])[0]
    g3_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0) & (g3_sn_flag == 1)), 1, 0)])[0]

    g12_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1) & (g3_sn_flag == 0)), 1, 0)])[0]
    g13_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 1)), 1, 0)])[0]
    g23_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 1) & (g3_sn_flag == 1)), 1, 0)])[0]

    g0_sn_pass = np.array([np.where(((g1_sn_flag == 0) & (g2_sn_flag == 0) & (g3_sn_flag == 0)), 1, 0)])[0]
    g123_sn_pass = np.array([np.where(((g1_sn_flag == 1) & (g2_sn_flag == 1) & (g3_sn_flag == 1)), 1, 0)])[0]

    #i1 = 90
    #j1 = 213
    i1 = _params['_i0']
    j1 = _params['_j0']
    g123_sn_flag = g1_sn_flag + g2_sn_flag + g3_sn_flag
    print("ng-23: ", "sn_pass: ", np.where(g123_sn_flag == 3))
    print("")
    print("ng-1: ", g1_sorted[j1, i1], "sn_flag: ", g1_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g1_sn_flag[j1, i1])
    print("ng-2: ", g2_sorted[j1, i1], "sn_flag: ", g2_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g2_sn_flag[j1, i1])
    print("ng-3: ", g3_sorted[j1, i1], "sn_flag: ", g3_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", g3_sn_flag[j1, i1])

#
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
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g3_opt_bf_snp0(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g0_sn_pass):
#________________________________________________________________________#
    #------------------------------------------------------#
    #------------------------------------------------------#
    # --> blank = -1
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_cond_sn = np.where(g0_sn_pass) # s/n < sn_limit
    g_opt[g_cond_sn] += 1

    g0_0 = np.array([np.where(g_opt > 0, -10, -10)])
    g3_opt_snp0 = g0_0[0]
    #print(g3_opt_snp0[0])

    return g3_opt_snp0
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g3_opt_bf_snp1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1_sn_pass, g01): # g01 : g indices (0, 1, 2, ...)
#________________________________________________________________________#
    # --> g01
    g_sort_n = g01
    _cond_N = 0
    _cond0 = ([g01, g01, '='], ) # dummy condition
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond0, _cond_N, g1_sn_pass)
    g3_opt_snp1 = g1_0 + 10*0
    #print(g3_opt_snp1[0])

    return g3_opt_snp1
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g3_opt_bf_snp2(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g12_sn_pass, g01, g02): # g01 < g02 : g indices (0, 1, 2, ...)
#________________________________________________________________________#

    #------------------------------------------------------#
    # G01-0
    #------------------------------------------------------#
    # if (z1/z2 > bf_limit): --> g01
    g_sort_n = g01
    _cond_N = 1
    _cond1 = ([g01, g02, '0'], )
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12_sn_pass)

    #------------------------------------------------------#
    # G02-0
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2): --> g02
    g_sort_n = g02
    _cond_N = 2
    _cond2 = ([g01, g02, '1'], [g01, g02, '0'])
    g2_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond2, _cond_N, g12_sn_pass)

    #------------------------------------------------------#
    # G01-1
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2): --> g01
    g_sort_n = g01
    _cond_N = 2
    _cond2 = ([g01, g02, '1'], [g01, g02, '1'])
    g1_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond2, _cond_N, g12_sn_pass)
    
    g3_opt_snp2 = g1_0 + g2_0 + g1_1 + 10*2
    #print(g3_opt_snp2[0])

    return g3_opt_snp2
#-- END OF SUB-ROUTINE____________________________________________________________#




#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g3_opt_bf_snp3(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g123_sn_pass, g01, g02, g03): # g01 < g02 < g03 : g indices (0, 1, 2, ...)
#________________________________________________________________________#

    #------------------------------------------------------#
    # G1-0
    #------------------------------------------------------#
    # if (z1/z2 > bf_limit): --> g1
    g_sort_n = g01
    _cond_N = 1
    _cond1 = ([g01, g02, '0'], )
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g123_sn_pass)

    #------------------------------------------------------#
    # G2-0
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 > bf_limit): --> g2
    g_sort_n = g02
    _cond_N = 3
    _cond3 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '0'])
    g2_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond3, _cond_N, g123_sn_pass)

    #------------------------------------------------------#
    # G3-0
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 > g3): --> g3
    g_sort_n = g03
    _cond_N = 4
    _cond4 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '1'], [g02, g03, '0'])
    g3_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond4, _cond_N, g123_sn_pass)
    
    #------------------------------------------------------#
    # G2-1
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 < g3): --> g2
    g_sort_n = g02
    _cond_N = 4
    _cond4 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '1'], [g02, g03, '1'])
    g2_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond4, _cond_N, g123_sn_pass)
    
    #------------------------------------------------------#
    # G1-1
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 > bf_limit): --> g1
    g_sort_n = g01
    _cond_N = 3
    _cond3 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '0'])
    g1_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond3, _cond_N, g123_sn_pass)


    #------------------------------------------------------#
    # G3-1
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 > g3): --> g3
    g_sort_n = g03
    _cond_N = 4
    _cond4 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '1'], [g01, g03, '0'])
    g3_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond4, _cond_N, g123_sn_pass)
    
    #------------------------------------------------------#
    # G1-2
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 < g3): --> g1
    g_sort_n = g01
    _cond_N = 4
    _cond4 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '1'], [g01, g03, '1'])
    g1_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond4, _cond_N, g123_sn_pass)

    #print(g1_2)
    g3_opt_snp3 =  g1_0 + g1_1 + g1_2 \
            + g2_0 + g2_1 \
            + g3_0 + g3_1 + 10*6 # -10 flag see above
    #print(g3_opt_snp3[0])

    return g3_opt_snp3
#-- END OF SUB-ROUTINE____________________________________________________________#




#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def snp0_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g0_sn_pass): #
#________________________________________________________________________#
    #------------------------------------------------------#
    #------------------------------------------------------#
    # --> blank = -1
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_cond_sn = np.where(g0_sn_pass) # s/n < sn_limit
    g_opt[g_cond_sn] += 1

    g0_0 = np.array([np.where(g_opt > 0, -10, -10)])
    g0_opt_snp0 = g0_0[0]
    #print(g0_opt_snp0[0])

    return g0_opt_snp0
#-- END OF SUB-ROUTINE____________________________________________________________#




#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def snp1_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1_sn_pass, g01): # g01: g indices (0, 1, 2, ...)
#________________________________________________________________________#
    # --> g01
    g_sort_n = g01
    _cond_N = 0
    _cond0 = ([g01, g01, '='], ) # dummy condition
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond0, _cond_N, g1_sn_pass)
    g1_opt_snp1 = g1_0 + 10*0
    #print(g1_opt_snp1[0])

    return g1_opt_snp1
#-- END OF SUB-ROUTINE____________________________________________________________#





#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def snp2_tree_opt_bf_bg(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g12_sn_pass, g01, g02): # g01 < g02: g indices (0, 1, 2, ...)
    # ----------------------------------------
    # ___g2_snp2 
    # ----------------------------------------
    # ----------------------------------------
    # G1-X
    # # e.g., 1-0, 1-1, 1-2
    # generate the condition tree
    print("")
    print("")
    g1x_t = set_cond_tree(1, 1, 2, 1) # [1]
    g1x = cp.deepcopy(g1x_t)

    print("1, 1, 2, 1")
    print(g1x[0])
    print(g1x[1])
    print("")
    print("")

    # n_G1-X
    n_G1x = 2 # from the condition matrix

    g_sort_n = g01 # for g01
    g1x_filtered = np.zeros((n_G1x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g1x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G1x):
        _cond_N = len(g1x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            # update the array with the current set: e.g., (1, 2) --> (2, 3), (3, 4), (4, 5) etc.
            # this is for the max_ngauss > 2  || base gauss numbers: g01=0, g02=1
            if g1x_t[i][j][0] == 0:
                g1x[i][j][0] = g1x_t[i][j][0] + (g01 - 0)
            if g1x_t[i][j][0] == 1:
                g1x[i][j][0] = g1x_t[i][j][0] + (g02 - 1)

            if g1x_t[i][j][1] == 0:
                g1x[i][j][1] = g1x_t[i][j][1] + (g01 - 0)
            if g1x_t[i][j][1] == 1:
                g1x[i][j][1] = g1x_t[i][j][1] + (g02 - 1)

            _cond1_list.append(g1x[i][j])

        # convert to tuple
        _cond1 = tuple(_cond1_list)
        g1x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12_sn_pass)
        g1x_filtered_sum[:, :] += g1x_filtered[i, :, :]

    print("after update")
    print(g1x[0])
    print(g1x[1])
    print("")
    print("")
    
    #print(np.where(g1x_filtered[0, :, :]>0))
    #print(g1x_filtered[1, :, :])
    #print(g1x_filtered[2, :, :])

    # ----------------------------------------
    # ----------------------------------------
    # G2-X
    # # e.g., 2-0, 2-1
    # generate the condition tree
    g2x_t = set_cond_tree(2, 2, 2, 0) # [0]
    g2x = cp.deepcopy(g2x_t)

    print("2, 2, 2, 0")
    print(g2x[0])
    print("")
    print("")

    # n_G2-X
    n_G2x = 1 # from the condition matrix
    g_sort_n = g02 # for g02
    g2x_filtered = np.zeros((n_G2x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g2x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G2x):
        _cond_N = len(g2x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            # update the array with the current set: e.g., (1, 2) --> (2, 3), (3, 4), (4, 5) etc.
            # this is for the max_ngauss > 2  || base gauss numbers: g01=0, g02=1
            if g2x_t[i][j][0] == 0:
                g2x[i][j][0] = g2x_t[i][j][0] + (g01 - 0)
            if g2x_t[i][j][0] == 1:
                g2x[i][j][0] = g2x_t[i][j][0] + (g02 - 1)

            if g2x_t[i][j][1] == 0:
                g2x[i][j][1] = g2x_t[i][j][1] + (g01 - 0)
            if g2x_t[i][j][1] == 1:
                g2x[i][j][1] = g2x_t[i][j][1] + (g02 - 1)

            _cond1_list.append(g2x[i][j])

        # convert to tuple
        _cond1 = tuple(_cond1_list)
        g2x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12_sn_pass)
        g2x_filtered_sum[:, :] += g2x_filtered[i, :, :]
    
    #print(g2x_filtered[0, :, :])
    #print(g2x_filtered[1, :, :])
    #print(g2x_filtered[2, :, :])

    print("after update")
    print(g2x[0])
    print("")
    print("")


    # ----------------------------------------
    # ----------------------------------------
    # combine the g1x and g2x results and return + -10 flag
    g2_opt_snp2 = g1x_filtered_sum + g2x_filtered_sum + 10*(n_G1x + n_G2x - 1)

    return g2_opt_snp2
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g12_sn_pass, g01, g02): # g01 < g02: g indices (0, 1, 2, ...)
    # ----------------------------------------
    # ___g2_snp2 
    # ----------------------------------------
    # ----------------------------------------
    # G1-X
    # # e.g., 1-0, 1-1, 1-2
    # generate the condition tree
    print("")
    print("")
    g1x_t = set_cond_tree(1, 1, 2, 1) # [1]
    g1x = cp.deepcopy(g1x_t)

    #print("1, 1, 2, 1")
    #print(g1x[0])
    #print(g1x[1])
    #print("")
    #print("")

    # n_G1-X
    n_G1x = 2 # from the condition matrix

    g_sort_n = g01 # for g01
    g1x_filtered = np.zeros((n_G1x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g1x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G1x):
        _cond_N = len(g1x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            # update the array with the current set: e.g., (1, 2) --> (2, 3), (3, 4), (4, 5) etc.
            # this is for the max_ngauss > 2  || base gauss numbers: g01=0, g02=1
            if g1x_t[i][j][0] == 0:
                g1x[i][j][0] = g1x_t[i][j][0] + (g01 - 0)
            if g1x_t[i][j][0] == 1:
                g1x[i][j][0] = g1x_t[i][j][0] + (g02 - 1)

            if g1x_t[i][j][1] == 0:
                g1x[i][j][1] = g1x_t[i][j][1] + (g01 - 0)
            if g1x_t[i][j][1] == 1:
                g1x[i][j][1] = g1x_t[i][j][1] + (g02 - 1)

            _cond1_list.append(g1x[i][j])

        # convert to tuple
        _cond1 = tuple(_cond1_list)
        g1x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12_sn_pass)
        g1x_filtered_sum[:, :] += g1x_filtered[i, :, :]

    #print("after update")
    #print(g1x[0])
    #print(g1x[1])
    #print("")
    #print("")
    
    #print(np.where(g1x_filtered[0, :, :]>0))
    #print(g1x_filtered[1, :, :])
    #print(g1x_filtered[2, :, :])

    # ----------------------------------------
    # ----------------------------------------
    # G2-X
    # # e.g., 2-0, 2-1
    # generate the condition tree
    g2x_t = set_cond_tree(2, 2, 2, 0) # [0]
    g2x = cp.deepcopy(g2x_t)

    #print("2, 2, 2, 0")
    #print(g2x[0])
    #print("")
    #print("")

    # n_G2-X
    n_G2x = 1 # from the condition matrix
    g_sort_n = g02 # for g02
    g2x_filtered = np.zeros((n_G2x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g2x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G2x):
        _cond_N = len(g2x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            # update the array with the current set: e.g., (1, 2) --> (2, 3), (3, 4), (4, 5) etc.
            # this is for the max_ngauss > 2  || base gauss numbers: g01=0, g02=1
            if g2x_t[i][j][0] == 0:
                g2x[i][j][0] = g2x_t[i][j][0] + (g01 - 0)
            if g2x_t[i][j][0] == 1:
                g2x[i][j][0] = g2x_t[i][j][0] + (g02 - 1)

            if g2x_t[i][j][1] == 0:
                g2x[i][j][1] = g2x_t[i][j][1] + (g01 - 0)
            if g2x_t[i][j][1] == 1:
                g2x[i][j][1] = g2x_t[i][j][1] + (g02 - 1)

            _cond1_list.append(g2x[i][j])

        # convert to tuple
        _cond1 = tuple(_cond1_list)
        g2x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12_sn_pass)
        g2x_filtered_sum[:, :] += g2x_filtered[i, :, :]
    
    #print(g2x_filtered[0, :, :])
    #print(g2x_filtered[1, :, :])
    #print(g2x_filtered[2, :, :])

    print("after update")
    print(g2x[0])
    print("")
    print("")


    # ----------------------------------------
    # ----------------------------------------
    # combine the g1x and g2x results and return + -10 flag
    g2_opt_snp2 = g1x_filtered_sum + g2x_filtered_sum + 10*(n_G1x + n_G2x - 1)

    return g2_opt_snp2
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g123_sn_pass, g01, g02, g03): # g01 < g02 < g03 : g indices (0, 1, 2, ...)
    # ----------------------------------------
    # ___g3_snp3 
    # ----------------------------------------
    # ----------------------------------------
    # G1-X
    # # e.g., 1-0, 1-1, 1-2
    # generate the condition tree
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

    # n_G1-X
    n_G1x = 3 # from the condition matrix

    g_sort_n = g01 # for g01
    g1x_filtered = np.zeros((n_G1x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g1x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G1x):
        _cond_N = len(g1x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            # update the array with the current set: e.g., (1, 2, 3) --> (1, 2, 4), (1, 3, 4), (2, 3, 4) etc.
            # this is for the max_ngauss > 3 
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

        # convert to tuple
        _cond1 = tuple(_cond1_list)
        g1x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g123_sn_pass)
        g1x_filtered_sum[:, :] += g1x_filtered[i, :, :]

    print("after update")
    print(g1x[0])
    print(g1x[1])
    print(g1x[2])
    print("")
    print("")
    
    #print(np.where(g1x_filtered[0, :, :]>0))
    #print(g1x_filtered[1, :, :])
    #print(g1x_filtered[2, :, :])

    # ----------------------------------------
    # ----------------------------------------
    # G2-X
    # # e.g., 2-0, 2-1
    # generate the condition tree
    g2x_t = set_cond_tree(2, 2, 3, 1) # [1]
    g2x = cp.deepcopy(g2x_t)

    print("2, 2, 3, 1")
    print(g2x[0])
    print(g2x[1])
    print("")
    print("")



    # n_G2-X
    n_G2x = 2 # from the condition matrix
    g_sort_n = g02 # for g02
    g2x_filtered = np.zeros((n_G2x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g2x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G2x):
        _cond_N = len(g2x[i])
        _cond1_list = []
        for j in range(0, _cond_N):

            # update the array with the current set: e.g., (1, 2, 3) --> (1, 2, 4), (1, 3, 4), (2, 3, 4) etc.
            # this is for the max_ngauss > 3 
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

        # convert to tuple
        _cond1 = tuple(_cond1_list)
        g2x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g123_sn_pass)
        g2x_filtered_sum[:, :] += g2x_filtered[i, :, :]
    
    #print(g2x_filtered[0, :, :])
    #print(g2x_filtered[1, :, :])
    #print(g2x_filtered[2, :, :])

    print("after update")
    print(g2x[0])
    print(g2x[1])
    print("")
    print("")

    # ----------------------------------------
    # ----------------------------------------
    # G3-X
    # # e.g., 3-0, 3-1
    # generate the condition tree
    g3x_t = set_cond_tree(3, 3, 3, 0) # [0]
    g3x = cp.deepcopy(g3x_t)

    print("3, 3, 3, 0")
    print(g3x[0])
    print(g3x[1])
    print("seheon")
    # n_G3-X
    n_G3x = 2 # from the condition matrix
    g_sort_n = g03 # for g03
    g3x_filtered = np.zeros((n_G3x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g3x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G3x):
        _cond_N = len(g3x[i])
        _cond1_list = []
        for j in range(0, _cond_N):

            # update the array with the current set: e.g., (1, 2, 3) --> (1, 2, 4), (1, 3, 4), (2, 3, 4) etc.
            # this is for the max_ngauss > 3 
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

        # convert to tuple
        _cond1 = tuple(_cond1_list)
        g3x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g123_sn_pass)
        g3x_filtered_sum[0, :, :] += g3x_filtered[i, :, :]
    
    #print(g3x_filtered[0, :, :])
    #print(g3x_filtered[1, :, :])
    #print(g3x_filtered[2, :, :])

    print("after update")
    print(g3x[0])
    print(g3x[1])
    print("seheon")

    # ----------------------------------------
    # ----------------------------------------
    # combine the g1x, g2x, and g3x results and return + -10 flag
    g3_opt_snp3 = g1x_filtered_sum + g2x_filtered_sum + g3x_filtered_sum + 10*(n_G1x + n_G2x + n_G3x - 1)

    #print(np.where(g3_opt_snp3==2))
    return g3_opt_snp3
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def snp4_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1234_sn_pass, g01, g02, g03, g04): # g01 < g02 < g03 : g indices (0, 1, 2, ...)
    # ----------------------------------------
    # ___g4_snp3 
    # ----------------------------------------
    # ----------------------------------------
    # G1-X
    # # e.g., 1-0, 1-1, 1-2
    # generate the condition tree
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

    # n_G1-X
    n_G1x = 4 # from the condition matrix

    g_sort_n = g01 # for g01
    g1x_filtered = np.zeros((n_G1x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g1x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G1x):
        _cond_N = len(g1x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            # update the array with the current set: e.g., (1, 2, 3, 4) --> (1, 2, 3, 5), (1, 3, 4, 5), (2, 3, 4, 5) etc.
            # this is for the max_ngauss > 4  || base gauss numbers: g01=0, g02=1, g03=2, g04=3
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

        # convert to tuple
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
    
    #print(np.where(g1x_filtered[0, :, :]>0))
    #print(g1x_filtered[1, :, :])
    #print(g1x_filtered[2, :, :])

    # ----------------------------------------
    # ----------------------------------------
    # G2-X
    # # e.g., 2-0, 2-1
    # generate the condition tree
    g2x_t = set_cond_tree(2, 2, 4, 2) # [2]
    g2x = cp.deepcopy(g2x_t)

    print("2, 2, 4, 2")
    print(g2x[0])
    print(g2x[1])
    print(g2x[2])
    print("")
    print("")

    # n_G2-X
    n_G2x = 3 # from the condition matrix
    g_sort_n = g02 # for g02
    g2x_filtered = np.zeros((n_G2x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g2x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G2x):
        _cond_N = len(g2x[i])
        _cond1_list = []
        for j in range(0, _cond_N):

            # update the array with the current set: e.g., (1, 2, 3, 4) --> (1, 2, 3, 5), (1, 3, 4, 5), (2, 3, 4, 5) etc.
            # this is for the max_ngauss > 4  || base gauss numbers: g01=0, g02=1, g03=2, g04=3
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

        # convert to tuple
        _cond1 = tuple(_cond1_list)
        g2x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g1234_sn_pass)
        g2x_filtered_sum[:, :] += g2x_filtered[i, :, :]
    
    #print(g2x_filtered[0, :, :])
    #print(g2x_filtered[1, :, :])
    #print(g2x_filtered[2, :, :])

    print("after update")
    print(g2x[0])
    print(g2x[1])
    print(g2x[2])
    print("")
    print("")

    # ----------------------------------------
    # ----------------------------------------
    # G3-X
    # # e.g., 3-0, 3-1
    # generate the condition tree
    g3x_t = set_cond_tree(3, 3, 4, 1) # [1]
    g3x = cp.deepcopy(g3x_t)

    print("3, 3, 4, 1")
    print(g3x[0])
    print(g3x[1])
    print(g3x[2])
    print(g3x[3])
    # n_G3-X
    n_G3x = 4 # from the condition matrix
    g_sort_n = g03 # for g03
    g3x_filtered = np.zeros((n_G3x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g3x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G3x):
        _cond_N = len(g3x[i])
        _cond1_list = []
        for j in range(0, _cond_N):

            # update the array with the current set: e.g., (1, 2, 3, 4) --> (1, 2, 3, 5), (1, 3, 4, 5), (2, 3, 4, 5) etc.
            # this is for the max_ngauss > 4  || base gauss numbers: g01=0, g02=1, g03=2, g04=3
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

        # convert to tuple
        _cond1 = tuple(_cond1_list)
        g3x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g1234_sn_pass)
        g3x_filtered_sum[0, :, :] += g3x_filtered[i, :, :]
    
    #print(g3x_filtered[0, :, :])
    #print(g3x_filtered[1, :, :])
    #print(g3x_filtered[2, :, :])

    print("after update")
    print(g3x[0])
    print(g3x[1])
    print(g3x[2])
    print(g3x[3])


    # ----------------------------------------
    # ----------------------------------------
    # G4-X
    # # e.g., 4-0, 4-1
    # generate the condition tree
    g4x_t = set_cond_tree(4, 4, 4, 0) # [0]
    g4x = cp.deepcopy(g4x_t)

    print("4, 4, 4, 0")
    print(g4x[0])
    print(g4x[1])
    print(g4x[2])
    print(g4x[3])
    # n_G4-X
    n_G4x = 4 # from the condition matrix
    g_sort_n = g04 # for g04
    g4x_filtered = np.zeros((n_G4x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g4x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G4x):
        _cond_N = len(g4x[i])
        _cond1_list = []
        for j in range(0, _cond_N):

            # update the array with the current set: e.g., (1, 2, 3, 4) --> (1, 2, 3, 5), (1, 3, 4, 5), (2, 3, 4, 5) etc.
            # this is for the max_ngauss > 4  || base gauss numbers: g01=0, g02=1, g03=2, g04=3
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

        # convert to tuple
        _cond1 = tuple(_cond1_list)
        g4x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g1234_sn_pass)
        g4x_filtered_sum[0, :, :] += g4x_filtered[i, :, :]
    
    #print(g4x_filtered[0, :, :])
    #print(g4x_filtered[1, :, :])
    #print(g4x_filtered[2, :, :])

    print("after update")
    print(g4x[0])
    print(g4x[1])
    print(g4x[2])
    print(g4x[3])

    # ----------------------------------------
    # ----------------------------------------
    # combine the g1x, g2x, g3x, and g4x results and return + -10 flag
    g4_opt_snp4 = g1x_filtered_sum + g2x_filtered_sum + g3x_filtered_sum + g4x_filtered_sum + 10*(n_G1x + n_G2x + n_G3x + n_G4x - 1)

    #print(np.where(g3_opt_snp3==2))
    return g4_opt_snp4
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def snp_gx_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, g1234_sn_pass, gx_list, n_GNx_list): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    # max_ngauss = 1
    # gx_list = [g01] # =< max_ngauss
    # n_GNx_list = [1] # from the condition matrix

    # max_ngauss = 2
    # gx_list = [g01, g02] # =< max_ngauss
    # n_GNx_list = [2, 1] # from the condition matrix

    # max_ngauss = 3
    # gx_list = [g01, g02, g03] # =< max_ngauss
    # n_GNx_list = [3, 2, 2] # from the condition matrix

    # max_ngauss = 4
    # gx_list = [g01, g02, g03, g04] # =< max_ngauss
    # n_GNx_list = [4, 3, 4, 4] # from the condition matrix

    # max_ngauss = 5
    # gx_list = [g01, g02, g03, g04, g05] # =< max_ngauss
    # n_GNx_list = [5, 4, 6, 8, 8] # from the condition matrix

    # ...
    # ...
    # ...

    # ----------------------------------------
    # ___gX_snp
    # ----------------------------------------
    # ----------------------------------------
    gNx_t = [0 for i in range(len(gx_list))]
    gNx =   [0 for i in range(len(gx_list))]

    # ----------------------------------------
    gNx_filtered_sum = np.zeros((len(gx_list), _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    gNx_filtered_sum_all = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    n_GNx_sum = 0
    for g in range(0, len(gx_list)):
        # ----------------------------------------
        # n_G1-X : refer to the condition matrix
        n_GNx = n_GNx_list[g]
        n_GNx_sum += n_GNx
        g_sorted_n = gx_list[g] # for g0X
        #print("g_sorted_n", g_sorted_n)

        # ----------------------------------------
        # GN-X
        # # e.g., 1-0, 1-1, 1-2
        # generate the condition tree
        #g1x_t = set_cond_tree(1, 1, 4, 3) # [3]
        gNx_t[g] = set_cond_tree(g+1, g+1, max_ngauss, max_ngauss-g-1) # [max_ngauss-g-1]
        gNx = cp.deepcopy(gNx_t)
    
        gNx_filtered = np.zeros((n_GNx, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
        gNx_filtered_sum_t = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

        # ----------------------------------------
        for i in range(0, n_GNx):
            _cond_N = len(gNx[g][i])
            #print("CHECK HERE", _cond_N, gNx[g][i])
            _cond1_list = []
            for j in range(0, _cond_N):
                # ----------------------------------------
                # update the array with the current set: e.g., (1, 2, 3, 4) --> (1, 2, 3, 5), (1, 3, 4, 5), (2, 3, 4, 5) etc.
                # this is for the max_ngauss > 4  || base gauss numbers: g01=0, g02=1, g03=2, g04=3
                #for k in range(0, max_ngauss):
                for k in range(0, len(gx_list)):
                    if gNx_t[g][i][j][0] == k:
                        gNx[g][i][j][0] = gNx_t[g][i][j][0] + (gx_list[k] - k)
    
                    if gNx_t[g][i][j][1] == k:
                        gNx[g][i][j][1] = gNx_t[g][i][j][1] + (gx_list[k] - k)
    
                _cond1_list.append(gNx[g][i][j])


            # ----------------------------------------
            # convert to tuple
            _cond1 = tuple(_cond1_list)
            #print("_con1", _cond1)
            gNx_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sorted_n, _cond1, _cond_N, g1234_sn_pass)
            gNx_filtered_sum_t[:, :] += gNx_filtered[i, :, :]

        # ----------------------------------------
        gNx_filtered_sum[g, :, :] = gNx_filtered_sum_t[:, :]

    # ----------------------------------------
    # ----------------------------------------
    # combine the g1x, g2x, g3x, and g4x results and return + -10 flag
    for g in range(0, len(gx_list)):
        gNx_filtered_sum_all[:, :] += gNx_filtered_sum[g, :, :]

    gNx_filtered_sum_all[:, :] += 10*(n_GNx_sum - 1) # -10 flag

    #print(np.where(g4_opt_snp4==2))
    # ----------------------------------------
    return gNx_filtered_sum_all[:, :]
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def snp_gx_tree_opt_bf1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, g1234_sn_pass, g01, g02, g03, g04): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    gx_list = [g01, g02, g03, g04]
    n_G1x = 4
    n_G2x = 3
    n_G3x = 4
    n_G4x = 4

    # ----------------------------------------
    # ___g4_snp3 
    # ----------------------------------------
    # ----------------------------------------
    # G1-X
    # # e.g., 1-0, 1-1, 1-2
    # generate the condition tree
    #g1x_t = set_cond_tree(1, 1, 4, 3) # [3]
    g1x_t = set_cond_tree(1, 1, 4, 3) # [n_gx-1]
    g1x = cp.deepcopy(g1x_t)

    # n_G1-X : refer to the condition matrix
    n_G1x = 4 # from the condition matrix

    g_sort_n = g01 # for g01
    g1x_filtered = np.zeros((n_G1x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g1x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G1x):
        _cond_N = len(g1x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            # update the array with the current set: e.g., (1, 2, 3, 4) --> (1, 2, 3, 5), (1, 3, 4, 5), (2, 3, 4, 5) etc.
            # this is for the max_ngauss > 4  || base gauss numbers: g01=0, g02=1, g03=2, g04=3
            for k in range(0, max_ngauss):
                if g1x_t[i][j][0] == k:
                    g1x[i][j][0] = g1x_t[i][j][0] + (gx_list[k] - k)

                if g1x_t[i][j][1] == k:
                    g1x[i][j][1] = g1x_t[i][j][1] + (gx_list[k] - k)

            _cond1_list.append(g1x[i][j])

        # convert to tuple
        _cond1 = tuple(_cond1_list)
        g1x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g1234_sn_pass)
        g1x_filtered_sum[:, :] += g1x_filtered[i, :, :]


    # ----------------------------------------
    # ----------------------------------------
    # G2-X
    # # e.g., 2-0, 2-1
    # generate the condition tree
    g2x_t = set_cond_tree(2, 2, 4, 2) # [2]
    g2x = cp.deepcopy(g2x_t)

    # n_G2-X
    n_G2x = 3 # from the condition matrix
    g_sort_n = g02 # for g02
    g2x_filtered = np.zeros((n_G2x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g2x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G2x):
        _cond_N = len(g2x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            # update the array with the current set: e.g., (1, 2, 3, 4) --> (1, 2, 3, 5), (1, 3, 4, 5), (2, 3, 4, 5) etc.
            # this is for the max_ngauss > 4  || base gauss numbers: g01=0, g02=1, g03=2, g04=3
            for k in range(0, max_ngauss):
                if g2x_t[i][j][0] == k:
                    g2x[i][j][0] = g2x_t[i][j][0] + (gx_list[k] - k)

                if g2x_t[i][j][1] == k:
                    g2x[i][j][1] = g2x_t[i][j][1] + (gx_list[k] - k)

            _cond1_list.append(g2x[i][j])

        # convert to tuple
        _cond1 = tuple(_cond1_list)
        g2x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g1234_sn_pass)
        g2x_filtered_sum[:, :] += g2x_filtered[i, :, :]
    
    # ----------------------------------------
    # ----------------------------------------
    # G3-X
    # # e.g., 3-0, 3-1
    # generate the condition tree
    g3x_t = set_cond_tree(3, 3, 4, 1) # [1]
    g3x = cp.deepcopy(g3x_t)

    # n_G3-X
    n_G3x = 4 # from the condition matrix
    g_sort_n = g03 # for g03
    g3x_filtered = np.zeros((n_G3x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g3x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G3x):
        _cond_N = len(g3x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            # update the array with the current set: e.g., (1, 2, 3, 4) --> (1, 2, 3, 5), (1, 3, 4, 5), (2, 3, 4, 5) etc.
            # this is for the max_ngauss > 4  || base gauss numbers: g01=0, g02=1, g03=2, g04=3
            for k in range(0, max_ngauss):
                if g3x_t[i][j][0] == k:
                    g3x[i][j][0] = g3x_t[i][j][0] + (gx_list[k] - k)

                if g3x_t[i][j][1] == k:
                    g3x[i][j][1] = g3x_t[i][j][1] + (gx_list[k] - k)

            _cond1_list.append(g3x[i][j])

        # convert to tuple
        _cond1 = tuple(_cond1_list)
        g3x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g1234_sn_pass)
        g3x_filtered_sum[0, :, :] += g3x_filtered[i, :, :]
    
    # ----------------------------------------
    # ----------------------------------------
    # G4-X
    # # e.g., 4-0, 4-1
    # generate the condition tree
    g4x_t = set_cond_tree(4, 4, 4, 0) # [0]
    g4x = cp.deepcopy(g4x_t)

    # n_G4-X
    n_G4x = 4 # from the condition matrix
    g_sort_n = g04 # for g04
    g4x_filtered = np.zeros((n_G4x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g4x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G4x):
        _cond_N = len(g4x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            # update the array with the current set: e.g., (1, 2, 3, 4) --> (1, 2, 3, 5), (1, 3, 4, 5), (2, 3, 4, 5) etc.
            # this is for the max_ngauss > 4  || base gauss numbers: g01=0, g02=1, g03=2, g04=3
            for k in range(0, max_ngauss):
                if g4x_t[i][j][0] == k:
                    g4x[i][j][0] = g4x_t[i][j][0] + (gx_list[k] - k)

                if g4x_t[i][j][1] == k:
                    g4x[i][j][1] = g4x_t[i][j][1] + (gx_list[k] - k)

            _cond1_list.append(g4x[i][j])

        # convert to tuple
        _cond1 = tuple(_cond1_list)
        g4x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g1234_sn_pass)
        g4x_filtered_sum[0, :, :] += g4x_filtered[i, :, :]
    
    # ----------------------------------------
    # ----------------------------------------
    # combine the g1x, g2x, g3x, and g4x results and return + -10 flag
    g4_opt_snp4 = g1x_filtered_sum + g2x_filtered_sum + g3x_filtered_sum + g4x_filtered_sum + 10*(n_G1x + n_G2x + n_G3x + n_G4x - 1)

    #print(np.where(g4_opt_snp4==2))
    return g4_opt_snp4
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def snp5_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g12345_sn_pass, g01, g02, g03, g04, g05): # g01 < g02 < g03 : g indices (0, 1, 2, ...)
    # ----------------------------------------
    # ___g5_snp5 
    # ----------------------------------------
    # ----------------------------------------
    # G1-X
    # # e.g., 1-0, 1-1, 1-2
    # generate the condition tree
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

    # n_G1-X
    n_G1x = 5 # from the condition matrix

    g_sort_n = g01 # for g01
    g1x_filtered = np.zeros((n_G1x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g1x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G1x):
        _cond_N = len(g1x[i])
        _cond1_list = []
        for j in range(0, _cond_N):
            # update the array with the current set: e.g., (1, 2, 3, 4, 5) --> (1, 2, 3, 5, 6), (1, 3, 4, 5, 6), (2, 3, 4, 5, 6) + (1, 2, 4, 5, 6??)etc.
            # this is for the max_ngauss > 5  || base gauss numbers: g01=0, g02=1, g03=2, g04=3
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

        # convert to tuple
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
    
    #print(np.where(g1x_filtered[0, :, :]>0))
    #print(g1x_filtered[1, :, :])
    #print(g1x_filtered[2, :, :])

    # ----------------------------------------
    # ----------------------------------------
    # G2-X
    # # e.g., 2-0, 2-1
    # generate the condition tree
    g2x_t = set_cond_tree(2, 2, 5, 3) # [3]
    g2x = cp.deepcopy(g2x_t)

    print("2, 2, 5, 3")
    print(g2x[0])
    print(g2x[1])
    print(g2x[2])
    print(g2x[3])
    print("")
    print("")

    # n_G2-X
    n_G2x = 4 # from the condition matrix
    g_sort_n = g02 # for g02
    g2x_filtered = np.zeros((n_G2x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g2x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G2x):
        _cond_N = len(g2x[i])
        _cond1_list = []
        for j in range(0, _cond_N):

            # update the array with the current set: e.g., (1, 2, 3, 4, 5) --> (1, 2, 3, 5, 6), (1, 3, 4, 5, 6), (2, 3, 4, 5, 6) etc.
            # this is for the max_ngauss > 5  || base gauss numbers: g01=0, g02=1, g03=2, g04=3
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

        # convert to tuple
        _cond1 = tuple(_cond1_list)
        g2x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12345_sn_pass)
        g2x_filtered_sum[:, :] += g2x_filtered[i, :, :]
    
    #print(g2x_filtered[0, :, :])
    #print(g2x_filtered[1, :, :])
    #print(g2x_filtered[2, :, :])

    print("after update")
    print(g2x[0])
    print(g2x[1])
    print(g2x[2])
    print(g2x[3])
    print("")
    print("")

    # ----------------------------------------
    # ----------------------------------------
    # G3-X
    # # e.g., 3-0, 3-1
    # generate the condition tree
    g3x_t = set_cond_tree(3, 3, 5, 2) # [2]
    g3x = cp.deepcopy(g3x_t)

    print("3, 3, 5, 2")
    print(g3x[0])
    print(g3x[1])
    print(g3x[2])
    print(g3x[3])
    print(g3x[4])
    print(g3x[5])
    # n_G3-X
    n_G3x = 6 # from the condition matrix
    g_sort_n = g03 # for g03
    g3x_filtered = np.zeros((n_G3x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g3x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G3x):
        _cond_N = len(g3x[i])
        _cond1_list = []
        for j in range(0, _cond_N):

            # update the array with the current set: e.g., (1, 2, 3, 4, 5) --> (1, 2, 3, 5, 6), (1, 3, 4, 5, 6), (2, 3, 4, 5, 6) etc.
            # this is for the max_ngauss > 5  || base gauss numbers: g01=0, g02=1, g03=2, g04=3
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

        # convert to tuple
        _cond1 = tuple(_cond1_list)
        g3x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12345_sn_pass)
        g3x_filtered_sum[0, :, :] += g3x_filtered[i, :, :]
    
    #print(g3x_filtered[0, :, :])
    #print(g3x_filtered[1, :, :])
    #print(g3x_filtered[2, :, :])

    print("after update")
    print(g3x[0])
    print(g3x[1])
    print(g3x[2])
    print(g3x[3])
    print(g3x[4])
    print(g3x[5])


    # ----------------------------------------
    # ----------------------------------------
    # G4-X
    # # e.g., 4-0, 4-1
    # generate the condition tree
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
    # n_G4-X
    n_G4x = 8 # from the condition matrix
    g_sort_n = g04 # for g04
    g4x_filtered = np.zeros((n_G4x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g4x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G4x):
        _cond_N = len(g4x[i])
        _cond1_list = []
        for j in range(0, _cond_N):

            # update the array with the current set: e.g., (1, 2, 3, 4, 5) --> (1, 2, 3, 5, 6), (1, 3, 4, 5, 6), (2, 3, 4, 5, 6) etc.
            # this is for the max_ngauss > 5  || base gauss numbers: g01=0, g02=1, g03=2, g04=3
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

        # convert to tuple
        _cond1 = tuple(_cond1_list)
        g4x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12345_sn_pass)
        g4x_filtered_sum[0, :, :] += g4x_filtered[i, :, :]
    
    #print(g4x_filtered[0, :, :])
    #print(g4x_filtered[1, :, :])
    #print(g4x_filtered[2, :, :])

    print("after update")
    print(g4x[0])
    print(g4x[1])
    print(g4x[2])
    print(g4x[3])
    print(g4x[4])
    print(g4x[5])
    print(g4x[6])
    print(g4x[7])


    # ----------------------------------------
    # ----------------------------------------
    # G5-X
    # # e.g., 5-0, 5-1
    # generate the condition tree
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

    # n_G5-X
    n_G5x = 8 # from the condition matrix
    g_sort_n = g05 # for g05
    g5x_filtered = np.zeros((n_G5x, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g5x_filtered_sum = np.zeros((1, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, n_G5x):
        _cond_N = len(g5x[i])
        _cond1_list = []
        for j in range(0, _cond_N):

            # update the array with the current set: e.g., (1, 2, 3, 4, 5) --> (1, 2, 3, 5, 6), (1, 3, 4, 5, 6), (2, 3, 4, 5, 6) etc.
            # this is for the max_ngauss > 5  || base gauss numbers: g01=0, g02=1, g03=2, g04=3
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

        # convert to tuple
        _cond1 = tuple(_cond1_list)
        g5x_filtered[i, :, :] = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12345_sn_pass)
        g5x_filtered_sum[0, :, :] += g5x_filtered[i, :, :]
    
    #print(g5x_filtered[0, :, :])
    #print(g5x_filtered[1, :, :])
    #print(g5x_filtered[2, :, :])

    print("after update")
    print(g5x[0])
    print(g5x[1])
    print(g5x[2])
    print(g5x[3])
    print(g5x[4])
    print(g5x[5])
    print(g5x[6])
    print(g5x[7])

    # ----------------------------------------
    # ----------------------------------------
    # combine the g1x, g2x, g3x, and g4x results and return + -10 flag
    g5_opt_snp5 = g1x_filtered_sum + g2x_filtered_sum + g3x_filtered_sum + g4x_filtered_sum + g5x_filtered_sum + 10*(n_G1x + n_G2x + n_G3x + n_G4x + n_G5x - 1)

    return g5_opt_snp5
#-- END OF SUB-ROUTINE____________________________________________________________#




#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g3_opt_bf_snp(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, sn_pass_ng_opt, g01, g02, g03):

    gx_sn_flag = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    for i in range(0, max_ngauss):
        # 1. g1 number of z1
        gx_sorted = g_num_sort[i, :, :]
        # 2. sn flag of z1
        # 3. take values of sn_pass_ng_opt given g1_sorted indices
        gx_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, gx_sorted[np.newaxis], axis=0)[0]
        # 4. gx_sn_flag[j, i] = (0 or 1)
        gx_sn_flag[i, :, :] = np.array([np.where(gx_sn_pass_ng_opt_value == (gx_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

        print(gx_sn_flag[i, :, :])
    #------------------------------------------------------#
    gx_list = [g01, g02, g03]

    #------------------------------------------------------#
    # count n_sn_pass
    n_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            n_sn_pass += 1
    #------------------------------------------------------#

    #------------------------------------------------------#
    gx_sn_pass = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)
    gx_opt_bf_snp = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)
    gx_opt_bf_tx = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)

    #------------------------------------------------------#
    # make a flag for the null condition
    #------------------------------------------------------#
    flag_string = ''
    for k3 in range(0, max_ngauss):
        flag_string_seg = '(gx_sn_flag[%d, :, :] == 0)' % k3
    
        flag_string = flag_string + flag_string_seg
        if k3 < max_ngauss-1:
            flag_string = flag_string + ' & '

    # gx_sn_pass[0, :, :] <-- put the null condition in the other loop
    gx_sn_pass[0, :, :] = np.array([np.where(eval(flag_string), 1, 0)])[0]

    # for the n_sn_pass >= 1 condition flags
    n_sn_pass = 1 # count from 1
    flag_string = ''
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            TF_flag = list(j) # True False flag
            print(TF_flag)

            # ---------------------------------------------------------------
            # make condition string
            # (g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 0) & (g4_sn_flag == 0)
            # (gx_sn_flag[0, :, :] == 1) & (gx_sn_flag[1, :, :] == 0) & (gx_sn_flag[2, :, :] == 0) & (gx_sn_flag[3, :, :] == 0)
            flag_string = ''
            #for k1 in range(0, max_ngauss):
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
                        
                        # move to the next True index if not reaching the end
                        if k2 < len(TF_flag)-1:
                            k2 += 1
                        # exit the current loop
                        break
                    else:
                        flag_string_seg = '(gx_sn_flag[%d, :, :] == 0)' % k3
                
                        flag_string = flag_string + flag_string_seg
                        if k3 < max_ngauss-1:
                            flag_string = flag_string + ' & '

                if k3 < max_ngauss-1:
                    # stay in the current loop until reaching the last point
                    # start from k3_s in the next loop
                    k3_s = k3+1
                else:
                    break # exit the while loop

            # ---------------------------------------------------------------

    
            print(flag_string)
            # gx_sn_pass[1, :, :] ~ gx_sn_pass[last, :, :]
            # gx_sn_pass[0, :, :] <-- put the null condition in the other loop
            gx_sn_pass[n_sn_pass, :, :] = np.array([np.where(eval(flag_string), 1, 0)])[0]
            n_sn_pass += 1

    #print(n_sn_pass)
    #print(gx_sn_pass[0, 410, 410])
    #print(gx_sn_pass[1, 410, 410])
    #print(gx_sn_pass[2, 410, 410])
    #print(gx_sn_pass[3, 410, 410])
    #print(gx_sn_pass[4, 410, 410])
    #print(gx_sn_pass[5, 410, 410])
    #print(gx_sn_pass[6, 410, 410])
    #print(gx_sn_pass[15, 410, 410])




    #i1 = 90
    #j1 = 213
    i1 = _params['_i0']
    j1 = _params['_j0']
    #g1234_sn_flag = gx_sn_pass[1, :, :] + gx_sn_pass[2, :, :] + gx_sn_pass[3, :, :] + gx_sn_pass[4, :, :]


    #print("ng-23: ", "sn_pass: ", np.where(g1234_sn_flag == 4))
    print("")
    #print("ng-1: ", gx_sorted[j1, i1], "sn_flag: ", gx_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", gx_sn_flag[0, j1, i1])
    #print("ng-2: ", gx_sorted[j1, i1], "sn_flag: ", gx_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", gx_sn_flag[1, j1, i1])
    #print("ng-3: ", gx_sorted[j1, i1], "sn_flag: ", gx_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", gx_sn_flag[2, j1, i1])
    #print("ng-4: ", gx_sorted[j1, i1], "sn_flag: ", gx_sn_pass_ng_opt_value[j1, i1], "sn_pass: ", gx_sn_flag[3, j1, i1])

    #g1234_sn_pass = np.array([np.where(((gx_sn_flag[0, :, :]== 1) & (gx_sn_flag[1, :, :] == 1) & (gx_sn_flag[2, :, :] == 1) & (gx_sn_flag[3, :, :] == 1)), 1, 0)])[0]

    #print("here")
    #print(g1234_sn_pass)
    #print("")
    #print(gx_sn_pass[15, :, :])
    #print("here")


    #------------------------------------------------------#
    # :: gx_opt_bf_snp[0, ] = snp0_tree_opt_bf
    gx_opt_bf_snp[0, :, :] = snp0_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[0, :, :])

    #------------------------------------------------------#
    # :: gx_opt_bf_snp[1~ ] = snp0_tree_opt_bf
    #------------------------------------------------------#
    n_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            #print(len(j))
            print(list(j)[0])
            #gx = 
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
    #------------------------------------------------------#


# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# WORK FROM HERE....


    #------------------------------------------------------#
    # :: gx_opt_bf_tx[0, ] = gx_opt_bf_snp[0, :, :]
    gx_opt_bf_tx[0, :, :] = np.array([np.where(gx_opt_bf_snp[0, :, :] > -1, gx_opt_bf_snp[0, :, :], -1)][0])

    #------------------------------------------------------#
    # :: gx_opt_bf_tx[1~ ] = gx_opt_bf_snp[1~, :, :]
    #------------------------------------------------------#
    n_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            #print(len(j))
            gx_opt_bf_tx[n_sn_pass, :, :] = np.array([np.where(gx_opt_bf_snp[n_sn_pass, :, :] > -1, gx_opt_bf_snp[n_sn_pass, :, :], gx_opt_bf_tx[n_sn_pass-1, :, :])][0])
            n_sn_pass += 1
    #------------------------------------------------------#

    print("optimal-ng:", gx_opt_bf_tx[n_sn_pass-1, j1, i1])
    return gx_opt_bf_tx[n_sn_pass-1, :, :]
#-- END OF SUB-ROUTINE____________________________________________________________#






#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g4_opt_bf_snp(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, sn_pass_ng_opt, g01, g02, g03, g04):

    gx_sn_flag = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    for i in range(0, max_ngauss):
        # 1. g1 number of z1
        gx_sorted = g_num_sort[i, :, :]
        # 2. sn flag of z1
        # 3. take values of sn_pass_ng_opt given g1_sorted indices
        gx_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, gx_sorted[np.newaxis], axis=0)[0]
        # 4. gx_sn_flag[j, i] = (0 or 1)
        gx_sn_flag[i, :, :] = np.array([np.where(gx_sn_pass_ng_opt_value == (gx_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

        print(gx_sn_flag[i, :, :])
    #------------------------------------------------------#
    gx_list = [g01, g02, g03, g04]

    #------------------------------------------------------#
    # count n_sn_pass
    n_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            n_sn_pass += 1
    #------------------------------------------------------#

    #------------------------------------------------------#
    gx_sn_pass = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)
    gx_opt_bf_snp = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)
    gx_opt_bf_tx = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)

    #------------------------------------------------------#
    # make a flag for the null condition
    #------------------------------------------------------#
    flag_string = ''
    for k3 in range(0, max_ngauss):
        flag_string_seg = '(gx_sn_flag[%d, :, :] == 0)' % k3
    
        flag_string = flag_string + flag_string_seg
        if k3 < max_ngauss-1:
            flag_string = flag_string + ' & '

    # gx_sn_pass[0, :, :] <-- put the null condition in the other loop
    gx_sn_pass[0, :, :] = np.array([np.where(eval(flag_string), 1, 0)])[0]

    # for the n_sn_pass >= 1 condition flags
    n_sn_pass = 1 # count from 1
    flag_string = ''
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            TF_flag = list(j) # True False flag
            print(TF_flag)

            # ---------------------------------------------------------------
            # make condition string
            # (g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 0) & (g4_sn_flag == 0)
            # (gx_sn_flag[0, :, :] == 1) & (gx_sn_flag[1, :, :] == 0) & (gx_sn_flag[2, :, :] == 0) & (gx_sn_flag[3, :, :] == 0)
            flag_string = ''
            #for k1 in range(0, max_ngauss):
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
                        
                        # move to the next True index if not reaching the end
                        if k2 < len(TF_flag)-1:
                            k2 += 1
                        # exit the current loop
                        break
                    else:
                        flag_string_seg = '(gx_sn_flag[%d, :, :] == 0)' % k3
                
                        flag_string = flag_string + flag_string_seg
                        if k3 < max_ngauss-1:
                            flag_string = flag_string + ' & '

                if k3 < max_ngauss-1:
                    # stay in the current loop until reaching the last point
                    # start from k3_s in the next loop
                    k3_s = k3+1
                else:
                    break # exit the while loop

            # ---------------------------------------------------------------

    
            print(flag_string)
            # gx_sn_pass[1, :, :] ~ gx_sn_pass[last, :, :]
            # gx_sn_pass[0, :, :] <-- put the null condition in the other loop
            gx_sn_pass[n_sn_pass, :, :] = np.array([np.where(eval(flag_string), 1, 0)])[0]
            n_sn_pass += 1

    #print(n_sn_pass)
    #print(gx_sn_pass[0, 410, 410])
    #print(gx_sn_pass[1, 410, 410])
    #print(gx_sn_pass[2, 410, 410])
    #print(gx_sn_pass[3, 410, 410])
    #print(gx_sn_pass[4, 410, 410])
    #print(gx_sn_pass[5, 410, 410])
    #print(gx_sn_pass[6, 410, 410])
    #print(gx_sn_pass[15, 410, 410])




    #i1 = 90
    #j1 = 213
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


    #------------------------------------------------------#
    # :: gx_opt_bf_snp[0, ] = snp0_tree_opt_bf
    gx_opt_bf_snp[0, :, :] = snp0_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[0, :, :])

    #------------------------------------------------------#
    # :: gx_opt_bf_snp[1~ ] = snp0_tree_opt_bf
    #------------------------------------------------------#
    n_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            #print(len(j))
            print(list(j)[0])
            #gx = 
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
    #------------------------------------------------------#


# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# WORK FROM HERE....


    #------------------------------------------------------#
    # :: gx_opt_bf_tx[0, ] = gx_opt_bf_snp[0, :, :]
    gx_opt_bf_tx[0, :, :] = np.array([np.where(gx_opt_bf_snp[0, :, :] > -1, gx_opt_bf_snp[0, :, :], -1)][0])

    #------------------------------------------------------#
    # :: gx_opt_bf_tx[1~ ] = gx_opt_bf_snp[1~, :, :]
    #------------------------------------------------------#
    n_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            #print(len(j))
            gx_opt_bf_tx[n_sn_pass, :, :] = np.array([np.where(gx_opt_bf_snp[n_sn_pass, :, :] > -1, gx_opt_bf_snp[n_sn_pass, :, :], gx_opt_bf_tx[n_sn_pass-1, :, :])][0])
            n_sn_pass += 1
    #------------------------------------------------------#

    print("optimal-ng:", gx_opt_bf_tx[15, j1, i1])
    return gx_opt_bf_tx[15, :, :]
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def gx_opt_bf_snp_org(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, sn_pass_ng_opt, gx_list):

    gx_sn_flag = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, max_ngauss):
        # 1. g1 number of z1
        gx_sorted = g_num_sort[i, :, :]
        # 2. sn flag of z1
        # 3. take values of sn_pass_ng_opt given g1_sorted indices
        gx_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, gx_sorted[np.newaxis], axis=0)[0]
        # 4. gx_sn_flag[j, i] = (0 or 1)
        gx_sn_flag[i, :, :] = np.array([np.where(gx_sn_pass_ng_opt_value == (gx_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

        #print(gx_sn_flag[i, :, :])
    #------------------------------------------------------#
    #gx_list = [g01, g02, g03, g04]

    #------------------------------------------------------#
    # count n_sn_pass
    n_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            n_sn_pass += 1
    #------------------------------------------------------#

    #------------------------------------------------------#
    gx_sn_pass = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)
    gx_opt_bf_snp = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)
    gx_opt_bf_tx = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)

    #------------------------------------------------------#
    # make a flag for the null condition
    #------------------------------------------------------#
    flag_string = ''
    for k3 in range(0, max_ngauss):
        flag_string_seg = '(gx_sn_flag[%d, :, :] == 0)' % k3
    
        flag_string = flag_string + flag_string_seg
        if k3 < max_ngauss-1:
            flag_string = flag_string + ' & '

    # gx_sn_pass[0, :, :] <-- put the null condition in the other loop
    n_sn_pass = 0
    gx_sn_pass[n_sn_pass, :, :] = np.array([np.where(eval(flag_string), 1, 0)])[0]

    # for the n_sn_pass >= 1 condition flags
    n_sn_pass = 1 # count from 1
    flag_string = ''
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            TF_flag = list(j) # True False flag
            print(TF_flag)

            # ---------------------------------------------------------------
            # make condition string
            # (g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 0) & (g4_sn_flag == 0)
            # (gx_sn_flag[0, :, :] == 1) & (gx_sn_flag[1, :, :] == 0) & (gx_sn_flag[2, :, :] == 0) & (gx_sn_flag[3, :, :] == 0)
            flag_string = ''
            #for k1 in range(0, max_ngauss):
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
                        
                        # move to the next True index if not reaching the end
                        if k2 < len(TF_flag)-1:
                            k2 += 1
                        # exit the current loop
                        break
                    else:
                        flag_string_seg = '(gx_sn_flag[%d, :, :] == 0)' % k3
                
                        flag_string = flag_string + flag_string_seg
                        if k3 < max_ngauss-1:
                            flag_string = flag_string + ' & '

                if k3 < max_ngauss-1:
                    # stay in the current loop until reaching the last point
                    # start from k3_s in the next loop
                    k3_s = k3+1
                else:
                    break # exit the while loop

            # ---------------------------------------------------------------

    
            #print(flag_string)
            # gx_sn_pass[1, :, :] ~ gx_sn_pass[last, :, :]
            # gx_sn_pass[0, :, :] <-- put the null condition in the other loop
            gx_sn_pass[n_sn_pass, :, :] = np.array([np.where(eval(flag_string), 1, 0)])[0]
            n_sn_pass += 1

    #i1 = 90
    #j1 = 213
    i1 = _params['_i0']
    j1 = _params['_j0']
    #------------------------------------------------------#
    # :: gx_opt_bf_snp[0, ] = snp0_tree_opt_bf
    n_sn_pass = 0
    gx_opt_bf_snp[n_sn_pass, :, :] = snp0_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :])

    #------------------------------------------------------#
    # :: gx_opt_bf_snp[1~ ] = snpX_tree_opt_bf
    #------------------------------------------------------#
    n_sn_pass = 1 # count from 1


    #______________________________________________________#
    # update n_GNx_list from the condition matrix for additional max_ngauss 
    # refer to the algorithm note by Se-Heon Oh
    # the below condition matrix is for up to max_ngauss=5
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
            #print(len(j))
            #print(list(j)[0])
            #gx = 

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
                #gx1 = list(j)[0]
                #gx2 = list(j)[1]
                #gx3 = list(j)[2]
                #gx_opt_bf_snp[n_sn_pass, :, :] = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2, gx3)
                gx_opt_bf_snp[n_sn_pass, :, :] = snp_gx_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, gx_sn_pass[n_sn_pass, :, :], gx_list, n_GNx_list)
            elif len(j) == 4:
                gx_list = list(j)
                n_GNx_list = [4, 3, 4, 4] # from the condition matrix
                #gx1 = list(j)[0]
                #gx2 = list(j)[1]
                #gx3 = list(j)[2]
                #gx4 = list(j)[3]
                #gx_opt_bf_snp[n_sn_pass, :, :] = snp4_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2, gx3, gx4)
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
#                gx1 = list(j)[0]
                gx2 = list(j)[1]
                gx3 = list(j)[2]
                gx4 = list(j)[3]
                gx5 = list(j)[4]
                gx6 = list(j)[5]
                gx7 = list(j)[6]
                gx_opt_bf_snp[n_sn_pass, :, :] = snp7_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2, gx3, gx4, gx5, gx6, gx7)

            n_sn_pass += 1
    #------------------------------------------------------#


    #------------------------------------------------------#
    #------------------------------------------------------#
    # :: gx_opt_bf_tx[0, ] = gx_opt_bf_snp[0, :, :]
    n_sn_pass = 0
    gx_opt_bf_tx[n_sn_pass, :, :] = np.array([np.where(gx_opt_bf_snp[0, :, :] > -1, gx_opt_bf_snp[n_sn_pass, :, :], -1)][0])

    #------------------------------------------------------#
    # :: gx_opt_bf_tx[1~ ] = gx_opt_bf_snp[1~, :, :]
    #------------------------------------------------------#
    n_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            #print(len(j))
            gx_opt_bf_tx[n_sn_pass, :, :] = np.array([np.where(gx_opt_bf_snp[n_sn_pass, :, :] > -1, gx_opt_bf_snp[n_sn_pass, :, :], gx_opt_bf_tx[n_sn_pass-1, :, :])][0])
            n_sn_pass += 1
    #------------------------------------------------------#

    print("optimal-ng:", gx_opt_bf_tx[n_sn_pass-1, j1, i1])
    return gx_opt_bf_tx[n_sn_pass-1, :, :]
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def find_gx_opt_bf_snp_dev(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, sn_pass_ng_opt, gx_list):

    gx_sn_flag = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, max_ngauss):
        # 1. g1 number of z1
        gx_sorted = g_num_sort[i, :, :]
        # 2. sn flag of z1
        # 3. take values of sn_pass_ng_opt given g1_sorted indices
        gx_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, gx_sorted[np.newaxis], axis=0)[0]
        # 4. gx_sn_flag[j, i] = (0 or 1)
        gx_sn_flag[i, :, :] = np.array([np.where(gx_sn_pass_ng_opt_value == (gx_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0

        #print(gx_sn_flag[i, :, :])
    #------------------------------------------------------#
    #gx_list = [g01, g02, g03, g04]

    #------------------------------------------------------#
    # count n_sn_pass
    n_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            n_sn_pass += 1
    #------------------------------------------------------#

    #------------------------------------------------------#
    gx_sn_pass = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)
    gx_opt_bf_snp = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)
    gx_opt_bf_tx = np.zeros((n_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)

    #------------------------------------------------------#
    # make a flag for the null condition
    #------------------------------------------------------#
    flag_string = ''
    for k3 in range(0, max_ngauss):
        flag_string_seg = '(gx_sn_flag[%d, :, :] == 0)' % k3
    
        flag_string = flag_string + flag_string_seg
        if k3 < max_ngauss-1:
            flag_string = flag_string + ' & '

    # gx_sn_pass[0, :, :] <-- put the null condition in the other loop
    n_sn_pass = 0
    gx_sn_pass[n_sn_pass, :, :] = np.array([np.where(eval(flag_string), 1, 0)])[0]

    # for the n_sn_pass >= 1 condition flags
    n_sn_pass = 1 # count from 1
    flag_string = ''
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            TF_flag = list(j) # True False flag
            print(TF_flag)

            # ---------------------------------------------------------------
            # make condition string
            # (g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 0) & (g4_sn_flag == 0)
            # (gx_sn_flag[0, :, :] == 1) & (gx_sn_flag[1, :, :] == 0) & (gx_sn_flag[2, :, :] == 0) & (gx_sn_flag[3, :, :] == 0)
            flag_string = ''
            #for k1 in range(0, max_ngauss):
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
                        
                        # move to the next True index if not reaching the end
                        if k2 < len(TF_flag)-1:
                            k2 += 1
                        # exit the current loop
                        break
                    else:
                        flag_string_seg = '(gx_sn_flag[%d, :, :] == 0)' % k3
                
                        flag_string = flag_string + flag_string_seg
                        if k3 < max_ngauss-1:
                            flag_string = flag_string + ' & '

                if k3 < max_ngauss-1:
                    # stay in the current loop until reaching the last point
                    # start from k3_s in the next loop
                    k3_s = k3+1
                else:
                    break # exit the while loop

            # ---------------------------------------------------------------

    
            #print(flag_string)
            # gx_sn_pass[1, :, :] ~ gx_sn_pass[last, :, :]
            # gx_sn_pass[0, :, :] <-- put the null condition in the other loop
            gx_sn_pass[n_sn_pass, :, :] = np.array([np.where(eval(flag_string), 1, 0)])[0]
            n_sn_pass += 1

    #------------------------------------------------------#
    # :: gx_opt_bf_snp[0, ] = snp0_tree_opt_bf
    n_sn_pass = 0
    gx_opt_bf_snp[n_sn_pass, :, :] = snp0_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :])

    #------------------------------------------------------#
    # :: gx_opt_bf_snp[1~ ] = snpX_tree_opt_bf
    #------------------------------------------------------#
    n_sn_pass = 1 # count from 1


    #______________________________________________________#
    # update n_GNx_list from the condition matrix for additional max_ngauss 
    # the below condition matrix is for up to max_ngauss=5
    # refer to the algorithm note by Se-Heon Oh
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

#            if len(j) == 1:
#                #gx1 = list(j)[0]
#                #gx_opt_bf_snp[n_sn_pass, :, :] = snp1_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1)
#
#                gx_list_comb = list(j)
#                GNx_index_comb = len(j)
#                gx_opt_bf_snp[n_sn_pass, :, :] = snp_gx_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, gx_sn_pass[n_sn_pass, :, :], gx_list_comb, n_GNx_list[GNx_index_comb])
#
#            elif len(j) == 2:
#                #gx1 = list(j)[0]
#                #gx2 = list(j)[1]
#                #gx_opt_bf_snp[n_sn_pass, :, :] = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2)
#
#                gx_list_comb = list(j)
#                GNx_index_comb = len(j)
#                gx_opt_bf_snp[n_sn_pass, :, :] = snp_gx_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, gx_sn_pass[n_sn_pass, :, :], gx_list_comb, n_GNx_list[GNx_index_comb])
#
#            elif len(j) == 3:
#                #gx1 = list(j)[0]
#                #gx2 = list(j)[1]
#                #gx3 = list(j)[2]
#                #gx_opt_bf_snp[n_sn_pass, :, :] = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2, gx3)
#
#                gx_list_comb = list(j)
#                GNx_index_comb = len(j)
#                gx_opt_bf_snp[n_sn_pass, :, :] = snp_gx_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, gx_sn_pass[n_sn_pass, :, :], gx_list_comb, n_GNx_list[GNx_index_comb])
#
#            elif len(j) == 4:
#                #gx1 = list(j)[0]
#                #gx2 = list(j)[1]
#                #gx3 = list(j)[2]
#                #gx4 = list(j)[3]
                #gx_opt_bf_snp[n_sn_pass, :, :] = snp4_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2, gx3, gx4)
#
#                #print("SSSSSSS", len(j))
#                gx_list_comb = list(j)
#                GNx_index_comb = len(j)
#                gx_opt_bf_snp[n_sn_pass, :, :] = snp_gx_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, gx_sn_pass[n_sn_pass, :, :], gx_list_comb, n_GNx_list[GNx_index_comb])
#
#            elif len(j) == 5:
#                #gx1 = list(j)[0]
#                #gx2 = list(j)[1]
#                #gx3 = list(j)[2]
#                #gx4 = list(j)[3]
#                #gx5 = list(j)[4]
#                #gx_opt_bf_snp[n_sn_pass, :, :] = snp5_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2, gx3, gx4, gx5)
#
#                gx_list_comb = list(j)
#                GNx_index_comb = len(j)
#                gx_opt_bf_snp[n_sn_pass, :, :] = snp_gx_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, gx_sn_pass[n_sn_pass, :, :], gx_list_comb, n_GNx_list[GNx_index_comb])
#
#
#        # seheon1
#            if len(j) == 3:
#                gx_list = list(j)
#                GNx_index = len(j)
#                print(gx_list, "gogo")
#                gx_opt_bf_snp[n_sn_pass, :, :] = snp_gx_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, gx_sn_pass[n_sn_pass, :, :], gx_list, n_GNx_list[GNx_index])

#            if len(j) == 1:
#                gx1 = list(j)[0]
#                gx_opt_bf_snp[n_sn_pass, :, :] = snp1_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1)
#            elif len(j) == 2:
#                gx1 = list(j)[0]
#                gx2 = list(j)[1]
#                gx_opt_bf_snp[n_sn_pass, :, :] = snp2_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2)
#                print("GGDDDDD")
#                print("GGDDDDD")
#                print("GGDDDDD")
#                print("GGDDDDD")


#            elif len(j) == 3:
#                gx_list = list(j)
#                GNx_index = len(j)
#                gx1 = list(j)[0]
#                gx2 = list(j)[1]
#                gx3 = list(j)[2]
#                #n_GNx_list = [3, 2, 2] # from the condition matrix
#                #gx1 = list(j)[0]
#                #gx2 = list(j)[1]
#                #gx3 = list(j)[2]
#                #gx_opt_bf_snp[n_sn_pass, :, :] = snp3_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2, gx3)
#                gx_opt_bf_snp[n_sn_pass, :, :] = snp_gx_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, gx_sn_pass[n_sn_pass, :, :], gx_list, n_GNx_list[GNx_index])
#            elif len(j) == 4:
#                gx_list = list(j)
#                GNx_index = len(j)
#                #n_GNx_list = [4, 3, 4, 4] # from the condition matrix
#                gx1 = list(j)[0]
#                gx2 = list(j)[1]
#                gx3 = list(j)[2]
#                gx4 = list(j)[3]
#                #gx_opt_bf_snp[n_sn_pass, :, :] = snp4_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2, gx3, gx4)
#                gx_opt_bf_snp[n_sn_pass, :, :] = snp_gx_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, gx_sn_pass[n_sn_pass, :, :], gx_list, n_GNx_list[GNx_index])
#                print('GOGOGOGOGOGOGOGOG')
#                print('GOGOGOGOGOGOGOGOG')
#                print('GOGOGOGOGOGOGOGOG')
#                print('GOGOGOGOGOGOGOGOG')
#                print('GOGOGOGOGOGOGOGOG')
#            elif len(j) == 5:
#                gx_list = list(j)
#                GNx_index = len(j)
#                gx1 = list(j)[0]
#                gx2 = list(j)[1]
#                gx3 = list(j)[2]
#                gx4 = list(j)[3]
#                gx5 = list(j)[4]
#                #gx_opt_bf_snp[n_sn_pass, :, :] = snp5_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2, gx3, gx4, gx5)
#                gx_opt_bf_snp[n_sn_pass, :, :] = snp_gx_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, gx_sn_pass[n_sn_pass, :, :], gx_list, n_GNx_list[GNx_index])
#            elif len(j) == 6:
#                gx_list = list(j)
#                GNx_index = len(j)
#                gx1 = list(j)[0]
#                gx2 = list(j)[1]
#                gx3 = list(j)[2]
#                gx4 = list(j)[3]
#                gx5 = list(j)[4]
#                gx6 = list(j)[5]
#                #gx_opt_bf_snp[n_sn_pass, :, :] = snp6_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2, gx3, gx4, gx5, gx6)
#                gx_opt_bf_snp[n_sn_pass, :, :] = snp_gx_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, gx_sn_pass[n_sn_pass, :, :], gx_list, n_GNx_list[GNx_index])
##            elif len(j) == 7:
###                gx1 = list(j)[0]
##                gx2 = list(j)[1]
##                gx3 = list(j)[2]
##                gx4 = list(j)[3]
##                gx5 = list(j)[4]
##                gx6 = list(j)[5]
##                gx7 = list(j)[6]
##                gx_opt_bf_snp[n_sn_pass, :, :] = snp7_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[n_sn_pass, :, :], gx1, gx2, gx3, gx4, gx5, gx6, gx7)
#
            n_sn_pass += 1
    #------------------------------------------------------#


    #------------------------------------------------------#
    #------------------------------------------------------#
    # :: gx_opt_bf_tx[0, ] = gx_opt_bf_snp[0, :, :]
    n_sn_pass = 0
    gx_opt_bf_tx[n_sn_pass, :, :] = np.array([np.where(gx_opt_bf_snp[0, :, :] > -1, gx_opt_bf_snp[n_sn_pass, :, :], -1)][0])

    #------------------------------------------------------#
    # :: gx_opt_bf_tx[1~ ] = gx_opt_bf_snp[1~, :, :]
    #------------------------------------------------------#
    n_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            #print(len(j))
            gx_opt_bf_tx[n_sn_pass, :, :] = np.array([np.where(gx_opt_bf_snp[n_sn_pass, :, :] > -1, gx_opt_bf_snp[n_sn_pass, :, :], gx_opt_bf_tx[n_sn_pass-1, :, :])][0])
            n_sn_pass += 1
    #------------------------------------------------------#

    #i1 = 90
    #j1 = 213
    i1 = _params['_i0']
    j1 = _params['_j0']
    print("optimal-ng-seheon:", gx_opt_bf_tx[n_sn_pass-1, j1, i1])
    return gx_opt_bf_tx[n_sn_pass-1, :, :]
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def find_gx_opt_bf_snp(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, sn_pass_ng_opt, gx_list):

    gx_sn_flag = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    for i in range(0, max_ngauss):
        #------------------------------------------------------#
        # 1. g number of z1 : gN_model with the highest log-z, e.g., gx_sorted[468, 600] = 2 --> ngauss=3 gaussian model has the hightest log-z
        gx_sorted = g_num_sort[i, :, :]
        #------------------------------------------------------#

        #------------------------------------------------------#
        # 2. sn flag of z1
        # 3. take values of sn_pass_ng_opt given gx_sorted indices
        # : gx_sn_pass_ng_opt_value for the gN_model with the highest log-z. <-- this value is from sn_pass_ng_opt array in main() routine
        # : if this value equals to gx_sorted+1 --> sn_pass 1 else --> sn_pass 0
        gx_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, gx_sorted[np.newaxis], axis=0)[0]
        #------------------------------------------------------#

        #------------------------------------------------------#
        # 4. gx_sn_flag[gN_model, j, i] = (0 or 1)
        gx_sn_flag[i, :, :] = np.array([np.where(gx_sn_pass_ng_opt_value == (gx_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0
        #------------------------------------------------------#

        # e.g.
        # gx_sorted[468, 600] == 2
        # gx_sn_pass_ng_opt_value[468, 600] <-- sn_pass_ng_opt[gx_sorted[468, 600], 468, 600] 
        # gx_sn_pass_ng_opt_value[468, 600] == 1.0
        # <-- For the profile at (468, 600), ngauss=2 (which is a tripe gaussian model) model has the highest log-z and so prefered as the best model.
        # <-- But only 1 gaussian (gx_sn_pass_ng_opt_value == 1) component out of the three passed the S/N limit.
        # <-- This can be checked by gx_sn_pass_ng_opt_value == (gx_sorted+1) : in this case, 1 != 2 + 1
        # <-- So the sn_flag for the triple gaussian model at (468, 600) is set to 0

        #print(gx_sn_flag[i, 468, 600], sn_pass_ng_opt[i, 468, 600], gx_sorted[468, 600], gx_sn_pass_ng_opt_value[468, 600])
        # --> 0.0 1.0 2 1.0
        # --> 0.0 1.0 1 1.0
        # --> 1.0 1.0 0 1.0
        # <-- only the model with ngauss=1 but with the lowest log-z (and thus gx_sn_flag[2, :, :] index is 2) passed the s/n limit
        # <-- so gx_sn_flag[2, 468, 600] = 1.0. Those for the others are all set to 0.0

        #print(gx_sn_flag[i, 468, 600], sn_pass_ng_opt[i, 468, 600], gx_sorted[468, 600], gx_sn_pass_ng_opt_value[468, 600])
        #print(gx_sn_flag[1, 468, 600], sn_pass_ng_opt[1, 468, 600])
        #print(gx_sn_flag[2, 468, 600], sn_pass_ng_opt[2, 468, 600])

    #------------------------------------------------------#
    #gx_list = [g01, g02, g03, g04]


    #------------------------------------------------------#
    # Count the cases of n_sn_pass
    # this is used for declaring the arrays below
    # : this is not the number of gaussians which pass the s/n limit
    # : but the numbering of combinations of gaussians which pass the s/n limit
    # : e.g., 0==[], 1==[0], 2==[1], 3==[2], 4==[0, 1], 5==[0, 2], 6==[1, 2], 7==[1, 2, 3] --> 8 combinations in case of max_ngass = 3
    loop_index_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            loop_index_sn_pass += 1
    #------------------------------------------------------#

    #------------------------------------------------------#
    gx_sn_pass = np.zeros((loop_index_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)
    gx_opt_bf_snp = np.zeros((loop_index_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)
    gx_opt_bf_tx = np.zeros((loop_index_sn_pass, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=int)

    #------------------------------------------------------#
    # make a flag for the null condition
    #------------------------------------------------------#
    flag_string = ''
    for k3 in range(0, max_ngauss):
        flag_string_seg = '(gx_sn_flag[%d, :, :] == 0)' % k3
    
        flag_string = flag_string + flag_string_seg
        if k3 < max_ngauss-1:
            flag_string = flag_string + ' & '


    #print(flag_string)
    #--> (gx_sn_flag[0, :, :] == 0) & (gx_sn_flag[1, :, :] == 0) & (gx_sn_flag[2, :, :] == 0)

    #------------------------------------------------------#
    # gx_sn_pass[0, :, :] <-- put the null condition in the other loop
    loop_index_sn_pass = 0
    gx_sn_pass[loop_index_sn_pass, :, :] = np.array([np.where(eval(flag_string), 1, 0)])[0]
    #------------------------------------------------------#

    # print(gx_sn_pass[0, 468, 600])
    # --> 0

    # for the loop_index_sn_pass >= 1 condition flags
    loop_index_sn_pass = 1 # count from 1
    flag_string = ''
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            TF_flag = list(j) # True False flag
            #print(TF_flag)

            # ---------------------------------------------------------------
            # make condition string
            # (g1_sn_flag == 1) & (g2_sn_flag == 0) & (g3_sn_flag == 0) & (g4_sn_flag == 0)
            # <-- (gx_sn_flag[0, :, :] == 1) & (gx_sn_flag[1, :, :] == 0) & (gx_sn_flag[2, :, :] == 0) & (gx_sn_flag[3, :, :] == 0)
            # ---------------------------------------------------------------
            flag_string = ''
            #for k1 in range(0, max_ngauss):
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
                        
                        # move to the next True index if not reaching the end
                        if k2 < len(TF_flag)-1:
                            k2 += 1
                        # exit the current loop
                        break
                    else:
                        flag_string_seg = '(gx_sn_flag[%d, :, :] == 0)' % k3
                
                        flag_string = flag_string + flag_string_seg
                        if k3 < max_ngauss-1:
                            flag_string = flag_string + ' & '

                if k3 < max_ngauss-1:
                    # stay in the current loop until reaching the last point
                    # start from k3_s in the next loop
                    k3_s = k3+1
                else:
                    break # exit the while loop

            # ---------------------------------------------------------------

            #print("FLAG STRING", flag_string)
            # gx_sn_pass[1, :, :] ~ gx_sn_pass[last, :, :]
            # gx_sn_pass[0, :, :] <-- put the null condition in the other loop
            gx_sn_pass[loop_index_sn_pass, :, :] = np.array([np.where(eval(flag_string), 1, 0)])[0]


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    # CHECK POINT
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
            # ---------------------------------------------------------------
            # USE THIS PRINT COMMAND TO UNDERSTAND THE N_SN_PASS STRUCTURE
            #i1 = 600
            #j1 = 468
            i1 = _params['_i0']
            j1 = _params['_j0']
            #print("N_SN_PASS COMBINATION %s" % TF_flag, "loop_index_sn_pass:", loop_index_sn_pass, "gx_sn_pass:", gx_sn_pass[loop_index_sn_pass, j1, i1], flag_string)
            # ---------------------------------------------------------------
            loop_index_sn_pass += 1


# nuree2
    #------------------------------------------------------#
    # :: gx_opt_bf_snp[0, ] = snp0_tree_opt_bf
    loop_index_sn_pass = 0 # the 1st combination
    gx_opt_bf_snp[loop_index_sn_pass, :, :] = snp0_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, gx_sn_pass[loop_index_sn_pass, :, :])

    #------------------------------------------------------#
    # :: gx_opt_bf_snp[1~ ] = snp_gX_tree_opt_bf
    #------------------------------------------------------#
    loop_index_sn_pass = 1 # count from 1, for the other combinations.. 

    #______________________________________________________#
    # update n_GNx_list from the condition matrix for additional max_ngauss 
    # the example below shows a condition matrix with up to max_ngauss=5
    # refer to the algorithm note by Se-Heon Oh
    #n_GNx_list[1] = [1] # from the condition matrix
    #n_GNx_list[2] = [2, 1] # from the condition matrix
    #n_GNx_list[3] = [3, 2, 2] # from the condition matrix
    #n_GNx_list[4] = [4, 3, 4, 4] # from the condition matrix
    #n_GNx_list[5] = [5, 4, 6, 8, 8] # from the condition matrix

    #______________________________________________________#
    # generate condition maxtrix given max_ngauss: refer to Oh's algorithm note
    n_GNx_list = [[0 for _i1 in range(0)] for _j1 in range(max_ngauss+1)]

    for gx_1 in range(1, max_ngauss+1):
        n_GNx_list[gx_1].append(gx_1)
        for gx_2 in range(0, gx_1-1):
            n_GNx_list[gx_1].append((2**gx_2)*(gx_1-1-gx_2))

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    # CHECK POINT
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    # check the generated condition matrix
    #for gx_1 in range(1, max_ngauss+1):
    #    print("BBBBBBBBBBBBB", gx_1, n_GNx_list[gx_1])

# nuree3 : THIS IS WHERE THE PROBLEM IS ..... check gx_sn_pass[loop_index_sn_pass, :, :] and snp_gx_tree_opt_bf
    #______________________________________________________#
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            gx_list_comb = list(j)
            GNx_index_comb = len(j)
            gx_opt_bf_snp[loop_index_sn_pass, :, :] = snp_gx_tree_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, gx_sn_pass[loop_index_sn_pass, :, :], gx_list_comb, n_GNx_list[GNx_index_comb])

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    # CHECK POINT
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
            #print("CHECK", loop_index_sn_pass, gx_sn_pass[loop_index_sn_pass, j1, i1], gx_opt_bf_snp[loop_index_sn_pass, j1, i1])
            loop_index_sn_pass += 1
    #------------------------------------------------------#

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    # CHECK POINT
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    #print("xxx", gx_opt_bf_snp[3, 468, 600])
    # if gx_opt_bf_snp[3, 468, 600] = 0 from snp_gx_tree_opt_bf, the current issue is resolved ... not -10
    # gx_opt_bf_snp[3, 468, 600] = 0

    #------------------------------------------------------#
    #------------------------------------------------------#
    # :: gx_opt_bf_tx[0, ] = gx_opt_bf_snp[0, :, :]
    loop_index_sn_pass = 0 # null case : [] : no gaussians pass the s/n limit  --> put -1
    gx_opt_bf_tx[loop_index_sn_pass, :, :] = np.array([np.where(gx_opt_bf_snp[0, :, :] > -1, gx_opt_bf_snp[loop_index_sn_pass, :, :], -1)][0])



    #------------------------------------------------------#
    # temporary gx_opt_bf_tx : which is updated below for each combination: [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]
    #------------------------------------------------------#
    # :: gx_opt_bf_tx[1~ ] = gx_opt_bf_snp[1~, :, :]
    #------------------------------------------------------#
    loop_index_sn_pass = 1 # count from 1
    for i in range(1, max_ngauss+1):
        for j in itt.combinations(gx_list, i):
            #------------------------------------------------------#
            # keep the previous case which is the most optimal case so far... --> s_sn_pass-1 (previous case)
            gx_opt_bf_tx[loop_index_sn_pass, :, :] = np.array([np.where(gx_opt_bf_snp[loop_index_sn_pass, :, :] > -1, gx_opt_bf_snp[loop_index_sn_pass, :, :], gx_opt_bf_tx[loop_index_sn_pass-1, :, :])][0])

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    # CHECK POINT
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
            #print("here", loop_index_sn_pass, gx_opt_bf_snp[loop_index_sn_pass, j1, i1])

            loop_index_sn_pass += 1
    #------------------------------------------------------#



    #------------------------------------------------------#
    #------------------------------------------------------#
    #------------------------------------------------------#
    #------------------------------------------------------#
    # set gx_opt_bf_tx with 0 (i.e., put single Gaussian model as the best model)
    # for the pixels with -1 (null model) but where the other gaussian models with > 1 gaussian
    # do not pass the S/N limit except for the single gaussian model.
    # check if the corresponding single gaussian model pass the S/N limit
    # if so put gx_opt_bf_tx[loop_index_sn_pass-1, y, x] = 0 (i.e., single gaussian model) --> 0
    # otherwise keep the current result --> gx_opt_bf_tx[loop_index_sn_pass-1, :, :]
    #------------------------------------------------------#
    gx_opt_bf_tx[loop_index_sn_pass-1, :, :] = np.array([np.where((gx_opt_bf_tx[loop_index_sn_pass-1, :, :] == -1) & (sn_pass_ng_opt[0, :, :] == 1.0) , 0, gx_opt_bf_tx[loop_index_sn_pass-1, :, :])][0])
    #------------------------------------------------------#


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    # CHECK POINT
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    #i1 = 600
    #j1 = 468
    i1 = _params['_i0']
    j1 = _params['_j0']
    #print("0:", gx_opt_bf_snp[0, j1, i1])
    #print("1:", gx_opt_bf_snp[1, j1, i1])
    #print("2:", gx_opt_bf_snp[2, j1, i1])
    #print("3:", gx_opt_bf_snp[3, j1, i1])
    #print("4:", gx_opt_bf_snp[4, j1, i1])
    #print("5:", gx_opt_bf_snp[5, j1, i1])
    #print("6:", gx_opt_bf_snp[6, j1, i1])
    #print("7:", gx_opt_bf_snp[7, j1, i1])
    #print("loop_index_sn_pass:", loop_index_sn_pass)
    #print("optimal-ng-seheon:", gx_opt_bf_tx[loop_index_sn_pass-1, j1, i1])


    return gx_opt_bf_tx[loop_index_sn_pass-1, :, :]
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g4_opt_bf_snp_org(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, sn_pass_ng_opt, g01, g02, g03, g04):

    #------------------------------------------------------#
    # 1. sn_pass flag for g1 
    # g1 number of z1
    g1_sorted = g_num_sort[g01, :, :]
    # sn flag of z1
    # take values of sn_pass_ng_opt given g1_sorted indices
    g1_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]
    # g1_sn_pass[j, i] = (0 or 1)
    g1_sn_flag = np.array([np.where(g1_sn_pass_ng_opt_value == (g1_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0
    #------------------------------------------------------#

    #------------------------------------------------------#
    # 2. sn_pass flag for g2 
    # g2 number of z2
    g2_sorted = g_num_sort[g02, :, :]
    # sn flag of z2
    # take values of sn_pass_ng_opt given g2_sorted indices
    g2_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g2_sorted[np.newaxis], axis=0)[0]
    # g2_sn_pass[j, i] = (0 or 1)
    g2_sn_flag = np.array([np.where(g2_sn_pass_ng_opt_value == (g2_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0
    #------------------------------------------------------#

    #------------------------------------------------------#
    # 3. sn_pass flag for g3 
    # g3 number of z3
    g3_sorted = g_num_sort[g03, :, :]
    # sn flag of z3
    # take values of sn_pass_ng_opt given g3_sorted indices
    g3_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g3_sorted[np.newaxis], axis=0)[0]
    # g3_sn_pass[j, i] = (0 or 1)
    g3_sn_flag = np.array([np.where(g3_sn_pass_ng_opt_value == (g3_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0
    #------------------------------------------------------#

    #------------------------------------------------------#
    # 4. sn_pass flag for g4 
    # g4 number of z4
    g4_sorted = g_num_sort[g04, :, :]
    # sn flag of z4
    # take values of sn_pass_ng_opt given g4_sorted indices
    g4_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g4_sorted[np.newaxis], axis=0)[0]
    # g4_sn_pass[j, i] = (0 or 1)
    g4_sn_flag = np.array([np.where(g4_sn_pass_ng_opt_value == (g4_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0
    #------------------------------------------------------#

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

    #i1 = 90
    #j1 = 213
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
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g5_opt_bf_snp(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, sn_pass_ng_opt, g01, g02, g03, g04, g05):

    #------------------------------------------------------#
    # 1. sn_pass flag for g1 
    # g1 number of z1
    g1_sorted = g_num_sort[g01, :, :]
    # sn flag of z1
    # take values of sn_pass_ng_opt given g1_sorted indices
    g1_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]
    # g1_sn_pass[j, i] = (0 or 1)
    g1_sn_flag = np.array([np.where(g1_sn_pass_ng_opt_value == (g1_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0
    #------------------------------------------------------#

    #------------------------------------------------------#
    # 2. sn_pass flag for g2 
    # g2 number of z2
    g2_sorted = g_num_sort[g02, :, :]
    # sn flag of z2
    # take values of sn_pass_ng_opt given g2_sorted indices
    g2_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g2_sorted[np.newaxis], axis=0)[0]
    # g2_sn_pass[j, i] = (0 or 1)
    g2_sn_flag = np.array([np.where(g2_sn_pass_ng_opt_value == (g2_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0
    #------------------------------------------------------#

    #------------------------------------------------------#
    # 3. sn_pass flag for g3 
    # g3 number of z3
    g3_sorted = g_num_sort[g03, :, :]
    # sn flag of z3
    # take values of sn_pass_ng_opt given g3_sorted indices
    g3_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g3_sorted[np.newaxis], axis=0)[0]
    # g3_sn_pass[j, i] = (0 or 1)
    g3_sn_flag = np.array([np.where(g3_sn_pass_ng_opt_value == (g3_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0
    #------------------------------------------------------#

    #------------------------------------------------------#
    # 4. sn_pass flag for g4 
    # g4 number of z4
    g4_sorted = g_num_sort[g04, :, :]
    # sn flag of z4
    # take values of sn_pass_ng_opt given g4_sorted indices
    g4_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g4_sorted[np.newaxis], axis=0)[0]
    # g4_sn_pass[j, i] = (0 or 1)
    g4_sn_flag = np.array([np.where(g4_sn_pass_ng_opt_value == (g4_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0
    #------------------------------------------------------#

    #------------------------------------------------------#
    # 5. sn_pass flag for g5 
    # g5 number of z5
    g5_sorted = g_num_sort[g05, :, :]
    # sn flag of z5
    # take values of sn_pass_ng_opt given g5_sorted indices
    g5_sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g5_sorted[np.newaxis], axis=0)[0]
    # g5_sn_pass[j, i] = (0 or 1)
    g5_sn_flag = np.array([np.where(g5_sn_pass_ng_opt_value == (g5_sorted+1), 1, 0)])[0] # if sn_pass 1, else 0
    #------------------------------------------------------#

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

    #i1 = 90
    #j1 = 213
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
#-- END OF SUB-ROUTINE____________________________________________________________#





#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g4_opt_bf_snp0(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g0_sn_pass):
#________________________________________________________________________#
    #------------------------------------------------------#
    #------------------------------------------------------#
    # --> blank = -1
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    g_cond_sn = np.where(g0_sn_pass) # s/n < sn_limit
    g_opt[g_cond_sn] += 1

    g0_0 = np.array([np.where(g_opt > 0, -10, -10)])
    g4_opt_snp0 = g0_0[0]
    #print(g4_opt_snp0[0])

    return g4_opt_snp0
#-- END OF SUB-ROUTINE____________________________________________________________#




#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g4_opt_bf_snp1(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1_sn_pass, g01): # g01 : g indices (0, 1, 2, ...)
#________________________________________________________________________#
    # --> g01
    g_sort_n = g01
    _cond_N = 0
    _cond0 = ([g01, g01, '='], ) # dummy condition
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond0, _cond_N, g1_sn_pass)
    g4_opt_snp1 = g1_0 + 10*0
    #print(g4_opt_snp1[0])

    return g4_opt_snp1
#-- END OF SUB-ROUTINE____________________________________________________________#




#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g4_opt_bf_snp2(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g12_sn_pass, g01, g02): # g01 < g02 : g indices (0, 1, 2, ...)
#________________________________________________________________________#

    #------------------------------------------------------#
    # G01-0
    #------------------------------------------------------#
    # if (z1/z2 > bf_limit): --> g01
    g_sort_n = g01
    _cond_N = 1
    _cond1 = ([g01, g02, '0'], )
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g12_sn_pass)

    #------------------------------------------------------#
    # G02-0
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2): --> g02
    g_sort_n = g02
    _cond_N = 2
    _cond2 = ([g01, g02, '1'], [g01, g02, '0'])
    g2_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond2, _cond_N, g12_sn_pass)

    #------------------------------------------------------#
    # G01-1
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2): --> g01
    g_sort_n = g01
    _cond_N = 2
    _cond2 = ([g01, g02, '1'], [g01, g02, '1'])
    g1_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond2, _cond_N, g12_sn_pass)
    
    g4_opt_snp2 = g1_0 + g2_0 + g1_1 + 10*2
    #print(g3_opt_snp2[0])

    return g4_opt_snp2
#-- END OF SUB-ROUTINE____________________________________________________________#






#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# set G3-X conditions for the baygaud output with _ng maximum gaussian fits
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

#    for i in range(0, 2):
#        print(seed_cond1[i])

    return seed_cond1
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# set G3-X conditions for the baygaud output with _ng maximum gaussian fits
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

#    for i in range(0, 2):
#        print(seed_cond1[i])

    return seed_cond1
#-- END OF SUB-ROUTINE____________________________________________________________#




#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# Gn-X conditions for the baygaud output with _ng maximum gaussian fits
def set_cond_tree_old(n_snp, _Gn, _max_ngauss, _gt_conds_index): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    #_Gn = 2
    #_max_ngauss = 5
    if n_snp == 1: # check here ...
        n_core_seeds = 1
    elif n_snp == 2: # check here ...
        n_core_seeds = 1
    else:
        n_core_seeds = 2**(n_snp-2) # check here...

    seed_cond0 = [[[0 for i in range(3)] for j in range(0)] for k in range(n_core_seeds)]
    # seed_cond1 : should be 2
    seed_cond1 = [[[0 for i in range(3)] for j in range(0)] for k in range(2)]

    _gt = [0 for i in range(_max_ngauss)]
    _gt_conds = [[0 for i in range(0)] for j in range(_max_ngauss-_Gn+1)]

    if n_snp == 1:
        _gt_conds[0].append(seed_cond0[0])

    # n_snp == 2
    if n_snp == 2:
        for i in range(0, 1):
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 1, '0'])

        _gt_conds[0].append(seed_cond0[0])

    # n_snp == 3
    if n_snp == 3:
        #for i in range(0, 1):
        #    seed_cond0[i].append([1, 2, '1'])
        #    seed_cond0[i].append([1, 2, '0'])
        #    seed_cond0[i].append([2, 3, '1'])
        #    seed_cond0[i].append([2, 3, '0'])

        #for i in range(1, 2):
        #    seed_cond0[i].append([1, 2, '1'])
        #    seed_cond0[i].append([1, 2, '1'])
        #    seed_cond0[i].append([1, 3, '1'])
        #    seed_cond0[i].append([1, 3, '0'])

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
    
        #_gt[0] = set_cond_child(seed_cond0[0], _Gn, _Gn+1)
        #_gt[0] = set_cond_child(seed_cond0[1], _Gn, _Gn+1)
    
        # G4-
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


#        for i in range(0, 1):
#            seed_cond0[i].append([1, 2, '1'])
#            seed_cond0[i].append([1, 2, '0'])
#            seed_cond0[i].append([2, 3, '1'])
#            seed_cond0[i].append([2, 3, '0'])
#            seed_cond0[i].append([3, 4, '1'])
#            seed_cond0[i].append([3, 4, '0'])
#            seed_cond0[i].append([3, 4, '0'])
#            seed_cond0[i].append([3, 4, '0'])

#       for i in range(1, 2):
#            seed_cond0[i].append([1, 2, '1'])
#            seed_cond0[i].append([1, 2, '1'])
#            seed_cond0[i].append([2, 3, '1'])
#            seed_cond0[i].append([2, 3, '0'])
#            seed_cond0[i].append([2, 4, '1'])
#            seed_cond0[i].append([2, 4, '0'])
#            seed_cond0[i].append([2, 4, '0'])
#            seed_cond0[i].append([2, 4, '0'])
    
#        for i in range(2, 3):
#            seed_cond0[i].append([1, 2, '1'])
#            seed_cond0[i].append([1, 2, '1'])
#            seed_cond0[i].append([1, 3, '1'])
#            seed_cond0[i].append([1, 3, '0'])
#            seed_cond0[i].append([3, 4, '1'])
#            seed_cond0[i].append([3, 4, '0'])
#            seed_cond0[i].append([3, 4, '0'])
#            seed_cond0[i].append([3, 4, '0'])
    
#        for i in range(3, 4):
#            seed_cond0[i].append([1, 2, '1'])
#            seed_cond0[i].append([1, 2, '1'])
#            seed_cond0[i].append([1, 3, '1'])
#            seed_cond0[i].append([1, 3, '0'])
#            seed_cond0[i].append([1, 4, '1'])
##            seed_cond0[i].append([1, 4, '0'])
#            seed_cond0[i].append([1, 4, '0'])
#            seed_cond0[i].append([1, 4, '0'])

#        for i in range(4, 5):
##            seed_cond0[i].append([1, 2, '1'])
#            seed_cond0[i].append([1, 2, '1'])
#            seed_cond0[i].append([1, 3, '1'])
#            seed_cond0[i].append([1, 3, '0'])
#            seed_cond0[i].append([1, 4, '1'])
#            seed_cond0[i].append([1, 4, '0'])
#            seed_cond0[i].append([1, 4, '0'])
#            seed_cond0[i].append([1, 4, '0'])

#        for i in range(5, 6):
##            seed_cond0[i].append([1, 2, '1'])
#            seed_cond0[i].append([1, 2, '1'])
#            seed_cond0[i].append([1, 3, '1'])
#            seed_cond0[i].append([1, 3, '0'])
#            seed_cond0[i].append([1, 4, '1'])
#            seed_cond0[i].append([1, 4, '0'])
#            seed_cond0[i].append([1, 4, '0'])
#            seed_cond0[i].append([1, 4, '0'])

#        for i in range(6, 7):
#            seed_cond0[i].append([1, 2, '1'])
#            seed_cond0[i].append([1, 2, '1'])
#            seed_cond0[i].append([1, 3, '1'])
#            seed_cond0[i].append([1, 3, '0'])
#            seed_cond0[i].append([1, 4, '1'])
#            seed_cond0[i].append([1, 4, '0'])
#            seed_cond0[i].append([1, 4, '0'])
#            seed_cond0[i].append([1, 4, '0'])

#        for i in range(7, 8):
#            seed_cond0[i].append([1, 2, '1'])
#            seed_cond0[i].append([1, 2, '1'])
#            seed_cond0[i].append([1, 3, '1'])
#            seed_cond0[i].append([1, 3, '0'])
#            seed_cond0[i].append([1, 4, '1'])
#            seed_cond0[i].append([1, 4, '0'])
#            seed_cond0[i].append([1, 4, '0'])
#            seed_cond0[i].append([1, 4, '0'])
    
        #_gt[0] = set_cond_child(seed_cond0[0], _Gn, _Gn+1)
        #_gt[0] = set_cond_child(seed_cond0[1], _Gn, _Gn+1)
    
        # G5-
        _gt_conds[0].append(seed_cond0[0])
        _gt_conds[0].append(seed_cond0[1])
        _gt_conds[0].append(seed_cond0[2])
        _gt_conds[0].append(seed_cond0[3])
        _gt_conds[0].append(seed_cond0[4])
        _gt_conds[0].append(seed_cond0[5])
        _gt_conds[0].append(seed_cond0[6])
        _gt_conds[0].append(seed_cond0[7])



    #set_cond_tree(n_snp, _Gn, _max_ngauss, _gt_conds_index): # g01 < g02 < g03 : g indices (0, 1, 2, ...)
    #g1x = set_cond_tree(1, 1, 4, 2) # [2]
    #g1x = set_cond_tree(1, 1, 3, 2) # [2]

    # for each seed cond[i] : 0 or 1
    for i in range(0, n_core_seeds): 
        # child cond for _Gn+1
        _gt[0] = set_cond_child(seed_cond0[i], _Gn, _Gn+1)

        # child cond for _Gn+1+j
        for j1 in range(1, _max_ngauss-_Gn): # 
            # pre version
            _gt[j1] = set_cond_child(_gt[j1-1][1], _Gn, _Gn+1+j1)


        for j2 in range(1, _max_ngauss-_Gn+1):
            for k1 in range(0, j2-1):
                _gt_conds[j2].append(_gt[k1][0])
                #print(_gt[k1][0])

            _gt_conds[j2].append(_gt[j2-1][0])
            _gt_conds[j2].append(_gt[j2-1][1])
            #print(_gt[j2][0])
            #print(_gt[j2][1])
            #print("")

    #for i in range(0, _max_ngauss-_Gn+1):
    #    for j in range(0, n_core_seeds*(i+1)):
    #        print(i, j, _gt_conds[i][j])
    #        #print("seheon")
    #    print("")

    #print(_gt_conds)

    #for i in range(0, _max_ngauss-_Gn+1):
    #    for j in range(0, 2**(_Gn-1)*(i+1)):
    #        print(_gt_conds[i][j])
    #    print("")

    return _gt_conds[_gt_conds_index]
#-- END OF SUB-ROUTINE____________________________________________________________#









#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# Gn-X conditions for the baygaud output with _ng maximum gaussian fits
# generate seed condition for each n_snp
def set_cond_tree(n_snp, _Gn, _max_ngauss, _gt_conds_index): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    if n_snp == 1: # (n-0)
        n_core_seeds = 1
    elif n_snp == 2: # 2**0 x (n-1)
        n_core_seeds = 1
    else: # 2**(n-2) x (n-(n-1))
        n_core_seeds = 2**(n_snp-2)

    seed_cond0 = [[[0 for i in range(3)] for j in range(0)] for k in range(n_core_seeds)]
    base_cond0 = [[[0 for i in range(3)] for j in range(0)] for k in range(n_core_seeds)]
    # seed_cond1 : should be 2
    # seed_cond1 = [[[0 for i in range(3)] for j in range(0)] for k in range(2)]

    _gt = [0 for i in range(_max_ngauss)]
    _gt_conds = [[0 for i in range(0)] for j in range(_max_ngauss-_Gn+1)]

    if n_snp == 1:
        _gt_conds[0].append(seed_cond0[0])

    # n_snp == 2
    if n_snp == 2:
        for i in range(0, 1):
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 1, '0'])

        _gt_conds[0].append(seed_cond0[0])


    #base00_list = [[0, 1, '1'], [0, 1, '0']]
    base_cond0[0] = [[0, 1, '1'], [0, 1, '0']]

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # make seeds for contition matrix 
    # additional conditions for _Gn will be added to these seeds below

    # n_snp >=3
    if n_snp >= 3:
        for n_snp_i in range(3, n_snp+1):
            # -------------------------------------------------------------------------
            base_cond0_i = 0
            for n0 in range(0, 2**(n_snp_i-2)):
    
                if n0 % 2 == 0: # even 
                    # -------------------------------------------------------------------------
                    # pair 1
                    #seed_cond0[n0].append(base00_list)
                    seed_cond0[n0] = cp.deepcopy(base_cond0[base_cond0_i])
            
                    _add_inc1 = cp.deepcopy(seed_cond0[n0])
                    delt_cond1 = _add_inc1[-1][1] - _add_inc1[-1][0]
    
                    _add_inc1[-2][0] += delt_cond1
                    _add_inc1[-2][1] = n_snp_i-1
    
                    _add_inc1[-1][0] += delt_cond1
                    _add_inc1[-1][1] = n_snp_i-1
    
                    seed_cond0[n0].append(_add_inc1[-2])
                    seed_cond0[n0].append(_add_inc1[-1])
                    #print(seed_cond0[n0])
    
                    # -------------------------------------------------------------------------
                    # pair 2
                    #seed_cond0[n0].append(base00_list)
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
                    #print(seed_cond0[n0+1])
    
                    # increase base_cond0_index
                    base_cond0_i += 1
    
            #print("")
            # update base_cond0 for the next n_snp
            for n0 in range(0, 2**(n_snp_i-2)):
                base_cond0[n0] = cp.deepcopy(seed_cond0[n0])
    
                # -------------------------------------------------------------------------
                # -------------------------------------------------------------------------
    
        for n_snp_i in range(0, 2**(n_snp-2)): # 3:2, 4:4, 5:8, 6:16 ...
            _gt_conds[0].append(seed_cond0[n_snp_i])


    #set_cond_tree(n_snp, _Gn, _max_ngauss, _gt_conds_index): # g01 < g02 < g03 : g indices (0, 1, 2, ...)
    #g1x = set_cond_tree(1, 1, 4, 2) # [2]
    #g1x = set_cond_tree(1, 1, 3, 2) # [2]

    # for each seed cond[i] : 0 or 1
    for i in range(0, n_core_seeds): 
        # child cond for _Gn+1
        _gt[0] = set_cond_child(seed_cond0[i], _Gn, _Gn+1)

        # child cond for _Gn+1+j
        for j1 in range(1, _max_ngauss-_Gn): # 
            # pre version
            _gt[j1] = set_cond_child(_gt[j1-1][1], _Gn, _Gn+1+j1)


        for j2 in range(1, _max_ngauss-_Gn+1):
            for k1 in range(0, j2-1):
                _gt_conds[j2].append(_gt[k1][0])
                #print(_gt[k1][0])

            _gt_conds[j2].append(_gt[j2-1][0])
            _gt_conds[j2].append(_gt[j2-1][1])
            #print(_gt[j2][0])
            #print(_gt[j2][1])
            #print("")

    #for i in range(0, _max_ngauss-_Gn+1):
    #    for j in range(0, n_core_seeds*(i+1)):
    #        print(i, j, _gt_conds[i][j])
    #        #print("seheon")
    #    print("")

    #print(_gt_conds)

    #for i in range(0, _max_ngauss-_Gn+1):
    #    for j in range(0, 2**(_Gn-1)*(i+1)):
    #        print(_gt_conds[i][j])
    #    print("")

    return _gt_conds[_gt_conds_index]
#-- END OF SUB-ROUTINE____________________________________________________________#




#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# Gn-X conditions for the baygaud output with _ng maximum gaussian fits
# generate seed condition for each n_snp
def set_cond_tree_dev(n_snp, _Gn, _max_ngauss, _gt_conds_index): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    if n_snp == 1: # (n-0)
        n_core_seeds = 1
    elif n_snp == 2: # 2**0 x (n-1)
        n_core_seeds = 1
    else: # 2**(n-2) x (n-(n-1))
        n_core_seeds = 2**(n_snp-2)

    seed_cond0 = [[[0 for i in range(3)] for j in range(0)] for k in range(n_core_seeds)]
    base_cond0 = [[[0 for i in range(3)] for j in range(0)] for k in range(n_core_seeds)]
    # seed_cond1 : should be 2
    # seed_cond1 = [[[0 for i in range(3)] for j in range(0)] for k in range(2)]

    _gt = [0 for i in range(_max_ngauss)]
    _gt_conds = [[0 for i in range(0)] for j in range(_max_ngauss-_Gn+1)]

    if n_snp == 1:
        _gt_conds[0].append(seed_cond0[0])

    # n_snp == 2
    if n_snp == 2:
        for i in range(0, 1):
            seed_cond0[i].append([0, 1, '1'])
            seed_cond0[i].append([0, 1, '0'])

        _gt_conds[0].append(seed_cond0[0])


    #base00_list = [[0, 1, '1'], [0, 1, '0']]
    base_cond0[0] = [[0, 1, '1'], [0, 1, '0']]

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # make seeds for contition matrix 
    # additional conditions for _Gn will be added to these seeds below

    # n_snp >=3
    if n_snp >= 3:
        for n_snp_i in range(3, n_snp+1):
            # -------------------------------------------------------------------------
            base_cond0_i = 0
            for n0 in range(0, 2**(n_snp_i-2)):
    
                if n0 % 2 == 0: # even 
                    # -------------------------------------------------------------------------
                    # pair 1
                    #seed_cond0[n0].append(base00_list)
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
    
                    # -------------------------------------------------------------------------
                    # pair 2
                    #seed_cond0[n0].append(base00_list)
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
    
                    # increase base_cond0_index
                    base_cond0_i += 1
    
            print("")
            # update base_cond0 for the next n_snp
            for n0 in range(0, 2**(n_snp_i-2)):
                base_cond0[n0] = cp.deepcopy(seed_cond0[n0])
    
                # -------------------------------------------------------------------------
                # -------------------------------------------------------------------------
    
        for n_snp_i in range(0, 2**(n_snp-2)): # 3:2, 4:4, 5:8, 6:16 ...
            _gt_conds[0].append(seed_cond0[n_snp_i])


    #set_cond_tree(n_snp, _Gn, _max_ngauss, _gt_conds_index): # g01 < g02 < g03 : g indices (0, 1, 2, ...)
    #g1x = set_cond_tree(1, 1, 4, 2) # [2]
    #g1x = set_cond_tree(1, 1, 3, 2) # [2]

    # for each seed cond[i] : 0 or 1
    for i in range(0, n_core_seeds): 
        # child cond for _Gn+1
        _gt[0] = set_cond_child(seed_cond0[i], _Gn, _Gn+1)

        # child cond for _Gn+1+j
        for j1 in range(1, _max_ngauss-_Gn): # 
            # pre version
            _gt[j1] = set_cond_child(_gt[j1-1][1], _Gn, _Gn+1+j1)


        for j2 in range(1, _max_ngauss-_Gn+1):
            for k1 in range(0, j2-1):
                _gt_conds[j2].append(_gt[k1][0])
                #print(_gt[k1][0])

            _gt_conds[j2].append(_gt[j2-1][0])
            _gt_conds[j2].append(_gt[j2-1][1])
            #print(_gt[j2][0])
            #print(_gt[j2][1])
            #print("")

    #for i in range(0, _max_ngauss-_Gn+1):
    #    for j in range(0, n_core_seeds*(i+1)):
    #        print(i, j, _gt_conds[i][j])
    #        #print("seheon")
    #    print("")

    #print(_gt_conds)

    #for i in range(0, _max_ngauss-_Gn+1):
    #    for j in range(0, 2**(_Gn-1)*(i+1)):
    #        print(_gt_conds[i][j])
    #    print("")

    return _gt_conds[_gt_conds_index]
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# set Gn-X conditions for the baygaud output with _ng maximum gaussian fits
def set_cond_tree_bg(n_snp, _Gn, _max_ngauss): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    #_Gn = 2
    #_max_ngauss = 5
    if _Gn == 1: # check here ...
        n_core_seeds = 2**(_Gn-1)
    else:
        n_core_seeds = 2**(_Gn-2)

    seed_cond0 = [[[0 for i in range(3)] for j in range(0)] for k in range(n_core_seeds)]
    # seed_cond1 : should be 2
    seed_cond1 = [[[0 for i in range(3)] for j in range(0)] for k in range(2)]

    _gt = [0 for i in range(_max_ngauss)]
    _gt_conds = [[0 for i in range(0)] for j in range(_max_ngauss-_Gn+1)]

    if n_snp == 1:
        _gt_conds[0].append(seed_cond0[0])

    # n_snp == 2
    if n_snp == 2:
        for i in range(0, 1):
            seed_cond0[i].append([1, 2, '1'])
            seed_cond0[i].append([1, 2, '0'])

        _gt_conds[0].append(seed_cond0[0])

    # n_snp == 3
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
    
        #_gt[0] = set_cond_child(seed_cond0[0], _Gn, _Gn+1)
        #_gt[0] = set_cond_child(seed_cond0[1], _Gn, _Gn+1)
    
        # G4-
        _gt_conds[0].append(seed_cond0[0])
        _gt_conds[0].append(seed_cond0[1])
        _gt_conds[0].append(seed_cond0[2])
        _gt_conds[0].append(seed_cond0[3])


    # for each seed cond[i] : 0 or 1
    for i in range(0, n_core_seeds): 
        # child cond for _Gn+1
        _gt[0] = set_cond_child(seed_cond0[i], _Gn, _Gn+1)

        # child cond for _Gn+1+j
        for j1 in range(1, _max_ngauss-_Gn): # 
            _gt[j1] = set_cond_child(_gt[j1-1][1], _Gn, _Gn+1+j1)


        for j2 in range(1, _max_ngauss-_Gn+1):
            for k1 in range(0, j2-1):
                _gt_conds[j2].append(_gt[k1][0])
                #print(_gt[k1][0])

            _gt_conds[j2].append(_gt[j2-1][0])
            _gt_conds[j2].append(_gt[j2-1][1])
            #print(_gt[j2][0])
            #print(_gt[j2][1])
            print("")

    for i in range(0, _max_ngauss-_Gn+1):
        for j in range(0, n_core_seeds*(i+1)):
            print(_gt_conds[i][j])
        print("")

    return _gt_conds
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# set G2-X conditions for the baygaud output with _ng maximum gaussian fits
def set_cond_tree_1(_Gn, _max_ngauss): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    #_Gn = 2
    #_max_ngauss = 5
    n_core_seeds = 2**(_Gn-1)

    seed_cond0 = [[[0 for i in range(3)] for j in range(0)] for k in range(n_core_seeds)]
    # seed_cond1 : should be 2
    seed_cond1 = [[[0 for i in range(3)] for j in range(0)] for k in range(2)]

    _gt = [0 for i in range(_max_ngauss)]
    _gt_conds = [[0 for i in range(0)] for j in range(_max_ngauss-_Gn+1)]

    #for i in range(0, 1):
    #    seed_cond0[i].append([1, 2, '0'])

    # G2-
    _gt_conds[0].append(seed_cond0[0])

    # for each seed cond[i] : 0 or 1
    for i in range(0, n_core_seeds): 
        # child cond for _Gn+1
        _gt[0] = set_cond_child(seed_cond0[i], _Gn, _Gn+1)

        # child cond for _Gn+1+j
        for j1 in range(1, _max_ngauss-_Gn): # 
            _gt[j1] = set_cond_child(_gt[j1-1][1], _Gn, _Gn+1+j1)


        for j2 in range(1, _max_ngauss-_Gn+1):
            for k1 in range(0, j2-1):
                _gt_conds[j2].append(_gt[k1][0])
                #print(_gt[k1][0])

            _gt_conds[j2].append(_gt[j2-1][0])
            _gt_conds[j2].append(_gt[j2-1][1])
            #print(_gt[j2][0])
            #print(_gt[j2][1])
            print("")

    # --------------------
    # G3 : 4
    #print(_gt_conds[0][0])
    #print(_gt_conds[0][1])
    #print(_gt_conds[0][2])
    #print(_gt_conds[0][3])
    #print("")
    # --------------------
    # G3 : 5
    #print(_gt_conds[1][0])
    #print(_gt_conds[1][1])
    #print(_gt_conds[1][2])
    #print(_gt_conds[1][3])
    #print(_gt_conds[1][4])
    #print(_gt_conds[1][5])
#
    #for i in range(0, _max_ngauss-_Gn):
    #    for j in range(0, 2**(_Gn-2)*(i-1+3)):
    #        print(_gt_conds[i][j])
    #    print("")

    for i in range(0, _max_ngauss-_Gn+1):
        for j in range(0, 2**(_Gn-1)*(i+1)):
            print(_gt_conds[i][j])
        print("")

    return _gt_conds
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# set G2-X conditions for the baygaud output with _ng maximum gaussian fits
def set_cond_tree_2(_Gn, _max_ngauss): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    #_Gn = 2
    #_max_ngauss = 5
    n_core_seeds = 2**(_Gn-2)

    seed_cond0 = [[[0 for i in range(3)] for j in range(0)] for k in range(n_core_seeds)]
    # seed_cond1 : should be 2
    seed_cond1 = [[[0 for i in range(3)] for j in range(0)] for k in range(2)]

    _gt = [0 for i in range(_max_ngauss)]
    _gt_conds = [[0 for i in range(0)] for j in range(_max_ngauss-_Gn+1)]

    for i in range(0, 1):
        seed_cond0[i].append([1, 2, '1'])
        seed_cond0[i].append([1, 2, '0'])
    # G2-
    _gt_conds[0].append(seed_cond0[0])

    # for each seed cond[i] : 0 or 1
    for i in range(0, n_core_seeds): 
        # child cond for _Gn+1
        _gt[0] = set_cond_child(seed_cond0[i], _Gn, _Gn+1)

        # child cond for _Gn+1+j
        for j1 in range(1, _max_ngauss-_Gn): # 
            _gt[j1] = set_cond_child(_gt[j1-1][1], _Gn, _Gn+1+j1)


        for j2 in range(1, _max_ngauss-_Gn+1):
            for k1 in range(0, j2-1):
                _gt_conds[j2].append(_gt[k1][0])
                #print(_gt[k1][0])

            _gt_conds[j2].append(_gt[j2-1][0])
            _gt_conds[j2].append(_gt[j2-1][1])
            #print(_gt[j2][0])
            #print(_gt[j2][1])
            print("")

    # --------------------
    # G3 : 4
    #print(_gt_conds[0][0])
    #print(_gt_conds[0][1])
    #print(_gt_conds[0][2])
    #print(_gt_conds[0][3])
    #print("")
    # --------------------
    # G3 : 5
    #print(_gt_conds[1][0])
    #print(_gt_conds[1][1])
    #print(_gt_conds[1][2])
    #print(_gt_conds[1][3])
    #print(_gt_conds[1][4])
    #print(_gt_conds[1][5])
#
    #for i in range(0, _max_ngauss-_Gn):
    #    for j in range(0, 2**(_Gn-2)*(i-1+3)):
    #        print(_gt_conds[i][j])
    #    print("")

    for i in range(0, _max_ngauss-_Gn+1):
        for j in range(0, 2**(_Gn-2)*(i+1)):
            print(_gt_conds[i][j])
        print("")

    return _gt_conds
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# set G3-X conditions for the baygaud output with _ng maximum gaussian fits
def set_cond_tree_3(_Gn, _max_ngauss): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    #_Gn = 3
    #_max_ngauss = 6
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

    #_gt[0] = set_cond_child(seed_cond0[0], _Gn, _Gn+1)
    #_gt[0] = set_cond_child(seed_cond0[1], _Gn, _Gn+1)

    # G3-
    _gt_conds[0].append(seed_cond0[0])
    _gt_conds[0].append(seed_cond0[1])

    #print(_gt_conds[0][0])
    #print(_gt_conds[0][1])


    # for each seed cond[i] : 0 or 1
    for i in range(0, n_core_seeds): 
        # child cond for _Gn+1
        _gt[0] = set_cond_child(seed_cond0[i], _Gn, _Gn+1)

        # child cond for _Gn+1+j
        for j1 in range(1, _max_ngauss-_Gn): # 
            _gt[j1] = set_cond_child(_gt[j1-1][1], _Gn, _Gn+1+j1)


        for j2 in range(1, _max_ngauss-_Gn+1):
            for k1 in range(0, j2-1):
                _gt_conds[j2].append(_gt[k1][0])
                #print(_gt[k1][0])

            _gt_conds[j2].append(_gt[j2-1][0])
            _gt_conds[j2].append(_gt[j2-1][1])
            #print(_gt[j2][0])
            #print(_gt[j2][1])
            print("")



    # --------------------
    # G3 : 4
    #print(_gt_conds[0][0])
    #print(_gt_conds[0][1])
    #print(_gt_conds[0][2])
    #print(_gt_conds[0][3])
    #print("")
    # --------------------
    # G3 : 5
    #print(_gt_conds[1][0])
    #print(_gt_conds[1][1])
    #print(_gt_conds[1][2])
    #print(_gt_conds[1][3])
    #print(_gt_conds[1][4])
    #print(_gt_conds[1][5])
#
    #for i in range(0, _max_ngauss-_Gn):
    #    for j in range(0, 2*(i+2)):
    #        print(_gt_conds[i][j])
    #    print("")

    for i in range(0, _max_ngauss-_Gn+1):
        for j in range(0, 2**(_Gn-2)*(i+1)):
            print(_gt_conds[i][j])
        print("")


    return _gt_conds
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# set G4-X conditions for the baygaud output with _ng maximum gaussian fits
def set_cond_tree_4(_Gn, _max_ngauss): # g01 < g02 < g03 : g indices (0, 1, 2, ...)

    #_Gn = 4
    #_max_ngauss = 6
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

    #_gt[0] = set_cond_child(seed_cond0[0], _Gn, _Gn+1)
    #_gt[0] = set_cond_child(seed_cond0[1], _Gn, _Gn+1)

    # G4-
    _gt_conds[0].append(seed_cond0[0])
    _gt_conds[0].append(seed_cond0[1])
    _gt_conds[0].append(seed_cond0[2])
    _gt_conds[0].append(seed_cond0[3])

    #print(_gt_conds[0][0])
    #print(_gt_conds[0][1])


    # for each seed cond[i] : 0 or 1
    for i in range(0, n_core_seeds): 
        # child cond for _Gn+1
        _gt[0] = set_cond_child(seed_cond0[i], _Gn, _Gn+1)

        # child cond for _Gn+1+j
        for j1 in range(1, _max_ngauss-_Gn): # 
            _gt[j1] = set_cond_child(_gt[j1-1][1], _Gn, _Gn+1+j1)


        for j2 in range(1, _max_ngauss-_Gn+1):
            for k1 in range(0, j2-1):
                _gt_conds[j2].append(_gt[k1][0])
                #print(_gt[k1][0])

            _gt_conds[j2].append(_gt[j2-1][0])
            _gt_conds[j2].append(_gt[j2-1][1])
            #print(_gt[j2][0])
            #print(_gt[j2][1])
            print("")

    for i in range(0, _max_ngauss-_Gn+1):
        for j in range(0, 2**(_Gn-2)*(i+1)):
            print(_gt_conds[i][j])
        print("")


    return _gt_conds
#-- END OF SUB-ROUTINE____________________________________________________________#




#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g4_opt_bf_snp3(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g123_sn_pass, g01, g02, g03): # g01 < g02 < g03 : g indices (0, 1, 2, ...)
#________________________________________________________________________#

    #------------------------------------------------------#
    # G1-0
    #------------------------------------------------------#
    # if (z1/z2 > bf_limit): --> g1
    g_sort_n = g01
    _cond_N = 1
    _cond1 = ([g01, g02, '0'], )
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g123_sn_pass)

    #------------------------------------------------------#
    # G2-0
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 > bf_limit): --> g2
    g_sort_n = g02
    _cond_N = 3
    _cond3 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '0'])
    g2_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond3, _cond_N, g123_sn_pass)

    #------------------------------------------------------#
    # G3-0
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 > g3): --> g3
    g_sort_n = g03
    _cond_N = 4
    _cond4 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '1'], [g02, g03, '0'])
    g3_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond4, _cond_N, g123_sn_pass)

    
    #------------------------------------------------------#
    # G2-1
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 < g3): --> g2
    g_sort_n = g02
    _cond_N = 4
    _cond4 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '1'], [g02, g03, '1'])
    g2_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond4, _cond_N, g123_sn_pass)

    
    #------------------------------------------------------#
    # G1-1
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 > bf_limit): --> g1
    g_sort_n = g01
    _cond_N = 3
    _cond3 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '0'])
    g1_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond3, _cond_N, g123_sn_pass)



    #------------------------------------------------------#
    # G3-1
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 > g3): --> g3
    g_sort_n = g03
    _cond_N = 4
    _cond4 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '1'], [g01, g03, '1'])
    g3_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond4, _cond_N, g123_sn_pass)

    
    #------------------------------------------------------#
    # G1-2
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 < g3): --> g1
    g_sort_n = g01
    _cond_N = 4
    _cond4 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '1'], [g01, g03, '1'])
    g1_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond4, _cond_N, g123_sn_pass)



    #print(g1_2)
    g4_opt_snp3 =  g1_0 + g1_1 + g1_2 \
            + g2_0 + g2_1 \
            + g3_0 + g3_1 + 10*6 # -10 flag see above
    #print(g4_opt_snp3[0])

    return g4_opt_snp3
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g4_opt_bf_snp4(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1234_sn_pass, g01, g02, g03, g04): # g01 < g02 < g03 < g04 : g indices (0, 1, 2, ...)
#________________________________________________________________________#

    #------------------------------------------------------#
    # G1-0
    #------------------------------------------------------#
    # if (z1/z2 > bf_limit): --> g1
    g_sort_n = g01
    _cond_N = 1
    _cond1 = ([g01, g02, '0'], )
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G2-0
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 > bf_limit): --> g2
    g_sort_n = g02
    _cond_N = 3
    _cond3 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '0'])
    g2_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond3, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G3-0
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 > g3) & (z3/z4 > bf_limit): --> g3
    g_sort_n = g03
    _cond_N = 5
    _cond5 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '1'], [g02, g03, '0'], [g03, g04, '0'])
    g3_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond5, _cond_N, g1234_sn_pass)
  
    #------------------------------------------------------#
    # G4-0
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 > g3) & (z3/z4 < bf_limit) & (g3 > g4): --> g4
    g_sort_n = g04
    _cond_N = 6
    _cond6 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '1'], [g02, g03, '0'], [g03, g04, '1'], [g03, g04, '0'])
    g4_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G3-1
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 > g3) & (z3/z4 < bf_limit) & (g3 < g4): --> g3
    g_sort_n = g03
    _cond_N = 6
    _cond6 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '1'], [g02, g03, '0'], [g03, g04, '1'], [g03, g04, '1'])
    g3_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G2-1
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 < g3) & (z2/z4 > bf_limit): --> g2
    g_sort_n = g02
    _cond_N = 5
    _cond5 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '1'], [g02, g03, '1'], [g02, g04, '0'])
    g2_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond5, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G4-1
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 < g3) & (z2/z4 < bf_limit) & (g2 > g4): --> g4
    g_sort_n = g04
    _cond_N = 6
    _cond6 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '1'], [g02, g03, '1'], [g02, g04, '1'], [g02, g04, '0'])
    g4_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G2-2
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 < g3) & (z2/z4 < bf_limit) & (g2 < g4): --> g2
    g_sort_n = g02
    _cond_N = 6
    _cond6 = ([g01, g02, '1'], [g01, g02, '0'], [g02, g03, '1'], [g02, g03, '1'], [g02, g04, '1'], [g02, g04, '1'])
    g2_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G1-1
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 > bf_limit): --> g1
    g_sort_n = g01
    _cond_N = 3
    _cond3 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '0'])
    g1_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond3, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G3-2
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 > g3) & (z3/z4 > bf_limit): --> g3
    g_sort_n = g03
    _cond_N = 5
    _cond5 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '1'], [g01, g03, '0'], [g03, g04, '0'])
    g3_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond5, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G4-2
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 > g3) & (z3/z4 < bf_limit) & (g3 > g4): --> g4
    g_sort_n = g04
    _cond_N = 6
    _cond6 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '1'], [g01, g03, '0'], [g03, g04, '1'], [g03, g04, '0'])
    g4_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G3-3
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 > g3) & (z3/z4 < bf_limit) & (g3 < g4): --> g3
    g_sort_n = g03
    _cond_N = 6
    _cond6 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '1'], [g01, g03, '0'], [g03, g04, '1'], [g03, g04, '1'])
    g3_3 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G1-2
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 < g3) & (z1/z4 > bf_limit): --> g1
    g_sort_n = g01
    _cond_N = 5
    _cond5 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '1'], [g01, g03, '1'], [g01, g04, '0'])
    g1_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond5, _cond_N, g1234_sn_pass)


    #------------------------------------------------------#
    # G4-3
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 < g3) & (z1/z4 < bf_limit) & (g1 > g4): --> g4
    g_sort_n = g04
    _cond_N = 6
    _cond6 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '1'], [g01, g03, '1'], [g01, g04, '1'], [g01, g04, '0'])
    g4_3 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G1-3
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 < g3) & (z1/z4 < bf_limit) & (g1 < g4): --> g1
    g_sort_n = g01
    _cond_N = 6
    _cond6 = ([g01, g02, '1'], [g01, g02, '1'], [g01, g03, '1'], [g01, g03, '1'], [g01, g04, '1'], [g01, g04, '1'])
    g1_3 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    g4_opt_snp4 = g1_0 + g1_1 + g1_2 + g1_3 \
            + g2_0 + g2_1 + g2_2 \
            + g3_0 + g3_1 + g3_2 + g3_3 \
            + g4_0 + g4_1 + g4_2 + g4_3 + 10*14 # -10 flag

    return g4_opt_snp4
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g5_opt_bf_snp5(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g1234_sn_pass, g01, g02, g03, g05): # g01 < g02 < g03 < g04 < g05: g indices (0, 1, 2, ...)
#________________________________________________________________________#

    #------------------------------------------------------#
    # G1-0
    #------------------------------------------------------#
    # if (z1/z2 > bf_limit): --> g1
    g_sort_n = g01
    _cond_N = 1
    _cond1 = ([g01, g02, '>'], )
    g1_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond1, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G2-0
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 > bf_limit): --> g2
    g_sort_n = g02
    _cond_N = 3
    _cond3 = ([g01, g02, '<'], [g01, g02, '>'], [g02, g03, '>'])
    g2_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond3, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G3-0
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 > g3) & (z3/z4 > bf_limit): --> g3
    g_sort_n = g03
    _cond_N = 5
    _cond5 = ([g01, g02, '<'], [g01, g02, '>'], [g02, g03, '<'], [g02, g03, '>'], [g03, g04, '>'])
    g3_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond5, _cond_N, g1234_sn_pass)
  
    #------------------------------------------------------#
    # G4-0
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 > g3) & (z3/z4 < bf_limit) & (g3 > g4): --> g4
    g_sort_n = g04
    _cond_N = 6
    _cond6 = ([g01, g02, '<'], [g01, g02, '>'], [g02, g03, '<'], [g02, g03, '>'], [g03, g04, '<'], [g03, g04, '>'])
    g4_0 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G3-1
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 > g3) & (z3/z4 < bf_limit) & (g3 < g4): --> g3
    g_sort_n = g03
    _cond_N = 6
    _cond6 = ([g01, g02, '<'], [g01, g02, '>'], [g02, g03, '<'], [g02, g03, '>'], [g03, g04, '<'], [g03, g04, '<'])
    g3_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G2-1
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 < g3) & (z2/z4 > bf_limit): --> g2
    g_sort_n = g02
    _cond_N = 5
    _cond5 = ([g01, g02, '<'], [g01, g02, '>'], [g02, g03, '<'], [g02, g03, '<'], [g02, g04, '>'])
    g2_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond5, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G4-1
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 < g3) & (z2/z4 < bf_limit) & (g2 > g4): --> g4
    g_sort_n = g04
    _cond_N = 6
    _cond6 = ([g01, g02, '<'], [g01, g02, '>'], [g02, g03, '<'], [g02, g03, '<'], [g02, g04, '<'], [g02, g04, '>'])
    g4_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G2-2
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 < g3) & (z2/z4 < bf_limit) & (g2 < g4): --> g2
    g_sort_n = g02
    _cond_N = 6
    _cond6 = ([g01, g02, '<'], [g01, g02, '>'], [g02, g03, '<'], [g02, g03, '<'], [g02, g04, '<'], [g02, g04, '<'])
    g2_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G1-1
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 > bf_limit): --> g1
    g_sort_n = g01
    _cond_N = 3
    _cond3 = ([g01, g02, '<'], [g01, g02, '<'], [g01, g03, '>'])
    g1_1 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond3, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G3-2
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 > g3) & (z3/z4 > bf_limit): --> g3
    g_sort_n = g03
    _cond_N = 5
    _cond5 = ([g01, g02, '<'], [g01, g02, '<'], [g01, g03, '<'], [g01, g03, '>'], [g03, g04, '>'])
    g3_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond5, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G4-2
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 > g3) & (z3/z4 < bf_limit) & (g3 > g4): --> g4
    g_sort_n = g04
    _cond_N = 6
    _cond6 = ([g01, g02, '<'], [g01, g02, '<'], [g01, g03, '<'], [g01, g03, '>'], [g03, g04, '<'], [g03, g04, '>'])
    g4_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G3-3
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 > g3) & (z3/z4 < bf_limit) & (g3 < g4): --> g3
    g_sort_n = g03
    _cond_N = 6
    _cond6 = ([g01, g02, '<'], [g01, g02, '<'], [g01, g03, '<'], [g01, g03, '>'], [g03, g04, '<'], [g03, g04, '<'])
    g3_3 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G1-2
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 < g3) & (z1/z4 > bf_limit): --> g1
    g_sort_n = g01
    _cond_N = 5
    _cond5 = ([g01, g02, '<'], [g01, g02, '<'], [g01, g03, '<'], [g01, g03, '<'], [g01, g04, '>'])
    g1_2 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond5, _cond_N, g1234_sn_pass)


    #------------------------------------------------------#
    # G4-3
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 < g3) & (z1/z4 < bf_limit) & (g1 > g4): --> g4
    g_sort_n = g04
    _cond_N = 6
    _cond6 = ([g01, g02, '<'], [g01, g02, '<'], [g01, g03, '<'], [g01, g03, '<'], [g01, g04, '<'], [g01, g04, '>'])
    g4_3 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    #------------------------------------------------------#
    # G1-3
    #------------------------------------------------------#
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 < g3) & (z1/z4 < bf_limit) & (g1 < g4): --> g1
    g_sort_n = g01
    _cond_N = 6
    _cond6 = ([g01, g02, '<'], [g01, g02, '<'], [g01, g03, '<'], [g01, g03, '<'], [g01, g04, '<'], [g01, g04, '<'])
    g1_3 = g_ms_bf_cond_N(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, g_sort_n, _cond6, _cond_N, g1234_sn_pass)

    g4_opt_snp4 = g1_0 + g1_1 + g1_2 + g1_3 \
            + g2_0 + g2_1 + g2_2 \
            + g3_0 + g3_1 + g3_2 + g3_3 \
            + g4_0 + g4_1 + g4_2 + g4_3 + 10*14 # -10 flag

    return g4_opt_snp4
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def g5_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, sn_pass_ng_opt, bf_limit):
#________________________________________________________________________#
    #------------------------------------------------------#
    # G1-0
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 > bf_limit): --> g1

    g1_sorted = g_num_sort[0, :, :]

    # take values of sn_pass_ng_opt given g1_sorted indices
    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]

    g1_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) > np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g_opt[g1_cond1] += 1
    g1_0 = np.array([np.where(g_opt > 0, g1_sorted, 0)])
    #print(g1_0)
    
    
    #------------------------------------------------------#
    # G2-0
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 > bf_limit): --> g2

    g2_sorted = g_num_sort[1, :, :]

    # take values of sn_pass_ng_opt given g2_sorted indices
    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g2_sorted[np.newaxis], axis=0)[0]

    g2_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond2 = np.where((g_num_sort[0, :, :] > g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    g2_cond3 = np.where(((bevidences_sort[1, :, :] - bevidences_sort[2, :, :]) > np.log(bf_limit)) & (sn_pass_ng_opt_value == (g2_sorted+1)))
    
    g_opt[g2_cond1] += 1
    g_opt[g2_cond2] += 1
    g_opt[g2_cond3] += 1
    g2_0 = np.array([np.where(g_opt > 2, g2_sorted, 0)])
    #print(g2_0)
    
    #------------------------------------------------------#
    # G3-0
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 > g3) & (z3/z4 > bf_limit): --> g3

    g3_sorted = g_num_sort[2, :, :]

    # take values of sn_pass_ng_opt given g3_sorted indices
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
    #print(g3_0)
    
    
    #------------------------------------------------------#
    # G4-0
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 > g3) & (z3/z4 < bf_limit) & (g3 > g4) & (z4/z5 > bf_limit): --> g4

    g4_sorted = g_num_sort[3, :, :]

    # take values of sn_pass_ng_opt given g4_sorted indices
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
    #print(g4_0)

    #------------------------------------------------------#
    # G5-0
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 > g3) & (z3/z4 < bf_limit) & (g3 > g4) & (z4/z5 < bf_limit) & (g4 > g5): --> g5

    g5_sorted = g_num_sort[4, :, :]

    # take values of sn_pass_ng_opt given g5_sorted indices
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
    #print(g5_0)


    #------------------------------------------------------#
    # G4-1
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 > g3) & (z3/z4 < bf_limit) & (g3 > g4) & (z4/z5 < bf_limit) & (g4 < g5): --> g4

    g4_sorted = g_num_sort[3, :, :]

    # take values of sn_pass_ng_opt given g4_sorted indices
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
    #print(g4_0)


    #------------------------------------------------------#
    # G3-1
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 > g3) & (z3/z4 < bf_limit) & (g3 < g4) & (z3/z5 > bf_limit): --> g3

    g3_sorted = g_num_sort[2, :, :]

    # take values of sn_pass_ng_opt given g3_sorted indices
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
    #print(g3_1)

    #------------------------------------------------------#
    # G5-1
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 > g3) & (z3/z4 < bf_limit) & (g3 < g4) & (z3/z5 < bf_limit) & (g3 > g5): --> g5

    g5_sorted = g_num_sort[4, :, :]

    # take values of sn_pass_ng_opt given g5_sorted indices
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
    #print(g5_1)

    #------------------------------------------------------#
    # G3-2
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 > g3) & (z3/z4 < bf_limit) & (g3 < g4) & (z3/z5 < bf_limit) & (g3 < g5): --> g3

    g3_sorted = g_num_sort[2, :, :]

    # take values of sn_pass_ng_opt given g3_sorted indices
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
    #print(g3_2)

    #------------------------------------------------------#
    # G2-1
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 < g3) & (z2/z4 > bf_limit): --> g2

    g2_sorted = g_num_sort[1, :, :]

    # take values of sn_pass_ng_opt given g2_sorted indices
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
    #print(g2_1)

    #------------------------------------------------------#
    # G4-2
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 < g3) & (z2/z4 < bf_limit) & (g2 > g4) & (z4/z5 > bf_limit): --> g4

    g4_sorted = g_num_sort[3, :, :]

    # take values of sn_pass_ng_opt given g4_sorted indices
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
    #print(g4_2)


    #------------------------------------------------------#
    # G5-2
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 < g3) & (z2/z4 < bf_limit) & (g2 > g4) & (z4/z5 < bf_limit) & (g4 > g5): --> g5

    g5_sorted = g_num_sort[4, :, :]

    # take values of sn_pass_ng_opt given g5_sorted indices
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
    #print(g5_2)

    #------------------------------------------------------#
    # G4-3
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 < g3) & (z2/z4 < bf_limit) & (g2 > g4) & (z4/z5 < bf_limit) & (g4 < g5): --> g4

    g4_sorted = g_num_sort[3, :, :]

    # take values of sn_pass_ng_opt given g4_sorted indices
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
    #print(g4_3)

    #------------------------------------------------------#
    # G2-2
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 < g3) & (z2/z4 < bf_limit) & (g2 < g4) & (z2/z5 > bf_limit): --> g2

    g2_sorted = g_num_sort[1, :, :]

    # take values of sn_pass_ng_opt given g2_sorted indices
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
    #print(g2_2)


    #------------------------------------------------------#
    # G5-3
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 < g3) & (z2/z4 < bf_limit) & (g2 < g4) & (z2/z5 < bf_limit) & (g2 > g5): --> g5

    g5_sorted = g_num_sort[4, :, :]

    # take values of sn_pass_ng_opt given g5_sorted indices
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
    #print(g5_3)


    #------------------------------------------------------#
    # G2-3
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 > g2) & (z2/z3 < bf_limit) & (g2 < g3) & (z2/z4 < bf_limit) & (g2 < g4) & (z2/z5 < bf_limit) & (g2 < g5): --> g2

    g2_sorted = g_num_sort[1, :, :]

    # take values of sn_pass_ng_opt given g2_sorted indices
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
    #print(g2_3)


    #------------------------------------------------------#
    # G1-1
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 > bf_limit): --> g1

    g1_sorted = g_num_sort[0, :, :]

    # take values of sn_pass_ng_opt given g1_sorted indices
    sn_pass_ng_opt_value = np.take_along_axis(sn_pass_ng_opt, g1_sorted[np.newaxis], axis=0)[0]

    g1_cond1 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[1, :, :]) < np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond2 = np.where((g_num_sort[0, :, :] < g_num_sort[1, :, :]) & (sn_pass_ng_opt_value == (g1_sorted+1)))
    g1_cond3 = np.where(((bevidences_sort[0, :, :] - bevidences_sort[2, :, :]) > np.log(bf_limit)) & (sn_pass_ng_opt_value == (g1_sorted+1)))

    g_opt[g1_cond1] += 1
    g_opt[g1_cond2] += 1
    g_opt[g1_cond3] += 1
    g1_1 = np.array([np.where(g_opt > 2, g1_sorted, 0)])
    #print(g1_1)

    #------------------------------------------------------#
    # G3-3
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 > g3) & (z3/z4 > bf_limit): --> g3

    g3_sorted = g_num_sort[2, :, :]

    # take values of sn_pass_ng_opt given g3_sorted indices
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
    #print(g3_3)


    #------------------------------------------------------#
    # G4-4
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 > g3) & (z3/z4 < bf_limit) & (g3 > g4) & (z4/z5 > bf_limit): --> g4

    g4_sorted = g_num_sort[3, :, :]

    # take values of sn_pass_ng_opt given g4_sorted indices
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
    #print(g4_4)

    #------------------------------------------------------#
    # G5-4
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 > g3) & (z3/z4 < bf_limit) & (g3 > g4) & (z4/z5 < bf_limit) & (g4 > g5): --> g5

    g5_sorted = g_num_sort[4, :, :]

    # take values of sn_pass_ng_opt given g5_sorted indices
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
    #print(g5_4)

    #------------------------------------------------------#
    # G4-5
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 > g3) & (z3/z4 < bf_limit) & (g3 > g4) & (z4/z5 < bf_limit) & (g4 < g5): --> g4

    g4_sorted = g_num_sort[4, :, :]

    # take values of sn_pass_ng_opt given g4_sorted indices
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
    #print(g4_5)

    #------------------------------------------------------#
    # G3-4
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 > g3) & (z3/z4 < bf_limit) & (g3 < g4) & (z3/z5 > bf_limit): --> g3

    g3_sorted = g_num_sort[2, :, :]

    # take values of sn_pass_ng_opt given g3_sorted indices
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
    #print(g3_4)

    #------------------------------------------------------#
    # G5-5
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 > g3) & (z3/z4 < bf_limit) & (g3 < g4) & (z3/z5 < bf_limit) & (g3 > g5): --> g5

    g5_sorted = g_num_sort[4, :, :]

    # take values of sn_pass_ng_opt given g5_sorted indices
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
    #print(g5_5)

    #------------------------------------------------------#
    # G3-5
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 > g3) & (z3/z4 < bf_limit) & (g3 < g4) & (z3/z5 < bf_limit) & (g3 < g5): --> g3

    g3_sorted = g_num_sort[2, :, :]

    # take values of sn_pass_ng_opt given g3_sorted indices
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
    #print(g3_5)

    #------------------------------------------------------#
    # G1-2
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 < g3) & (z1/z4 > bf_limit): --> g1

    g1_sorted = g_num_sort[0, :, :]

    # take values of sn_pass_ng_opt given g1_sorted indices
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
    #print(g1_2)

    #------------------------------------------------------#
    # G4-6
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 < g3) & (z1/z4 < bf_limit) & (g1 > g4) & (z4/z5 > bf_limit): --> g4

    g4_sorted = g_num_sort[3, :, :]

    # take values of sn_pass_ng_opt given g4_sorted indices
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
    #print(g4_6)

    #------------------------------------------------------#
    # G5-6
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 < g3) & (z1/z4 < bf_limit) & (g1 > g4) & (z4/z5 < bf_limit) & (g4 > g5): --> g5

    g5_sorted = g_num_sort[4, :, :]

    # take values of sn_pass_ng_opt given g5_sorted indices
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
    #print(g5_6)

    #------------------------------------------------------#
    # G4-7
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 < g3) & (z1/z4 < bf_limit) & (g1 > g4) & (z4/z5 < bf_limit) & (g4 < g5): --> g4

    g4_sorted = g_num_sort[3, :, :]

    # take values of sn_pass_ng_opt given g4_sorted indices
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
    #print(g4_7)

    #------------------------------------------------------#
    # G1-3
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 < g3) & (z1/z4 < bf_limit) & (g1 < g4) & (z1/z5 > bf_limit): --> g1

    g1_sorted = g_num_sort[0, :, :]

    # take values of sn_pass_ng_opt given g1_sorted indices
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
    #print(g1_3)

    #------------------------------------------------------#
    # G5-7
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 < g3) & (z1/z4 < bf_limit) & (g1 < g4) & (z1/z5 < bf_limit) & (g1 > g5): --> g5

    g5_sorted = g_num_sort[4, :, :]

    # take values of sn_pass_ng_opt given g5_sorted indices
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
    #print(g5_7)

    #------------------------------------------------------#
    # G1-4
    #------------------------------------------------------#
    g_opt = np.zeros((_fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    # if (z1/z2 < bf_limit) & (g1 < g2) & (z1/z3 < bf_limit) & (g1 < g3) & (z1/z4 < bf_limit) & (g1 < g4) & (z1/z5 < bf_limit) & (g1 < g5): --> g1

    g1_sorted = g_num_sort[0, :, :]

    # take values of sn_pass_ng_opt given g1_sorted indices
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
    #print(g1_4)


    # add all the filters : N-gauss
    g5_opt = g1_0 + g1_1 + g1_2 + g1_3 + g1_4 \
            + g2_0 + g2_1 + g2_2 + g2_3 \
            + g3_0 + g3_1 + g3_2 + g3_3 + g3_4 + g3_5 \
            + g4_0 + g4_1 + g4_2 + g4_3 + g4_4 + g4_5 + g4_6 + g4_7 \
            + g5_0 + g5_1 + g5_2 + g5_3 + g5_4 + g5_5 + g5_6 + g5_7
    #print(g5_opt[0])

    return g5_opt
#-- END OF SUB-ROUTINE____________________________________________________________#





if __name__ == "__main__":

    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    # This is to ignore runtime warning message in the code;
    # particularly for np.where conditions which are processed no matter what the conditions are. <-- this is normal.
    np.seterr(divide='ignore', invalid='ignore')

    # start time
    _time_start = datetime.now()

    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> read fits header ...]")
    print("")
    print("")
    # ............................................................................. #
    # header info of the input datacube
    with fits.open(_params['wdir'] + '/' + _params['input_datacube'], 'update') as hdu:
        naxis1 = hdu[0].header['NAXIS1']
        naxis2 = hdu[0].header['NAXIS2']
        naxis3 = hdu[0].header['NAXIS3']
        cdelt3 = abs(hdu[0].header['CDELT3'])

    #ngauss = int(sys.argv[1])
    #ngauss = _params['max_ngauss']
    max_ngauss = _params['max_ngauss']
    outputdir_segs = _params['wdir'] + '/' + _params['_segdir']

    # read the input cube for extracting kinematic info
    cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s) # in km/s

    _x = np.linspace(0, 1, naxis3, dtype=np.float32)
    _vel_min = cube.spectral_axis.min().value
    _vel_max = cube.spectral_axis.max().value
    _params['vel_min'] = _vel_min   
    _params['vel_max'] = _vel_max


    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> load baygaud-segs from the output dir ...]")
    print("")
    print("")
    # ............................................................................. #
    _list_segs_bf = os.listdir(outputdir_segs)
    _list_segs_bf.sort(key = lambda x: x.split('.x')[1], reverse=False) # reverse with x pixels
    
    #combined_segs_bf = open('output.npy', "wb")
    #for _segs_bf in _list_segs_bf:
    #    _open_segs_bf = open(os.path.join(outputdir_segs, _segs_bf), "rb")
    #    shutil.copyfileobj(_open_segs_bf, combined_segs_bf)
    #    _open_segs_bf.close()
    #combined_segs_bf.close()

    print(_list_segs_bf)
    print()
    print()

    nparams = 2*(3*max_ngauss+2) + max_ngauss + 7
    gfit_results = np.full((naxis1, naxis2, max_ngauss, nparams), fill_value=-1E9, dtype=float)
    print(gfit_results.shape)
    
    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> load gfit_results from the baygaud-segs output ...]")
    print("")
    print("")
    # ............................................................................. #
    for _segs_bf in _list_segs_bf:

        #gfit_results.append(np.load('%s/%s' % (outputdir_segs, _segs_bf)))
        _slab_t = np.load('%s/%s' % (outputdir_segs, _segs_bf))
        # _slat_b[x, y, ng, nparams] : 4D
        nx = _slab_t.shape[0]
        ny = _slab_t.shape[1]
        # extract slicing info from the current slab
        x0 = int(_slab_t[0, 0, 0, nparams-2])
        y0 = int(_slab_t[0, 0, 0, nparams-1])
        x1 = int(_slab_t[0, nx-1, 0, nparams-2]) + 1
        y1 = int(_slab_t[0, ny-1, 0, nparams-1]) + 1
        gfit_results[x0:x1, y0:y1, :, :] = _slab_t
        #print(x0, x1, y0, y1)

        #(1, 5, 2, 25)
        #print(gfit_results[500, 504, 0, 2])
        #print('%s' % _segs_bf)
        #print()

    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> make baygaud-classified output directories ...]")
    print("")
    print("")
    # ............................................................................. #
    #-----------------------------------------------#
    # create baygaud output directories
    #_______________________________________________#
    # baygaud_combined
    make_dirs("%s/%s" % (_params['wdir'], 'baygaud_combined'))
    _dir_baygaud_combined = _params['wdir'] + '/' + 'baygaud_combined'

    make_dirs("%s/%s/sgfit" % (_params['wdir'], 'baygaud_combined'))
    make_dirs("%s/%s/psgfit" % (_params['wdir'], 'baygaud_combined'))

    make_dirs("%s/%s/cool" % (_params['wdir'], 'baygaud_combined'))
    make_dirs("%s/%s/warm" % (_params['wdir'], 'baygaud_combined'))
    make_dirs("%s/%s/hot" % (_params['wdir'], 'baygaud_combined'))

    make_dirs("%s/%s/ngfit" % (_params['wdir'], 'baygaud_combined'))
    #make_dirs("%s/%s/bulk" % (_params['wdir'], 'baygaud_combined'))



    #-----------------------------------------------#
    # OBSOLETE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    #-----------------------------------------------#
    # write fits
    #_______________________________________________#
    # _nparray_gfit_results = np.array(gfit_results) # 5d array
    #-----------------------------------------------#
    # _nparray_gfit_results : 5D np array
    #_______________________________________________#
    #                                               #
    # 0: x
    # 1: array itself
    # 2: y
    # 3: ngauss
    # 4: gauss params (for N gaussians)
    #_______________________________________________#
    # [x, array_itself, y, ngauss, params]
    #_fitsarray_gfit_results1 = np.transpose(_nparray_gfit_results, axes=[1, 3, 4, 2, 0])[0] # --> 4d array
    #_______________________________________________#


    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> declare _fitsarray_gfit_results1 : 4d numpy array ...]")
    print("")
    print("")
    # ............................................................................... #
    #-----------------------------------------------#
    # ---> _fitsarray_gfit_results1 : 4D np array
    #_______________________________________________#
    #                                               #
    # 0: ngauss 
    # 1: gauss params (for N Gaussians)
    # 2: y
    # 3: x
    #_______________________________________________#
    # [ngauss, params, y, x]
    #_______________________________________________#
    _fitsarray_gfit_results1 = np.transpose(gfit_results, axes=[2, 3, 1, 0]) # 4d array


    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> declare _fitsarray_gfit_results2 : 3d numpy array ...]")
    print("")
    print("")
    # ............................................................................... #
    # e.g., 3 gaussian model
    # reduce the 4D to 3D array by slicing it with ngauss axis
    # extract 3D array for each gaussian
    #_g1 = _fitsarray_gfit_results1[0,:,:,:] # G1 : [params, y, x]
    #_g2 = _fitsarray_gfit_results1[1,:,:,:] # G2 : [params, y, x]
    #_g3 = _fitsarray_gfit_results1[2,:,:,:] # G3 : [params, y, x]
    #print(_g1[: ,50, 50])
    #print(_g2[: ,50, 50])
    #print(_g3[: ,50, 50])

    # e.g., 3 gaussian model
    # concatenate the 3-3D arrays: now the params are concatenated, 3x(params x 3)=87 [params, y, x]
    #_g = np.concatenate((_g1, _g2, _g3), axis=0) 
    #_fitsarray_gfit_results2 = np.concatenate((_g1, _g2, _g3), axis=0)
    _fitsarray_gfit_results2 = np.concatenate(_fitsarray_gfit_results1 , axis=0)
    #print(_fitsarray_gfit_results2.shape)

    #-----------------------------------------------#
    # ---> _fitsarray_gfit_results2 : 3D np array
    #_______________________________________________#
    #                                               #
    # 0: params [n-gaussians * n-gaussian-params] 
    # 1: y
    # 2: x
    #_______________________________________________#
    # [params, y, x]
    # shape[0]: params
    # shape[1]: y
    # shape[2]: x
    #_______________________________________________#


    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> declare bevidences : 3d numpy array ...]")
    print("")
    print("")
    # ............................................................................... #
    bevidences = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    #-----------------------------------------------#
    # ---> bevidences : 3D np array
    #_______________________________________________#
    #                                               #
    # 0: n-gauss
    # 1: y
    # 2: x
    #_______________________________________________#
    # [ngauss, y, x]
    # shape[0]: ngauss <-- log-Z values
    # shape[1]: y
    # shape[2]: x
    #_______________________________________________#


    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> declare g_num_sort : 3d numpy array ...]")
    print("")
    print("")
    # ............................................................................... #
    g_num_sort = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    #-----------------------------------------------#
    # ---> g_num_sort : 3D np array
    #_______________________________________________#
    #                                               #
    # 0: n-gauss
    # 1: y
    # 2: x
    #_______________________________________________#
    # [ngauss, y, x]
    # shape[0]: ngauss <-- gaussian components (e.g., 0, 1, 2)
    # shape[1]: y
    # shape[2]: x
    #_______________________________________________#

    #|-----------------------------------------|
    # example: 3 gaussians : bg + 3 * (x, std, peak) 
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


    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> declare numpy arrays for S/N slices: 3d numpy arrays ...]")
    print("")
    print("")
    # ............................................................................... #
    # extract S/N slice 
    # max ngauss
    # peak_sn_limit_for_ng_opt 
    peak_sn_pass_for_ng_opt = _params['peak_sn_pass_for_ng_opt']
    # number of parameters per step
    nparams_step = 2*(3*max_ngauss+2) + (max_ngauss + 7)

    sn_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_ng_opt = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_pass_ng_opt = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    sn_pass_ng_opt_t = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    x_ng_opt_slice = np.zeros((max_ngauss, max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)
    x_ng_opt = np.zeros((max_ngauss, _fitsarray_gfit_results2.shape[1], _fitsarray_gfit_results2.shape[2]), dtype=float)

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    # CHECK POINT
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    i1 = _params['_i0']
    j1 = _params['_j0']
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> check _fitsarray_gfit_results2 for (x:%d, y:%d)...]" % (i1, j1))
    print("")
    print("")
    # ............................................................................... #
    print("_"*50)
    print("_"*50)
    print(i1, j1)
    print("_"*50)
    print(_fitsarray_gfit_results2[:, j1, i1])
    print("")
    print("")

    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> extract sn_ng_opt_slice from _fitsarray_Gfit_results2 array[params, y, x] ...]")
    print("")
    print("")
    # ............................................................................... #
    for i in range(0, max_ngauss):
        for j in range(0, max_ngauss):
            #---------------------------------------------------
            # S/N slice : for convenience, sn_ng_opt_slice only includes the Gaussian components which pass the VDISP criteria (e.g., VDISP > 1 channel & < XXX km/s)
            sn_ng_opt_slice[i, j, :, :] = np.array([np.where( \
                                                    (j >= i) & \
                                                    (_fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :] > 0) & \
                                                    (_fitsarray_gfit_results2[nparams_step*j + 3 + 3*i, :, :] >= _params['vdisp_lower']) & \
                                                    (_fitsarray_gfit_results2[nparams_step*j + 3 + 3*i, :, :] < _params['vdisp_upper']), \
                                                    _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, :, :] / _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, :, :], 0.0)])[0]

            if j >= i:
                print(i, j, _fitsarray_gfit_results2[nparams_step*j + 2 + 3*i, j1, i1], _fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, j1, i1], _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, j1, i1])
        print("")


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    # CHECK POINT
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> check sn_ng_opt_slice ...]")
    print("")
    print("")
    # ............................................................................... #
    #i1 = 510
    #j1 = 510
    i1 = _params['_i0']
    j1 = _params['_j0']
    #print(_fitsarray_gfit_results2[nparams_step*j + 4 + 3*i, j1, i1],  _fitsarray_gfit_results2[nparams_step*(j+1)-max_ngauss-7+j, j1, i1])

    # sn_ng_opt_slice[x_st Gaussian, max_ngauss, j1, ij]
    for i in range(0, max_ngauss):
        for j in range(0, max_ngauss):
            print(sn_ng_opt_slice[j, i, j1, i1])
        print("")
    print("")

    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> check sn_pass_ng_opt from sn_ng_opt_slice ...]")
    print("")
    print("")
    # ............................................................................... #
    # --------------------------------------------------------------------- #
    #for i in range(0, max_ngauss):
    #    if sn_pass_ng_opt[i, :, :] == i+1 --> S/N pass
    #    else --> S/N non pass
    # sn_pass_ng_opt[0, j1, i1] == 1 <-- max_ngauss=1 passed
    # sn_pass_ng_opt[1, j1, i1] == 2 <-- max_ngauss=2 passed
    # sn_pass_ng_opt[2, j1, i1] == 3 <-- max_ngauss=3 passed
    # check below print statements
    # _____________________________________________________________________ #
    for i in range(0, max_ngauss):
        for j in range(0, max_ngauss):
            sn_pass_ng_opt[i, :, :] += np.array([np.where( \
                                            (sn_ng_opt_slice[j, i, :, :] > peak_sn_pass_for_ng_opt), \
                                            1, 0)][0])

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    # CHECK POINT
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    #print(sn_pass_ng_opt[0, j1, i1])
    #print(sn_pass_ng_opt[1, j1, i1])
    #print(sn_pass_ng_opt[2, j1, i1])


    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> extract log-Z from _fitsarray_gfit_results2 array ...]")
    print("[--> to bevidences array ...]")
    print("")
    print("")
    # ............................................................................... #
    # extract log-Z and put it into bevidences array
    nparams_step = 2*(3*max_ngauss+2) + max_ngauss + 7
    for i in range(max_ngauss):
        # e.g., for 3 gaussians  : i:0, 1, 2
        bevidences[i, :, :] = _fitsarray_gfit_results2[nparams_step*(i+1)-7, :, :] # corresponing log-Z


    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> sort the coupled (max_ngauss, log-Z) with log-Z ...]")
    print("[--> in descending order : max(log-Z) first ...]")
    print("")
    print("")
    # ............................................................................... #
    # sort the coupled (max_ngauss, log-Z) with log-Z : descending order : max(log-Z) first
    g_num_sort = bevidences.argsort(axis=0)[::-1] # descening order : arg
    bevidences_sort = np.sort(bevidences, axis=0)[::-1] # descening order : log-Z
    #print(bevidences_argsort.shape)

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    # CHECK POINT
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    #print(bevidences_sort[:, 468, 600])
    #print(g_num_sort[:, 468, 600])


    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> derive the optimal number of Gaussian components ...]")
    print("[--> given the sn_pass + bayes factor limit ...]")
    print("[--> opt_ngmap_gmax_ng array : optimal n-gauss array ...]")
    print("[--> max_ngauss: %d ...]" % max_ngauss)
    print("")
    print("")
    # ............................................................................... #
    #|----------------------------------------------------------------------------------------|
    # 1. determine the optimal number of gaussian components
    #    sn_pass_ng_opt[0, nax2, nax1] <-- g1 : 1
    #    sn_pass_ng_opt[1, nax2, nax1] <-- g2 : 2
    #    sn_pass_ng_opt[2, nax2, nax1] <-- g3 : 3
    #    
    #   --> gx_opt : optimal n-gauss array
    #|________________________________________________________________________________________|
    # Bayesian model selection : based on bayes factor
    bf_limit = _params['bayes_factor_limit']

    if max_ngauss == 1:
        opt_ngmap_gmax_ng = g1_opt_bf(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit)
        print() 
        #print(opt_ngmap_gmax_ng)
        #print(np.where(g1_opt < 1))

    elif max_ngauss > 1:
        #______________________________________________________#
        # generate gx list
        gx_list = [] # [0, 1, 2, ...] == [1, 2, 3, ...]
        for i in range(0, max_ngauss):
            gx_list.append(i)

        opt_ngmap_gmax_ng = find_gx_opt_bf_snp(_fitsarray_gfit_results2, bevidences_sort, g_num_sort, bf_limit, max_ngauss, sn_pass_ng_opt, gx_list)
    #______________________________________________________#


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    # CHECK POINT
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> (%d, %d) -- optimal ng: %d ...]" % (i1, j1, opt_ngmap_gmax_ng[j1, i1]))
    print("")
    print("")

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    # CHECK POINT
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
    i1 = _params['_i0']
    j1 = _params['_j0']
    #print("i: ", i1, "j: ", j1, "ng_opt: ", opt_ngmap_gmax_ng[j1, i1], "bf_limit: ", bf_limit)
    #print(np.where(opt_ngmap_gmax_ng == 1))



    #|----------------------------------------------------------------------------------------|
    # extract maps and write them in fits
    #|________________________________________________________________________________________|
    #|----------------------------------------------------------------------------------------|
    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> extract :: single Gaussian component:: given the optimal n-gauss map ...]")
    print("")
    print("")
    # 1. single gaussian components --> write to fits
    extract_maps(_fitsarray_gfit_results2, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='sgfit', ng_opt=opt_ngmap_gmax_ng, _hdu=hdu)

    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> extract :: perfect single Gaussian component:: given the optimal n-gauss map ...]")
    print("")
    print("")
    # 2. perfect single gaussian components--> write to fits
    extract_maps(_fitsarray_gfit_results2, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='psgfit', ng_opt=opt_ngmap_gmax_ng, _hdu=hdu)

    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> extract :: kinematically cool Gaussian component:: given the optimal n-gauss map ...]")
    print("")
    print("")
    # 3. kinematically cool gaussian components--> write to fits
    extract_maps(_fitsarray_gfit_results2, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='cool', ng_opt=opt_ngmap_gmax_ng, _hdu=hdu)

    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    # 4. kinematically warm gaussian components--> write to fits
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> extract :: kinematically warm Gaussian component:: given the optimal n-gauss map ...]")
    print("")
    print("")
    # 4. kinematically warm gaussian components--> write to fits
    extract_maps(_fitsarray_gfit_results2, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='warm', ng_opt=opt_ngmap_gmax_ng, _hdu=hdu)


    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> extract :: kinematically hot Gaussian component:: given the optimal n-gauss map ...]")
    print("")
    print("")
    # 5. kinematically hot gaussian components--> write to fits
    extract_maps(_fitsarray_gfit_results2, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='hot', ng_opt=opt_ngmap_gmax_ng, _hdu=hdu)


    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> extract :: all the Gaussian components:: given max_ngauss ...]")
    print("")
    print("")
    # 6. extract all the Gaussian components--> write to fits
    #extract_maps_ngfit(_fitsarray_gfit_results2, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='ngfit', ng_opt=opt_ngmap_gmax_ng, _hdu=hdu)
    extract_maps(_fitsarray_gfit_results2, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='ngfit', ng_opt=opt_ngmap_gmax_ng, _hdu=hdu)

    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    if _params['_bulk_extraction'] == 'Y':
        print(" ____________________________________________")
        print("[____________________________________________]")
        print("[--> extract ::bulk motions:: given the optimal n-gauss map ...]")
        print("")
        print("")
        # 6. bulk motion components--> write to fits
        # REFERENCE VELOICTY FIELD should be given
        # check sgfit part in extract_maps()

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


        #-----------------------------------------------#
        # ---> make bulk & non_bulk directories if not present
        #_______________________________________________#
        if not os.path.exists("%s/%s/bulk" % (_params['wdir'], 'baygaud_combined')):
            make_dirs("%s/%s/bulk" % (_params['wdir'], 'baygaud_combined'))

        if not os.path.exists("%s/%s/non_bulk" % (_params['wdir'], 'baygaud_combined')):
            make_dirs("%s/%s/non_bulk" % (_params['wdir'], 'baygaud_combined'))

        # bulk_ref_vf : reference 2d map
        bulk_ref_vf = fitsio.read(_params['_bulk_model_dir'] + '/' + _params['_bulk_ref_vf'])
        # bulk_delv_limit : velocity limit 2d map
        bulk_delv_limit = fitsio.read(_params['_bulk_model_dir'] + '/' + _params['_bulk_delv_limit']) * _params['_bulk_delv_limit_factor']
        #print(bulk_ref_vf.shape)
        #print(bulk_delv_limit.shape)

        extract_maps_bulk(_fitsarray_gfit_results2, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='bulk', \
                            ng_opt=opt_ngmap_gmax_ng, _bulk_ref_vf=bulk_ref_vf, _bulk_delv_limit=bulk_delv_limit, _hdu=hdu)


    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> extract :: hvc component:: given the optimal n-gauss map ...]")
    print("")
    print("")
    # 7. hvc components--> write to fits
    # bulk_ref_vf, bulk_delv_limit
    #extract_maps_bulk(_fitsarray_gfit_results2, params=_params, _output_dir=_dir_baygaud_combined, _kin_comp='bulk', \
    #                    ng_opt=opt_ngmap_gmax_ng, _bulk_ref_vf=bulk_ref_vf, _bulk_delv_limit=bulk_delv_limit)


    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> save the full baygaud_gfit_results in fits format...]")
    print("")
    print("")
    #-----------------------------------------------#
    # ---> _fitsarray_gfit_results2 : 3D np array
    #_______________________________________________#
    #                                               #
    # 0: params [n-gaussians * n-gaussian-params] 
    # 1: y
    # 2: x
    #_______________________________________________#
    # [params, y, x]
    # shape[0]: params
    # shape[1]: y
    # shape[2]: x
    #_______________________________________________#
    _hdu_nparray_gfit_results = fits.PrimaryHDU(_fitsarray_gfit_results2)
    _hdulist_nparray_gfit_result = fits.HDUList([_hdu_nparray_gfit_results])
    # write the baygaud gfit results to fits | save the results in binaries
    _hdulist_nparray_gfit_result.writeto('%s/baygaud_gfit_results.fits' % _dir_baygaud_combined, overwrite=True)
    _hdulist_nparray_gfit_result.close()

    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
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

    #-----------------------------------------#
#
#-- END OF SUB-ROUTINE____________________________________________________________#
