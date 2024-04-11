#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _baygaud_params.py
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|

import sys
import numpy as np
import yaml


global _inputDataCube
global _is, _ie, _js, _je
global parameters
global nparams
global ngauss
global ndim
global max_ngauss
global gfit_results
global _x
global nchannels


def read_configfile(configfile):
    with open(configfile, "r") as file:
        _params = yaml.safe_load(file)
    return _params

def default_params():
    _params = {
    'main_baygaud':'/home/seheon/research/code/_python/baygaud_py/test/baygaud-PI/src',  # <---- UPDATE HERE

    'wdir':'/home/seheon/research/code/_python/baygaud_py/test/baygaud-PI/demo/test_cube',  # <---- UPDATE HERE

    '_segdir':'baygaud_segs_output',
    '_combdir':'baygaud_segs_merged',

    'input_datacube':'ngc2403.regrid.testcube.0.fits',  # <---- UPDATE HERE
    '_cube_mask':'N', # Y(if available) or N(if not)  # <---- UPDATE HERE
    '_cube_mask_2d':'2403_mosaic_5kms_r05_HI_mwfilt_mask-2d.fits',  # <---- UPDATE HERE

    'naxis1':0,
    'naxis2':0,
    'naxis3':0,
    'cdelt1':0,
    'cdelt2':0,
    'cdelt3':0,
    'vel_min':0, # km/s
    'vel_max':0, # km/s
    '_rms_med':0,
    '_bg_med':0,

    'naxis1_s0':5,  # <---- UPDATE HERE
    'naxis1_e0':35, # <---- UPDATE HERE
    'naxis2_s0':5,  # <---- UPDATE HERE
    'naxis2_e0':35, # <---- UPDATE HERE

    'nsteps_x_rms':3, # <---- UPDATE HERE
    'nsteps_y_rms':3, # <---- UPDATE HERE

    '_dynesty_class_':'static', # (recommended)

    'nlive':100,
    'sample':'rwalk', # or auto
    'dlogz':0.05,
    'maxiter':999999,
    'maxcall':999999,
    'update_interval':2.0, # ===> 2.0 x nlilve
    'vol_dec':0.2,
    'vol_check':2,
    'facc':0.5,
    'fmove':0.9,
    'walks':25,
    'rwalk':100,
    'max_move':100,
    'bound':'multi', # or 'single' for unimodal  :: if it complains about sampling efficiency, 'multi' is recommended..


    'bayes_factor_limit':100, # <---- UPDATE HERE
    'max_ngauss':3, # maximum number of Gaussian components : for initial baygaud run (non-recoverble)  # <---- UPDATE HERE
    'mom0_nrms_limit':3, # only profiles with peak fluxes > mom0_nrms_limit * rms + bg are used for computing mom0 : this is for deriving integrated S/N (for initial baygaud run - non-recoverble) # <---- UPDATE HERE
    'peak_sn_pass_for_ng_opt':3, # count Gaussians > peak_sn_pass_for_ng_opt (recoverble) # <---- UPDATE HERE
    'peak_sn_limit':2, # for initial baygaud run : do profile-decomposition for profiles only with > peak_sn_limit (for initial baygaud run: non-recoverable) # <---- UPDATE HERE
    'int_sn_limit':4.0, # for initial baygaud run : do profile-decomposition for profiles only with > peak_sn_limit (for initial baygaud run: non-recoverable) # <---- UPDATE HERE
    '_bulk_model_dir':'/home/seheon/research/mhongoose/ngc2403/things_model_vf/',  # <---- UPDATE HERE
    '_bulk_extraction':'N', # Y or N  # <---- UPDATE HERE
    '_bulk_ref_vf':'ngc2403.bulk.vf.model.fits',  # <---- UPDATE HERE
    '_bulk_delv_limit':'sgfit.G5_1.2.fits',  # <---- UPDATE HERE
    '_bulk_delv_limit_factor':0.3,

    'vlos_lower':0, # km/s
    'vlos_upper':400, # km/s
    'vdisp_lower':5, # km/s
    'vdisp_upper':999, # km/s

    'vlos_lower_cool':0, # km/s
    'vlos_upper_cool':400, # km/s
    'vdisp_lower_cool':5, # km/s
    'vdisp_upper_cool':12, # km/s

    'vlos_lower_warm':0, # km/s
    'vlos_upper_warm':400, # km/s
    'vdisp_lower_warm':12, # km/s
    'vdisp_upper_warm':40, # km/s

    'vlos_lower_hot':0, # km/s
    'vlos_upper_hot':400, # km/s
    'vdisp_lower_hot':40, # km/s
    'vdisp_upper_hot':999, # km/s

    '_i0':20, # x-pixel
    '_j0':20, # y-pixel


    'x_lowerbound_gfit':0.1,
    'x_upperbound_gfit':0.9,
    'nsigma_prior_range_gfit':3.0, # for the N-gaussians in the N+1 gaussian fit
    'sigma_prior_lowerbound_factor':0.2,
    'sigma_prior_upperbound_factor':2.0,
    'bg_prior_lowerbound_factor':0.2,
    'bg_prior_upperbound_factor':2.0,
    'x_prior_lowerbound_factor':5.0, # --> X_min - lowerbound_factor * STD_max
    'x_prior_upperbound_factor':5.0, # --> X_max + upperbound_factor * STD_max
    'std_prior_lowerbound_factor':0.1,
    'std_prior_upperbound_factor':3.0,
    'p_prior_lowerbound_factor':0.05,
    'p_prior_upperbound_factor':1.0,

    'ncolm_per_core':'',
    'nsegments_nax2':'',
    'num_cpus':12,  # <---- UPDATE HERE
    }
    
    return _params


