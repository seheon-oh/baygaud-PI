#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| combine_segs.py
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|

#|-----------------------------------------|
# Python 3 compatability
from __future__ import division, print_function
from six.moves import range

#|-----------------------------------------|
import time, sys, os
from datetime import datetime

#|-----------------------------------------|
import numpy as np
from numpy import linalg, array, sum, log, exp, pi, std, diag, concatenate
import gc


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

#........................................ 
# global parameters
global _inputDataCube
global _x
global _spect
global _is, _ie, _js, _je
global parameters
global nparams
global ngauss
_x = np.linspace(0, 1, 143, dtype=np.float32)
_spect = None
global ndim
global max_ngauss


import shutil


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
outputdir_segs = '/home/seheon/research/libs/dynesty/output'

combined_segs_bf = open('output.npy', "wb")
_list_segs_bf = os.listdir(outputdir_segs)

#for _segs_bf in _list_segs_bf:
#    _open_segs_bf = open(os.path.join(outputdir_segs, _segs_bf), "rb")
#    shutil.copyfileobj(_open_segs_bf, combined_segs_bf)
#    _open_segs_bf.close()
#combined_segs_bf.close()

gfit_results = []
print(_list_segs_bf)
for _segs_bf in _list_segs_bf:

    gfit_results.append(np.load('%s/%s' % (outputdir_segs, _segs_bf)))

    #gfit_results = np.array(gfit_results) # 5d array
    #print(gfit_results[0,0,50,2,:])
    #print(gfit_results.shape)
    #sys.exit()
#gfit_results = np.load('./test.bin')
#print(gfit_results)
print(len(gfit_results))


#-----------------------------------------#
# WRITE FITS
#_nparray_gfit_results = np.array(gfit_results).reshape(1, 50, 10, 17)[0] # 4d array
#_nparray_gfit_results = np.array(gfit_results).reshape(1, 50, 10, 17) # 4d array
_nparray_gfit_results = np.array(gfit_results) # 5d array
print(_nparray_gfit_results.shape)
#_fitsarray_gfit_results = np.transpose(_nparray_gfit_results, axes=[2, 1, 0]) # 3d array
_fitsarray_gfit_results1 = np.transpose(_nparray_gfit_results, axes=[1, 3, 4, 2, 0])[0] # 4d array

a1 = _fitsarray_gfit_results1[0,:,:,:]
a2 = _fitsarray_gfit_results1[1,:,:,:]
a3 = _fitsarray_gfit_results1[2,:,:,:]
print(a1[: ,50, 50])
print(a2[: ,50, 50])
print(a3[: ,50, 50])
a = np.concatenate((a1, a2, a3), axis=0)
print(a[: ,50, 50])
_fitsarray_gfit_results2 = np.concatenate((a1, a2, a3), axis=0)

#_fitsarray_gfit_results2 = _fitsarray_gfit_results1.reshape(_fitsarray_gfit_results1.shape[0]*_fitsarray_gfit_results1.shape[1],_fitsarray_gfit_results1.shape[2], _fitsarray_gfit_results1.shape[3])
#print(_fitsarray_gfit_results[10][10])

_hdu_nparray_gfit_results = fits.PrimaryHDU(_fitsarray_gfit_results2)
_hdulist_nparray_gfit_result = fits.HDUList([_hdu_nparray_gfit_results])
_hdulist_nparray_gfit_result.writeto('test1.fits', overwrite=True)
_hdulist_nparray_gfit_result.close()
print(_fitsarray_gfit_results2.shape)
#-----------------------------------------#

#-----------------------------------------#
# WRITE FITS
#_nparray_gfit_results = np.array(gfit_results).reshape(1, 50, 10, 17)[1] # 3d array
#_fitsarray_gfit_results = np.transpose(_nparray_gfit_results, axes=[2, 1, 0])[2] # 2d map slice
#print(_fitsarray_gfit_results[10][10])
#print(_fitsarray_gfit_results.shape)

#_hdu_nparray_gfit_results = fits.PrimaryHDU(_fitsarray_gfit_results)
#_hdulist_nparray_gfit_result = fits.HDUList([_hdu_nparray_gfit_results])
#_hdulist_nparray_gfit_result.writeto('test1.fits', overwrite=True)
#_hdulist_nparray_gfit_result.close()
#-- END OF SUB-ROUTINE____________________________________________________________#