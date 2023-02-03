#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _fits_io.py
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|

#|-----------------------------------------|
import numpy as np
import fitsio
import sys

import astropy.units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from spectral_cube import SpectralCube
from spectral_cube import BooleanArrayMask

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def read_datacube(_params):
    global _inputDataCube

    with fits.open(_params['wdir'] + '/' + _params['input_datacube'], 'update') as hdu:
        _naxis1 = hdu[0].header['NAXIS1']
        _naxis2 = hdu[0].header['NAXIS2']
        _naxis3 = hdu[0].header['NAXIS3']

        _cdelt1 = hdu[0].header['CDELT1']
        _cdelt2 = hdu[0].header['CDELT2']
        _cdelt3 = hdu[0].header['CDELT3']
        try:
            hdu[0].header['RESTFREQ'] = hdu[0].header['FREQ0'] # THIS IS NEEDED WHEN INPUTING FITS PROCESSED WITH GIPSY
        except:
            pass

        try:
            if(hdu[0].header['CUNIT3']=='M/S' or hdu[0].header['CUNIT3']=='m/S'):
                hdu[0].header['CUNIT3'] = 'm/s'
        except KeyError:
            hdu[0].header['CUNIT3'] = 'm/s'

    _params['naxis1'] = _naxis1   
    _params['naxis2'] = _naxis2  
    _params['naxis3'] = _naxis3   
    _params['cdelt1'] = _cdelt1   
    _params['cdelt2'] = _cdelt2   
    _params['cdelt3'] = _cdelt3   

    cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s) # in km/s

    # normalise velocity-axis to 0-1 scale
    _x = np.linspace(0, 1, _naxis3, dtype=np.float32)
    _vel_min = cube.spectral_axis.min().value
    _vel_max = cube.spectral_axis.max().value
    _params['vel_min'] = _vel_min   
    _params['vel_max'] = _vel_max  

    #  _____________________________________________________________________________  #
    # [_____________________________________________________________________________] #
    print(" ____________________________________________")
    print("[____________________________________________]")
    print("[--> check cube dimension...]")
    print("[--> naxis1: ", _naxis1)
    print("[--> naxis2: ", _naxis2)
    print("[--> naxis3: ", _naxis3)
    print(" ____________________________________________")
    print("[--> check cube velocity range :: velocities should be displayed in [KM/S] here...]")
    print("[--> If the velocity units are displayed with [km/s] then the input cube fortmat is fine for the baygaud analysis...]")
    print("[--> The spectral axis of the input data cube should be in m/s ...]")
    print("")
    print("The lowest velocity [km/s]: ", _vel_min)
    print("The highest velocity [km/s]: ", _vel_max)
    print("CDELT3 [m/s]: ", _cdelt3)
    if _cdelt3 < 0:
        print("[--> Spctral axis with decreasing order...]")
    else:
        print("[--> Spctral axis with increasing order...]")
    print("")
    print("")
    print(_x)

    #_inputDataCube = fitsio.read(_params['wdir'] + _params['input_datacube'], dtype=np.float32)
    _inputDataCube = fitsio.read(_params['wdir'] + '/' + _params['input_datacube'])
    #_spect = _inputDataCube[:,516,488]
    return _inputDataCube, _x

    #plot profile
    #plt.figure(figsize=(12, 5))
    #plt.plot(_x, _spect, color='black', marker='x', 
    #        ls='none', alpha=0.9, markersize=10)
    #plt.plot(_x, _spect, marker='o', color='red', ls='none', alpha=0.7)
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.tight_layout()
    #plt.show()
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def update_header_cube_to_2d(_hdulist_nparray, _hdu_cube):
    # _hdulist_nparray: numpy array for fits whose header info is updated.
    # _hdu_cube : input data cube whose header info is used for updating the 2d fits

    #_hdulist_nparray[0].header.update(NAXIS1=_hdu[0].header['NAXIS1'])
    #_hdulist_nparray[0].header.update(NAXIS2=_hdu[0].header['NAXIS2'])
    _hdulist_nparray[0].header.insert('NAXIS2', ('CDELT1', _hdu_cube[0].header['CDELT1']), after=True)
    #_hdulist_nparray[0].header.insert('CDELT1', ('CROTA1', _hdu_cube[0].header['CROTA1']), after=True)
    #_hdulist_nparray[0].header.insert('CROTA1', ('CRPIX1', _hdu_cube[0].header['CRPIX1']), after=True)
    _hdulist_nparray[0].header.insert('CDELT1', ('CRPIX1', _hdu_cube[0].header['CRPIX1']), after=True)
    _hdulist_nparray[0].header.insert('CRPIX1', ('CRVAL1', _hdu_cube[0].header['CRVAL1']), after=True)
    _hdulist_nparray[0].header.insert('CRVAL1', ('CTYPE1', _hdu_cube[0].header['CTYPE1']), after=True)
    try:
        _hdulist_nparray[0].header.insert('CTYPE1', ('CUNIT1', _hdu_cube[0].header['CUNIT1']), after=True)
    except:
        _hdulist_nparray[0].header.insert('CTYPE1', ('CUNIT1', 'deg'), after=True)


    _hdulist_nparray[0].header.insert('CUNIT1', ('CDELT2', _hdu_cube[0].header['CDELT2']), after=True)
    #_hdulist_nparray[0].header.insert('CDELT2', ('CROTA2', _hdu_cube[0].header['CROTA2']), after=True)
    #_hdulist_nparray[0].header.insert('CROTA2', ('CRPIX2', _hdu_cube[0].header['CRPIX2']), after=True)
    _hdulist_nparray[0].header.insert('CDELT2', ('CRPIX2', _hdu_cube[0].header['CRVAL2']), after=True)
    _hdulist_nparray[0].header.insert('CRPIX2', ('CRVAL2', _hdu_cube[0].header['CRVAL2']), after=True)
    _hdulist_nparray[0].header.insert('CRVAL2', ('CTYPE2', _hdu_cube[0].header['CTYPE2']), after=True)

    try:
        _hdulist_nparray[0].header.insert('CTYPE2', ('CUNIT2', _hdu_cube[0].header['CUNIT2']), after=True)
    except:
        _hdulist_nparray[0].header.insert('CTYPE2', ('CUNIT2', 'deg'), after=True)

    #_hdulist_nparray[0].header.insert('CUNIT2', ('EPOCH', _hdu_cube[0].header['EPOCH']), after=True)


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def write_fits_seg(_segarray, _segfitsfile):
    hdu = fits.PrimaryHDU(data=_segarray)
    hdu.writeto(_segfitsfile, overwrite=True)
#-- END OF SUB-ROUTINE____________________________________________________________#


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def moment_analysis(_params):

    with fits.open(_params['wdir'] + '/' + _params['input_datacube'], 'update') as hdu:
        _naxis3 = hdu[0].header['NAXIS3']
        cdelt3 = abs(hdu[0].header['CDELT3'])
        try:
            hdu[0].header['RESTFREQ'] = hdu[0].header['FREQ0']    # in case the input cube is pre-processed using GIPSY
        except:
            pass
    
        try:
            if(hdu[0].header['CUNIT3']=='M/S' or hdu[0].header['CUNIT3']=='m/S'):
                hdu[0].header['CUNIT3'] = 'm/s'
        except KeyError:
            hdu[0].header['CUNIT3'] = 'm/s'


    #_____________________________________
    #-------------------------------------
    # 0. load the input cube
    #cubedata = fitsio.read(_params['wdir'] + _params['input_datacube'], dtype=np.float32)
    _input_cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s, velocity_convention='radio')


    # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    # set _input_cube.beam_threshold > 0.01, e.g., 0.1 : 10%, normally < 1%
    _input_cube.beam_threshold = 0.1

   
    #_____________________________________
    #-------------------------------------
    # 1. make a mask
    _flux_threshold = _params['mom0_nrms_limit']*_params['_rms_med'] + _params['_bg_med']
    peak_sn_mask = _input_cube > _flux_threshold*u.Jy/u.beam
    #print("flux_threshold:", _flux_threshold)

    # 2. extract profiles > _flux_threhold
    _input_cube_peak_sn_masked = _input_cube.with_mask(peak_sn_mask)

    # 3. extract mom0
    mom0 = _input_cube_peak_sn_masked.with_spectral_unit(u.km/u.s).moment(order=0, axis=0)

    # 4. extrac N
    # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    # hdulist should be used instead of hdu
    if _input_cube.beam_threshold > 0.09: # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
        _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdulist[0].data), -1E5, _input_cube_peak_sn_masked.hdulist[0].data)
    #_N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdu.data), -1E5, _input_cube_peak_sn_masked.hdu.data)

    # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403 : VLA + SINGLE DISH
    # hdulist should be used instead of hdu
    if _input_cube.beam_threshold > 0.09: # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
        _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdulist[0].data), -1E5, _input_cube_peak_sn_masked.hdulist[0].data)
    #_N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdu.data), -1E5, _input_cube_peak_sn_masked.hdu.data)

    _N_masked = np.where(np.isinf(_N_masked), -1E5, _N_masked)
    _N_masked = np.where(np.isinf(-1*_N_masked), -1E5, _N_masked)
    _N = (_N_masked > -1E5).sum(axis=0)

    # 5. derive integerated rms
    _rms_int = np.sqrt(_N) * _params['_rms_med'] * (cdelt3/1000.)


    # 6. integrated s/n map --> spectralcube array
    _sn_int_map = mom0 / _rms_int
    #print(_params['_rms_med'], _params['_bg_med'])
    #print(mom0)

    # 7. integrated s/n map: numpy array : being returned 

    # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    # hdulist should be used instead of hdu
    #if _input_cube.beam_threshold > 0.09: # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    #    _sn_int_map_nparray = _sn_int_map.hdulist[0].data
    _sn_int_map_nparray = _sn_int_map.hdu.data
    _sn_int_map_nparray = np.where(np.isnan(_sn_int_map_nparray), 0, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(_sn_int_map_nparray), 1E5, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(-1*_sn_int_map_nparray), -1E5, _sn_int_map_nparray)


    #_____________________________________
    #-------------------------------------
    # make a peak s/n map
    # 1. extract the peak flux map
    # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    # hdulist should be used instead of hdu
    if _input_cube.beam_threshold > 0.09: # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
        peak_flux_map = _input_cube.hdulist[0].data.max(axis=0)
    #peak_flux_map = _input_cube.hdu.data.max(axis=0)
    peak_flux_map = np.where(np.isnan(peak_flux_map), -1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(peak_flux_map), 1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(-1*peak_flux_map), -1E5, peak_flux_map)

    # 2. peak s/n map
    peak_sn_map = (peak_flux_map - _params['_bg_med']) / _params['_rms_med']


    #-------------------------------------
    # write fits
    # moment0
    mom0.write('test1.mom0.fits', overwrite=True)

    # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    # hdulist should be used instead of hdu
    #if _input_cube.beam_threshold > 0.09: # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
    #    _sn_int_map.hdulist[0].header['BUNIT'] = 's/n'
    _sn_int_map.hdu.header['BUNIT'] = 's/n'
    # write fits
    # _sn_int_map
    _sn_int_map.write('test1.sn_int.fits', overwrite=True)
    #print('moment0_unit:', mom0.unit)

    return peak_sn_map, _sn_int_map_nparray
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def moment_analysis_alternate(_params):

    with fits.open(_params['wdir'] + '/' + _params['input_datacube'], 'update') as hdu:
        _naxis3 = hdu[0].header['NAXIS3']
        cdelt3 = abs(hdu[0].header['CDELT3'])
        try:
            hdu[0].header['RESTFREQ'] = hdu[0].header['FREQ0']    # in case the input cube is pre-processed using GIPSY
        except:
            pass
    
        try:
            if(hdu[0].header['CUNIT3']=='M/S' or hdu[0].header['CUNIT3']=='m/S'):
                hdu[0].header['CUNIT3'] = 'm/s'
        except KeyError:
            hdu[0].header['CUNIT3'] = 'm/s'
    
    
    #cubedata = fitsio.read(_params['wdir'] + _params['input_datacube'], dtype=np.float32)
    cubedata = fitsio.read(_params['wdir'] + '/' + _params['input_datacube'])
   
    # peak s/n
    _chan_linefree1 = np.mean(fits.open(_params['wdir'] + '/' + _params['input_datacube'])[0].data[0:int(_naxis3*0.05):1, :, :], axis=0) # first 5% channels
    _chan_linefree2 = np.mean(fits.open(_params['wdir'] + '/' + _params['input_datacube'])[0].data[int(_naxis3*0.95):_naxis3-1:1, :, :], axis=0) # last 5% channels
    _chan_linefree = (_chan_linefree1 + _chan_linefree2)/2.

    # masking nan, inf, -inf
    _chan_linefree = np.where(np.isnan(_chan_linefree), -1E5, _chan_linefree)
    _chan_linefree = np.where(np.isinf(_chan_linefree), 1E5, _chan_linefree)
    _chan_linefree = np.where(np.isinf(-1*_chan_linefree), -1E5, _chan_linefree)
    #print(_chan_linefree.shape)
    _mean_bg, _median_bg, _std_bg = sigma_clipped_stats(_chan_linefree, sigma=3.0)
    #print(_mean_bg, _median_bg, _std_bg)
    # use _params['_rms_med'] instead of _std_bg which tends to be lower

    _flux_threshold = _params['mom0_nrms_limit']*_params['_rms_med'] + _median_bg
    _input_cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s, velocity_convention='radio')

    # make a mask
    peak_sn_mask = _input_cube > _flux_threshold*u.Jy/u.beam
    # extract lines > peak_sn
    _input_cube_peak_sn_masked = _input_cube.with_mask(peak_sn_mask)
    # extract mom0
    mom0 = _input_cube_peak_sn_masked.with_spectral_unit(u.km/u.s).moment(order=0, axis=0)

    # extract the peak flux map
    peak_flux_map = _input_cube.hdu.data.max(axis=0)
    peak_flux_map = np.where(np.isnan(peak_flux_map), -1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(peak_flux_map), 1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(-1*peak_flux_map), -1E5, peak_flux_map)


    # make a peak s/n map
    peak_sn_map = (peak_flux_map - _median_bg) / _params['_rms_med']
    #print("peak sn")
    #print(peak_sn_map)

    _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdu.data), -1E5, _input_cube_peak_sn_masked.hdu.data)
    _N_masked = np.where(np.isinf(_N_masked), -1E5, _N_masked)
    _N_masked = np.where(np.isinf(-1*_N_masked), -1E5, _N_masked)

    #print(_N_masked)
    _N = (_N_masked > -1E5).sum(axis=0)
    #print(_N)
    _rms_int = np.sqrt(_N) * _params['_rms_med'] * (_params['cdelt3']/1000.)

    print(mom0)
    print(_rms_int)
    # integrated s/n map: spectralcube array
    _sn_int_map = mom0 / _rms_int
    #print("int sn")
    #print(_sn_int_map)
    # integrated s/n map: numpy array : being returned 
    #print(_sn_int_map)

    _sn_int_map_nparray = _sn_int_map.hdu.data
    _sn_int_map_nparray = np.where(np.isnan(_sn_int_map_nparray), 0, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(_sn_int_map_nparray), 1E5, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(-1*_sn_int_map_nparray), -1E5, _sn_int_map_nparray)

    # write fits
    # moment0
    mom0.write('test1.mom0.fits', overwrite=True)
    _sn_int_map.hdu.header['BUNIT'] = 's/n'
    # write fits
    # _sn_int_map
    _sn_int_map.write('test1.sn_int.fits', overwrite=True)
    #print('moment0_unit:', mom0.unit)

    return peak_sn_map, _sn_int_map_nparray
#-- END OF SUB-ROUTINE____________________________________________________________#



