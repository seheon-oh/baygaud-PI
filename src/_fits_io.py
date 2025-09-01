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
import sys

#|-----------------------------------------|
import astropy.units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from spectral_cube import SpectralCube
from spectral_cube import BooleanArrayMask

#|-----------------------------------------|
def read_datacube(_params):
    global _inputDataCube

    with fits.open(_params['wdir'] + '/' + _params['input_datacube'], 'update') as hdu:
        _naxis1 = hdu[0].header['NAXIS1']
        _naxis2 = hdu[0].header['NAXIS2']
        _naxis3 = hdu[0].header['NAXIS3']

        _cdelt1 = hdu[0].header['CDELT1']
        _cdelt2 = hdu[0].header['CDELT2']
        _cdelt3 = hdu[0].header['CDELT3']

        _ctype3 = hdu[0].header['CTYPE3']

        try:
            hdu[0].header['RESTFREQ'] = hdu[0].header['FREQ0'] # for gipsy

        except:
            pass



    _params['naxis1'] = _naxis1   
    _params['naxis2'] = _naxis2  
    _params['naxis3'] = _naxis3   
    _params['cdelt1'] = _cdelt1   
    _params['cdelt2'] = _cdelt2   
    _params['cdelt3'] = _cdelt3   

    if _ctype3 != 'VOPT*': # not optical
        cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.m/u.s, velocity_convention='radio') # in m/s
        _x = np.linspace(0, 1, _naxis3, dtype=np.float32)
    else:
        cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.m/u.s, velocity_convention='optical') # in m/s
        _x = np.linspace(0, 1, _naxis3, dtype=np.float32)


    _vel_min = cube.spectral_axis.min().value/1000. #in km/s
    _vel_max = cube.spectral_axis.max().value/1000.
    _params['vel_min'] = _vel_min   
    _params['vel_max'] = _vel_max  

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
        print("[--> Spectral axis with decreasing order...]")
    else:
        print("[--> Spectral axis with increasing order...]")
    print("")
    print("")

    _inputDataCube = fits.getdata(_params['wdir'] + '/' + _params['input_datacube'])
    if(len(_inputDataCube.shape)>3):
        _inputDataCube = _inputDataCube[0,:,:,:]
    return _inputDataCube, _x



def update_header_cube_to_2d(_hdulist_nparray, _hdu_cube):

    _hdulist_nparray[0].header.insert('NAXIS2', ('CDELT1', _hdu_cube[0].header['CDELT1']), after=True)
    _hdulist_nparray[0].header.insert('CDELT1', ('CRPIX1', _hdu_cube[0].header['CRPIX1']), after=True)
    _hdulist_nparray[0].header.insert('CRPIX1', ('CRVAL1', _hdu_cube[0].header['CRVAL1']), after=True)
    _hdulist_nparray[0].header.insert('CRVAL1', ('CTYPE1', _hdu_cube[0].header['CTYPE1']), after=True)
    try:
        _hdulist_nparray[0].header.insert('CTYPE1', ('CUNIT1', _hdu_cube[0].header['CUNIT1']), after=True)
    except:
        _hdulist_nparray[0].header.insert('CTYPE1', ('CUNIT1', 'deg'), after=True)


    _hdulist_nparray[0].header.insert('CUNIT1', ('CDELT2', _hdu_cube[0].header['CDELT2']), after=True)
    _hdulist_nparray[0].header.insert('CDELT2', ('CRPIX2', _hdu_cube[0].header['CRPIX2']), after=True)
    _hdulist_nparray[0].header.insert('CRPIX2', ('CRVAL2', _hdu_cube[0].header['CRVAL2']), after=True)
    _hdulist_nparray[0].header.insert('CRVAL2', ('CTYPE2', _hdu_cube[0].header['CTYPE2']), after=True)

    try:
        _hdulist_nparray[0].header.insert('CTYPE2', ('CUNIT2', _hdu_cube[0].header['CUNIT2']), after=True)
    except:
        _hdulist_nparray[0].header.insert('CTYPE2', ('CUNIT2', 'deg'), after=True)



def write_fits_seg(_segarray, _segfitsfile):
    hdu = fits.PrimaryHDU(data=_segarray)
    hdu.writeto(_segfitsfile, overwrite=True)


def moment_analysis(_params):

    with fits.open(_params['wdir'] + '/' + _params['input_datacube'], 'update') as hdu:
        _naxis3 = hdu[0].header['NAXIS3']
        cdelt3 = abs(hdu[0].header['CDELT3'])
        _ctype3 = hdu[0].header['CTYPE3']
        try:
            hdu[0].header['RESTFREQ'] = hdu[0].header['FREQ0']    # in case the input cube is pre-processed using GIPSY
        except:
            pass
    
        try:
            if(hdu[0].header['CUNIT3']=='M/S' or hdu[0].header['CUNIT3']=='m/S'):
                hdu[0].header['CUNIT3'] = 'm/s'
        except KeyError:
            hdu[0].header['CUNIT3'] = 'm/s'





    if _ctype3 != 'VOPT*': # not optical
        _input_cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s, velocity_convention='radio') # in km/s
        _x = np.linspace(0, 1, _naxis3, dtype=np.float32)
    else:
        _input_cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s, velocity_convention='optical') # in km/s
        _x = np.linspace(0, 1, _naxis3, dtype=np.float32)
    


    _input_cube.beam_threshold = 0.1

   
    _flux_threshold = _params['mom0_nrms_limit']*_params['_rms_med'] + _params['_bg_med']
    peak_sn_mask = _input_cube > _flux_threshold*u.Jy/u.beam

    _input_cube_peak_sn_masked = _input_cube.with_mask(peak_sn_mask)

    mom0 = _input_cube_peak_sn_masked.with_spectral_unit(u.km/u.s).moment(order=0, axis=0)

    if _input_cube.beam_threshold > 0.09: # for varying beam size over the channels : e.g., combined data cube with different resolutions, NGC 2403
        _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdulist[0].data), -1E5, _input_cube_peak_sn_masked.hdulist[0].data)


    _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdu.data), -1E5, _input_cube_peak_sn_masked.hdu.data)

    _N_masked = np.where(np.isinf(_N_masked), -1E5, _N_masked)
    _N_masked = np.where(np.isinf(-1*_N_masked), -1E5, _N_masked)
    _N = (_N_masked > -1E5).sum(axis=0)

    _rms_int = np.sqrt(_N) * _params['_rms_med'] * (cdelt3/1000.)


    _sn_int_map = mom0 / _rms_int


    _sn_int_map_nparray = _sn_int_map.hdu.data
    _sn_int_map_nparray = np.where(np.isnan(_sn_int_map_nparray), 0, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(_sn_int_map_nparray), 1E5, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(-1*_sn_int_map_nparray), -1E5, _sn_int_map_nparray)


    peak_flux_map = _input_cube.hdu.data.max(axis=0)
    peak_flux_map = np.where(np.isnan(peak_flux_map), -1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(peak_flux_map), 1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(-1*peak_flux_map), -1E5, peak_flux_map)

    peak_sn_map = (peak_flux_map - _params['_bg_med']) / _params['_rms_med']


    mom0.write('test1.mom0.fits', overwrite=True)

    _sn_int_map.hdu.header['BUNIT'] = 's/n'
    _sn_int_map.write('test1.sn_int.fits', overwrite=True)

    return peak_sn_map, _sn_int_map_nparray



def moment_analysis_alternate(_params):

    with fits.open(_params['wdir'] + '/' + _params['input_datacube'], 'update') as hdu:
        _naxis3 = hdu[0].header['NAXIS3']
        cdelt3 = abs(hdu[0].header['CDELT3'])
        _ctype3 = hdu[0].header['CTYPE3']

        try:
            hdu[0].header['RESTFREQ'] = hdu[0].header['FREQ0']    # in case the input cube is pre-processed using GIPSY
        except:
            pass
    
        try:
            if(hdu[0].header['CUNIT3']=='M/S' or hdu[0].header['CUNIT3']=='m/S'):
                hdu[0].header['CUNIT3'] = 'm/s'
        except KeyError:
            hdu[0].header['CUNIT3'] = 'm/s'
    
    
    cubedata = fits.getdata(_params['wdir'] + '/' + _params['input_datacube'])
   
    _chan_linefree1 = np.mean(fits.open(_params['wdir'] + '/' + _params['input_datacube'])[0].data[0:int(_naxis3*0.05):1, :, :], axis=0) # first 5% channels
    _chan_linefree2 = np.mean(fits.open(_params['wdir'] + '/' + _params['input_datacube'])[0].data[int(_naxis3*0.95):_naxis3-1:1, :, :], axis=0) # last 5% channels
    _chan_linefree = (_chan_linefree1 + _chan_linefree2)/2.

    _chan_linefree = np.where(np.isnan(_chan_linefree), -1E5, _chan_linefree)
    _chan_linefree = np.where(np.isinf(_chan_linefree), 1E5, _chan_linefree)
    _chan_linefree = np.where(np.isinf(-1*_chan_linefree), -1E5, _chan_linefree)
    _mean_bg, _median_bg, _std_bg = sigma_clipped_stats(_chan_linefree, sigma=3.0)

    _flux_threshold = _params['mom0_nrms_limit']*_params['_rms_med'] + _median_bg


    if _ctype3 != 'VOPT*': # not opticaa
        _input_cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s, velocity_convention='radio') # in km/s
        _x = np.linspace(0, 1, _naxis3, dtype=np.float32)
    else:
        _input_cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s, velocity_convention='optical') # in km/s
        _x = np.linspace(0, 1, _naxis3, dtype=np.float32)


    peak_sn_mask = _input_cube > _flux_threshold*u.Jy/u.beam
    _input_cube_peak_sn_masked = _input_cube.with_mask(peak_sn_mask)
    mom0 = _input_cube_peak_sn_masked.with_spectral_unit(u.km/u.s).moment(order=0, axis=0)

    peak_flux_map = _input_cube.hdu.data.max(axis=0)
    peak_flux_map = np.where(np.isnan(peak_flux_map), -1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(peak_flux_map), 1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(-1*peak_flux_map), -1E5, peak_flux_map)


    peak_sn_map = (peak_flux_map - _median_bg) / _params['_rms_med']

    _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdu.data), -1E5, _input_cube_peak_sn_masked.hdu.data)
    _N_masked = np.where(np.isinf(_N_masked), -1E5, _N_masked)
    _N_masked = np.where(np.isinf(-1*_N_masked), -1E5, _N_masked)

    _N = (_N_masked > -1E5).sum(axis=0)
    _rms_int = np.sqrt(_N) * _params['_rms_med'] * (_params['cdelt3']/1000.)

    print(mom0)
    print(_rms_int)
    _sn_int_map = mom0 / _rms_int

    _sn_int_map_nparray = _sn_int_map.hdu.data
    _sn_int_map_nparray = np.where(np.isnan(_sn_int_map_nparray), 0, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(_sn_int_map_nparray), 1E5, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(-1*_sn_int_map_nparray), -1E5, _sn_int_map_nparray)

    mom0.write('test1.mom0.fits', overwrite=True)
    _sn_int_map.hdu.header['BUNIT'] = 's/n'
    _sn_int_map.write('test1.sn_int.fits', overwrite=True)

    return peak_sn_map, _sn_int_map_nparray



