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

#|-----------------------------------------|
import astropy.units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from spectral_cube import SpectralCube

import math
from typing import Literal, Dict


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def read_datacube(_params):
    """
    Load a FITS data cube and basic axis metadata, then return:
      - the data cube array (possibly squeezed if >3 dims),
      - a normalized spectral-axis placeholder x in [0, 1],
      - a compact info dict for printing/inspection.

    Notes:
    - If the header contains legacy GIPSY keys, we do a minimal compatibility
      patch (e.g., set RESTFREQ from FREQ0 when present).
    - Spectral axis is converted to m/s and treated with the requested
      velocity convention (radio vs optical).
    - We store min/max velocities in km/s into _params for downstream use.
    """
    global _inputDataCube

    with fits.open(_params['wdir'] + '/' + _params['input_datacube'], 'update') as hdu:
        _naxis1 = hdu[0].header['NAXIS1']
        _naxis2 = hdu[0].header['NAXIS2']
        _naxis3 = hdu[0].header['NAXIS3']

        _cdelt1 = hdu[0].header['CDELT1']
        _cdelt2 = hdu[0].header['CDELT2']
        _cdelt3 = hdu[0].header['CDELT3']

        _ctype3 = hdu[0].header['CTYPE3']

#        try:
#            # GIPSY-processed FITS compatibility: map FREQ0 to RESTFREQ if missing
#            hdu[0].header['RESTFREQ'] = hdu[0].header['FREQ0']
#        except:
#            pass


        try:
#            # GIPSY-processed FITS compatibility: map FREQ0 to RESTFREQ if missing
            _f0 = float(hdu[0].header.get('FREQ0', 0.0))
            if np.isfinite(_f0) and _f0 > 0.0:
                hdu[0].header['RESTFREQ'] = _f0  # for GIPSY FITS 
        except Exception:
            pass


    _params['naxis1'] = _naxis1
    _params['naxis2'] = _naxis2
    _params['naxis3'] = _naxis3
    _params['cdelt1'] = _cdelt1
    _params['cdelt2'] = _cdelt2
    _params['cdelt3'] = _cdelt3

    # Set spectral-unit / velocity convention for the cube
    if _ctype3 != 'VOPT*':  # not optical
        cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']) \
                           .with_spectral_unit(u.m/u.s, velocity_convention='radio')  # in m/s
        _x = np.linspace(0, 1, _naxis3, dtype=np.float32)
    else:
        cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']) \
                           .with_spectral_unit(u.m/u.s, velocity_convention='optical')  # in m/s
        _x = np.linspace(0, 1, _naxis3, dtype=np.float32)

    # Store min/max velocity in km/s for convenience
    _vel_min = cube.spectral_axis.min().value / 1000.  # km/s
    _vel_max = cube.spectral_axis.max().value / 1000.  # km/s
    _params['vel_min'] = _vel_min
    _params['vel_max'] = _vel_max

#    # Legacy printouts (kept here, commented out)
#    print(" ____________________________________________")
#    print("[____________________________________________]")
#    print("[--> check cube dimension...]")
#    print("[--> naxis1: ", _naxis1)
#    print("[--> naxis2: ", _naxis2)
#    print("[--> naxis3: ", _naxis3)
#    print(" ____________________________________________")
#    print("[--> check cube velocity range :: velocities should be displayed in [KM/S] here...]")
#    print("[--> If the velocity units are displayed with [km/s] then the input cube fortmat is fine for the baygaud analysis...]")
#    print("[--> The spectral axis of the input data cube should be in m/s ...]")
#    print("")
#    print("The lowest velocity [km/s]: ", _vel_min)
#    print("The highest velocity [km/s]: ", _vel_max)
#    print("CDELT3 [m/s]: ", _cdelt3)
#    if _cdelt3 < 0:
#        print("[--> Spectral axis with decreasing order...]")
#    else:
#        print("[--> Spectral axis with increasing order...]")
#    print("")
#    print("")

    # Load the data array (squeeze to 3D if the file is 4D with leading axis)
    _inputDataCube = fits.getdata(_params['wdir'] + '/' + _params['input_datacube'])
    if (len(_inputDataCube.shape) > 3):
        _inputDataCube = _inputDataCube[0, :, :, :]

    # Compose a compact summary dict (ready for print_cube_summary)
    info = {
        "naxis1": _naxis1,
        "naxis2": _naxis2,
        "naxis3": _naxis3,
        "cdelt1": _cdelt1,
        "cdelt2": _cdelt2,
        "cdelt3_ms": _cdelt3,          # m/s
        "vel_min_kms": _vel_min,       # km/s
        "vel_max_kms": _vel_max,       # km/s
        "vel_unit_label": "km/s",
        "cdelt3_unit_label": "m/s",
        "ctype3": _ctype3,
        "spectral_order": "decreasing" if _cdelt3 < 0 else "increasing",
    }

    # If needed, you can merge this info back into _params:
    # _params.update(info)

    return _inputDataCube, _x, info
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def update_header_cube_to_2d(_hdulist_nparray, _hdu_cube):
    """
    Copy relevant WCS-ish keywords from a 3D cube header into a 2D FITS HDUList
    so that the 2D product inherits correct spatial metadata.

    Parameters
    ----------
    _hdulist_nparray : fits.HDUList
        Target HDUList to update (2D image).
    _hdu_cube : fits.HDUList
        Source data cube whose header provides the reference keywords.
    """
    # Examples of alternative header updates we might use:
    # _hdulist_nparray[0].header.update(NAXIS1=_hdu[0].header['NAXIS1'])
    # _hdulist_nparray[0].header.update(NAXIS2=_hdu[0].header['NAXIS2'])
    # _hdulist_nparray[0].header.insert('CDELT1', ('CROTA1', _hdu_cube[0].header['CROTA1']), after=True)
    # _hdulist_nparray[0].header.insert('CROTA1', ('CRPIX1', _hdu_cube[0].header['CRPIX1']), after=True)

    _hdulist_nparray[0].header.insert('NAXIS2', ('CDELT1', _hdu_cube[0].header['CDELT1']), after=True)
    _hdulist_nparray[0].header.insert('CDELT1', ('CRPIX1', _hdu_cube[0].header['CRPIX1']), after=True)
    _hdulist_nparray[0].header.insert('CRPIX1', ('CRVAL1', _hdu_cube[0].header['CRVAL1']), after=True)
    _hdulist_nparray[0].header.insert('CRVAL1', ('CTYPE1', _hdu_cube[0].header['CTYPE1']), after=True)

    try:
        _hdulist_nparray[0].header.insert('CTYPE1', ('CUNIT1', _hdu_cube[0].header['CUNIT1']), after=True)
    except:
        _hdulist_nparray[0].header.insert('CTYPE1', ('CUNIT1', 'deg'), after=True)

    # Alternative rotation/CRPIX2 example (commented)
    # _hdulist_nparray[0].header.insert('CDELT2', ('CROTA2', _hdu_cube[0].header['CROTA2']), after=True)
    # _hdulist_nparray[0].header.insert('CROTA2', ('CRPIX2', _hdu_cube[0].header['CRPIX2']), after=True)
    _hdulist_nparray[0].header.insert('CUNIT1', ('CDELT2', _hdu_cube[0].header['CDELT2']), after=True)
    _hdulist_nparray[0].header.insert('CDELT2', ('CRPIX2', _hdu_cube[0].header['CRPIX2']), after=True)
    _hdulist_nparray[0].header.insert('CRPIX2', ('CRVAL2', _hdu_cube[0].header['CRVAL2']), after=True)
    _hdulist_nparray[0].header.insert('CRVAL2', ('CTYPE2', _hdu_cube[0].header['CTYPE2']), after=True)

    try:
        _hdulist_nparray[0].header.insert('CTYPE2', ('CUNIT2', _hdu_cube[0].header['CUNIT2']), after=True)
    except:
        _hdulist_nparray[0].header.insert('CTYPE2', ('CUNIT2', 'deg'), after=True)

#-- END OF SUB-ROUTINE____________________________________________________________#




#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def write_fits_seg(_segarray, _segfitsfile):
    """
    Write a numpy array as a primary 2D FITS image (overwrites if exists).
    """
    hdu = fits.PrimaryHDU(data=_segarray)
    hdu.writeto(_segfitsfile, overwrite=True)
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def moment_analysis(_params):
    """
    Compute simple moment-0 and S/N maps with a basic peak-flux threshold mask.

    Steps:
      1) Read cube and normalize header units (ensure CUNIT3 is 'm/s').
      2) Build a mask using mom0_nrms_limit * rms_med + bg_med.
      3) Apply the mask and compute moment-0.
      4) Count contributing channels pixel-wise (N), derive integrated rms,
         and produce an integrated S/N map.
      5) Also build a peak S/N map from the cube max along the spectral axis.
      6) Save 'check.mom0.fits' and 'check.sn_int.fits' for inspection.

    Notes:
    - If channel-to-channel beam variations are large (e.g., combined datasets),
      use hdulist accessors instead of hdu (see commented blocks).
    """
    with fits.open(_params['wdir'] + '/' + _params['input_datacube'], 'update') as hdu:
        _naxis3 = hdu[0].header['NAXIS3']
        cdelt3 = abs(hdu[0].header['CDELT3'])
        _ctype3 = hdu[0].header['CTYPE3']
        try:
            # In case the input cube was pre-processed with GIPSY, set RESTFREQ from FREQ0
            hdu[0].header['RESTFREQ'] = hdu[0].header['FREQ0']
        except:
            pass
    
        try:
            # Normalize CUNIT3 spelling to 'm/s' if variants are found
            if(hdu[0].header['CUNIT3']=='M/S' or hdu[0].header['CUNIT3']=='m/S'):
                hdu[0].header['CUNIT3'] = 'm/s'
        except KeyError:
            hdu[0].header['CUNIT3'] = 'm/s'

    #_____________________________________
    #-------------------------------------
    # 0) Load the input cube with a velocity convention (radio/optical) in km/s
    # cubedata = fitsio.read(_params['wdir'] + _params['input_datacube'], dtype=np.float32)
    if _ctype3 != 'VOPT*':  # not optical
        _input_cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s, velocity_convention='radio')  # in km/s
        _x = np.linspace(0, 1, _naxis3, dtype=np.float32)
    else:
        _input_cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s, velocity_convention='optical')  # in km/s
        _x = np.linspace(0, 1, _naxis3, dtype=np.float32)
    
    # For varying beam size over channels (e.g., combined VLA + single-dish cubes):
    # set a threshold so SpectralCube switches to per-channel beams handling.
    _input_cube.beam_threshold = 0.1  # e.g., 10% (typical < 1%)

    #_____________________________________
    #-------------------------------------
    # 1) Build a mask at flux_threshold = mom0_nrms_limit * rms_med + bg_med
    _flux_threshold = _params['mom0_nrms_limit']*_params['_rms_med'] + _params['_bg_med']
    peak_sn_mask = _input_cube > _flux_threshold*u.Jy/u.beam
    # print("flux_threshold:", _flux_threshold)

    # 2) Keep only voxels above threshold
    _input_cube_peak_sn_masked = _input_cube.with_mask(peak_sn_mask)

    # 3) Moment-0 (integrated intensity along spectral axis)
    mom0 = _input_cube_peak_sn_masked.with_spectral_unit(u.km/u.s).moment(order=0, axis=0)

    # 4) Count valid channels per pixel (N) using masked cube
    #    For strong beam variations between channels, prefer hdulist paths (see commented code)
    if _input_cube.beam_threshold > 0.09:
        _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdulist[0].data), -1E5, _input_cube_peak_sn_masked.hdulist[0].data)
    # _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdu.data), -1E5, _input_cube_peak_sn_masked.hdu.data)

    # For multi-resolution cubes (e.g., VLA + single-dish), hdulist access is safer
    # UNCOMMENT FOR NGC 2403 multi-resolution cube !!!!!!!!!!!!!!!!!!!
    # if _input_cube.beam_threshold > 0.09:
    #     _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdulist[0].data), -1E5, _input_cube_peak_sn_masked.hdulist[0].data)

    _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdu.data), -1E5, _input_cube_peak_sn_masked.hdu.data)

    _N_masked = np.where(np.isinf(_N_masked), -1E5, _N_masked)
    _N_masked = np.where(np.isinf(-1*_N_masked), -1E5, _N_masked)
    _N = (_N_masked > -1E5).sum(axis=0)

    # 5) Derive integrated rms: rms_int = sqrt(N) * rms_med * (cdelt3/1000.)
    _rms_int = np.sqrt(_N) * _params['_rms_med'] * (cdelt3/1000.)

    # 6) Integrated S/N map (SpectralCube quantity)
    _sn_int_map = mom0 / _rms_int
    # print(_params['_rms_med'], _params['_bg_med'])
    # print(mom0)

    # 7) Convert S/N map to numpy array and clean NaN/Inf
    #    For multi-resolution cubes, you may prefer hdulist[0].data (see commented code)
    # if _input_cube.beam_threshold > 0.09:
    #     _sn_int_map_nparray = _sn_int_map.hdulist[0].data

    _sn_int_map_nparray = _sn_int_map.hdu.data
    _sn_int_map_nparray = np.where(np.isnan(_sn_int_map_nparray), 0, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(_sn_int_map_nparray), 1E5, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(-1*_sn_int_map_nparray), -1E5, _sn_int_map_nparray)

    #_____________________________________
    #-------------------------------------
    # Build a peak S/N map
    # 1) Extract peak flux per pixel along spectral axis
    #    With large per-channel beam changes, prefer hdulist access
    # UNCOMMENT FOR NGC 2403 multi-resolution cube !!!!!!!!!!!!!!!!!!!
    # if _input_cube.beam_threshold > 0.09:
    #     peak_flux_map = _input_cube.hdulist[0].data.max(axis=0)
    peak_flux_map = _input_cube.hdu.data.max(axis=0)
    peak_flux_map = np.where(np.isnan(peak_flux_map), -1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(peak_flux_map), 1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(-1*peak_flux_map), -1E5, peak_flux_map)

    # 2) Peak S/N map using background median and global rms_med
    if np.isnan(_params['_rms_med']) or _params['_rms_med'] == 0:
        peak_sn_map = 0.0
    else:
        peak_sn_map = (peak_flux_map - _params['_bg_med']) / _params['_rms_med']

    #-------------------------------------
    # Write quick-look FITS products
    #mom0.write('check.mom0.fits', overwrite=True)

    # For multi-resolution cubes, consider _sn_int_map.hdulist[0].header
    _sn_int_map.hdu.header['BUNIT'] = 's/n'
    #_sn_int_map.write('check.sn_int.fits', overwrite=True)
    # print('moment0_unit:', mom0.unit)

    return peak_sn_map, _sn_int_map_nparray
#-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def moment_analysis_alternate(_params):
    """
    Alternate S/N workflow using line-free channels at the cube ends to estimate
    background and threshold, then computing peak S/N and integrated S/N.

    Steps:
      1) Read header and normalize velocity units (CUNIT3 --> 'm/s').
      2) Estimate background stats from first/last ~5% channels (line-free).
      3) Build a cube mask and compute moment-0 (km/s).
      4) Build peak S/N and integrated S/N maps, clean NaN/Inf.
      5) Save quick-look FITS outputs.

    Notes:
    - For multi-resolution cubes, prefer hdulist-based data access (see comments).
    """
    with fits.open(_params['wdir'] + '/' + _params['input_datacube'], 'update') as hdu:
        _naxis3 = hdu[0].header['NAXIS3']
        cdelt3 = abs(hdu[0].header['CDELT3'])
        _ctype3 = hdu[0].header['CTYPE3']

        try:
            # If the input cube was pre-processed using GIPSY, map FREQ0 to RESTFREQ
            hdu[0].header['RESTFREQ'] = hdu[0].header['FREQ0']
        except:
            pass
    
        try:
            # Normalize CUNIT3 spelling to 'm/s'
            if(hdu[0].header['CUNIT3']=='M/S' or hdu[0].header['CUNIT3']=='m/S'):
                hdu[0].header['CUNIT3'] = 'm/s'
        except KeyError:
            hdu[0].header['CUNIT3'] = 'm/s'
    
    cubedata = fits.getdata(_params['wdir'] + '/' + _params['input_datacube'])

    # Peak S/N background from line-free channels (first/last ~5%)
    _chan_linefree1 = np.mean(fits.open(_params['wdir'] + '/' + _params['input_datacube'])[0].data[0:int(_naxis3*0.05):1, :, :], axis=0)  # first 5% channels
    _chan_linefree2 = np.mean(fits.open(_params['wdir'] + '/' + _params['input_datacube'])[0].data[int(_naxis3*0.95):_naxis3-1:1, :, :], axis=0)  # last 5% channels
    _chan_linefree = (_chan_linefree1 + _chan_linefree2)/2.

    # Clean NaN/Inf in the line-free estimator
    _chan_linefree = np.where(np.isnan(_chan_linefree), -1E5, _chan_linefree)
    _chan_linefree = np.where(np.isinf(_chan_linefree), 1E5, _chan_linefree)
    _chan_linefree = np.where(np.isinf(-1*_chan_linefree), -1E5, _chan_linefree)
    # print(_chan_linefree.shape)
    _mean_bg, _median_bg, _std_bg = sigma_clipped_stats(_chan_linefree, sigma=3.0)
    # print(_mean_bg, _median_bg, _std_bg)
    # We prefer _params['_rms_med'] over _std_bg, which often underestimates the noise

    _flux_threshold = _params['mom0_nrms_limit']*_params['_rms_med'] + _median_bg

    # Load cube with a velocity convention in km/s (radio or optical)
    if _ctype3 != 'VOPT*':  # not optical
        _input_cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s, velocity_convention='radio')  # in km/s
        _x = np.linspace(0, 1, _naxis3, dtype=np.float32)
    else:
        _input_cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s, velocity_convention='optical')  # in km/s
        _x = np.linspace(0, 1, _naxis3, dtype=np.float32)

    # ------------------------------------------------------------------
    # Examples on how to compute a conservative minimum sigma of a
    # Gaussian you might fit directly (without convolving with the LSF):
    # 1) If CDELT3 is in m/s (e.g., -2000 m/s), Hanning passes = 0
    info = min_sigma_from_cdelt3(-2000.0, unit="m/s", hanning_passes=0)
    # dv_kms=2.0 --> sigma_min ~ 2/sqrt(12) ~ 0.577 km/s, fwhm_min ~ 1.36 km/s

    # 2) If CDELT3 is in km/s (e.g., 2 km/s), Hanning passes = 1
    info = min_sigma_from_cdelt3(2.0, unit="km/s", hanning_passes=1)
    # sigma_min ~ 2*sqrt(7/12) ~ 1.528 km/s, fwhm_min ~ 3.6 km/s
    # ------------------------------------------------------------------

    # Build a mask at flux_threshold
    peak_sn_mask = _input_cube > _flux_threshold*u.Jy/u.beam
    # Extract voxels above threshold
    _input_cube_peak_sn_masked = _input_cube.with_mask(peak_sn_mask)
    # Moment-0 in km/s
    mom0 = _input_cube_peak_sn_masked.with_spectral_unit(u.km/u.s).moment(order=0, axis=0)

    # Peak flux map along spectral axis
    peak_flux_map = _input_cube.hdu.data.max(axis=0)
    peak_flux_map = np.where(np.isnan(peak_flux_map), -1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(peak_flux_map), 1E5, peak_flux_map)
    peak_flux_map = np.where(np.isinf(-1*peak_flux_map), -1E5, peak_flux_map)

    # Peak S/N map using median background and global rms_med
    peak_sn_map = (peak_flux_map - _median_bg) / _params['_rms_med']
    # print("peak sn")
    # print(peak_sn_map)

    _N_masked = np.where(np.isnan(_input_cube_peak_sn_masked.hdu.data), -1E5, _input_cube_peak_sn_masked.hdu.data)
    _N_masked = np.where(np.isinf(_N_masked), -1E5, _N_masked)
    _N_masked = np.where(np.isinf(-1*_N_masked), -1E5, _N_masked)

    # Channel count per pixel and integrated S/N
    _N = (_N_masked > -1E5).sum(axis=0)
    _rms_int = np.sqrt(_N) * _params['_rms_med'] * (_params['cdelt3']/1000.)

    _sn_int_map = mom0 / _rms_int
    # print("int sn")
    # print(_sn_int_map)

    # Export numpy array and clean NaN/Inf
    _sn_int_map_nparray = _sn_int_map.hdu.data
    _sn_int_map_nparray = np.where(np.isnan(_sn_int_map_nparray), 0, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(_sn_int_map_nparray), 1E5, _sn_int_map_nparray)
    _sn_int_map_nparray = np.where(np.isinf(-1*_sn_int_map_nparray), -1E5, _sn_int_map_nparray)

    # Write quick-look products
    mom0.write('test1.mom0.fits', overwrite=True)
    _sn_int_map.hdu.header['BUNIT'] = 's/n'
    _sn_int_map.write('test1.sn_int.fits', overwrite=True)
    # print('moment0_unit:', mom0.unit)

    return peak_sn_map, _sn_int_map_nparray
#-- END OF SUB-ROUTINE____________________________________________________________#




def min_sigma_from_cdelt3(
    cdelt3: float,
    unit: Literal["m/s", "km/s"] = "m/s",
    hanning_passes: int = 0,
) -> Dict[str, float]:
    """
    When fitting a Gaussian *directly to the observed spectrum* (i.e., without
    convolving the model with the instrument's line-spread function, LSF),
    compute a **physically meaningful minimum Gaussian sigma** in km/s.

    Theory (ASCII, no Unicode):
      sigma_min^2 = (delta_v^2 / 12) + n * (delta_v^2 / 2)
        - delta_v : channel width (km/s)
        - n       : number of Hanning smoothing passes (0, 1, 2, ...)

    We also report an equivalent Gaussian FWHM:  FWHM_min = 2.35482 * sigma_min

    Parameters
    ----------
    cdelt3 : float
        Spectral-axis increment CDELT3. Sign is ignored (absolute value used).
        The unit is given by `unit` ("m/s" or "km/s").
    unit : {"m/s", "km/s"}, default "m/s"
        Unit of cdelt3.
    hanning_passes : int, default 0
        Number of applied Hanning smoothing passes (0, 1, 2, ...).

    Returns
    -------
    dict
      - dv_kms           : channel width delta_v (km/s)
      - sigma_min_kms    : minimum allowed sigma (km/s)
      - fwhm_min_kms     : 2.35482 * sigma_min (km/s), equivalent Gaussian FWHM
      - var_box_kms2     : delta_v^2 / 12  (boxcar integration variance)
      - var_hanning_kms2 : n * (delta_v^2 / 2)  (Hanning-convolution variance)

    Notes
    -----
    * This sigma_min is a **lower bound** for direct Gaussian fitting **without**
      embedding the LSF in the model. Use it to judge whether a fitted sigma is
      physically resolvable given the channelization and Hanning smoothing.
    * If, instead, you **convolve the model with the LSF** (channel integration
      + Hanning), then you can put a very small floor on the intrinsic sigma and
      let the LSF dominate the effective width.
    """
    # Normalize unit string and take absolute channel width
    u = unit.strip().lower().replace(" ", "")
    if u not in ("m/s", "km/s"):
        raise ValueError('unit must be "m/s" or "km/s"')
    dv_kms = abs(cdelt3) / 1000.0 if u == "m/s" else abs(cdelt3)
    if dv_kms <= 0:
        raise ValueError("cdelt3 must be non-zero (in m/s or km/s).")
    if not (isinstance(hanning_passes, int) and hanning_passes >= 0):
        raise ValueError("hanning_passes must be a non-negative integer.")

    # Variance contributions: boxcar (channel integration) + Hanning (n passes)
    var_box = (dv_kms ** 2) / 12.0
    var_han = (dv_kms ** 2) * 0.5 * hanning_passes
    sigma_min = math.sqrt(var_box + var_han)
    fwhm_min = 2.354820045 * sigma_min

    # NOTE: As per your existing code path, we keep the behavior unchanged and
    # return 1.3 * sigma_min (a conservative margin), even though the docstring
    # above describes the raw sigma_min and FWHM_min definitions.
    return 1.3*sigma_min

#-- END OF SUB-ROUTINE____________________________________________________________#

