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

from datetime import datetime, timezone
#|-----------------------------------------|
import astropy.units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from spectral_cube import SpectralCube

import math
from typing import Literal, Dict

from pathlib import Path




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
    #if _ctype3 != 'VOPT*':  # not optical
    if 'VOPT' not in _ctype3:  # not optical
        cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']) \
                           .with_spectral_unit(u.m/u.s, velocity_convention='radio')  # in m/s

        if _cdelt3 > 0: # positive channel width : increasing x
            _x = np.linspace(0, 1, _naxis3, dtype=np.float32)
        else: # negative : decreasing x
            _x = np.linspace(1, 0, _naxis3, dtype=np.float32)
    else:
        cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']) \
                           .with_spectral_unit(u.m/u.s, velocity_convention='optical')  # in m/s

        if _cdelt3 > 0: # positive channel width : increasing x
            _x = np.linspace(0, 1, _naxis3, dtype=np.float32)
        else: # negative : decreasing x
            _x = np.linspace(1, 0, _naxis3, dtype=np.float32)

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
# Get bygaud-PI version from _version.py in the same directory
try:
    from _version import __version__ as _BYGAUD_VERSION
except Exception:
    _BYGAUD_VERSION = "unknown"

# Map integer codes to BUNIT strings
_BUNIT_MAP = {
    0: "Jy/beam",           # integrated intensity
    1: "km/s",              # line-of-sight velocity 
    2: "km/s",              # velocity dispersion (gaussian sigma)
    3: "Jy/beam",           # background
    4: "Jy/beam",           # rms
    5: "Jy/beam",           # peakflux
    6: "peakflux-s/n",      # peakflux S/N
    7: "n-gauss",           # N-Gauss
}

def update_header_cube_to_2d(_hdulist_nparray,
                             _hdu_cube,
                             _params=None,
                             bunit_code: int = None):
    """
    Copy spatial WCS keywords (only those that exist) from a 3D cube header
    into a 2D FITS header. Also set ORIGIN, BUNIT, and add HISTORY lines.

    What this function does:
      - Copies spatial WCS keys for axes 1 and 2 if they exist in the cube.
      - If WCSAXES exists in the cube, set it to 2 in the 2D header.
      - Sets ORIGIN to 'bygaud-PI <version> (<UTC time>)'.
      - Sets BUNIT from an integer code (0..7).
      - Adds HISTORY lines:
          * 'input.data.cube = <input_datacube>' (if present in parameters)
          * A block of dynesty parameters
          * A block of classification parameters
          * A block of global kinematics parameters
    """

    # -------- helpers (short, safe, ASCII-only where needed) --------
    def _get_header(hdu_like):
        # Return a Header from an HDU or an HDUList
        return (hdu_like.header if hasattr(hdu_like, "header") else hdu_like[0].header)

    def _copy_if_present(dst, src, keys):
        # Copy keys that exist in source into destination
        for k in keys:
            if k in src:
                dst[k] = src[k]

    def _copy_matrix(dst, src, base):
        # Copy 2x2 PC/CD matrices for spatial axes (1,2) if present
        for i in (1, 2):
            for j in (1, 2):
                key = f"{base}{i}_{j}"
                if key in src:
                    dst[key] = src[key]

    def _add_history_line(hdr, text):
        # Add a HISTORY line; enforce ASCII to avoid FITS errors
        s = str(text)
        try:
            hdr.add_history(s)
        except Exception:
            hdr.add_history(s.encode("ascii", "ignore").decode("ascii"))

    def _add_history_block(hdr, title, kv_pairs, line_width=70):
        # Add a titled block of "k=v" pairs, wrapped to ~line_width
        if not kv_pairs:
            return
        _add_history_line(hdr, f"{title}:")
        line = ""
        for idx, (k, v) in enumerate(kv_pairs):
            frag = f"{k}={v}"
            if idx == 0:
                line = frag
            elif len(line) + 2 + len(frag) <= line_width:
                line += ", " + frag
            else:
                _add_history_line(hdr, "  " + line)
                line = frag
        if line:
            _add_history_line(hdr, "  " + line)

    # -------- get headers --------
    dst_hdr = _hdulist_nparray[0].header
    src_hdr = _get_header(_hdu_cube)

    # -------- copy basic spatial WCS keys if present --------
    scalar_keys = [
        # axis 1
        "CTYPE1", "CUNIT1", "CRVAL1", "CRPIX1", "CDELT1", "CROTA1",
        # axis 2
        "CTYPE2", "CUNIT2", "CRVAL2", "CRPIX2", "CDELT2", "CROTA2",
        # frame/system
        "RADESYS", "EQUINOX", "WCSNAME", "LONPOLE", "LATPOLE", "MJD-OBS", "DATE-OBS",
        # beam (copy if the cube has these)
        "BMAJ", "BMIN", "BPA",
        # data unit (will be overridden by bunit_code below)
        "BUNIT",
    ]
    _copy_if_present(dst_hdr, src_hdr, scalar_keys)

    # Copy 2x2 PC/CD if present
    _copy_matrix(dst_hdr, src_hdr, "PC")
    _copy_matrix(dst_hdr, src_hdr, "CD")

    # If cube has WCSAXES, set it to 2 for a 2D product
    if "WCSAXES" in src_hdr:
        try:
            dst_hdr["WCSAXES"] = 2
        except Exception:
            pass  # be tolerant if header library is picky about types

    # -------- ORIGIN (always set) --------
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    dst_hdr["ORIGIN"] = f"bygaud-PI {_BYGAUD_VERSION} ({now_iso})"

    # -------- BUNIT (must come from a code 0..7) --------
    if bunit_code is None:
        raise ValueError("You must pass bunit_code (0..7).")
    if bunit_code not in _BUNIT_MAP:
        raise ValueError(f"bunit_code must be in 0..7; got {bunit_code}")
    dst_hdr["BUNIT"] = _BUNIT_MAP[bunit_code]

    # -------- HISTORY from YAML parameters (dict or file) --------
    params = _params
    if isinstance(params, dict):
        # Record input data cube name if present
        in_cube = params.get("input_datacube")
        if in_cube:
            _add_history_line(dst_hdr, f"input.data.cube = {in_cube}")

        # Keys to record in HISTORY as simple k=v pairs
        dynesty_keys = [
            "_dynesty_class_", "nlive", "sample", "dlogz", "maxiter", "maxcall",
            "update_interval", "vol_dec", "vol_check", "facc", "fmove",
            "walks", "rwalk", "max_move", "bound",
        ]
        classif_keys = [
            "bayes_factor_limit", "max_ngauss", "mom0_nrms_limit",
            "peak_sn_pass_for_ng_opt", "peak_sn_limit", "int_sn_limit",
        ]
        global_kin_keys = [
            "g_vlos_lower", "g_vlos_upper", "g_sigma_lower", "g_sigma_upper",
        ]

        def _collect(keys):
            out = []
            for k in keys:
                if k in params:
                    v = params[k]
                    if isinstance(v, (list, tuple, np.ndarray)):
                        v = list(v)
                    out.append((k, v))
            return out

        _add_history_block(dst_hdr, "DYNESTY PARAMS", _collect(dynesty_keys))
        _add_history_block(dst_hdr, "CLASSIFICATION PARAMS", _collect(classif_keys))
        _add_history_block(dst_hdr, "GLOBAL KINEMATICS", _collect(global_kin_keys))
    else:
        _add_history_line(dst_hdr, "YAML history not added (no params dict).")

    return _hdulist_nparray




# [_____________________________________________________________________________] #
def update_header_cube_to_2d_org(_hdulist_nparray, _hdu_cube):
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
    #if _ctype3 != 'VOPT*':  # not optical
    if 'VOPT' not in _ctype3:  # not optical
        _input_cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s, velocity_convention='radio')  # in km/s

        if cdelt3 > 0: # positive channel width : increasing x
            _x = np.linspace(0, 1, _naxis3, dtype=np.float32)
        else: # negative : decreasing x
            _x = np.linspace(1, 0, _naxis3, dtype=np.float32)


    else:
        _input_cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s, velocity_convention='optical')  # in km/s

        if cdelt3 > 0: # positive channel width : increasing x
            _x = np.linspace(0, 1, _naxis3, dtype=np.float32)
        else: # negative : decreasing x
            _x = np.linspace(1, 0, _naxis3, dtype=np.float32)
    
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
    #if _ctype3 != 'VOPT*':  # not optical
    if 'VOPT' not in _ctype3:  # not optical
        _input_cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s, velocity_convention='radio')  # in km/s

        if cdelt3 > 0: # positive channel width : increasing x
            _x = np.linspace(0, 1, _naxis3, dtype=np.float32)
        else: # negative : decreasing x
            _x = np.linspace(1, 0, _naxis3, dtype=np.float32)

    else:
        _input_cube = SpectralCube.read(_params['wdir'] + '/' + _params['input_datacube']).with_spectral_unit(u.km/u.s, velocity_convention='optical')  # in km/s

        if cdelt3 > 0: # positive channel width : increasing x
            _x = np.linspace(0, 1, _naxis3, dtype=np.float32)
        else: # negative : decreasing x
            _x = np.linspace(1, 0, _naxis3, dtype=np.float32)

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



def _save_mask_outputs_to_fits1(
    valid2d,
    vmin2d=None,
    vmax2d=None,
    *,
    save_dir=None,
    prefix="mask",
    mef=False,
    overwrite=True,
    ref_header=None,
    vphys_min=None,
    vphys_max=None,
    vphys_unit="m/s"
):
    """
    Write mask products to FITS.

    Notes
    -----
    - If vmin2d/vmax2d are provided, they are assumed to be in *physical units*
      (e.g., m/s or km/s). This function does not normalize them.
    - HISTORY and header keywords (VMINPHYS/VMAXPHYS/VUNIT) are added
      for traceability of the physical velocity system.

    Parameters
    ----------
    valid2d : np.ndarray (uint8), shape (ny, nx)
        0: masked, 1: valid.
    vmin2d, vmax2d : np.ndarray or None
        2D arrays in physical velocity units to be written (float32 recommended).
        If None, they won't be written.
    save_dir, prefix, mef, overwrite, ref_header :
        Standard I/O options.
    vphys_min, vphys_max : float | None
        Global min/max velocity of the spectral axis for documentation purposes.
    vphys_unit : str
        Unit string for velocities (e.g. 'm/s', 'km/s').
    """


    save_dir = Path(save_dir) if save_dir is not None else Path(".")
    save_dir.mkdir(parents=True, exist_ok=True)

    hdr = fits.Header()
    if ref_header is not None:
        hdr = ref_header.copy()

    if (vphys_min is not None) and (vphys_max is not None):
        hdr.add_history("Velocity values are PHYSICAL (not normalized).")
        hdr["VMINPHYS"] = (float(vphys_min), f"Global min velocity [{vphys_unit}]")
        hdr["VMAXPHYS"] = (float(vphys_max), f"Global max velocity [{vphys_unit}]")
        hdr["VUNIT"] = (vphys_unit, "Velocity unit")

    written = {}

    if mef:
        hdus = [fits.PrimaryHDU()]
        h_valid = fits.ImageHDU(valid2d.astype(np.uint8), header=hdr, name="VALID")
        hdus.append(h_valid)

        if vmin2d is not None:
            h_vmin = fits.ImageHDU(vmin2d.astype(np.float32), header=hdr, name="VMIN")
            hdus.append(h_vmin)
        if vmax2d is not None:
            h_vmax = fits.ImageHDU(vmax2d.astype(np.float32), header=hdr, name="VMAX")
            hdus.append(h_vmax)

        out_path = save_dir / f"{prefix}.mef.fits"
        fits.HDUList(hdus).writeto(out_path, overwrite=overwrite)
        written["mef"] = out_path
    else:
        valid_path = save_dir / f"{prefix}.valid.fits"
        fits.writeto(valid_path, valid2d.astype(np.uint8), header=hdr, overwrite=overwrite)
        written["valid"] = valid_path

        if vmin2d is not None:
            vmin_path = save_dir / f"{prefix}.vmin.fits"
            fits.writeto(vmin_path, vmin2d.astype(np.float32), header=hdr, overwrite=overwrite)
            written["vmin"] = vmin_path
        if vmax2d is not None:
            vmax_path = save_dir / f"{prefix}.vmax.fits"
            fits.writeto(vmax_path, vmax2d.astype(np.float32), header=hdr, overwrite=overwrite)
            written["vmax"] = vmax_path

    return written
#-- END OF SUB-ROUTINE____________________________________________________________#


def _prepare_mask_2d1(
    mask2d_fits_path,
    target_shape=None,
    return_vrange=False,
    *,
    save_to_fits=False,
    save_dir=None,
    prefix=None,
    overwrite=True,
    ref_header=None,
    mef=False,
    # NEW: if you want to save physical vmin/vmax for a 2D mask,
    # provide the global spectral range of the *data cube* here.
    vphys_min=None,
    vphys_max=None,
    vphys_unit="m/s",
):
    """
    Read a 2D mask (FITS) and return a (ny, nx) validity map.

    Returns
    -------
    valid2d : uint8 (ny, nx)
    vmin2d_norm, vmax2d_norm : float32, optional (only if return_vrange=True)
    """
    mask2d_fits_path = Path(mask2d_fits_path)
    data = fits.getdata(mask2d_fits_path)

    if data.ndim != 2:
        raise ValueError(f"Expected a 2D mask; got ndim={data.ndim} for {mask2d_fits_path}")

    ny, nx = data.shape
    if target_shape is not None:
        ty, tx = target_shape
        if (ny, nx) != (ty, tx):
            raise ValueError(f"Shape mismatch: mask({ny},{nx}) != target({ty},{tx})")

    valid = (data > 0).astype(np.uint8)

    # If not provided, reuse input header for spatial metadata
    if ref_header is None:
        try:
            _, ref_header = fits.getdata(mask2d_fits_path, header=True)
        except Exception:
            ref_header = fits.Header()

    # Optional: normalized outputs for compatibility
    if not return_vrange:
        # Save if requested (VALID only, unless vphys_* provided)
        if save_to_fits:
            vmin_phys = vmax_phys = None
            if (vphys_min is not None) and (vphys_max is not None):
                vmin_phys = np.where(valid > 0, float(vphys_min), -1.0).astype(np.float32)
                vmax_phys = np.where(valid > 0, float(vphys_max), -1.0).astype(np.float32)
            _save_mask_outputs_to_fits(
                valid,
                vmin2d=vmin_phys,
                vmax2d=vmax_phys,
                save_dir=(save_dir or mask2d_fits_path.parent),
                prefix=(prefix or mask2d_fits_path.stem),
                mef=mef,
                overwrite=overwrite,
                ref_header=ref_header,
                vphys_min=vphys_min,
                vphys_max=vphys_max,
                vphys_unit=vphys_unit,
            )
        return valid

    # Normalized placeholders for callers that expect (valid, vmin, vmax)
    vmin_norm = np.where(valid > 0, 0.0, -1.0).astype(np.float32)
    vmax_norm = np.where(valid > 0, 1.0, -1.0).astype(np.float32)

    if save_to_fits:
        vmin_phys = vmax_phys = None
        if (vphys_min is not None) and (vphys_max is not None):
            vmin_phys = np.where(valid > 0, float(vphys_min), -1.0).astype(np.float32)
            vmax_phys = np.where(valid > 0, float(vphys_max), -1.0).astype(np.float32)

        _save_mask_outputs_to_fits(
            valid,
            vmin2d=vmin_phys,   # PHYSICAL values in file
            vmax2d=vmax_phys,   # PHYSICAL values in file
            save_dir=(save_dir or mask2d_fits_path.parent),
            prefix=(prefix or mask2d_fits_path.stem),
            mef=mef,
            overwrite=overwrite,
            ref_header=ref_header,
            vphys_min=vphys_min,
            vphys_max=vphys_max,
            vphys_unit=vphys_unit,
        )

    return valid, vmin_norm, vmax_norm
#-- END OF SUB-ROUTINE____________________________________________________________#


def _prepare_mask_3d_org(
    mask3d_fits_path,
    *,
    save_to_fits=False,
    save_dir=None,
    prefix=None,
    overwrite=True,
    ref_header=None,
    mef=False
):
    """
    Read a 3D mask (FITS) and produce:
      1) 2D validity map
      2) 2D min velocity (normalized to 0..1 for return)
      3) 2D max velocity (normalized to 0..1 for return)

    : SOFIA-2 mask cube can be used.

    Returns
    -------
    valid2d : uint8 (ny, nx)
    vmin_norm2d : float32 (ny, nx)  in [0,1] or -1 if invalid
    vmax_norm2d : float32 (ny, nx)  in [0,1] or -1 if invalid
    """
    mask3d_fits_path = Path(mask3d_fits_path)
    data, hdr = fits.getdata(mask3d_fits_path, header=True)

    if data.ndim != 3:
        raise ValueError(f"Expected a 3D mask; got ndim={data.ndim} for {mask3d_fits_path}")

    # Data axes: (nz, ny, nx)
    nz, ny, nx = data.shape

    # Spectral WCS
    try:
        crval3 = float(hdr.get("CRVAL3"))
        cdelt3 = float(hdr.get("CDELT3"))
        crpix3 = float(hdr.get("CRPIX3"))
    except (TypeError, ValueError):
        raise ValueError("3D mask header must include CRVAL3, CDELT3, and CRPIX3.")

    # Build spectral axis in physical units (e.g., m/s). FITS indices are 1-based.
    k = np.arange(nz, dtype=np.float64)
    v_axis = crval3 + (k + 1.0 - crpix3) * cdelt3  # (nz,)
    v_unit = hdr.get("CUNIT3", "m/s")

    v_min = np.nanmin(v_axis)
    v_max = np.nanmax(v_axis)
    if not np.isfinite(v_min) or not np.isfinite(v_max) or (v_max == v_min):
        raise ValueError("Invalid spectral axis (check CRVAL3/CDELT3/CRPIX3).")

    # Normalized axis for return arrays
    v_axis_norm = (v_axis - v_min) / (v_max - v_min)  # 0..1

    # Boolean mask
    mask_bool = data > 0

    # Valid pixels
    valid2d = mask_bool.any(axis=0).astype(np.uint8)  # (ny, nx)

    # First/last channel indices with True
    first_idx = np.argmax(mask_bool, axis=0)              # (ny, nx)
    last_idx_rev = np.argmax(mask_bool[::-1, :, :], axis=0)
    last_idx = (nz - 1) - last_idx_rev

    # Normalized vmin/vmax for returns
    vmin_norm2d = v_axis_norm[first_idx].astype(np.float32)
    vmax_norm2d = v_axis_norm[last_idx].astype(np.float32)

    # Physical vmin/vmax for FITS output
    vmin_phys2d = v_axis[first_idx].astype(np.float32)
    vmax_phys2d = v_axis[last_idx].astype(np.float32)

    # Invalidate where no channels are True
    invalid = (valid2d == 0)
    vmin_norm2d[invalid] = -1.0
    vmax_norm2d[invalid] = -1.0
    vmin_phys2d[invalid] = -1.0
    vmax_phys2d[invalid] = -1.0

    # Prepare 2D header if not supplied
    if ref_header is None:
        ref_header = hdr.copy()

    if save_to_fits:
        _save_mask_outputs_to_fits(
            valid2d,
            vmin2d=vmin_phys2d,   # PHYSICAL velocities
            vmax2d=vmax_phys2d,   # PHYSICAL velocities
            save_dir=(save_dir or mask3d_fits_path.parent),
            prefix=(prefix or mask3d_fits_path.stem),
            mef=mef,
            overwrite=overwrite,
            ref_header=ref_header,
            vphys_min=v_min,
            vphys_max=v_max,
            vphys_unit=v_unit,
        )

    return valid2d, vmin_norm2d, vmax_norm2d

#-- END OF SUB-ROUTINE____________________________________________________________#

def _velocity_axis_from_header1(hdr):
    """
    Build physical velocity/frequency axis from FITS header, respecting sign of CDELTn.
    Returns:
        vaxis : 1D np.ndarray (length = N along the velocity-like axis, in header's CUNITn)
        axnum : int (1-based FITS axis number that is spectral)
        vunit : str (unit string from CUNITn, e.g. 'm/s' or 'Hz')
    """
    import re
    spectral_keys = ('VELO', 'VRAD', 'VOPT', 'FREQ', 'LOGLAM', 'WAVE')

    for k in (1, 2, 3, 4, 5):
        ctype = (hdr.get(f'CTYPE{k}', '') or '').upper()
        if any(key in ctype for key in spectral_keys):
            crval = float(hdr.get(f'CRVAL{k}'))
            crpix = float(hdr.get(f'CRPIX{k}'))
            cdelt = float(hdr.get(f'CDELT{k}'))
            naxis = int(hdr.get(f'NAXIS{k}'))
            # FITS is 1-based in header, numpy is 0-based in memory
            pix = np.arange(naxis, dtype=float) + 1.0  # 1..N
            vaxis = crval + (pix - crpix) * cdelt      # sign of cdelt handled here
            vunit = (hdr.get(f'CUNIT{k}', '') or '').strip()
            return vaxis, k, vunit

    # Fallback (legacy): assume axis 3 is spectral
    crval = float(hdr.get('CRVAL3'))
    crpix = float(hdr.get('CRPIX3'))
    cdelt = float(hdr.get('CDELT3'))
    naxis = int(hdr.get('NAXIS3'))
    pix = np.arange(naxis, dtype=float) + 1.0
    vaxis = crval + (pix - crpix) * cdelt
    vunit = (hdr.get('CUNIT3', '') or '').strip()
    return vaxis, 3, vunit

#-- END OF SUB-ROUTINE____________________________________________________________#


def _prepare_mask_3d1(mask3d_fits_path, save_fits=False, out_prefix=None, mef=True):
    """
    Returns
        valid2d      : (ny, nx)  uint8  (0/1)
        vmin_norm2d  : (ny, nx)  float32 in [0,1] or -1 if invalid
        vmax_norm2d  : (ny, nx)  float32 in [0,1] or -1 if invalid
    Saves (if save_fits):
        VALID, VMIN (phys), VMAX (phys) — vmin/vmax are saved in physical units (e.g., m/s)
    """

    data, hdr = fits.getdata(mask3d_fits_path, header=True)
    if data.ndim != 3:
        raise ValueError("3D mask must be a cube (ndim=3).")

    vaxis, axnum, vunit = _velocity_axis_from_header(hdr)  # ← 부호 반영된 물리 속도축
    nz_spec = len(vaxis)

    # 어떤 축이 스펙트럴인지 데이터 모양과 맞춰 추론
    # (hdr의 NAXISk와 data.shape 중 같은 길이를 가진 축을 스펙트럴 축으로 간주)
    # 기본값: 마지막 축이 스펙트럴이라고 가정
    spec_axis = np.argmax([s == nz_spec for s in data.shape])
    # (nz, ny, nx) 형태로 정렬
    cube = np.moveaxis(data, spec_axis, 0).astype(np.float32)
    nz, ny, nx = cube.shape

    valid2d = np.zeros((ny, nx), dtype=np.uint8)
    vmin_phys2d = np.full((ny, nx), -1.0, dtype=np.float32)
    vmax_phys2d = np.full((ny, nx), -1.0, dtype=np.float32)

    # 전체 속도축의 물리적 min/max (정규화에 사용; CDELT 부호 자동 반영)
    v_global_min = float(np.nanmin(vaxis))
    v_global_max = float(np.nanmax(vaxis))
    v_span = v_global_max - v_global_min if v_global_max > v_global_min else np.nan

    for j in range(ny):
        col = cube[:, j, :]  # shape: (nz, nx)
        # >0인 채널 마스크
        m = col > 0
        any_valid = m.any(axis=0)  # (nx,)
        valid2d[j, any_valid] = 1

        # 물리 속도값에서의 최소/최대
        if any_valid.any():
            # 각 픽셀별 유효 채널의 물리 속도 배열
            for i in np.where(any_valid)[0]:
                v_sel = vaxis[m[:, i]]
                if v_sel.size:
                    vmin_phys2d[j, i] = float(np.nanmin(v_sel))
                    vmax_phys2d[j, i] = float(np.nanmax(v_sel))

    # 정규화된 0..1 범위 (필요할 때 내부용으로 사용)
    vmin_norm2d = np.full((ny, nx), -1.0, dtype=np.float32)
    vmax_norm2d = np.full((ny, nx), -1.0, dtype=np.float32)
    if np.isfinite(v_span) and v_span > 0:
        mask_ok = valid2d > 0
        vmin_norm2d[mask_ok] = (vmin_phys2d[mask_ok] - v_global_min) / v_span
        vmax_norm2d[mask_ok] = (vmax_phys2d[mask_ok] - v_global_min) / v_span

    # ── Optional: 저장 (vmin/vmax는 "물리 속도값"으로 저장) ─────────────────
    if save_fits:
        ref_header = hdr.copy()
        # 단일-HDU 여러 파일(mef=False) 또는 MEF 한 파일(mef=True)
        if mef:
            hdul = fits.HDUList([
                fits.PrimaryHDU(header=ref_header),
                fits.ImageHDU(data=valid2d,      name='VALID', header=ref_header),
                fits.ImageHDU(data=vmin_phys2d,  name='VMIN',  header=ref_header),
                fits.ImageHDU(data=vmax_phys2d,  name='VMAX',  header=ref_header),
            ])
            # 유닛 정보 기록
            hdul['VMIN'].header['BUNIT'] = (vunit or '', 'physical velocity unit')
            hdul['VMAX'].header['BUNIT'] = (vunit or '', 'physical velocity unit')
            out_name = (out_prefix or mask3d_fits_path.replace('.fits','')) + '_mask3d_products.fits'
            hdul.writeto(out_name, overwrite=True)
        else:
            base = (out_prefix or mask3d_fits_path.replace('.fits',''))
            fits.writeto(f'{base}_mask3d_valid.fits', valid2d, header=ref_header, overwrite=True)
            h = ref_header.copy(); h['BUNIT'] = (vunit or '', 'physical velocity unit')
            fits.writeto(f'{base}_mask3d_vmin_phys.fits', vmin_phys2d, header=h, overwrite=True)
            fits.writeto(f'{base}_mask3d_vmax_phys.fits', vmax_phys2d, header=h, overwrite=True)

    # 이전 코드와의 호환성: (valid2d, vmin_norm2d, vmax_norm2d) 반환
    return valid2d, vmin_norm2d, vmax_norm2d

















import numpy as np
from pathlib import Path
from astropy.io import fits

# ---------- 1) Build physical velocity axis (sign-safe) ----------
def _velocity_axis_from_header(hdr):
    """
    Create a physical spectral axis from a FITS header, respecting the sign of CDELTn.

    Returns
    -------
    vaxis : np.ndarray
        1D array of physical spectral coordinate (e.g., velocity in m/s, frequency in Hz).
    axnum : int
        1-based FITS axis index that is spectral.
    vunit : str
        Unit string from CUNITn (e.g., 'm/s', 'Hz').
    """
    spectral_keys = ('VELO', 'VRAD', 'VOPT', 'FREQ', 'LOGLAM', 'WAVE')

    for k in (1, 2, 3, 4, 5):
        ctype = (hdr.get(f'CTYPE{k}', '') or '').upper()
        if any(key in ctype for key in spectral_keys):
            crval = float(hdr.get(f'CRVAL{k}'))
            crpix = float(hdr.get(f'CRPIX{k}'))
            cdelt = float(hdr.get(f'CDELT{k}'))
            naxis = int(hdr.get(f'NAXIS{k}'))
            pix = np.arange(naxis, dtype=float) + 1.0  # FITS is 1-based
            vaxis = crval + (pix - crpix) * cdelt      # sign handled naturally
            vunit = (hdr.get(f'CUNIT{k}', '') or '').strip()
            return vaxis, k, vunit

    # Fallback to axis 3 if no spectral CTYPE was found
    crval = float(hdr.get('CRVAL3'))
    crpix = float(hdr.get('CRPIX3'))
    cdelt = float(hdr.get('CDELT3'))
    naxis = int(hdr.get('NAXIS3'))
    pix = np.arange(naxis, dtype=float) + 1.0
    vaxis = crval + (pix - crpix) * cdelt
    vunit = (hdr.get('CUNIT3', '') or '').strip()

    print("----")
    print(vaxis)
    print("----")
    return vaxis, 3, vunit


# ---------- 2) Save helpers ----------
def _save_mask_outputs_to_fits(
    valid2d,
    vmin2d=None,
    vmax2d=None,
    *,
    save_dir,
    prefix,
    mef=True,
    overwrite=True,
    ref_header=None,
    vphys_min=None,
    vphys_max=None,
    vphys_unit=''
):
    """
    Save mask products to FITS.
    - If mef=True: one MEF file with HDUs: PRIMARY, VALID, (optional) VMIN, VMAX
    - If mef=False: write separate files for each product.

    Parameters
    ----------
    valid2d : np.ndarray (ny, nx), uint8 or bool
    vmin2d, vmax2d : np.ndarray (ny, nx) or None
        Physical velocity min/max per pixel (same unit as vphys_unit). If None, the HDU is skipped.
    save_dir : str or Path
    prefix : str
        Output filename prefix (without extension).
    mef : bool
    overwrite : bool
    ref_header : fits.Header or None
        Header to attach to the images (if None, a minimal header is used).
    vphys_min, vphys_max : float or None
        Global min/max of physical spectral axis; stored in headers when provided.
    vphys_unit : str
        Unit string for vmin/vmax arrays (e.g., 'm/s').
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    base = save_dir / prefix

    # Normalize dtypes
    valid_img = valid2d.astype(np.uint8)

    if mef:
        phdr = (ref_header.copy() if ref_header is not None else fits.Header())
        hdul = fits.HDUList([fits.PrimaryHDU(header=phdr)])

        h_valid = (ref_header.copy() if ref_header is not None else fits.Header())
        hdul.append(fits.ImageHDU(data=valid_img, name='VALID', header=h_valid))

        if vmin2d is not None:
            h_vmin = (ref_header.copy() if ref_header is not None else fits.Header())
            if vphys_unit:
                h_vmin['BUNIT'] = (vphys_unit, 'physical velocity unit')
            if vphys_min is not None:
                h_vmin['VGLBMIN'] = (float(vphys_min), 'global min of physical spectral axis')
            if vphys_max is not None:
                h_vmin['VGLBMAX'] = (float(vphys_max), 'global max of physical spectral axis')
            hdul.append(fits.ImageHDU(data=np.asarray(vmin2d, dtype=np.float32), name='VMIN', header=h_vmin))

        if vmax2d is not None:
            h_vmax = (ref_header.copy() if ref_header is not None else fits.Header())
            if vphys_unit:
                h_vmax['BUNIT'] = (vphys_unit, 'physical velocity unit')
            if vphys_min is not None:
                h_vmax['VGLBMIN'] = (float(vphys_min), 'global min of physical spectral axis')
            if vphys_max is not None:
                h_vmax['VGLBMAX'] = (float(vphys_max), 'global max of physical spectral axis')
            hdul.append(fits.ImageHDU(data=np.asarray(vmax2d, dtype=np.float32), name='VMAX', header=h_vmax))

        out_path = base.with_suffix('.fits')
        hdul.writeto(out_path, overwrite=overwrite)

    else:
        # VALID
        h_valid = (ref_header.copy() if ref_header is not None else fits.Header())
        fits.writeto(base.with_name(base.name + '_valid.fits'), valid_img, header=h_valid, overwrite=overwrite)

        # VMIN
        if vmin2d is not None:
            h_vmin = (ref_header.copy() if ref_header is not None else fits.Header())
            if vphys_unit:
                h_vmin['BUNIT'] = (vphys_unit, 'physical velocity unit')
            if vphys_min is not None:
                h_vmin['VGLBMIN'] = (float(vphys_min), 'global min of physical spectral axis')
            if vphys_max is not None:
                h_vmin['VGLBMAX'] = (float(vphys_max), 'global max of physical spectral axis')
            fits.writeto(base.with_name(base.name + '_vmin_phys.fits'),
                         np.asarray(vmin2d, dtype=np.float32), header=h_vmin, overwrite=overwrite)

        # VMAX
        if vmax2d is not None:
            h_vmax = (ref_header.copy() if ref_header is not None else fits.Header())
            if vphys_unit:
                h_vmax['BUNIT'] = (vphys_unit, 'physical velocity unit')
            if vphys_min is not None:
                h_vmax['VGLBMIN'] = (float(vphys_min), 'global min of physical spectral axis')
            if vphys_max is not None:
                h_vmax['VGLBMAX'] = (float(vphys_max), 'global max of physical spectral axis')
            fits.writeto(base.with_name(base.name + '_vmax_phys.fits'),
                         np.asarray(vmax2d, dtype=np.float32), header=h_vmax, overwrite=overwrite)


# ---------- 3) 2D mask ----------
def _prepare_mask_2d(
    mask2d_fits_path,
    *,
    save_to_fits=False,
    save_dir=None,
    prefix=None,
    mef=True,
    overwrite=True,
    ref_header=None,
):
    """
    Prepare 2D mask:
      - valid2d[j, i] = 1 if mask(j,i) > 0 else 0
      - vmin_norm2d, vmax_norm2d are returned as 0 and 1 everywhere

    Returns
    -------
    valid2d : (ny, nx) uint8
    vmin_norm2d : (ny, nx) float32 filled with 0 
    vmax_norm2d : (ny, nx) float32 filled with 1
    """
    mask2d_fits_path = Path(mask2d_fits_path)

    data, hdr = fits.getdata(mask2d_fits_path, header=True)
    if data.ndim != 2:
        raise ValueError("2D mask must be a 2D image.")

    ny, nx = data.shape
    valid2d = (np.asarray(data, dtype=float) > 0).astype(np.uint8)
    vmin_norm2d = np.full((ny, nx), 0, dtype=np.float32)
    vmax_norm2d = np.full((ny, nx), 1.0, dtype=np.float32)

    if save_to_fits:
        _save_mask_outputs_to_fits(
            valid2d,
            vmin2d=None,
            vmax2d=None,
            save_dir=(save_dir or mask2d_fits_path.parent),
            prefix=(prefix or (mask2d_fits_path.stem + '_mask2d')),
            mef=mef,
            overwrite=overwrite,
            ref_header=(ref_header or hdr),
            vphys_min=None,
            vphys_max=None,
            vphys_unit='',
        )

    return valid2d, vmin_norm2d, vmax_norm2d



# ---------- 4) 3D mask (supports 3D and 4D cubes with STOKES axis) ----------
def _prepare_mask_3d(
    mask3d_fits_path,
    *,
    save_to_fits=False,
    save_dir=None,
    prefix=None,
    mef=True,
    overwrite=True,
    ref_header=None,
):
    """
    Prepare a 3D mask from a FITS cube. If the input is 4D and has a STOKES axis,
    use only the first STOKES plane (index 0). If the input is 3D, use it as-is.

    For each spatial pixel (j, i):
      - If all spectral channels at (j, i) are <= 0 => valid2d[j, i] = 0
      - Else valid2d[j, i] = 1 and:
          vmin_phys2d[j, i] = min physical velocity where mask > 0
          vmax_phys2d[j, i] = max physical velocity where mask > 0
      - Also returns normalized [0..1] versions (relative to the global velocity
        axis min/max). Invalid pixels are set to -1.

    Returns
    -------
    valid2d      : (ny, nx) uint8  (0/1)
    vmin_norm2d  : (ny, nx) float32 in [0,1] or -1 if invalid
    vmax_norm2d  : (ny, nx) float32 in [0,1] or -1 if invalid
    """
    mask3d_fits_path = Path(mask3d_fits_path)

    data, hdr = fits.getdata(mask3d_fits_path, header=True)
    if data.ndim not in (3, 4):
        raise ValueError(f"Mask FITS must be 3D or 4D; got ndim={data.ndim}")

    # Build physical velocity axis from FITS header (handles CDELT sign)
    vaxis, _spectral_axnum, v_unit = _velocity_axis_from_header(hdr)
    nz_spec = int(len(vaxis))

    # --- Find which numpy axis is spectral by matching length (fallback: last) ---
    axis_matches = [int(dim) == nz_spec for dim in data.shape]
    if any(axis_matches):
        spec_axis = int(np.argmax(axis_matches))
    else:
        spec_axis = data.ndim - 1  # fallback: last axis

    # --- Move spectral axis to position 0 -> shape becomes (nz, ...) ---
    arr = np.moveaxis(data, spec_axis, 0)

    # --- If 4D, try to locate a STOKES axis among the remaining ones and take plane 0 ---
    if arr.ndim == 4:
        # Try to find STOKES axis using header CTYPEk="STOKES"
        naxis_hdr = int(hdr.get("NAXIS", arr.ndim))
        stokes_len_hdr = None
        stokes_ctypes = []
        for k in range(1, naxis_hdr + 1):
            ctype_k = str(hdr.get(f"CTYPE{k}", "")).upper()
            if "STOKES" in ctype_k:
                stokes_ctypes.append(k)
                try:
                    stokes_len_hdr = int(hdr.get(f"NAXIS{k}", 0) or 0)
                except Exception:
                    stokes_len_hdr = None
                break

        # Among axes 1..3 (since 0 is spectral), choose the one matching header NAXISk if available
        st_ax = None
        if stokes_len_hdr is not None and stokes_len_hdr > 0:
            for ax in range(1, arr.ndim):  # skip spectral axis at 0
                if int(arr.shape[ax]) == stokes_len_hdr:
                    st_ax = ax
                    break

        # If still not found, pick a "small" axis (len <= 4) as likely STOKES axis; else fallback to axis 1
        if st_ax is None:
            small_axes = [ax for ax in range(1, arr.ndim) if int(arr.shape[ax]) <= 4]
            st_ax = small_axes[0] if small_axes else 1

        # Take the first plane along the chosen STOKES axis -> now 3D: (nz, ny, nx)
        arr = np.take(arr, indices=0, axis=st_ax)

    # At this point we expect a 3D array: (nz, ny, nx)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected dimensionality after reduction: ndim={arr.ndim}")

    # Optional sanity: if spectral length disagrees with vaxis length, still proceed
    # (normalization uses vaxis min/max, which is fine)
    nz, ny, nx = arr.shape
    arr = arr.astype(np.float32, copy=False)

    # --- Allocate outputs ---
    valid2d = np.zeros((ny, nx), dtype=np.uint8)
    vmin_phys2d = np.full((ny, nx), 0.0, dtype=np.float32)
    vmax_phys2d = np.full((ny, nx), 1.0, dtype=np.float32)

    # Global min/max of physical velocity axis (for normalization)
    v_min = float(np.nanmin(vaxis))
    v_max = float(np.nanmax(vaxis))
    v_span = float(v_max - v_min)

    # --- Per-pixel scan across spectral channels ---
    # For each spatial pixel, check where mask>0 across channels.
    for j in range(ny):
        col = arr[:, j, :]          # shape: (nz, nx)
        m = col > 0                 # True where mask>0
        any_valid = m.any(axis=0)   # (nx,)
        valid2d[j, any_valid] = 1

        idxs = np.where(any_valid)[0]
        for i in idxs:
            v_sel = vaxis[m[:, i]]
            if v_sel.size > 0:
                vmin_phys2d[j, i] = float(np.nanmin(v_sel))
                vmax_phys2d[j, i] = float(np.nanmax(v_sel))

    # --- Normalized [0..1] maps (invalid -> -1) ---
    vmin_norm2d = np.full((ny, nx), -1.0, dtype=np.float32)
    vmax_norm2d = np.full((ny, nx), -1.0, dtype=np.float32)
    if np.isfinite(v_span) and v_span > 0:
        ok = valid2d > 0
        vmin_norm2d[ok] = (vmin_phys2d[ok] - v_min) / v_span
        vmax_norm2d[ok] = (vmax_phys2d[ok] - v_min) / v_span

    # --- Optional: save outputs (vmin/vmax are PHYSICAL velocities) ---
    if save_to_fits:
        _save_mask_outputs_to_fits(
            valid2d,
            vmin2d=vmin_phys2d,   # physical velocities
            vmax2d=vmax_phys2d,   # physical velocities
            save_dir=(save_dir or mask3d_fits_path.parent),
            prefix=(prefix or (mask3d_fits_path.stem + "_mask3d")),
            mef=mef,
            overwrite=overwrite,
            ref_header=(ref_header or hdr),
            vphys_min=v_min,
            vphys_max=v_max,
            vphys_unit=v_unit,
        )

    return valid2d, vmin_norm2d, vmax_norm2d




# ---------- 4) 3D mask ----------
def _prepare_mask_3d_nostokes(
    mask3d_fits_path,
    *,
    save_to_fits=False,
    save_dir=None,
    prefix=None,
    mef=True,
    overwrite=True,
    ref_header=None,
):
    """
    Prepare 3D mask cube:
      - If all spectral channels at (j,i) are <= 0 => valid2d[j,i] = 0
      - Else valid2d[j,i] = 1 and:
          vmin_phys2d[j,i] = min physical velocity where mask > 0
          vmax_phys2d[j,i] = max physical velocity where mask > 0
      - Also returns normalized [0..1] versions (relative to the global v-axis min/max).
        Invalid pixels are set to -1.

    Parameters
    ----------
    mask3d_fits_path : str or Path
    save_to_fits, save_dir, prefix, mef, overwrite, ref_header : see _save_mask_outputs_to_fits

    Returns
    -------
    valid2d : (ny, nx) uint8 (0/1)
    vmin_norm2d : (ny, nx) float32 in [0,1] (or -1 if invalid)
    vmax_norm2d : (ny, nx) float32 in [0,1] (or -1 if invalid)
    """
    mask3d_fits_path = Path(mask3d_fits_path)

    data, hdr = fits.getdata(mask3d_fits_path, header=True)
    if data.ndim != 3:
        raise ValueError("3D mask must be a cube (ndim=3).")

    # Build physical spectral axis (sign of CDELT handled)
    vaxis, spectral_axnum, v_unit = _velocity_axis_from_header(hdr)
    nz_spec = len(vaxis)

    # Find which array axis is spectral by matching length (fallback: last axis)
    axis_matches = [dim == nz_spec for dim in data.shape]
    if any(axis_matches):
        spec_axis = int(np.argmax(axis_matches))
    else:
        spec_axis = data.ndim - 1  # fallback

    # Reorder to (nz, ny, nx)
    cube = np.moveaxis(data, spec_axis, 0).astype(np.float32)
    nz, ny, nx = cube.shape

    # Initialize outputs
    valid2d = np.zeros((ny, nx), dtype=np.uint8)
    vmin_phys2d = np.full((ny, nx), 0, dtype=np.float32)
    vmax_phys2d = np.full((ny, nx), 1.0, dtype=np.float32)

    # Global min/max of physical spectral axis (for normalization)
    v_min = float(np.nanmin(vaxis))
    v_max = float(np.nanmax(vaxis))
    v_span = v_max - v_min

    # Per-pixel scan
    # m[:, i] is True for channels > 0 at pixel (j, i)
    for j in range(ny):
        col = cube[:, j, :]        # shape: (nz, nx)
        m = col > 0
        any_valid = m.any(axis=0)  # (nx,)
        valid2d[j, any_valid] = 1

        idxs = np.where(any_valid)[0]
        for i in idxs:
            v_sel = vaxis[m[:, i]]
            if v_sel.size > 0:
                vmin_phys2d[j, i] = float(np.nanmin(v_sel))
                vmax_phys2d[j, i] = float(np.nanmax(v_sel))

    # Normalized 0..1 (invalid -> -1)
    vmin_norm2d = np.full((ny, nx), -1, dtype=np.float32)
    vmax_norm2d = np.full((ny, nx), -1, dtype=np.float32)
    if np.isfinite(v_span) and v_span > 0:
        ok = valid2d > 0
        vmin_norm2d[ok] = (vmin_phys2d[ok] - v_min) / v_span
        vmax_norm2d[ok] = (vmax_phys2d[ok] - v_min) / v_span

    # Optional: save products (vmin/vmax saved in PHYSICAL units)
    if save_to_fits:
        _save_mask_outputs_to_fits(
            valid2d,
            vmin2d=vmin_phys2d,   # PHYSICAL velocities
            vmax2d=vmax_phys2d,   # PHYSICAL velocities
            save_dir=(save_dir or mask3d_fits_path.parent),
            prefix=(prefix or (mask3d_fits_path.stem + '_mask3d')),
            mef=mef,
            overwrite=overwrite,
            ref_header=(ref_header or hdr),
            vphys_min=v_min,
            vphys_max=v_max,
            vphys_unit=v_unit,
        )

    return valid2d, vmin_norm2d, vmax_norm2d
