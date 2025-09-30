#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#|-----------------------------------------|
#| baygaud.py
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|


#|-----------------------------------------|
# Python 3 compatibility
from __future__ import division, print_function


# ── filter noisy warnings from dependencies (narrow-scoped) ─────────────
import warnings, re
# ── silence specific UserWarning from spectral_cube about pkg_resources deprecation
# spectral_cube: pkg_resources deprecation
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
    category=UserWarning,
    module=r"^spectral_cube(\.|$)"
)
# astropy.units: divide-by-zero during quantity ops (e.g., bad RESTFREQ)
warnings.filterwarnings(
    "ignore",
    message=r"divide by zero encountered in divide",
    category=RuntimeWarning,
    module=r"^astropy\.units\.quantity(\.|$)"
)


#|-----------------------------------------|
# system functions
import time, sys, os
from datetime import datetime

#|-----------------------------------------|
# python packages
import numpy as np
from numpy import array
from multiprocessing import cpu_count

from astropy.io import fits

#|-----------------------------------------|
# import ray
os.environ["RAY_DEDUP_LOGS"] = "0"
import ray

#|-----------------------------------------|
# load baygaudpy modules
#|-----------------------------------------|
# _params.py
from _handle_yaml import read_configfile, _get_threading_env_from_params, update_yaml_param, recommend_tiling_from_yaml
global _x

#|-----------------------------------------|
# _dynesty_sampler.py
from _dynesty_sampler import dynamic_baygaud_nested_sampling
# from _dynesty_sampler import derive_rms_npoints

#|-----------------------------------------|
# _fits_io.py
from _fits_io import read_datacube, moment_analysis, min_sigma_from_cdelt3, _prepare_mask_2d, _prepare_mask_3d

#|-----------------------------------------|
# import make_dirs
from _dirs_files import make_dirs


#|-----------------------------------------|
# import _plot.py
from _progress_bar import _fmt_ddhhmmss, _print_progress_classic

#|-----------------------------------------|
from collections import deque

# _combine_segs.py
# import _combine_segs

from _banner import print_banner, LEFT_MARGIN

from _info_summary import print_cube_summary_from_info, get_ray_info, get_runtime_resource_info


import matplotlib.pyplot as plt

from pathlib import Path


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def main():
    start = datetime.now()

    if len(sys.argv) == 2:
        if not os.path.exists(sys.argv[1]):
            print("")
            print(" ____________________________________________")
            print("[____________________________________________]")
            print("")
            print(" :: WARNING: File '%s' does not exist." % sys.argv[1])
            print("")
            print("")
            sys.exit()

        configfile = sys.argv[1]
        _params = read_configfile(configfile)
        print("")
        print(" ____________________________________________")
        print("[____________________________________________]")
        print("")
        print(" :: Running baygaud.py with config: %s" % configfile)
        print("")
    else:
        print("")
        print(" ____________________________________________")
        print("[____________________________________________]")
        print("")
        print(" :: USAGE: python3 baygaud.py <config.yaml>")
        print(" :: Example: python3 baygaud.py _baygaud_params.ngc2403.yaml")
        print("")
        print("")
        sys.exit()


    _is = int(_params['naxis1_s0'])
    _ie = int(_params['naxis1_e0'])
    _js = int(_params['naxis2_s0'])
    _je = int(_params['naxis2_e0'])

    max_ngauss = int(_params['max_ngauss'])
    _nparams   = 3*max_ngauss + 2
    _base      = 2*_nparams  # same parameter vector offset as the remote side

    # ─────────────────────────────────────────────────────────
    # Parallel setup (Ray + worker thread pinning)
    # ─────────────────────────────────────────────────────────
    num_cpus_ray = int(_params['num_cpus_ray'])
    num_cpus_nested_sampling = int(_params['num_cpus_nested_sampling'])  # recommended: 1
    num_cpus_total = num_cpus_ray * num_cpus_nested_sampling

    # 1) Load env dict from YAML
    env_vars = _get_threading_env_from_params(_params)

    # 2) Apply env vars to the driver process (use setdefault to preserve pre-set values if desired)
    for k, v in env_vars.items():
        os.environ[k] = v  # overwrite
        # os.environ.setdefault(k, v)  # use this line instead if you want to preserve existing values

    # 3) Pass the same env to Ray workers at init
    ray.init(
        num_cpus=num_cpus_total,
        runtime_env={"env_vars": env_vars}
    )

    #------------------------------
    # load the input datacube
    _inputDataCube, _x, cube_info = read_datacube(_params)

    #------------------------------
    # compute the minimum Gaussian sigma:
    # cdelt3 (boxcar) + hanning info + x1.3 factor for a conservative lower limit
    _g_sigma_lower = min_sigma_from_cdelt3(cube_info['cdelt3_ms'], unit="m/s", hanning_passes=_params['num_hanning_passes'])
    _params['g_sigma_lower'] = _g_sigma_lower
    update_yaml_param(configfile, "g_sigma_lower", _g_sigma_lower)

    _tile_opt = recommend_tiling_from_yaml(configfile, write_back=False)
    update_yaml_param(configfile, "y_chunk_size", _tile_opt['y_chunk_size'])
    update_yaml_param(configfile, "gather_batch", _tile_opt['gather_batch'])

    # load 2D mask if provided
    if _params['_cube_mask_2d'] == 'Y':
        #_cube_mask_2d = fits.getdata(_params['wdir'] + '/' + _params['_cube_mask_2d_fits'])

        mask_valid, mask_vmin_norm, mask_vmax_norm = _prepare_mask_2d(
            Path(_params['wdir']) / _params['_cube_mask_2d_fits'],
            save_to_fits=True,
            save_dir=Path(_params['wdir']) / "mask",
            prefix="mask2d",
            mef=False,
            overwrite=True
        )
    else:
        mask_valid = np.full((_params['naxis2'], _params['naxis1']), fill_value=1, dtype=np.float32)
        mask_vmin_norm = np.full((_params['naxis2'], _params['naxis1']), 0.0, dtype=np.float32)
        mask_vmax_norm = np.full((_params['naxis2'], _params['naxis1']), 1.0, dtype=np.float32)


    # load sofia-2 3D mask if provided
    if _params['_cube_mask_3d'] == 'Y':
        #_cube_mask_3d = fits.getdata(_params['wdir'] + '/' + _params['_cube_mask_3d_fits'])

        mask_valid, mask_vmin_norm, mask_vmax_norm = _prepare_mask_3d(
            Path(_params['wdir']) / _params['_cube_mask_3d_fits'],
            save_to_fits=True,
            save_dir=Path(_params['wdir']) / "mask",
            prefix="mask3d",
            mef=False,
            overwrite=True
        )

    else:
        mask_valid = np.full((_params['naxis2'], _params['naxis1']), fill_value=1, dtype=np.float32)
        mask_vmin_norm = np.full((_params['naxis2'], _params['naxis1']), 0.0, dtype=np.float32)
        mask_vmax_norm = np.full((_params['naxis2'], _params['naxis1']), 1.0, dtype=np.float32)


    #------------------------------
    # make the segs output directory
    make_dirs(f"{_params['wdir']}/{_params['_segdir']}")

    #------------------------------
    # derive a rms_med using npoints lines via sgfit --> _params['_rms_med']
    # derive_rms_npoints(_inputDataCube, _cube_mask_2d, _x, _params, 1)

    #------------------------------
    # derive peak S/N map + integrated S/N map
    _peak_sn_map, _sn_int_map = moment_analysis(_params)

    #------------------------------
    # Put large objects into the Object Store and pass by reference
    _inputDataCube_id = ray.put(_inputDataCube)
    _peak_sn_map_id   = ray.put(_peak_sn_map)
    _sn_int_map_id    = ray.put(_sn_int_map)
    _mask_valid_id  = ray.put(mask_valid)
    _mask_vmin_norm_id  = ray.put(mask_vmin_norm)
    _mask_vmax_norm_id  = ray.put(mask_vmax_norm)
    _x_id             = ray.put(_x)
    _params_id        = ray.put(_params)

    #------------------------------
    # Tiling / batch gather settings
    tile    = int(_params['y_chunk_size'])  # recommended starting point: 20–50
    # YAML: gather_batch
    batch_k = int(_params['gather_batch'])
    # Prevent overload: do not exceed CPU budget or set too large
    batch_k = max(1, min(batch_k, num_cpus_total))
    #------------------------------

    ray_info = get_ray_info()  # may be None if Ray is not used
    runtime_info = get_runtime_resource_info(_params)

    print_banner()

    print_cube_summary_from_info(
        _params,
        cube_info=cube_info,
        yaml_path=configfile,
        left_margin=LEFT_MARGIN,
        title="Data cube / key params",
        ray_info=runtime_info
    )
    print()

    save_fmt = f"{_params['wdir']}/{_params['_segdir']}/G{max_ngauss:02d}.x{{curi}}.ys{_js}ye{_je}"

    # remote function handle
    baygaud_nested_sampling_remote = dynamic_baygaud_nested_sampling(num_cpus_nested_sampling, _params)

    # Accumulators for delayed writes / remaining tile counters
    acc = {}        # i -> ndarray((je-js), max_ngauss, _base+7)
    remaining = {}  # i -> remaining tile count
    pending = {}    # ObjectRef -> (i, j0, j1)

    # ── Build tile queue ──
    tiles = deque()
    for i in range(_is, _ie):
        j0 = _js
        while j0 < _je:
            j1 = min(j0 + tile, _je)
            tiles.append((i, j0, j1))
            j0 = j1

    # ── Pre-allocation (no delayed allocation) ──
    P = 2*(2 + 3*max_ngauss) + 7 + max_ngauss
    acc       = {i: np.empty(((_je - _js), max_ngauss, P), dtype=np.float32) for i in range(_is, _ie)}
    remaining = {i: ((_je - _js + tile - 1)//tile) for i in range(_is, _ie)}

    total_pixels     = (_ie - _is) * (_je - _js)
    completed_pixels = 0
    t0 = time.perf_counter()

    t0 = time.perf_counter()
    # Print progress bar at 0%
    _print_progress_classic(0, total_pixels, t0, width=85, divisions=20, min_interval=0.2)

    # ── In-flight window settings ──
    # After building the tile queue:
    inflight_factor = int(_params.get("inflight_factor", 1))  # start at 1
    MAX_INFLIGHT    = max(1, min(len(tiles), num_cpus_total * inflight_factor))
    wait_timeout_s  = float(_params.get("wait_timeout_s", 1.0))   # lengthen to 1.0
    print_min_int   = float(_params.get("print_min_interval_s", 1.0))  # print less frequently

    # ── Submission/collection state ──
    inflight_refs = []
    meta = {}  # ref -> (i, j0, j1)

    def _submit_one():
        if not tiles:
            return
        i, j0, j1 = tiles.popleft()
        ref = baygaud_nested_sampling_remote.remote(
            _inputDataCube_id, _x_id,
            _peak_sn_map_id, _sn_int_map_id,
            _params_id,
            _is, _ie, int(i), int(j0), int(j1),
            _mask_valid_id,
            _mask_vmin_norm_id,
            _mask_vmax_norm_id
        )
        inflight_refs.append(ref)
        meta[ref] = (i, j0, j1)

    for _ in range(MAX_INFLIGHT):
        _submit_one()

    # ── Collection loop: single wait + timeout + lightweight heartbeat ──
    while inflight_refs:
        done_ids, _ = ray.wait(inflight_refs, num_returns=1, timeout=wait_timeout_s)
        if not done_ids:
            time.sleep(1)  # let the driver breathe (avoid freezing)
            _print_progress_classic(completed_pixels, total_pixels, t0, width=85, divisions=20, min_interval=1.0)
            continue

        ref = done_ids[0]
        inflight_refs.remove(ref)
        i, j0, j1 = meta.pop(ref)

        arr = np.asarray(ray.get(ref), dtype=np.float32)  # shape: (j1-j0, max_ngauss, P)
        acc[i][(j0 - _js):(j1 - _js), :, :] = arr

        remaining[i] -= 1
        if remaining[i] == 0:
            out = acc[i][None, ...]
            np.save(save_fmt.format(curi=i), out)

        completed_pixels += (j1 - j0)
        _print_progress_classic(
            completed_pixels, total_pixels, t0,
            cur_i=i, cur_j0=j0, cur_j1=j1,
            width=50, divisions=20, min_interval=print_min_int
        )

        # Keep the window full: submit as many new tiles as have finished
        _submit_one()

    sys.stdout.write("\n"); sys.stdout.flush()

    ray.shutdown()
    dur = (datetime.now() - start).total_seconds()
    print(f"total processing time = {_fmt_ddhhmmss(dur)}\n")
    print("")
#-- END OF SUB-ROUTINE____________________________________________________________#


if __name__ == '__main__':
    main()


#-- END OF SUB-ROUTINE____________________________________________________________#

