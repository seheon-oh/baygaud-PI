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
# Python 3 compatability
from __future__ import division, print_function

#|-----------------------------------------|
# system functions
import time, sys, os
from datetime import datetime

#|-----------------------------------------|
# python packages
import numpy as np
from numpy import array
import psutil
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
from _baygaud_params import read_configfile
global _x

#|-----------------------------------------|
# _dynesty_sampler.py
from _dynesty_sampler import dynamic_baygaud_nested_sampling
from _dynesty_sampler import derive_rms_npoints

#|-----------------------------------------|
# _fits_io.py
from _fits_io import read_datacube, moment_analysis

#|-----------------------------------------|
# import make_dirs
from _dirs_files import make_dirs

#|-----------------------------------------|
#|-----------------------------------------|
# _combine_segs.py
#import _combine_segs 


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
            print(" :: WARNING: No ' %s ' exist.." % sys.argv[1])
            print("")
            print("")
            sys.exit()

        configfile = sys.argv[1]
        _params=read_configfile(configfile)
        print("")
        print(" ____________________________________________")
        print("[____________________________________________]")
        print("")
        print(" :: Running baygaud.py with %s ::" % configfile)
        print("")

    else:
        print("")
        print(" ____________________________________________")
        print("[____________________________________________]")
        print("")
        print(" :: Usage: running baygaud.py with baygaud_params file")
        print(" :: > python3 baygaud.py [ARG1: _baygaud_params.yaml]")
        print(" :: e.g.,")
        print(" :: > python3 baygaud.py _baygaud_params.ngc2403.yaml")
        print("")
        print("")
        sys.exit()

    _is = int(_params['naxis1_s0'])
    _ie = int(_params['naxis1_e0'])
    _js = int(_params['naxis2_s0'])
    _je = int(_params['naxis2_e0'])

    max_ngauss = _params['max_ngauss']
    gfit_results = np.zeros(((_je-_js), max_ngauss, 2*(2+3*max_ngauss)+7), dtype=np.float32)


    required_num_cpus = _ie - _is
    #ray.init(num_cpus = _params['num_cpus'], dashboard_port=8265, logging_level='DEBUG')
    #num_cpus = psutil.cpu_count(logical=False)

    num_cpus_ray = int(_params['num_cpus_ray'])
    num_cpus_nested_sampling = int(_params['num_cpus_nested_sampling'])
    num_cpus_total = num_cpus_ray * num_cpus_nested_sampling


    # Numba/BLAS 스레딩 고정 (워커 상속용)
    os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    ray.init(
        num_cpus=num_cpus_total,
        runtime_env={"env_vars": {
            "NUMBA_THREADING_LAYER": "workqueue",
            "NUMBA_NUM_THREADS": "2",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }}
    )

    print(" ____________________________________________")
    print("[____________________________________________]")
    print("")
    print(" :: Running baygaud.py with %4d cores in total   ::" % num_cpus_total)
    print(" ::                       : %4d cores (rays)     ::" % num_cpus_ray)
    print(" ::                       : %4d cores (sampling) ::" % num_cpus_nested_sampling)
    print("")


    #------------------------------
    # load the input datacube
    _inputDataCube, _x = read_datacube(_params) # --> _inputDataCub=


    # load cube_mask if provided
    if _params['_cube_mask'] == 'Y':
        _cube_mask_2d = fits.getdata(_params['wdir'] + '/' + _params['_cube_mask_2d'])
    elif _params['_cube_mask'] == 'N':
        # put a dummy array filled with 1
        _cube_mask_2d = np.full((_params['naxis2'], _params['naxis1']), fill_value=1, dtype=np.float32)

    #------------------------------
    # make the segs output directory
    make_dirs("%s/%s" % (_params['wdir'], _params['_segdir']))

    #------------------------------
    # derive a rms_med using npoints lines via sgfit --> _params['_rms_med']
    derive_rms_npoints(_inputDataCube, _cube_mask_2d, _x, _params, 1) 

    #------------------------------
    # derive peak s/n map + integrated s/n map
    _peak_sn_map, _sn_int_map = moment_analysis(_params) #
    #print(_peak_sn_map)


    max_ngauss = _params['max_ngauss']
    # nparams: 3*ngauss(x, std, p) + bg + sigma
    _nparams   = 3*max_ngauss + 2
    _base      = 2*_nparams  # 파라미터 벡터 공통 오프셋 (서브루틴과 동일 계산식)
    _meta_idx  = {
        "logz": _base + max_ngauss,
        "xs":   _base + max_ngauss + 1,
        "xe":   _base + max_ngauss + 2,
        "ys":   _base + max_ngauss + 3,
        "ye":   _base + max_ngauss + 4,
        "curi": _base + max_ngauss + 5,
        "curj": _base + max_ngauss + 6,
    }


# ray.put : put object store --> speed up 
    _inputDataCube_id = ray.put(_inputDataCube)
    _peak_sn_map_id   = ray.put(_peak_sn_map)
    _sn_int_map_id    = ray.put(_sn_int_map)
    _cube_mask_2d_id  = ray.put(_cube_mask_2d)
    _x_id             = ray.put(_x)
    _params_id        = ray.put(_params)     # 추가: params도 반드시 put

    baygaud_nested_sampling = dynamic_baygaud_nested_sampling(num_cpus_nested_sampling)

    results_ids = [
        baygaud_nested_sampling.remote(
            _inputDataCube_id, _x_id,
            _peak_sn_map_id, _sn_int_map_id,
            _params_id,                  # ← ObjectRef 전달
            _is, _ie, i, _js, _je,       # ← 스칼라는 값으로 전달
            _cube_mask_2d_id
        )
        for i in range(_is, _ie)
    ]

    # 파일 저장 경로 포맷도 미리 준비
    save_fmt = f"{_params['wdir']}/{_params['_segdir']}/G{max_ngauss:02d}.x{{curi}}.ys{{ys}}ye{{ye}}"

    while results_ids:
        done_ids, results_ids = ray.wait(results_ids, num_returns=1)
        if done_ids:
            arr = np.asarray(ray.get(done_ids))   # ← get은 1회만
            # 기대 shape: (je-js, max_ngauss, _base+7)

            # i=given, j=0, k=max_ngauss-1 파라미터 벡터
            block = arr[0, 0, max_ngauss-1]

            # ------------------------------------------
            # block description
            #_xs = int(ray.get(done_ids)[0][0][max_ngauss-1][2*_nparams+max_ngauss+1])
            # block = ray.get(done_ids)[0][0][max_ngauss-1]
            # ------------------------------------------

            _xs   = int(block[_meta_idx["xs"]])
            _xe   = int(block[_meta_idx["xe"]])
            _ys   = int(block[_meta_idx["ys"]])
            _ye   = int(block[_meta_idx["ye"]])
            _curi = int(block[_meta_idx["curi"]])
            _curj = int(block[_meta_idx["curj"]])  # in case

            np.save(save_fmt.format(curi=_curi, ys=_ys, ye=_ye), arr)

    #results_compile = ray.get(results_ids)
    #print(results_compile)
    ray.shutdown()

    print("duration =", datetime.now() - start)
#-- END OF SUB-ROUTINE____________________________________________________________#

if __name__ == '__main__':
    main()

