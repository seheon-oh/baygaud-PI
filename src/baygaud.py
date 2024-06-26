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

import numpy as np
from numpy import array
import psutil
from multiprocessing import cpu_count

from astropy.io import fits

os.environ["RAY_DEDUP_LOGS"] = "0"
import ray

from _baygaud_params import read_configfile
global _x

from _dynesty_sampler import dynamic_baygaud_nested_sampling
from _dynesty_sampler import derive_rms_npoints

from _fits_io import read_datacube, moment_analysis

from _dirs_files import make_dirs



#|-----------------------------------------|
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

    num_cpus_ray = int(_params['num_cpus_ray'])
    num_cpus_nested_sampling = int(_params['num_cpus_nested_sampling'])
    num_cpus_total = num_cpus_ray * num_cpus_nested_sampling


    print(" ____________________________________________")
    print("[____________________________________________]")
    print("")
    print(" :: Running baygaud.py with %4d cores in total   ::" % num_cpus_total)
    print(" ::                       : %4d cores (rays)     ::" % num_cpus_ray)
    print(" ::                       : %4d cores (sampling) ::" % num_cpus_nested_sampling)
    print("")

    ray.init(num_cpus = num_cpus_total)

    _inputDataCube, _x = read_datacube(_params) # --> _inputDataCub=


    if _params['_cube_mask'] == 'Y':
        _cube_mask_2d = fits.getdata(_params['wdir'] + '/' + _params['_cube_mask_2d'])
    elif _params['_cube_mask'] == 'N':
        _cube_mask_2d = np.full((_params['naxis2'], _params['naxis1']), fill_value=1, dtype=np.float32)

    make_dirs("%s/%s" % (_params['wdir'], _params['_segdir']))

    derive_rms_npoints(_inputDataCube, _cube_mask_2d, _x, _params, 1) 

    _peak_sn_map, _sn_int_map = moment_analysis(_params) #


    _inputDataCube_id = ray.put(_inputDataCube)
    _peak_sn_map_id = ray.put(_peak_sn_map)
    _sn_int_map_id = ray.put(_sn_int_map)
    _cube_mask_2d_id = ray.put(_cube_mask_2d)

    _x_id = ray.put(_x)
    _is_id = ray.put(_is)
    _ie_id = ray.put(_ie)
    _js_id = ray.put(_js)
    _je_id = ray.put(_je)

    _nparams = 3*max_ngauss + 2

    baygaud_nested_sampling = dynamic_baygaud_nested_sampling(num_cpus_nested_sampling)
    results_ids = [baygaud_nested_sampling.remote(_inputDataCube_id, _x_id, \
                                                            _peak_sn_map_id, _sn_int_map_id, \
                                                            _params, \
                                                            _is_id, _ie_id, i, _js_id, _je_id, _cube_mask_2d_id) for i in range(_is, _ie)]

    while len(results_ids):
        done_ids, results_ids = ray.wait(results_ids)
        if done_ids:
            _xs = int(ray.get(done_ids)[0][0][max_ngauss-1][2*_nparams+max_ngauss+1])
            _xe = int(ray.get(done_ids)[0][0][max_ngauss-1][2*_nparams+max_ngauss+2])
            _ys = int(ray.get(done_ids)[0][0][max_ngauss-1][2*_nparams+max_ngauss+3])
            _ye = int(ray.get(done_ids)[0][0][max_ngauss-1][2*_nparams+max_ngauss+4])
            _curi = int(ray.get(done_ids)[0][0][max_ngauss-1][2*_nparams+max_ngauss+5])
            _curj = int(ray.get(done_ids)[0][0][max_ngauss-1][2*_nparams+max_ngauss+6])

            _segid = _curi-_is+1 # current_i - i_start


            np.save('%s/%s/G%02d.x%d.ys%dye%d' % (_params['wdir'], _params['_segdir'], max_ngauss, _curi, _ys, _ye), array(ray.get(done_ids)))

    ray.shutdown()
    print("duration =", datetime.now() - start)

if __name__ == '__main__':
    main()

