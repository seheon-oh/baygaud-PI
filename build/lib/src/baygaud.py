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

import fitsio

#|-----------------------------------------|
# import ray
import ray

#|-----------------------------------------|
# load baygaudpy modules
#|-----------------------------------------|
# _params.py
from _baygaud_params import default_params, read_configfile
global _x

#|-----------------------------------------|
# _dynesty_sampler.py
from _dynesty_sampler import run_dynesty_sampler_uniform_priors
from _dynesty_sampler import run_dynesty_sampler_optimal_priors
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
    # read the input datacube
    start = datetime.now()


    if len(sys.argv) == 1:
        ("WARNING: No configfile supplied, trying default values")
        print(91*"-")
        print(" usage-2: running baygaud.py with the DEFAULT baygaud_params file")
        print("        : the DEFAULT baygaud_params file: _baygaud_params.py")
        print(" e.g.,")
        print(" > python3 baygaud.py")
        print(91*"_")
        print("")
        print("")
        _params=default_params()

    elif len(sys.argv) == 2:
        print("")
        print(91*"_")
        print(91*"")
        print(" :: baygaud.py usage ::")
        print(91*"")
        print(" usage-1: running baygaud.py with baygaud_params file")
        print(" > python3 baygaud.py [ARG1: _baygaud_params.txt]")
        print(" e.g.,")
        print(" > python3 baygaud.py _baygaud_params.ngc2403.txt")
        print("")
        print("")

        configfile = sys.argv[1]
        _params=read_configfile(configfile)


    _is = int(_params['naxis1_s0'])
    _ie = int(_params['naxis1_e0'])
    _js = int(_params['naxis2_s0'])
    _je = int(_params['naxis2_e0'])

    max_ngauss = _params['max_ngauss']
    gfit_results = np.zeros(((_je-_js), max_ngauss, 2*(2+3*max_ngauss)+7), dtype=np.float32)

    #ray.init(num_cpus=3, num_gpus=3)
    #ray.init(num_cpus=1, ignore_reinit_error=True, object_store_memory=2*10**9)
    required_num_cpus = _ie - _is
    num_cpus = psutil.cpu_count(logical=False)
    #ray.init(num_cpus=num_cpus)
    ray.init(num_cpus = _params['num_cpus'])

    #------------------------------
    # load the input datacube
    _inputDataCube, _x = read_datacube(_params) # --> _inputDataCub=

    # load cube_mask if provided
    if _params['_cube_mask'] == 'Y':
        _cube_mask_2d = fitsio.read(_params['wdir'] + '/' + _params['_cube_mask_2d'])
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


    # ray.put : speed up
    _inputDataCube_id = ray.put(_inputDataCube)
    _peak_sn_map_id = ray.put(_peak_sn_map)
    _sn_int_map_id = ray.put(_sn_int_map)
    _cube_mask_2d_id = ray.put(_cube_mask_2d)

    _x_id = ray.put(_x)
    _is_id = ray.put(_is)
    _ie_id = ray.put(_ie)
    _js_id = ray.put(_js)
    _je_id = ray.put(_je)

    # nparams: 3*ngauss(x, std, p) + bg + sigma
    _nparams = 3*max_ngauss + 2


    #results_ids = [run_dynesty_sampler_uniform_priors.remote(_x_id, _inputDataCube_id, _is_id, _ie_id, i, _js, _je, _max_ngauss_id, _vel_min_id, _vel_max_id) for i in range(_is, _ie)]

    results_ids = [run_dynesty_sampler_optimal_priors.remote(_inputDataCube_id, _x_id, \
                                                            _peak_sn_map_id, _sn_int_map_id, \
                                                            _params, \
                                                            _is_id, _ie_id, i, _js_id, _je_id, _cube_mask_2d_id) for i in range(_is, _ie)]

    while len(results_ids):
        time.sleep(0.1)
        done_ids, results_ids = ray.wait(results_ids)
        if done_ids:
            # _xs, _xe, _ys, _ye : variables inside the loop
            _xs = int(ray.get(done_ids)[0][0][max_ngauss-1][2*_nparams+max_ngauss+1])
            _xe = int(ray.get(done_ids)[0][0][max_ngauss-1][2*_nparams+max_ngauss+2])
            _ys = int(ray.get(done_ids)[0][0][max_ngauss-1][2*_nparams+max_ngauss+3])
            _ye = int(ray.get(done_ids)[0][0][max_ngauss-1][2*_nparams+max_ngauss+4])
            _curi = int(ray.get(done_ids)[0][0][max_ngauss-1][2*_nparams+max_ngauss+5])
            _curj = int(ray.get(done_ids)[0][0][max_ngauss-1][2*_nparams+max_ngauss+6])

            #print(_xs, _curi, _ys, _curj)
            _segid = _curi-_is+1 # current_i - i_start
            #makedir_for_curprocess('%s/_seg%d/output_xs%dxe%dys%dye%di%d'
            #    % (_segdir_, _segid, xs, _xe, _ys, _ye, _segid))
            #makedir_for_curprocess('%s/_seg%d' % (_segdir_, _segid))

            #print(ray.get(done_ids))
            #print(array(ray.get(done_ids)).shape)

            # save the fits reults to a binary file
            np.save('%s/%s/G%02d.x%d.ys%dye%d' % (_params['wdir'], _params['_segdir'], max_ngauss, _curi, _ys, _ye), array(ray.get(done_ids)))

    #results_compile = ray.get(results_ids)
    #print(results_compile)
    ray.shutdown()
    print("duration =", datetime.now() - start)
#-- END OF SUB-ROUTINE____________________________________________________________#

if __name__ == '__main__':
    main()

