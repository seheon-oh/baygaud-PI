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
# import _plot.py
from _plot import _fmt_ddhhmmss, _print_progress_classic

#|-----------------------------------------|
#|-----------------------------------------|
# _combine_segs.py
#import _combine_segs 


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
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
        _params = read_configfile(configfile)
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

    max_ngauss = int(_params['max_ngauss'])
    _nparams   = 3*max_ngauss + 2
    _base      = 2*_nparams  # remote와 동일한 파라미터 벡터 오프셋

    # 병렬 설정
    num_cpus_ray = int(_params['num_cpus_ray'])
    num_cpus_nested_sampling = int(_params['num_cpus_nested_sampling'])  # 권장: 1
    num_cpus_total = num_cpus_ray * num_cpus_nested_sampling

    # Numba/BLAS 스레딩 고정 (워커 상속용)
    os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
    os.environ.setdefault("NUMBA_NUM_THREADS",     str(_params['numba_num_threads']))
    os.environ.setdefault("OMP_NUM_THREADS",       "1")
    os.environ.setdefault("MKL_NUM_THREADS",       "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS",  "1")

    ray.init(
        num_cpus=num_cpus_total,
        runtime_env={"env_vars": {
            "NUMBA_THREADING_LAYER": os.environ["NUMBA_THREADING_LAYER"],
            "NUMBA_NUM_THREADS":     os.environ["NUMBA_NUM_THREADS"],
            "OMP_NUM_THREADS":       "1",
            "MKL_NUM_THREADS":       "1",
            "OPENBLAS_NUM_THREADS":  "1",
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
    _inputDataCube, _x = read_datacube(_params)

    # load cube_mask if provided
    if _params['_cube_mask'] == 'Y':
        _cube_mask_2d = fits.getdata(_params['wdir'] + '/' + _params['_cube_mask_2d'])
    else:
        _cube_mask_2d = np.full((_params['naxis2'], _params['naxis1']), fill_value=1, dtype=np.float32)

    #------------------------------
    # make the segs output directory
    make_dirs(f"{_params['wdir']}/{_params['_segdir']}")

    #------------------------------
    # derive a rms_med using npoints lines via sgfit --> _params['_rms_med']
    derive_rms_npoints(_inputDataCube, _cube_mask_2d, _x, _params, 1)

    #------------------------------
    # derive peak s/n map + integrated s/n map
    _peak_sn_map, _sn_int_map = moment_analysis(_params)

    #------------------------------
    # 대형 객체는 Object Store에 올려 참조로 전달
    _inputDataCube_id = ray.put(_inputDataCube)
    _peak_sn_map_id   = ray.put(_peak_sn_map)
    _sn_int_map_id    = ray.put(_sn_int_map)
    _cube_mask_2d_id  = ray.put(_cube_mask_2d)
    _x_id             = ray.put(_x)
    _params_id        = ray.put(_params)



    #------------------------------
    # 타일링/배치 수집 설정
    tile    = int(_params['y_chunk_size'])  # 권장 시작: 20~50
    # YAML: gather_batch
    batch_k = int(_params['gather_batch'])
    # 과도한 부하 방지: CPU 예산을 넘지 않게, 너무 크지 않게
    batch_k = max(1, min(batch_k, num_cpus_total))

    print("| -------------------------------------------------------------- |")
    print(f" :: y_chunk_size = {tile} | gather_batch = {batch_k}")
    print("| -------------------------------------------------------------- |")
    print("")
    print("avail:", ray.available_resources())   # 남은 CPU 리소스
    print("total:", ray.cluster_resources())     # 총 CPU 리소스
    print("| -------------------------------------------------------------- |")

    # (i, 전체 y-range) 파일로 저장 (기존 포맷 유지)
    save_fmt = f"{_params['wdir']}/{_params['_segdir']}/G{max_ngauss:02d}.x{{curi}}.ys{_js}ye{_je}"

    # remote 함수 핸들(한 번만 생성해서 재사용)
    baygaud_nested_sampling_remote = dynamic_baygaud_nested_sampling(num_cpus_nested_sampling)

    # 지연할당용 누적 버퍼/남은 타일 카운터
    acc = {}        # i -> ndarray((je-js), max_ngauss, _base+7)
    remaining = {}  # i -> 남은 타일 수
    pending = {}    # ObjectRef -> (i, j0, j1)

    # --- total_tiles 카운트 ---
    total_tiles = 0

    # 작업 제출: (i, j0:j1) 타일 단위
    for i in range(_is, _ie):
        j0 = _js
        while j0 < _je:
            j1 = min(j0 + tile, _je)
            ref = baygaud_nested_sampling_remote.remote(
                _inputDataCube_id, _x_id,
                _peak_sn_map_id, _sn_int_map_id,
                _params_id,
                _is, _ie, i, j0, j1,
                _cube_mask_2d_id
            )
            pending[ref] = (i, j0, j1)
            j0 = j1
            total_tiles += 1  # ★ 타일 1개 제출할 때마다 증가




    # ── 결과 텐서 3축 길이 P는 리모트 결과와 동일해야 함 ──
    P = 2*(2 + 3*max_ngauss) + 7 + max_ngauss  # (당신 코드의 리모트 gfit_results와 동일 공식)

    # 모든 i에 대해 미리 버퍼/카운터 생성 (지연할당 X)
    acc       = {i: np.empty(((_je - _js), max_ngauss, P), dtype=np.float32) for i in range(_is, _ie)}
    remaining = {i: ((_je - _js + tile - 1) // tile) for i in range(_is, _ie)}

    # 픽셀 기준 진행률
    total_pixels     = (_ie - _is) * (_je - _js) # 전체 (i,y) 픽셀 수
    completed_pixels = 0 # 누적 완료 픽셀 수
    # 결과 수집: 타일을 acc[i]에 끼워넣고, 해당 i의 모든 타일 수신 시 한 번만 저장
    t0 = time.perf_counter()
    print("\n |")
    # 초기 0% 표시 (픽셀 기준)
    _print_progress_classic(0, total_pixels, t0, width=50, divisions=20, min_interval=1.0)


    while pending:
        done_ids, _ = ray.wait(list(pending.keys()), num_returns=1, timeout=0.5)
        if not done_ids:
            _print_progress_classic(completed_pixels, total_pixels, t0, width=50, divisions=20, min_interval=2.0)
            continue

        ref = done_ids[0]
        i, j0, j1 = pending.pop(ref)
        i = int(i)  # 키 타입 안전하게
        # (디버그) i 범위 확인
        # assert _is <= i < _ie, f"unexpected i {i}"

        arr = np.asarray(ray.get(ref), dtype=np.float32)  # (j1-j0, max_ngauss, P)

        # ── 선할당된 버퍼에 바로 삽입 ──
        acc[i][(j0 - _js):(j1 - _js), :, :] = arr

        remaining[i] -= 1
        if remaining[i] == 0:
            out = acc[i][None, ...]  # (1, Y, max_ngauss, P)
            np.save(save_fmt.format(curi=i), out)
            # 메모리 비우고 싶으면 아래 두 줄 활성화
            # del acc[i]
            # del remaining[i]

        # ── 픽셀 기반 진행률 ──
        completed_pixels += (j1 - j0)
        _print_progress_classic(
            completed_pixels, total_pixels, t0,
            cur_i=i, cur_j0=j0, cur_j1=j1,
            width=50, divisions=20, min_interval=2.0
        )

    # 진행률 줄바꿈 마무리
    sys.stdout.write("\n"); sys.stdout.flush()
    print(" |\n")

    ray.shutdown()
    dur = (datetime.now() - start).total_seconds()
    print(f"duration = {_fmt_ddhhmmss(dur)}\n")
    print("")
#-- END OF SUB-ROUTINE____________________________________________________________#

if __name__ == '__main__':
    main()


#-- END OF SUB-ROUTINE____________________________________________________________#
