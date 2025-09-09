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
from collections import deque

# _combine_segs.py
#import _combine_segs 


import matplotlib.pyplot as plt


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

    # ─────────────────────────────────────────────────────────
    # 병렬 설정 (Ray + 워커 스레드 핀ning)
    # ─────────────────────────────────────────────────────────
    num_cpus_ray = int(_params['num_cpus_ray'])
    num_cpus_nested_sampling = int(_params['num_cpus_nested_sampling'])  # 권장: 1
    num_cpus_total = num_cpus_ray * num_cpus_nested_sampling

    # 워커 환경(스레드 1로 고정). 필요시 NUMBA_* 기본값은 파라미터에서 가져옴
    env_vars = {
        "OMP_NUM_THREADS":        "1",
        "MKL_NUM_THREADS":        "1",
        "OPENBLAS_NUM_THREADS":   "1",
        "BLIS_NUM_THREADS":       "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS":    "1",
        "NUMBA_THREADING_LAYER":  str(_params.get("numba_threading_layer", "workqueue")),
        "NUMBA_NUM_THREADS":      str(_params.get("numba_num_threads", 1)),
    }

    # (옵션) 드라이버 프로세스도 동일하게 핀ning하고 싶다면 주석 해제
    # for k, v in env_vars.items():
    #     os.environ.setdefault(k, v)

    ray.init(
        num_cpus=num_cpus_total,
        runtime_env={"env_vars": env_vars}
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
    #derive_rms_npoints(_inputDataCube, _cube_mask_2d, _x, _params, 1)

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
    #------------------------------

    print("")
    print("| -------------------------------------------------------------- |")
    print(f" :: y_chunk_size = {tile} | gather_batch = {batch_k}")
    print("| -------------------------------------------------------------- |")
    print("")
    print("|................................................................|")
    print(" -- Ray dashboard info --")
    print(" :: available:", ray.available_resources())   # 남은 CPU 리소스
    print(" :: total    :", ray.cluster_resources())     # 총 CPU 리소스
    print("|................................................................|")

    save_fmt = f"{_params['wdir']}/{_params['_segdir']}/G{max_ngauss:02d}.x{{curi}}.ys{_js}ye{_je}"

    # remote 함수 핸들
    baygaud_nested_sampling_remote = dynamic_baygaud_nested_sampling(num_cpus_nested_sampling, _params)

    # 지연할당용 누적 버퍼/남은 타일 카운터
    acc = {}        # i -> ndarray((je-js), max_ngauss, _base+7)
    remaining = {}  # i -> 남은 타일 수
    pending = {}    # ObjectRef -> (i, j0, j1)

    # ── 타일 큐 만들기 ──
    tiles = deque()
    for i in range(_is, _ie):
        j0 = _js
        while j0 < _je:
            j1 = min(j0 + tile, _je)
            tiles.append((i, j0, j1))
            j0 = j1

    # ── 선할당(지연할당 없이) ──
    P = 2*(2 + 3*max_ngauss) + 7 + max_ngauss
    acc       = {i: np.empty(((_je - _js), max_ngauss, P), dtype=np.float32) for i in range(_is, _ie)}
    remaining = {i: ((_je - _js + tile - 1)//tile) for i in range(_is, _ie)}

    total_pixels     = (_ie - _is) * (_je - _js)
    completed_pixels = 0
    t0 = time.perf_counter()
    print("\n |")

    # 진행바 0% 출력
    _print_progress_classic(0, total_pixels, t0, width=50, divisions=20, min_interval=0.2)

    # ── in-flight 윈도우 설정 ──
    # 타일 큐 만든 뒤
    inflight_factor = int(_params.get("inflight_factor", 1))  # ← 1 로 시작
    MAX_INFLIGHT    = max(1, min(len(tiles), num_cpus_total * inflight_factor))
    wait_timeout_s  = float(_params.get("wait_timeout_s", 1.0))   # ← 1.0 로 늘림
    print_min_int   = float(_params.get("print_min_interval_s", 1.0))  # 진행 출력도 느긋하게

    # ── 제출/수집 상태 ──
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
            _cube_mask_2d_id
        )
        inflight_refs.append(ref)
        meta[ref] = (i, j0, j1)

    for _ in range(MAX_INFLIGHT):
        _submit_one()

    # ── 수집 루프: 단건 wait + timeout + 가벼운 하트비트 ──
    while inflight_refs:
        done_ids, _ = ray.wait(inflight_refs, num_returns=1, timeout=wait_timeout_s)
        if not done_ids:
            time.sleep(1)  # 드라이버에 휴식 제공(프리징 방지)
            _print_progress_classic(completed_pixels, total_pixels, t0, width=50, divisions=20, min_interval=1.0)
            continue

        ref = done_ids[0]
        inflight_refs.remove(ref)
        i, j0, j1 = meta.pop(ref)

        arr = np.asarray(ray.get(ref), dtype=np.float32)  # (j1-j0, max_ngauss, P)
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

        # 창 유지: 끝난 만큼 새로 제출
        _submit_one()

    sys.stdout.write("\n"); sys.stdout.flush()
    print("\n")

    ray.shutdown()
    dur = (datetime.now() - start).total_seconds()
    print(f"duration = {_fmt_ddhhmmss(dur)}\n")
    print("")
#-- END OF SUB-ROUTINE____________________________________________________________#


if __name__ == '__main__':
    main()


#-- END OF SUB-ROUTINE____________________________________________________________#
