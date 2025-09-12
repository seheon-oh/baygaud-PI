#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _handle_yaml.py
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|

import sys
import numpy as np
import yaml

#  ______________________________________________________  #
# [______________________________________________________] #
# [ global parameters
# _______________________________________________________  #

#global _inputDataCube
#global _is, _ie, _js, _je
#global parameters
#global nparams
#global ngauss
#global ndim
#global max_ngauss
#global gfit_results
#global _x
#global nchannels

#  ______________________________________________________  #
# [______________________________________________________] #
# [ read yaml file
# _______________________________________________________  #
def read_configfile(configfile):
    with open(configfile, "r") as file:
        _params = yaml.safe_load(file)
    return _params
#-- END OF SUB-ROUTINE____________________________________#


#  ______________________________________________________  #
# [______________________________________________________] #
# [ read threading setup from yaml file
# _______________________________________________________  #
def _get_threading_env_from_params(_params):
    """
    YAML의 'threading' 섹션(권장) 또는 top-level 키(레거시)를 읽어
    Ray 워커/드라이버에서 사용할 환경 변수 dict를 생성합니다.
    모든 값은 문자열이어야 하므로 str()로 감쌉니다.
    """
    thr = _params.get("threading", {}) or {}

    def _get(key, default):
        # threading 섹션 우선, 없으면 top-level 키를 fallback
        val = thr.get(key, _params.get(key, default))
        return str(val)

    env_vars = {
        "OMP_NUM_THREADS":        _get("omp_num_threads", 1),
        "MKL_NUM_THREADS":        _get("mkl_num_threads", 1),
        "OPENBLAS_NUM_THREADS":   _get("openblas_num_threads", 1),
        "BLIS_NUM_THREADS":       _get("blis_num_threads", 1),
        "VECLIB_MAXIMUM_THREADS": _get("veclib_maximum_threads", 1),
        "NUMEXPR_NUM_THREADS":    _get("numexpr_num_threads", 1),
        "NUMBA_THREADING_LAYER":  _get("numba_threading_layer", "workqueue"),
        "NUMBA_NUM_THREADS":      _get("numba_num_threads", 1),
    }
    return env_vars
#-- END OF SUB-ROUTINE____________________________________#



#  ______________________________________________________  #
# [______________________________________________________] #
# [ read yaml parameters 
# _______________________________________________________  #
def _sec(params, name):
    """섹션 딕셔너리(없으면 {})"""
    s = params.get(name, {})
    return {} if s is None else s
#-- END OF SUB-ROUTINE____________________________________#


#  ______________________________________________________  #
# [______________________________________________________] #
# [ _as_int_or_none
# _______________________________________________________  #
def _as_int_or_none(v):
    if v is None or v == "null":
        return None
    return int(v)
#-- END OF SUB-ROUTINE____________________________________#


#  ______________________________________________________  #
# [______________________________________________________] #
# [ _as_float_or_none
# _______________________________________________________  #
def _as_float_or_none(v):
    if v is None or v == "null":
        return None
    return float(v)
#-- END OF SUB-ROUTINE____________________________________#


#  ______________________________________________________  #
# [______________________________________________________] #
# [ build_dynesty_run_config
# _______________________________________________________  #
def build_dynesty_run_config_dynesty_v2_1_5(dy: dict, _params: dict):
    """
    dynesty 섹션(dict) -> 실행에 필요한 설정을 정리해 반환.
    반환:
      - dyn_class            : 'static' | 'dynamic'
      - common_kwargs        : NestedSampler / DynamicNestedSampler 공통 kwargs
      - static_run_kwargs    : sampler.run_nested(...) 인자 (static 전용)
      - dynamic_run_kwargs   : sampler.run_nested(...) 인자 (dynamic 전용)
    """

    # (recommended) 사용할 샘플러 모드: static | dynamic
    dyn_class = str(dy.get("class", "static")).lower()   # 추천: static

    # 라이브 포인트 개수
    nlive     = int(dy.get("nlive", 100))

    # 샘플링 방식: auto | unif | rwalk | slice | rslice
    sample    = dy.get("sample", "rwalk")                # 또는 auto

    # 수렴 기준 (static 모드)
    dlogz     = float(dy.get("dlogz", 0.01))
    maxiter   = _as_int_or_none(dy.get("maxiter", 999999))
    maxcall   = _as_int_or_none(dy.get("maxcall", 999999))

    # 업데이트 주기: 2.0이면 "2.0 × nlive"로 이해해도 무방
    update_interval = float(dy.get("update_interval", 2.0))     # ===> 2.0 x nlive : CAUTION !!! : update_interval should be float not integer!!! (e.g., not 2 but 2.0)

    # 볼륨 축소/체크(샘플링 효율 튜닝)
    #vol_dec   = float(dy.get("vol_dec", 0.2))
    #vol_check = int(dy.get("vol_check", 2))

    # 이동/수용률 튜닝
    facc      = float(dy.get("facc", 0.5))
    fmove     = float(dy.get("fmove", 0.9))
    max_move  = int(dy.get("max_move", 100))

    # rwalk 전용 스텝 수 (둘 다 있을 때 rwalk 값이 우선)
    walks = dy.get("rwalk", None)
    if walks is None:
        walks = dy.get("walks", 25)
    walks = int(walks)

    # 경계 설정: single | multi | none | balls ...
    bound     = dy.get("bound", "multi")  # 단봉형이면 'single' 가능. 효율 문제 시 'multi' 권장.

    # ── 아래는 dynamic 모드에서만 유효 ────────────────────────────────
    queue_size = dy.get("queue_size", None)
    if queue_size is None or queue_size == "null":
        queue_size = int(_params.get("num_cpus_nested_sampling", 1))
    else:
        queue_size = int(queue_size)

    dlogz_init   = float(dy.get("dlogz_init", dlogz))
    maxiter_init = _as_int_or_none(dy.get("maxiter_init", None))
    maxcall_init = _as_int_or_none(dy.get("maxcall_init", None))

    # 재현성을 위한 RNG 시드
    seed   = int(dy.get("seed", 2))
    rstate = np.random.default_rng(seed)

    # 공통 kwargs (dynesty가 kwargs로 받는 것들 포함)
    common_kwargs = dict(
        nlive=nlive,
        sample=sample,
        bound=bound,
        rstate=rstate,
        # run-time 튜닝 관련
        facc=facc,
        fmove=fmove,
        max_move=max_move,
        #vol_dec=vol_dec,
        #vol_check=vol_check,
        update_interval=update_interval,  # float/정수 그대로 전달
    )

    # sample='rwalk'이면 walks 전달
    if str(sample).lower() == "rwalk":
        common_kwargs["walks"] = walks

    # run_nested 인자들
    static_run_kwargs  = dict(dlogz=dlogz, maxiter=maxiter, maxcall=maxcall, print_progress=False)
    dynamic_run_kwargs = dict(dlogz_init=dlogz_init, maxiter_init=maxiter_init,
                              maxcall_init=maxcall_init, print_progress=False,
                              queue_size=queue_size)

    return dyn_class, common_kwargs, static_run_kwargs, dynamic_run_kwargs
#-- END OF SUB-ROUTINE____________________________________#


from typing import Any, List, Tuple, Union
import os, shutil, re

try:
    from ruamel.yaml import YAML
except ImportError:
    raise SystemExit("Please install ruamel.yaml:  pip install ruamel.yaml")

_PATH_TOKEN = re.compile(r"""
    ([^. \[\]]+)      # a key segment (no dot/[]/space)
  | \[(\d+)\]         # or an index like [0]
""", re.VERBOSE)

def _parse_param_path(param_path: str) -> List[Union[str, int]]:
    """
    Parse 'a.b[0].c' -> ['a','b',0,'c']
    """
    tokens: List[Union[str, int]] = []
    for m in _PATH_TOKEN.finditer(param_path.strip()):
        key, idx = m.groups()
        if key is not None:
            tokens.append(key)
        else:
            tokens.append(int(idx))
    if not tokens:
        raise ValueError(f"Invalid param_path: {param_path!r}")
    return tokens

def _walk_to_parent(doc: Any, tokens: List[Union[str,int]], create_missing: bool):
    """
    Walk to the parent container of the final token.
    Returns (parent_container, last_token).
    Creates intermediate dict/list if create_missing=True.
    """
    cur = doc
    for i, tok in enumerate(tokens[:-1]):
        nxt = tokens[i+1]
        if isinstance(tok, str):
            # Ensure mapping
            if not isinstance(cur, dict):
                if not create_missing:
                    raise KeyError(f"Expected mapping at {tokens[:i+1]}, got {type(cur).__name__}")
                # replace with mapping
                # (ruamel.yaml CommentedMap 권장)
                from ruamel.yaml.comments import CommentedMap
                parent = cur
                cur = CommentedMap()
                # we cannot set back here without the parent ref; better require mapping
                raise KeyError(f"Path segment {tok!r} not found and parent is not a mapping.")
            if tok not in cur:
                if not create_missing:
                    raise KeyError(f"Key {tok!r} not found (use create_missing=True to create).")
                from ruamel.yaml.comments import CommentedMap
                # Decide what to create based on next token type
                cur[tok] = [] if isinstance(nxt, int) else CommentedMap()
            cur = cur[tok]
        else:  # index
            if not isinstance(cur, list):
                if not create_missing:
                    raise KeyError(f"Expected sequence at {tokens[:i]}, got {type(cur).__name__}")
                # cannot auto-convert safely; require existing list
                raise KeyError(f"Sequence not present at {tokens[:i]} (create_missing doesn't create lists at this level).")
            idx = tok
            if idx >= len(cur):
                if not create_missing:
                    raise IndexError(f"Index {idx} out of range at {tokens[:i+1]}")
                # extend with None up to idx
                cur.extend([None] * (idx - len(cur) + 1))
            if isinstance(nxt, (str,)):
                if cur[idx] is None:
                    from ruamel.yaml.comments import CommentedMap
                    cur[idx] = CommentedMap()
            elif isinstance(nxt, int):
                if cur[idx] is None:
                    cur[idx] = []
            cur = cur[idx]
    return cur, tokens[-1]

def update_yaml_param(
    yaml_path: str,
    param_path: str,
    new_value: Any,
    *,
    create_missing: bool = False,
    backup: bool = True,
    indent: Tuple[int,int,int] = (2, 2, 2),
) -> dict:
    """
    Update a single parameter in a YAML file.

    Parameters
    ----------
    yaml_path : str
        Path to YAML file.
    param_path : str
        Dot path with optional list indices, e.g.:
        'sgfit_bounds.model_sigma_bounds[0]', 'threading.numexpr_num_threads'
    new_value : Any
        Value to write (Python types: int/float/str/bool/list/dict, etc.).
    create_missing : bool, default False
        If True, create missing intermediate mappings (and extend sequences).
        (Lists at non-existent locations are not auto-created for safety.)
    backup : bool, default True
        If True, create a '.bak' copy once before first write.
    indent : (int,int,int), default (2,2,2)
        YAML indentation (mapping, sequence, offset).

    Returns
    -------
    dict with keys:
      - "old_value": previous value (or None if not set)
      - "new_value": new_value
      - "path": param_path
      - "written": True/False
      - "backup": backup path or None
    """
    yaml = YAML()
    yaml.preserve_quotes = True

    with open(yaml_path, "r", encoding="utf-8") as f:
        doc = yaml.load(f)

    if backup:
        bak = yaml_path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(yaml_path, bak)
    else:
        bak = None

    tokens = _parse_param_path(param_path)
    parent, last = _walk_to_parent(doc, tokens, create_missing=create_missing)

    # Get old value
    old_value = None
    if isinstance(last, str):
        if isinstance(parent, dict) and last in parent:
            old_value = parent[last]
        elif not create_missing:
            raise KeyError(f"Key {last!r} not found at path {param_path!r}")
        # set
        if isinstance(parent, dict):
            parent[last] = new_value
        else:
            raise TypeError(f"Parent is not a mapping at {param_path!r}")
    else:  # last is int index
        if not isinstance(parent, list):
            raise TypeError(f"Parent is not a sequence at {param_path!r}")
        idx = last
        if idx >= len(parent):
            if not create_missing:
                raise IndexError(f"Index {idx} out of range at {param_path!r}")
            parent.extend([None] * (idx - len(parent) + 1))
        old_value = parent[idx]
        parent[idx] = new_value

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.indent(*indent)
        yaml.dump(doc, f)

    return {
        "old_value": old_value,
        "new_value": new_value,
        "path": param_path,
        "written": True,
        "backup": bak,
    }
