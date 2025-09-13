#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _handle_yaml.py
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|

from __future__ import annotations
import math
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
    Read the 'threading' section from YAML (preferred) or fall back to
    top-level keys (legacy), then build an environment variable dict for
    Ray workers/drivers. All values must be strings, so we wrap with str().
    """
    thr = _params.get("threading", {}) or {}

    def _get(key, default):
        # Prefer the 'threading' section; otherwise use top-level key as fallback.
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
    """Return a section dict if present, otherwise {}."""
    s = params.get(name, {})
    return {} if s is None else s
#-- END OF SUB-ROUTINE____________________________________#


#  ______________________________________________________  #
# [______________________________________________________] #
# [ _as_int_or_none
# _______________________________________________________  #
def _as_int_or_none(v):
    """Cast to int unless v is None or the string 'null'."""
    if v is None or v == "null":
        return None
    return int(v)
#-- END OF SUB-ROUTINE____________________________________#


#  ______________________________________________________  #
# [______________________________________________________] #
# [ _as_float_or_none
# _______________________________________________________  #
def _as_float_or_none(v):
    """Cast to float unless v is None or the string 'null'."""
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
    Convert the 'dynesty' section (dict) into a set of run configurations.

    Returns
    -------
    dyn_class : str
        'static' or 'dynamic'.
    common_kwargs : dict
        Keyword args shared by NestedSampler / DynamicNestedSampler.
    static_run_kwargs : dict
        Arguments for sampler.run_nested(...) (static-only).
    dynamic_run_kwargs : dict
        Arguments for sampler.run_nested(...) (dynamic-only).
    """

    # (recommended) sampler mode: static | dynamic
    dyn_class = str(dy.get("class", "static")).lower()   # recommended: static

    # number of live points
    nlive     = int(dy.get("nlive", 100))

    # sampling method: auto | unif | rwalk | slice | rslice
    sample    = dy.get("sample", "rwalk")                # or 'auto'

    # convergence target (static mode)
    dlogz     = float(dy.get("dlogz", 0.01))
    maxiter   = _as_int_or_none(dy.get("maxiter", 999999))
    maxcall   = _as_int_or_none(dy.get("maxcall", 999999))

    # update interval: 2.0 can be interpreted as "2.0 x nlive"
    update_interval = float(dy.get("update_interval", 2.0))     # ===> 2.0 x nlive : CAUTION !!! : update_interval should be float not integer!!! (e.g., not 2 but 2.0)

    # volume shrink/check (sampling efficiency tuning)
    #vol_dec   = float(dy.get("vol_dec", 0.2))
    #vol_check = int(dy.get("vol_check", 2))

    # move/acceptance tuning
    facc      = float(dy.get("facc", 0.5))
    fmove     = float(dy.get("fmove", 0.9))
    max_move  = int(dy.get("max_move", 100))

    # rwalk-specific steps (if both are present, 'rwalk' takes precedence)
    walks = dy.get("rwalk", None)
    if walks is None:
        walks = dy.get("walks", 25)
    walks = int(walks)

    # bound setting: single | multi | none | balls ...
    bound     = dy.get("bound", "multi")  # for unimodal, 'single' may work; 'multi' often more robust.

    # -- the following are only used for dynamic mode --
    queue_size = dy.get("queue_size", None)
    if queue_size is None or queue_size == "null":
        queue_size = int(_params.get("num_cpus_nested_sampling", 1))
    else:
        queue_size = int(queue_size)

    dlogz_init   = float(dy.get("dlogz_init", dlogz))
    maxiter_init = _as_int_or_none(dy.get("maxiter_init", None))
    maxcall_init = _as_int_or_none(dy.get("maxcall_init", None))

    # RNG seed for reproducibility
    seed   = int(dy.get("seed", 2))
    rstate = np.random.default_rng(seed)

    # common kwargs (as accepted by dynesty)
    common_kwargs = dict(
        nlive=nlive,
        sample=sample,
        bound=bound,
        rstate=rstate,
        # runtime tuning
        facc=facc,
        fmove=fmove,
        max_move=max_move,
        #vol_dec=vol_dec,
        #vol_check=vol_check,
        update_interval=update_interval,  # pass through as float/int as given
    )

    # if sample == 'rwalk', include 'walks'
    if str(sample).lower() == "rwalk":
        common_kwargs["walks"] = walks

    # run_nested argument sets
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
    Parse 'a.b[0].c' -> ['a','b',0,'c'].
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
                # (ruamel.yaml CommentedMap recommended)
                from ruamel.yaml.comments import CommentedMap
                parent = cur
                cur = CommentedMap()
                # cannot set back here without the parent ref; require mapping
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

    # Get old value and set the new value
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


# _tiling_auto.py
# ---------------------------------------------------------------------------
# Purpose
#   Compute good values for y_chunk_size and gather_batch automatically
#   from your YAML config. This helps you use your CPUs efficiently when
#   running baygaud with Ray.
#
# What it does
#   - Reads naxis2_s0 / naxis2_e0 and CPU settings from the YAML
#   - Estimates a tile height along Y ("y_chunk_size") so that there are
#     enough tiles in the queue to keep all CPU workers busy
#   - Picks a reasonable "gather_batch" (how many finished tasks to gather
#     in one wait cycle) and a safe "inflight_factor" (how many tasks to
#     have in flight per CPU)
#   - Optionally writes the recommendations back into the YAML
#
# How to use (programmatically)
#   from _tiling_auto import recommend_tiling_from_yaml
#   out = recommend_tiling_from_yaml("_baygaud_params.yaml", write_back=True)
#
# How to use (CLI)
#   python _tiling_auto.py /path/to/_baygaud_params.yaml [--write]
#   (with --write it will update the YAML in-place)
# ---------------------------------------------------------------------------



def _compute_tiling_params(
    y_span: int,
    num_cpus_ray: int,
    num_cpus_nested_sampling: int,
    *,
    # --- Tuning knobs with safe defaults ---
    # We want several tiles per core so the queue never runs dry.
    target_tiles_per_core: float = 3.0,
    # Avoid extremely small tiles (too much overhead) or overly big ones (bad load balance).
    min_tile: int = 10,
    max_tile: int = 50,
    # Do not gather too many results at once; it only delays progress printing.
    gather_cap: int = 16,
) -> dict:
    """
    Compute recommended tiling parameters.

    Parameters
    ----------
    y_span : int
        Number of rows along Y to process = (naxis2_e0 - naxis2_s0).
    num_cpus_ray : int
        Number of CPU workers Ray may use (from YAML: num_cpus_ray).
    num_cpus_nested_sampling : int
        Threads/cores used inside each nested-sampling task (from YAML: num_cpus_nested_sampling).
        Usually 1 is best to avoid oversubscription.

    Keyword-only parameters (safe defaults)
    ---------------------------------------
    target_tiles_per_core : float
        Aim for this many tiles per core in the queue.
        More tiles → better load balancing; too many → not needed.
    min_tile, max_tile : int
        Clamp the tile height to this range.
        Experience shows 10–50 is a good balance on most machines.
    gather_cap : int
        Upper limit for gather_batch. Keep it modest (e.g., ≤16).

    Returns
    -------
    dict
        {
          "y_chunk_size": int,      # tile height along Y to request from Ray workers
          "gather_batch": int,      # how many results we gather per wait cycle
          "inflight_factor": int,   # how many in-flight tasks per CPU (3 is safe if RAM is plenty)
          "y_span": int,            # echo input
          "cpus_total": int         # total CPU concurrency considered (ray * nested_sampling)
        }

    Notes
    -----
    Heuristic for y_chunk_size:
      T ≈ H / (target_tiles_per_core * C)
      where H = y_span, and C = num_cpus_ray * num_cpus_nested_sampling.
      Then clamp T into [min_tile, max_tile].

    Rationale
    ---------
    - Too small tiles: too much Python/Ray overhead.
    - Too big tiles: some workers finish early and sit idle while others
      process huge tiles (stragglers).
    - A small fixed gather_batch keeps the UI/progress responsive and
      avoids long waits for output.
    """
    # Sanitize inputs
    H = max(1, int(y_span))
    # Total "parallelism" is Ray workers times per-task threads.
    C = max(1, int(num_cpus_ray) * int(num_cpus_nested_sampling))

    # First guess for tile height T:
    # If we want k tiles per core, and we have C cores,
    # we want ~k*C tiles in total → each tile ~ H / (k*C).
    raw = H / max(1.0, target_tiles_per_core * C)
    T = int(math.floor(raw))

    # Keep T within a healthy range [min_tile, max_tile].
    T = max(min_tile, min(max_tile, T))

    # If Y is very small, T can exceed H; fix that.
    if T > H:
        # Use the smaller of H and min_tile, but at least 1.
        T = max(1, min(H, min_tile))

    # How many results to collect per wait.
    # A good rule: near the amount of concurrent work C, but cap it.
    gather_batch = max(1, min(C, gather_cap))

    # Memory is available (as you said), so we can keep more tasks in flight.
    # This helps absorb variability in task durations.
    inflight_factor = 3

    return dict(
        y_chunk_size=int(T),
        gather_batch=int(gather_batch),
        inflight_factor=int(inflight_factor),
        y_span=int(H),
        cpus_total=int(C),
    )


def recommend_tiling_from_yaml(configfile: str, *, write_back: bool = False) -> dict:
    """
    Read required fields from a baygaud YAML file and compute tiling parameters.

    Parameters
    ----------
    configfile : str
        Path to your _baygaud_params.yaml.
    write_back : bool, default False
        If True, immediately write y_chunk_size, gather_batch, and inflight_factor
        back into the YAML (top-level keys). If False, only return the values.

    Returns
    -------
    dict
        Same structure as _compute_tiling_params(...) plus input echo.

    Raises
    ------
    KeyError
        If the YAML is missing required keys (naxis2_s0/e0, num_cpus_ray).

    Usage (code)
    ------------
    >>> out = recommend_tiling_from_yaml("_baygaud_params.yaml", write_back=True)
    >>> print(out["y_chunk_size"], out["gather_batch"])
    """
    # Load YAML into a dict
    p = read_configfile(configfile)

    # Required keys. We keep the error messages clear if any are missing.
    try:
        y0 = int(p["naxis2_s0"])
        y1 = int(p["naxis2_e0"])
        num_cpus_ray = int(p["num_cpus_ray"])
        num_cpus_nested_sampling = int(p.get("num_cpus_nested_sampling", 1))
    except KeyError as e:
        raise KeyError(f"Missing required YAML key: {e!s}")

    # Compute recommendations
    out = _compute_tiling_params(
        y_span=(y1 - y0),
        num_cpus_ray=num_cpus_ray,
        num_cpus_nested_sampling=num_cpus_nested_sampling,
    )

    # Optionally write them back to the YAML
    if write_back:
        # If keys do not exist yet, create them (create_missing=True).
        update_yaml_param(configfile, "y_chunk_size",     out["y_chunk_size"],     create_missing=True)
        update_yaml_param(configfile, "gather_batch",     out["gather_batch"],     create_missing=True)
        update_yaml_param(configfile, "inflight_factor",  out["inflight_factor"],  create_missing=True)

    return out


#-- END OF SUB-ROUTINE____________________________________#


