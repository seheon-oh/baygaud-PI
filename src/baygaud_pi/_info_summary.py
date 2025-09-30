#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _info_summary.py
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|


# -- Subroutines to print a compact summary table under a banner --

from typing import Dict, List, Tuple, Optional
import shutil
import math

def _term_width(default: int = 100) -> int:
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except Exception:
        return default

def _box_print(rows,
               title: str | None = None,
               left_margin: int = 0,
               max_width: int | None = None,
               labels: tuple[str, str, str] = ("Quantity", "Value", "Note")) -> None:
    """
    Print a 3-column ASCII table.

    Parameters
    ----------
    rows : list of tuples
        Each row as (col1, col2, col3).
    labels : tuple of str
        Header labels (default: Quantity / Value / Note).
    """
    import shutil

    pad = " " * max(left_margin, 0)
    tw = shutil.get_terminal_size((100, 24)).columns
    width = min(max_width or tw, tw)

    def cut(s: str, w: int) -> str:
        return s if len(s) <= w else (s[:max(0, w-1)] + "…")

    # Compute column widths
    name_w = max([len(r[0]) for r in rows] + [len(labels[0]), 4])
    val_w  = max([len(r[1]) for r in rows] + [len(labels[1]), 5])
    note_w = max([len(r[2]) for r in rows] + [len(labels[2]), 4])

    total = name_w + val_w + note_w + 8
    if total > width - left_margin:
        overflow = total - (width - left_margin)
        # Reduce width preferentially: note -> value -> name
        for target in (2, 1, 0):
            if overflow <= 0:
                break
            take = min(overflow, [name_w, val_w, note_w][target] - 4)
            if target == 0:
                name_w -= take
            elif target == 1:
                val_w -= take
            else:
                note_w -= take
            overflow -= take

    hbar = pad + "+" + "-"*(name_w+2) + "+" + "-"*(val_w+2) + "+" + "-"*(note_w+2) + "+"
    if title:
        cap = pad + "+" + "-"*(name_w+val_w+note_w+6) + "+"
        inner_w = name_w + val_w + note_w + 6
        print(cap)
        print(pad + "|" + f" {title} ".center(inner_w) + "|")
        print(cap)

    print(hbar)
    col1, col2, col3 = labels
    # Header: right-align the Value column
    header = pad + f"| {col1.ljust(name_w)} | {col2.rjust(val_w)} | {col3.ljust(note_w)} |"
    print(header)
    print(hbar)

    for c1, c2, c3 in rows:
        c1 = cut(c1, name_w)
        c2 = cut(c2, val_w)
        c3 = cut(c3, note_w)

        # Value column is always right-aligned
        c1 = c1.ljust(name_w)
        c2 = c2.rjust(val_w)
        c3 = c3.ljust(note_w)

        print(pad + f"| {c1} | {c2} | {c3} |")
    print(hbar)




def get_ray_info():
    """
    Return a dict summarizing Ray runtime status.
    If Ray is not installed/initialized, return minimal info or None,
    so that the caller can omit the Ray section in the table gracefully.
    """
    try:
        import ray
    except ImportError:
        return None  # If Ray is not used, skip the section in the table

    info = {"Ray initialized": str(ray.is_initialized())}

    if not ray.is_initialized():
        return info  # If not initialized, only report this flag

    try:
        # Total and available resources
        resources = ray.cluster_resources()       # totals
        avail     = ray.available_resources()     # currently available

        # Number of nodes
        try:
            nodes = ray.nodes()
            info["nodes"] = str(len(nodes))
        except Exception:
            pass

        # CPU/GPU
        info["total_CPUs"] = str(int(resources.get("CPU", 0)))
        info["avail_CPUs"] = str(int(avail.get("CPU", 0)))
        if "GPU" in resources or "GPU" in avail:
            info["total_GPUs"] = str(int(resources.get("GPU", 0)))
            info["avail_GPUs"] = str(int(avail.get("GPU", 0)))

        # Memory (Ray 2.x may expose byte-based keys)
        mem_bytes = int(resources.get("memory", 0))
        obj_bytes = int(resources.get("object_store_memory", 0))
        if mem_bytes:
            info["memory_GB"] = f"{mem_bytes/1e9:.1f}"
        if obj_bytes:
            info["object_store_GB"] = f"{obj_bytes/1e9:.1f}"

    except Exception as e:
        info["note"] = f"ray info collection error: {e}"

    return info




# -- System/allocated resource summary: total_CPUs, avail_CPUs, memory_GB, object_store_GB --
def get_runtime_resource_info(_params: dict | None = None) -> dict | None:
    """
    Return a compact resource summary. Example:
      {
        "Ray initialized": "True" | "False",
        "total_CPUs": "12",                 # physical core count (psutil); fallback to logical
        "avail_CPUs": "8",                  # CPUs allocated to baygaud (priority: _params -> Ray limit -> CPU affinity -> logical)
        "memory_GB": "64.0",                # total system memory (GB)
        "object_store_GB": "3.2",           # current process RSS memory (GB)
      }
    """
    info = {}
    # (1) Ray status
    try:
        import ray
        info["Ray initialized"] = "True" if ray.is_initialized() else "False"
    except Exception:
        info["Ray initialized"] = "False"

    # (2) CPUs: physical/logical/affinity/parameters/environment variables
    import os
    try:
        import psutil
    except Exception:
        psutil = None

    # total_CPUs: prefer physical core count
    total_phys = None
    if psutil:
        try:
            total_phys = psutil.cpu_count(logical=False)
        except Exception:
            total_phys = None
    if not total_phys:
        # Fallback to logical if physical fails
        total_phys = os.cpu_count() or 1
    info["Total physical cores"] = str(total_phys)

    # avail_CPUs: CPUs "allocated" to baygaud
    # Priority: _params['num_cpus'] -> CPU affinity size -> Ray limit/env -> logical
    avail = None
    if _params and isinstance(_params, dict):
        nc = _params.get("num_cpus_ray")
        if nc:
            try:
                avail = int(nc)
            except Exception:
                try:
                    avail = int(float(nc))
                except Exception:
                    pass

    if avail is None and psutil:
        try:
            # On Linux, if CPU affinity is restricted, its length is usable core count
            p = psutil.Process()
            if hasattr(p, "cpu_affinity"):
                aff = p.cpu_affinity()
                if aff:
                    avail = len(aff)
        except Exception:
            pass

    if avail is None:
        # If Ray is initialized and a num_cpus limit exists, estimate conservatively
        try:
            if info["Ray initialized"] == "True":
                # (1) If ray.init(num_cpus=...) was used, sometimes env var exists
                env_nc = os.environ.get("RAY_NUM_CPUS")
                if env_nc:
                    avail = int(float(env_nc))
                else:
                    # (2) Otherwise, fall back to logical core count
                    avail = os.cpu_count() or total_phys
        except Exception:
            pass

    if avail is None:
        avail = os.cpu_count() or total_phys

    info["Ray allocated cores"] = str(avail)
    info["Sampler allocated cores"] = str(_params['num_cpus_nested_sampling'])
    info["Numba allocated threads"] = str(_params['numba_num_threads'])

    # (3) Memory: total system / current process RSS
    total_mem_gb = None
    proc_rss_gb = None
    if psutil:
        try:
            total_mem_gb = psutil.virtual_memory().total / (1024**3)
        except Exception:
            pass
        try:
            proc_rss_gb = psutil.Process().memory_info().rss / (1024**3)
        except Exception:
            pass
    if total_mem_gb is None:
        total_mem_gb = 0.0
    if proc_rss_gb is None:
        proc_rss_gb = 0.0

    info["System memory (GB)"] = f"{total_mem_gb:.1f}"
    # Note: for "object_store", we show memory currently used by this process (RSS)
    info["Process memory (GB)"] = f"{proc_rss_gb:.1f}"

    info["(y chunk size)"] = '(' + str(_params['y_chunk_size']) + ')'
    info["(gather batch)"] = '(' + str(_params['gather_batch']) + ')'

    return info



from typing import Optional, Dict, List

# ---------- helpers (no color) ----------
def _mk_row(c1: str, c2: str, c3: str, w1: int, w2: int, w3: int, *, left_margin: int = 0) -> str:
    sp = " " * left_margin
    return f"{sp}| {c1:<{w1}} | {c2:>{w2}} | {c3:<{w3}}|"


def _mk_cube_wdir_row(
    _params: str,
    c1: str, c2: str, c3: str,  # keep signature; c2,c3 are unused
    w1: int, w2: int, w3: int,
    *, left_margin: int = 0,
    gap: str = "   ",
) -> str:
    """
    Render something like:
      | Input Cube:   test_input.fits                                      |
    If the filename/path overflows, wrap to a second line that starts
    at the SAME column where the filename began on the first line.
    Total inner width matches _mk_row(c1,c2,c3,w1,w2,w3,...).
    """
    sp = " " * left_margin
    inner = w1 + w2 + w3 + 6   # same inner width as _mk_row lines

    prefix = f"{c1}{gap}"      # label + gap
    fname  = str(_params)

    # Safety: if prefix alone already eats (almost) the line, ellipsize one-liner
    if len(prefix) >= inner - 1:
        text = (prefix + fname)
        if len(text) > inner:
            text = text[:inner-1] + "…"
        else:
            text = text.ljust(inner)
        return f"{sp}| {text}|"

    # Room left on line 1 after prefix:
    avail1 = inner - len(prefix)

    # Case 1: everything fits on one line
    if len(fname) <= avail1:
        line1 = (prefix + fname).ljust(inner)
        return f"{sp}| {line1}|"

    # Case 2: wrap to second line; align under filename start
    part1 = fname[:avail1]
    rest  = fname[avail1:]

    line1 = (prefix + part1).ljust(inner)

    # Second line content starts with spaces equal to prefix width
    pad   = " " * len(prefix)
    avail2 = inner - len(prefix)
    if len(rest) > avail2:
        line2_body = rest[:avail2-1] + "…"
    else:
        line2_body = rest.ljust(avail2)
    line2 = (pad + line2_body)

    return f"{sp}| {line1}|\n{sp}| {line2}|"




def _hline(w1: int, w2: int, w3: int, *, left_margin: int = 0) -> str:
    sp = " " * left_margin
    inner = (w1 + w2 + w3) + 9  # include spaces/separators
    return f"{sp}+" + "-" * (inner - 2) + "+"

# ---------- main table ----------
def print_cube_summary(
    _params,
    naxis1: int,
    naxis2: int,
    naxis3: int,
    vel_min_kms: float,
    vel_max_kms: float,
    cdelt3_ms: float,
    *,
    vel_unit_label: str = "km/s",
    cdelt3_unit_label: str = "m/s",
    # Optional (shown if provided)
    naxis1_s0: int | None = None, naxis1_e0: int | None = None,
    naxis2_s0: int | None = None, naxis2_e0: int | None = None,
    max_ngauss: int | None = None,
    peak_sn_limit: float | None = None,
    y_chunk_size: int | None = None,
    gather_batch: int | None = None,
    # Runtime/Ray
    left_margin: int = 0,
    title: str = "Data cube / key params",
    ray_info: Optional[Dict[str, str]] = None,
) -> None:
    """Render a plain 3-column ASCII table (monochrome), similar to screenshots."""

    # Fixed widths (tuned for screenshot-like appearance)
    W1, W2, W3 = 23, 9, 33

    # Header
    print(_hline(W1, W2, W3, left_margin=left_margin))
    _inputcube = _mk_cube_wdir_row(_params['input_datacube'], "Input Cube:", "", "", W1, W2, W3, left_margin=0)
    _wdir = _mk_cube_wdir_row(_params['wdir'], "WDIR:", "", "", W1, W2, W3, left_margin=0)

    if _params['_cube_mask_2d'] == 'Y':
        _mask = _mk_cube_wdir_row(_params['_cube_mask_2d_fits'], "MASK:", "", "", W1, W2, W3, left_margin=0)
    if _params['_cube_mask_3d'] == 'Y':
        _mask = _mk_cube_wdir_row(_params['_cube_mask_3d_fits'], "MASK:", "", "", W1, W2, W3, left_margin=0)
    #--------------------------
    print(_inputcube)
    print(_wdir)

    if _params['_cube_mask_2d'] == 'Y' or _params['_cube_mask_3d'] == 'Y':
        print(_mask)

    #--------------------------
    print(_hline(W1, W2, W3, left_margin=left_margin))
    #--------------------------
    _subtitle = 'Key header params'
    print(_mk_row(_subtitle, "Value", "Note", W1, W2, W3, left_margin=left_margin))
    #--------------------------
    print(_hline(W1, W2, W3, left_margin=left_margin))

    # (1) Key header params
    roi1 = f"[{naxis1_s0} : {naxis1_e0}]" if (naxis1_s0 is not None and naxis1_e0 is not None) else "[:]"
    roi2 = f"[{naxis2_s0} : {naxis2_e0}]" if (naxis2_s0 is not None and naxis2_e0 is not None) else "[:]"
    print(_mk_row("naxis1 (pixels)",      str(naxis1), roi1, W1, W2, W3, left_margin=left_margin))
    print(_mk_row("naxis2 (pixels)",      str(naxis2), roi2, W1, W2, W3, left_margin=left_margin))
    print(_mk_row("naxis3 (channels)",    str(naxis3), "[:]", W1, W2, W3, left_margin=left_margin))
    # (2) Velocity / spectral axis
    sign_note = "(+) spectral axis increasing" if cdelt3_ms >= 0 else "(-) spectral axis decreasing"
    print(_mk_row(f"Velocity min ({vel_unit_label})", f"{vel_min_kms:.2f}", "", W1, W2, W3, left_margin=left_margin))
    print(_mk_row(f"Velocity max ({vel_unit_label})", f"{vel_max_kms:.2f}", "", W1, W2, W3, left_margin=left_margin))
    print(_mk_row(f"CDELT3 ({cdelt3_unit_label})", f"{cdelt3_ms:+.2f}", sign_note,
                  W1, W2, W3, left_margin=left_margin))
    print(_mk_row("Spec axis unit check", vel_unit_label, "<- displayed here should be km/s",
                  W1, W2, W3, left_margin=left_margin))

    #--------------------------
    print(_hline(W1, W2, W3, left_margin=left_margin))
    _subtitle = 'Key baygaud params'
    print(_mk_row(_subtitle, "Value", "Note", W1, W2, W3, left_margin=left_margin))
    print(_hline(W1, W2, W3, left_margin=left_margin))
    #--------------------------
    if max_ngauss is not None:
        print(_mk_row("max_ngauss (number)", str(max_ngauss), "Maximum Gaussian components",
                      W1, W2, W3, left_margin=left_margin))
    if peak_sn_limit is not None:
        print(_mk_row("peak-flux S/N limit", f"{float(peak_sn_limit):.1f}", "Minimum peak-flux S/N",
                      W1, W2, W3, left_margin=left_margin))
    print(_hline(W1, W2, W3, left_margin=left_margin))
    #--------------------------

    # (3) Runtime (Ray)
    print(_mk_row("Runtime (Ray)", "Value", "", W1, W2, W3, left_margin=left_margin))
    print(_hline(W1, W2, W3, left_margin=left_margin))

    if ray_info:
        def _get(k, default=""):
            return str(ray_info.get(k, default))
        for label in [
            "Ray initialized",
            "Total physical cores",
            "Ray allocated cores",
            "Sampler allocated cores",
            "Numba allocated threads",
            "System memory (GB)",
            "Process memory (GB)",
        ]:
            print(_mk_row(label, _get(label), "", W1, W2, W3, left_margin=left_margin))

    if y_chunk_size is not None:
        print(_mk_row("(y chunk size)", f"({y_chunk_size})", "", W1, W2, W3, left_margin=left_margin))
    if gather_batch is not None:
        print(_mk_row("(gather batch)", f"({gather_batch})", "", W1, W2, W3, left_margin=left_margin))

    print(_hline(W1, W2, W3, left_margin=left_margin))






# Simple YAML loader (prefer PyYAML; fall back to ruamel.yaml)
def _yaml_load(path: str) -> dict:
    try:
        import yaml  # PyYAML
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        from ruamel.yaml import YAML
        y = YAML()
        with open(path, "r", encoding="utf-8") as f:
            return y.load(f) or {}

def print_cube_summary_from_info(
    _params,
    info: dict | None = None,
    *,
    cube_info: dict | None = None,
    yaml_path: str | None = None,       # YAML file path (optional)
    left_margin: int = 0,
    title: str = "Data cube / key params",
    ray_info: dict | None = None,
    prefer: str = "info",               # Merge rule: "info" (default, info overrides cube) or "cube"
) -> None:
    """
    Merge info with (cube_info or YAML) and call print_cube_summary.

    Behavior
    --------
    - If yaml_path is provided, load YAML and merge into cube_info
      (explicit cube_info values override YAML values).
    - prefer="info": in the final merge, values from `info` override cube values (default).
      prefer="cube": cube-side values override `info`.

    Raises
    ------
    ValueError
        If none of (info, cube_info, yaml_path) are provided.
    """
    if (info is None) and (cube_info is None) and (yaml_path is None):
        raise ValueError("Provide at least one of: info, cube_info, yaml_path.")

    # (1) Build cube-side source: YAML --> cube_info (explicit dict has priority)
    base_cube: dict = {}
    if yaml_path:
        base_cube.update(_yaml_load(yaml_path))
    if cube_info:
        base_cube.update(cube_info)

    # (2) Final merge (choose precedence)
    a = info or {}
    b = base_cube
    merged = ({**b, **a} if prefer == "info" else {**a, **b})

    # (3) Helper to get value by synonyms/keys
    def g(keys, default=None):
        if isinstance(keys, str):
            return merged.get(keys, default)
        for k in keys:
            if k in merged:
                return merged[k]
        return default

    # (4) Call print_cube_summary
    print_cube_summary(
        _params,
        naxis1=g("naxis1", 0),
        naxis2=g("naxis2", 0),
        naxis3=g("naxis3", 0),
        vel_min_kms=g(["vel_min_kms", "vel_min"], 0.0),
        vel_max_kms=g(["vel_max_kms", "vel_max"], 0.0),
        cdelt3_ms=g(["cdelt3_ms", "cdelt3"], 0.0),
        vel_unit_label=g("vel_unit_label", "km/s"),
        cdelt3_unit_label=g("cdelt3_unit_label", "m/s"),
        naxis1_s0=g("naxis1_s0"),
        naxis1_e0=g("naxis1_e0"),
        naxis2_s0=g("naxis2_s0"),
        naxis2_e0=g("naxis2_e0"),
        max_ngauss=g("max_ngauss"),
        peak_sn_limit=g("peak_sn_limit"),
        y_chunk_size=g("y_chunk_size"),
        gather_batch=g("gather_batch"),
        left_margin=left_margin,
        title=title,
        ray_info=ray_info,
    )

#-- END OF SUB-ROUTINE____________________________________________________________#