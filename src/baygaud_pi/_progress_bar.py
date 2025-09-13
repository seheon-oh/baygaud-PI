#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _progress_bar.py
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|


import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time, sys, os, math

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# Re-define plotting defaults (tick padding/size/width and base font size)
from matplotlib import rcParams
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'font.size': 30})

#-- END OF SUB-ROUTINE____________________________________________________________#



def _fmt_ddhhmmss(seconds: float) -> str:
    """seconds --> dd:hh:mm:ss (also guards against negative/NaN/inf)"""
    if not (seconds == seconds) or seconds < 0:  # NaN or negative
        seconds = 0.0
    days = int(seconds // 86400)
    seconds -= days * 86400
    hours = int(seconds // 3600)
    seconds -= hours * 3600
    minutes = int(seconds // 60)
    seconds = int(seconds - minutes * 60)
    return f"{days:02d}:{hours:02d}:{minutes:02d}:{seconds:02d}"





def _render_bar_classic(pct: float, width: int = 90, divisions: int = 20) -> str:
    """
    Uniform-tick progress bar.
    - Snap the requested width upward so that inner-1 is a multiple of 'divisions'.
    - Place ticks (:) exactly every L/divisions --> perfectly uniform spacing.
    """
    pct = max(0.0, min(1.0, pct))

    # Snap requested width up to nearest "uniform width": inner = n*divisions + 1
    inner_req = max(10, width)
    n = max(1, math.ceil((inner_req - 1) / divisions))
    inner = n * divisions + 1   # => (inner-1) % divisions == 0
    last  = inner - 1           # arrow '>' candidate position (0..last)

    # Tick placement: exactly at multiples of L/divisions
    L = last                    # effective length used for fill/ticks
    buf = [' '] * inner
    step = L // divisions       # integer, uniform spacing
    for k in range(1, divisions):       # exclude 0%/100%
        p = k * step             # 0 < p < L
        if 0 <= p < last:
            buf[p] = ':'

    # Fill ('-') and arrow ('>') placement
    filled = int(pct * L)        # 0..L
    end = min(filled, last)
    for i in range(end):
        if buf[i] == ' ':
            buf[i] = '-'
    buf[min(filled, last)] = '>'

    return "|" + "".join(buf) + "|"




def _print_progress_classic(done: int, total: int, t0: float,
                             cur_i=None, cur_j0=None, cur_j1=None,
                             state=[None, 0.0, 0.0, 0.0, 0, False, 0],  # [t_last, elapsed, eta, sps, done_last, rendered, printed_lines]
                             width: int = 99, divisions: int = 20,
                             min_interval: float = 0.5, _last_print=[0.0]):
    """
    Print a 5-line block (top border / bar / stats / current tile / bottom border).
    - The "current tile" line keeps the previous indices until a new index arrives.
    - The index line is right-aligned to the right pipe of the left column (ETA column is on the right).
    """
    import sys, time

    MIN_LEFT   = 8
    BASE_RIGHT = 8  # fixed width for percentage column

    # Pad/trim helper
    def _fit(s: str, w: int, *, right=False):
        if len(s) > w: s = s[:w]
        return s.rjust(w) if right else s.ljust(w)

    def _fmt(sec: float) -> str:
        return _fmt_ddhhmmss(sec)

    # Initialize state (+ add last_idx_raw cache: state[7])
    now = time.perf_counter()
    if state[0] is None:
        state[0] = now; state[1] = 0.0; state[2] = 0.0
        state[3] = 0.0; state[4] = done
        state[5] = False; state[6] = 0
    if len(state) < 8:
        state.append(None)             # state[7] = last_idx_raw

    # Rate-limit output
    if (now - _last_print[0] < min_interval) and (done < total):
        return None
    _last_print[0] = now

    # Progress/time bookkeeping
    pct = 1.0 if total <= 0 else (done / max(1, total))
    if (now - state[0] >= 2.0) or (done >= total):
        elapsed = now - t0
        eta = (elapsed / pct - elapsed) if pct > 0 else 0.0
        state[0], state[1], state[2] = now, elapsed, eta
        state[3] = done / max(1e-9, elapsed)

    elapsed = state[1]; eta = state[2]
    sps     = state[3] if state[3] > 0 else (done / max(1e-9, now - t0))
    eltxt   = _fmt(elapsed); etatxt = _fmt(eta)

    # Text fields
    pct_txt = f"{pct*100:6.2f}%"
    # Update when new indices come in; otherwise keep cached state[7]
    if cur_i is not None:
        state[7] = (f"last processed tile: x[{cur_i}]...y[{cur_j0}:{cur_j1}]"
                    if (cur_j0 is not None and cur_j1 is not None)
                    else f"last processed tile: x[{cur_i}]")
    idx_raw = state[7] or ""  # use cache

    stats_text = " | ".join([
        f" {done}/{total} profiles",
        f"{sps:5.2f} profiles/s",
        f"elapsed {eltxt}",
        f"eta {etatxt}",
    ])

    # Width/columns layout

    inner_w = max(20, width - 2)
    right_w = BASE_RIGHT
    left_w  = inner_w - 3 - right_w

    need_inner = max(len(stats_text), MIN_LEFT) + 3 + right_w
    if inner_w < need_inner:
        inner_w = need_inner; width = inner_w + 2; left_w = inner_w - 3 - right_w
    if left_w < MIN_LEFT:
        grow = MIN_LEFT - left_w
        inner_w += grow; width += grow; left_w = MIN_LEFT
    border = "|" + "-" * inner_w + "|"

    # 2) progress bar
    groups = max(1, left_w // 4)
    filled = int(groups * pct + 1e-12)
    tokens = (["---:"] * max(0, filled - 1)) + (["--> "] if filled > 0 else []) + (["  : "] * (groups - filled))
    bar_left = _fit("".join(tokens), left_w)
    line2 = "|" + bar_left + " | " + _fit(pct_txt, right_w, right=True) + " |"

    # 3) stats
    line3 = "|" + _fit(stats_text, left_w) + " | " + _fit("", right_w, right=True) + " |"

    # 4) current tile (right-align inside the left column; if too long, prefix-truncate with '...')
    if idx_raw and len(idx_raw) > left_w:
        idx_show = ("..." + idx_raw[-(left_w - 3):]) if left_w > 3 else idx_raw[-left_w:]
    else:
        idx_show = idx_raw
    line4 = "|" + _fit(idx_show, left_w, right=True) + " | " + _fit("", right_w, right=True) + " |"

    # Print (overwrite previous 5 lines)
    def _out(line: str):
        if len(line) < width: line = line + (" " * (width - len(line)))
        elif len(line) > width: line = line[:width]
        sys.stdout.write("\r" + line + "\x1b[0K\n")


    if state[5] and state[6] == 5:
        sys.stdout.write("\x1b[5A")
    _out(border); _out(line2); _out(line3); _out(line4); _out(border)
    sys.stdout.flush()

    state[5] = True; state[6] = 5
    return line2

#-- END OF SUB-ROUTINE____________________________________________________________#