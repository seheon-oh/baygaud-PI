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
# re-defining plotting defaults
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
    """초 → dd:hh:mm:ss (음수/NaN/inf 방지 포함)"""
    if not (seconds == seconds) or seconds < 0:  # NaN 또는 음수
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
    균등 눈금 진행바.
    - 내부 길이 L = inner-1 이 divisions의 배수가 되도록 width를 상향 스냅(ceil).
    - 눈금(:)은 정확히 L/divisions 간격으로 배치 → 완전 균등.
    """
    pct = max(0.0, min(1.0, pct))

    # 요청 폭을 가장 가까운 '균등 폭'으로 상향 스냅: inner = n*divisions + 1
    inner_req = max(10, width)
    n = max(1, math.ceil((inner_req - 1) / divisions))
    inner = n * divisions + 1   # => (inner-1) % divisions == 0
    last  = inner - 1           # 화살표 '>' 위치 후보 (0..last)

    # 눈금 배치: 정확히 L/divisions 간격
    L = last                    # 채움에 쓰는 유효 길이 (눈금/채움 모두 이 길이 기준)
    buf = [' '] * inner
    step = L // divisions       # 정수, 균등 간격
    for k in range(1, divisions):       # 0%/100% 제외
        p = k * step             # 0< p < L
        if 0 <= p < last:
            buf[p] = ':'

    # 채움(’-’)과 화살표(’>’) 배치
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
    5줄 블록(상/바/통계/인덱스/하) 출력.
    - current tile 라인은 새 인덱스가 오기 전까지 이전 값을 계속 보여줍니다.
    - 인덱스는 ETA가 있는 왼쪽 칼럼의 오른쪽 파이프에 맞춰 우측 정렬됩니다.
    """
    import sys, time

    MIN_LEFT   = 8
    BASE_RIGHT = 8  # 퍼센트 칼럼 고정 폭

    # pad/trim 유틸
    def _fit(s: str, w: int, *, right=False):
        if len(s) > w: s = s[:w]
        return s.rjust(w) if right else s.ljust(w)

    def _fmt(sec: float) -> str:
        return _fmt_ddhhmmss(sec)

    # 상태 초기화 (+ last_idx_raw 캐시 추가: state[7])
    now = time.perf_counter()
    if state[0] is None:
        state[0] = now; state[1] = 0.0; state[2] = 0.0
        state[3] = 0.0; state[4] = done
        state[5] = False; state[6] = 0
    if len(state) < 8:
        state.append(None)             # state[7] = last_idx_raw

    # 출력 빈도 제한
    if (now - _last_print[0] < min_interval) and (done < total):
        return None
    _last_print[0] = now

    # 진행률/시간
    pct = 1.0 if total <= 0 else (done / max(1, total))
    if (now - state[0] >= 2.0) or (done >= total):
        elapsed = now - t0
        eta = (elapsed / pct - elapsed) if pct > 0 else 0.0
        state[0], state[1], state[2] = now, elapsed, eta
        state[3] = done / max(1e-9, elapsed)

    elapsed = state[1]; eta = state[2]
    sps     = state[3] if state[3] > 0 else (done / max(1e-9, now - t0))
    eltxt   = _fmt(elapsed); etatxt = _fmt(eta)

    # 텍스트
    pct_txt = f"{pct*100:6.2f}%"
    # 새 인덱스가 오면 업데이트, 없으면 기존 캐시(state[7]) 유지
    if cur_i is not None:
        state[7] = (f"last processed tile: x[{cur_i}]...y[{cur_j0}:{cur_j1}]"
                    if (cur_j0 is not None and cur_j1 is not None)
                    else f"last processed tile: x[{cur_i}]")
    idx_raw = state[7] or ""  # 캐시 사용

    stats_text = " | ".join([
        f" {done}/{total} profiles",
        f"{sps:5.2f} profiles/s",
        f"elapsed {eltxt}",
        f"eta {etatxt}",
    ])

    # 폭/칼럼

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

    # 2) 바
    groups = max(1, left_w // 4)
    filled = int(groups * pct + 1e-12)
    tokens = (["---:"] * max(0, filled - 1)) + (["--> "] if filled > 0 else []) + (["  : "] * (groups - filled))
    bar_left = _fit("".join(tokens), left_w)
    line2 = "|" + bar_left + " | " + _fit(pct_txt, right_w, right=True) + " |"

    # 3) 통계
    line3 = "|" + _fit(stats_text, left_w) + " | " + _fit("", right_w, right=True) + " |"

    # 4) current tile (왼쪽 칼럼에 우측 정렬, 길면 앞을 '...'로 접기)
    if idx_raw and len(idx_raw) > left_w:
        idx_show = ("..." + idx_raw[-(left_w - 3):]) if left_w > 3 else idx_raw[-left_w:]
    else:
        idx_show = idx_raw
    line4 = "|" + _fit(idx_show, left_w, right=True) + " | " + _fit("", right_w, right=True) + " |"

    # 출력 (덮어쓰기)
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



