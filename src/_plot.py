
#|-----------------------------------------|
# plotting
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
                            state=[None, 0.0, 0.0, 0.0, 0, 0, False],  # [t_last_time, elapsed, eta, s_per_pixel, done_at_last_win]
                            width: int = 90, divisions: int = 20,
                            min_interval: float = 1.0, _last_print=[0.0]):
    """
    타일 1개 완료 때마다 호출.
    - 바/퍼센트: 즉시 갱신
    - elapsed/eta, profiles/s: 2초마다(또는 완료 시) 갱신
    - 출력 최소 간격(min_interval)로 콘솔 부하 감소
    """
    # --- 추가: 내부 캐시 필드 확장 (기존 state는 그대로 유지, 뒤에 붙여서 사용) ---
    # state[7] : last_bar_ticks  (int)
    # state[8] : last_bar_str    (str)
    # state[9] : last_elapsed_txt(str)
    # state[10]: last_eta_txt    (str)

    now = time.perf_counter()
    if state[0] is None:
        state[0] = now      # t_last_time (ETA/elapsed 갱신 기준)
        state[1] = 0.0      # elapsed
        state[2] = 0.0      # eta
        state[3] = 0.0      # profiles_per_s
        state[4] = done     # done_at_last_win
        # 캐시 초기화
        if len(state) < 11:
            state.extend([ -1, "", "", "" ])  # [7..10]

    # 너무 잦은 출력 방지
    if (now - _last_print[0] < min_interval) and (done < total):
        return None
    _last_print[0] = now

    pct = 1.0 if total == 0 else (done / total)

    # elapsed / eta 갱신 (2초마다)
    if (now - state[0] >= 2.0) or (done >= total):
        elapsed = now - t0
        eta = (elapsed / pct - elapsed) if pct > 0.0 else 0.0
        state[0] = now
        state[1] = elapsed
        state[2] = eta
        # pixel/s (최근 1초 창)
        dd = max(0, done - state[4])
        dt = max(1e-9, 1.0)  # 창을 1초로 가정
        if dd > 0:
            state[3] = dd / dt
        state[4] = done
        # --- 추가: 포맷 문자열 캐싱 (매 호출마다 포맷하지 않도록) ---
        state[9]  = _fmt_ddhhmmss(elapsed)  # last_elapsed_txt
        state[10] = _fmt_ddhhmmss(eta)      # last_eta_txt

    elapsed = state[1]
    eta     = state[2]
    #spt     = state[3] if state[3] > 0 else ( max(1, done) / (now - t0) )
    spt     = done / (now - t0)

    # --- 추가: 프로그레스 바 렌더링 최소화 (tick 변화시에만 다시 그림) ---
    # divisions 칸 중 채워진 칸 수
    filled = int(pct * divisions + 1e-12)
    if filled != state[7]:
        # _render_bar_classic가 있으면 사용, 없으면 빠른 기본 바 생성
        try:
            bar = _render_bar_classic(pct, width=width, divisions=divisions)
        except NameError:
            # 빠른 기본 바: [====.....] 형태
            left  = "=" * filled
            right = "." * (divisions - filled)
            bar   = f"[{left}{right}]".ljust(width)
        state[7] = filled
        state[8] = bar
    else:
        bar = state[8]

    pct_txt     = f"{pct*100:5.2f}%"
    elapsed_txt = state[9] if state[9] else _fmt_ddhhmmss(elapsed)
    eta_txt     = state[10] if state[10] else _fmt_ddhhmmss(eta)

    idx_txt = ""
    if cur_i is not None:
        idx_txt = f"x[{cur_i}]"
        if (cur_j0 is not None) and (cur_j1 is not None):
            idx_txt += f" | y[{cur_j0}:{cur_j1}] |"

    msg = (f"\r {bar} {pct_txt} | {done}/{total} profiles "
           f"| {spt:5.2f} profiles/s | elapsed {elapsed_txt} | eta {eta_txt} | {idx_txt}")
    
    # --- 여기부터: 다음 줄에 '|'를 붙여서 누적 갱신 ---
    tick_count = state[5] + 1
    state[5] = tick_count
    tick_line = " |"

    if not state[6]:
        # 커서를 위로 한 줄 올려서(ANSI), 두 줄을 다시 그린다
        sys.stdout.write("\x1b[1A")          # cursor up 1 line
        # 한 번에 write하여 flush 호출 수 최소화
        sys.stdout.write("\r" + msg + "\n" + tick_line)
        sys.stdout.flush()
        # 필요 시 최초 1회 렌더링 이후 플래그 on (원래 코드와 동작 동일하게 유지하고 싶으면 주석 처리)
        # state[6] = True

    return msg


