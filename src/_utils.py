import numpy as np
from numba import njit

# ---------------------------------------------- #
# 유틸: 중앙값, 표준편차, 분위수(Numba 호환)       #
# ---------------------------------------------- #

@njit(cache=True, fastmath=True)
def _median_nb(a):
    """a(1D)를 복사-정렬해 중앙값 반환"""
    n = a.size
    if n == 0:
        return 0.0
    b = a.copy()
    b.sort()
    if (n & 1) == 1:
        return b[n // 2]
    else:
        return 0.5 * (b[n // 2 - 1] + b[n // 2])

@njit(cache=True, fastmath=True)
def _std_nb(a):
    """a(1D)의 표준편차(MLE 아님; N으로 나눔)"""
    n = a.size
    if n == 0:
        return 0.0
    s = 0.0
    for i in range(n):
        s += a[i]
    mu = s / n
    v = 0.0
    for i in range(n):
        d = a[i] - mu
        v += d * d
    return np.sqrt(v / n)

@njit(cache=True, fastmath=True)
def _mad_scale_nb(a):
    """
    a(1D)의 중앙값 m, 그리고 1.4826*MAD 기반 robust σ 추정치 반환.
    MAD==0이면 표준편차로 폴백.
    """
    m = _median_nb(a)
    n = a.size
    dev = np.empty(n, dtype=a.dtype)
    for i in range(n):
        d = a[i] - m
        dev[i] = d if d >= 0.0 else -d
    mad = _median_nb(dev)
    if mad > 0.0:
        s = 1.4826 * mad
        return m, s
    else:
        return m, max(_std_nb(a), 1e-12)

@njit(cache=True, fastmath=True)
def _quantile_nb(a, q):
    """
    a(1D)의 q-분위수(선형보간). a는 복사-정렬.
    q∈[0,1].
    """
    n = a.size
    if n == 0:
        return 0.0
    b = a.copy()
    b.sort()
    pos = q * (n - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if hi == lo:
        return b[lo]
    frac = pos - lo
    return b[lo] * (1.0 - frac) + b[hi] * frac

# ---------------------------------------------- #
# 보조: comps를 peakflux 내림차순으로 정렬하는 인덱스 #
# (Numba 호환 selection sort; ncomp가 작아 부담 적음) #
# ---------------------------------------------- #

@njit(cache=True, fastmath=True)
def _argsort_desc_by_peak(comps):
    n = comps.shape[0]
    idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        idx[i] = i
    # selection sort by comps[:,2] (peakflux) desc
    for i in range(n):
        best = i
        for j in range(i+1, n):
            if comps[idx[j], 2] > comps[idx[best], 2]:
                best = j
        if best != i:
            tmp = idx[i]
            idx[i] = idx[best]
            idx[best] = tmp
    return idx

# -------------------------------------------------------- #
# 핵심: 시드 여러 개 → ±kσ 배제(적응형) → robust bg/rms   #
# comps 순서: [x, sigma, peakflux]                         #
# -------------------------------------------------------- #

@njit(cache=True, fastmath=True)
def robust_bg_rms_from_seeds_norm_adaptive_nb(
    f_norm, x_norm,
    comps,               # shape (ncomp, 3) = [x_center, sigma, peakflux] in norm units
    exclude_k=5.0,       # 기본 배제 폭 (±kσ)
    k_min=2.0,           # k 축소 하한
    shrink_factor=0.85,  # k ← k * shrink_factor (0.5~0.95 권장)
    max_shrink_steps=6,  # 축소 최대 횟수
    clip_sigma=3.0,      # sigma-clipping 강도
    max_iter=8,
    min_bg_frac=0.25,    # 전체 중 최소 배경 비율
    emission_positive=True  # 방출선 가정: 상단만 강하게 컷
):
    """
    반환: (bg, rms) in normalized units.

    절차:
      1) 모든 컴포넌트에 대해 ±kσ 배제 마스크 생성.
      2) 배경 비율 < min_bg_frac 이면 k를 단계적으로 축소(shrink).
      3) 그래도 부족하면 peakflux 내림차순으로 컴포넌트를 훑으며
         - 강한 성분부터 배제 시도
         - 배제 후 배경이 너무 작아지면 '그 성분은 배제하지 않음' (약한 성분은 배경으로 흡수)
      4) 남은 배경 샘플로 MAD + σ-clipping으로 bg/rms 추정.
      5) 배경이 지나치게 적으면 분위수(하위 20~30%) 폴백.
    """
    N = f_norm.size
    if N == 0:
        return 0.0, 1.0

    # 0) 배경 최소 개수
    min_bg_cnt = int(min_bg_frac * N)
    if min_bg_cnt < 8:
        min_bg_cnt = 8

    # 1) k를 줄여가며 전체 배제 시도
    ncomp = comps.shape[0]
    k_cur = exclude_k

    for _ in range(max_shrink_steps + 1):
        # 전부 배제
        mask = np.ones(N, dtype=np.uint8)
        for c in range(ncomp):
            xc = comps[c, 0]
            sc = comps[c, 1]
            if sc <= 0.0:
                continue
            left  = xc - k_cur * sc
            right = xc + k_cur * sc
            for i in range(N):
                xi = x_norm[i]
                if (xi >= left) and (xi <= right):
                    mask[i] = 0

        # 배경 샘플 수
        cnt = 0
        for i in range(N):
            if mask[i] == 1:
                cnt += 1

        if cnt >= min_bg_cnt:
            break  # 충분히 배경 있음 → 이 마스크 사용
        # 배경 부족 → k 축소
        if k_cur <= k_min:
            # 더는 못 줄임 → 선택적 배제 전략으로 넘어감
            break
        k_cur = k_cur * shrink_factor
        if k_cur < k_min:
            k_cur = k_min

    # 2) 선택적 배제 전략 (필요한 경우에만)
    # 위 루프에서 충분하면 mask 이미 준비됨
    if cnt < min_bg_cnt:
        # mask를 전부 True로 초기화
        for i in range(N):
            mask[i] = 1
        cnt = N

        # peakflux 내림차순 인덱스
        order = _argsort_desc_by_peak(comps)

        # 강한 성분부터 배제 시도하되, 배경이 임계 이하로 줄어드는 성분은 스킵
        for t in range(ncomp):
            c = order[t]
            xc = comps[c, 0]
            sc = comps[c, 1]
            if sc <= 0.0:
                continue
            left  = xc - k_cur * sc
            right = xc + k_cur * sc

            # 임시로 적용해 보고 배경 수 확인
            # 적용 전 남은 개수
            old_cnt = cnt
            # 몇 개가 새로 제외될지 세고, 일단 플래그만 세팅
            new_excluded = 0
            for i in range(N):
                if mask[i] == 1:
                    xi = x_norm[i]
                    if (xi >= left) and (xi <= right):
                        new_excluded += 1

            # 적용 후 남을 개수
            new_cnt = old_cnt - new_excluded

            if new_cnt >= min_bg_cnt:
                # 실제 적용
                for i in range(N):
                    if mask[i] == 1:
                        xi = x_norm[i]
                        if (xi >= left) and (xi <= right):
                            mask[i] = 0
                cnt = new_cnt
            else:
                # 이 성분은 배제하지 않음 (약한 성분일수록 스킵될 가능성 큼)
                continue

    # 3) 최종 배경 벡터 추출
    if cnt < min_bg_cnt:
        # 그래도 부족하면 전 채널로 진행 (다음 단계에서 분위수 폴백)
        fb = f_norm.copy()
        cnt = N
    else:
        fb = np.empty(cnt, dtype=f_norm.dtype)
        j = 0
        for i in range(N):
            if mask[i] == 1:
                fb[j] = f_norm[i]
                j += 1

    # 4) 초기 robust 통계 (MAD)
    m, s = _mad_scale_nb(fb)

    # 5) 반복 σ-clipping
    for _ in range(max_iter):
        if emission_positive:
            # 상단만 컷: (f - m) < clip*s
            # 새로 남을 개수 세기
            cnt2 = 0
            for i in range(cnt):
                if (fb[i] - m) < (clip_sigma * s):
                    cnt2 += 1

            if cnt2 < min_bg_cnt:
                # 분위수 폴백 (하단 20% 또는 30%)
                q = 0.20 if N >= 20 else 0.30
                cutoff = _quantile_nb(fb, q)
                tmp_cnt = 0
                for i in range(cnt):
                    if fb[i] <= cutoff:
                        tmp_cnt += 1
                if tmp_cnt < 5:
                    k = int(0.3 * cnt)
                    if k < 5:
                        k = 5
                    bb = fb.copy()
                    bb.sort()
                    if k > bb.size:
                        k = bb.size
                    fb2 = bb[:k]
                else:
                    fb2 = np.empty(tmp_cnt, dtype=fb.dtype)
                    jj = 0
                    for i in range(cnt):
                        if fb[i] <= cutoff:
                            fb2[jj] = fb[i]
                            jj += 1
                m2, s2 = _mad_scale_nb(fb2)
                return m2, s2

            # 정상 적용
            fb2 = np.empty(cnt2, dtype=fb.dtype)
            jj = 0
            for i in range(cnt):
                if (fb[i] - m) < (clip_sigma * s):
                    fb2[jj] = fb[i]
                    jj += 1
        else:
            # 양쪽 컷: |f - m| < clip*s
            cnt2 = 0
            for i in range(cnt):
                d = fb[i] - m
                if (d if d >= 0.0 else -d) < (clip_sigma * s):
                    cnt2 += 1

            if cnt2 < min_bg_cnt:
                q = 0.20 if N >= 20 else 0.30
                cutoff = _quantile_nb(fb, q)
                tmp_cnt = 0
                for i in range(cnt):
                    if fb[i] <= cutoff:
                        tmp_cnt += 1
                if tmp_cnt < 5:
                    k = int(0.3 * cnt)
                    if k < 5:
                        k = 5
                    bb = fb.copy()
                    bb.sort()
                    if k > bb.size:
                        k = bb.size
                    fb2 = bb[:k]
                else:
                    fb2 = np.empty(tmp_cnt, dtype=fb.dtype)
                    jj = 0
                    for i in range(cnt):
                        if fb[i] <= cutoff:
                            fb2[jj] = fb[i]
                            jj += 1
                m2, s2 = _mad_scale_nb(fb2)
                return m2, s2

            fb2 = np.empty(cnt2, dtype=fb.dtype)
            jj = 0
            for i in range(cnt):
                d = fb[i] - m
                if (d if d >= 0.0 else -d) < (clip_sigma * s):
                    fb2[jj] = fb[i]
                    jj += 1

        # 수렴 체크(간단): 길이 동일 + 중앙값 변화 미미 → 종료
        if fb2.size == fb.size:
            m2, s2 = _mad_scale_nb(fb2)
            if np.abs(m2 - m) < 1e-12 and np.abs(s2 - s) < 1e-12:
                return m2, s2
            m, s = m2, s2
            fb = fb2
            cnt = fb.size
            continue

        # 갱신
        fb = fb2
        cnt = fb.size
        m, s = _mad_scale_nb(fb)

    return m, s



def robust_bg_rms_from_seed_dict_norm(
    x_norm, f_norm, seeds,
    *,
    # 라인-프리 마스크 파라미터 (적응형)
    exclude_k=5.0,        # 시작은 ±5σ 배제
    k_min=2.0,            # 최소 ±2σ까지 축소 허용
    shrink_factor=0.85,   # k ← k*0.85 단계 축소
    max_shrink_steps=6,   # 최대 6단계 축소
    min_bg_frac=0.25,     # 배경 채널 최소 비율
    # robust σ-클리핑
    clip_sigma=3.0,
    max_iter=8,
    emission_positive=True # 방출선이면 상단만 클리핑
):
    """
    f_norm, x_norm: 정규화 스펙트럼/축(0~1)
    seeds: {'ncomp', 'components', 'bg', 'rms', ...}, components는 [x, sigma, peak] (정규화 단위)
    반환: (bg_norm, rms_norm)
    """
    # components 꺼내기 (+ 유효성)
    comps = seeds.get("components", None)
    if comps is None or np.size(comps) == 0:
        comps_arr = np.zeros((0, 3), dtype=np.float64)
    else:
        comps_arr = np.ascontiguousarray(np.asarray(comps, dtype=np.float64))
        if comps_arr.ndim != 2 or comps_arr.shape[1] < 3:
            raise ValueError("seeds['components'] must be (ncomp, 3) = [x, sigma, peak].")
        # [x, sigma, peak]만 사용
        comps_arr = comps_arr[:, :3]

    # Numba 함수는 float64 1D 배열을 기대
    f64 = np.ascontiguousarray(f_norm, dtype=np.float64).ravel()
    x64 = np.ascontiguousarray(x_norm, dtype=np.float64).ravel()

    # ← 여기서 이전에 드린 Numba 핵심 함수 호출
    bg_norm, rms_norm = robust_bg_rms_from_seeds_norm_adaptive_nb(
        f64, x64, comps_arr,
        exclude_k=exclude_k, k_min=k_min,
        shrink_factor=shrink_factor, max_shrink_steps=max_shrink_steps,
        clip_sigma=clip_sigma, max_iter=max_iter,
        min_bg_frac=min_bg_frac, emission_positive=emission_positive
    )
    return bg_norm, rms_norm


def update_bg_rms_to_seeds(seeds, bg_norm, rms_norm, *,
                          inplace=True,
                          bg_clip=(0.0, 1.0),
                          rms_floor=1e-9):
    """
    이미 계산된 bg_norm, rms_norm을 시드 dict에 반영.
    seeds 포맷: {"ncomp": int, "components": (n,3), "bg": float, "rms": float, "indices": ..., "debug": ...}

    Parameters
    ----------
    seeds : dict
        _gaussian_seeds (정규화 단위)
    bg_norm : float
        정규화 배경 (0~1 권장)
    rms_norm : float
        정규화 RMS (>=0)
    inplace : bool
        True면 seeds를 제자리 업데이트, False면 사본 반환
    bg_clip : (lo, hi)
        bg를 클리핑할 구간
    rms_floor : float
        너무 작은 RMS 보호용 하한

    Returns
    -------
    dict
        업데이트된 시드 dict (inplace=True면 원본이 곧 반환값)
    """
    tgt = seeds if inplace else dict(seeds)

    # 안전 처리
    bg  = float(bg_norm) if np.isfinite(bg_norm) else float(tgt.get("bg", 0.0))
    rms = float(rms_norm) if np.isfinite(rms_norm) else float(tgt.get("rms", 0.1))

    # 클리핑/하한
    bg  = float(np.clip(bg, bg_clip[0], bg_clip[1]))
    rms = float(max(rms, rms_floor))

    # 디버그 로그 보존/추가
    dbg_old = tgt.get("debug", {})
    dbg = dict(dbg_old) if isinstance(dbg_old, dict) else {}
    dbg.update({
        "bg_rms_updated": True,
        "bg_old": float(seeds.get("bg", np.nan)),
        "rms_old": float(seeds.get("rms", np.nan)),
        "bg_new": bg,
        "rms_new": rms,
    })

    # 업데이트
    tgt["bg"] = bg
    tgt["rms"] = rms
    tgt["debug"] = dbg

    return tgt





