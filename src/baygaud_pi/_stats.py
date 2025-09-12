#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _stats.py
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|






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














#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# parameters are sigma, bg, _x01, _std01, _p01, _x02, _std02, _p02...

@njit(cache=True, fastmath=True)  # parallel 제거
def _gaussian_sum_norm(x, theta, ngauss):
    """
    (벡터화) model = bg + Σ amp * exp(-0.5*((x-mu)/sig)^2)
    """
    bg = float(theta[1])
    model = np.full(x.size, bg, dtype=np.float64)
    for m in range(ngauss):
        mu  = float(theta[2 + 3*m])
        sig = float(theta[3 + 3*m])
        amp = float(theta[4 + 3*m])
        invs = 1.0 / sig
        # x 전 구간에 대해 한 번에 계산
        dx = (x - mu) * invs
        model += amp * np.exp(-0.5 * dx * dx)
    return model

@njit(cache=True, fastmath=True)  # parallel 제거
def _rms_of_residual(data_norm, model):
    """
    (벡터화) sqrt(mean((data-model)^2))
    """
    r = data_norm - model
    return np.sqrt(np.mean(r * r))

@njit(cache=True, fastmath=True)  # parallel 제거
def _neg_half_chi2(data_norm, model, inv_sigma2):
    """
    (벡터화) -0.5 * Σ r^2 / σ^2
    """
    r = data_norm - model
    return -0.5 * inv_sigma2 * np.sum(r * r)


# =========================
# Robust RMS 헬퍼 (Numba)
# =========================

@njit(cache=True, fastmath=True)
def _median_nb(a):
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
    n = a.size
    if n == 0:
        return 0.0
    mu = np.mean(a)
    v = np.mean((a - mu) * (a - mu))
    return np.sqrt(v)

@njit(cache=True, fastmath=True)
def _mad_sigma_nb(a):
    """
    1. 중앙값 m
    2. MAD = median(|a - m|)
    3. robust σ ≈ 1.4826 * MAD (MAD==0이면 std로 폴백)
    """
    m = _median_nb(a)
    dev = np.abs(a - m)
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

@njit(cache=True, fastmath=True)
def _argsort_desc_abs_amp_from_theta(theta, ngauss):
    """
    theta에서 각 컴포넌트의 |amp| 를 기준으로 내림차순 정렬한 인덱스 반환.
    (선택 정렬)
    """
    n = int(ngauss)
    idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        idx[i] = i
    for i in range(n):
        best = i
        a_best = theta[4 + 3*idx[best]]
        if a_best < 0.0:
            a_best = -a_best
        for j in range(i+1, n):
            a_j = theta[4 + 3*idx[j]]
            if a_j < 0.0:
                a_j = -a_j
            if a_j > a_best:
                best = j
                a_best = a_j
        if best != i:
            tmp = idx[i]; idx[i] = idx[best]; idx[best] = tmp
    return idx

@njit(cache=True, fastmath=True)
def _build_mask_excluding_windows(x, theta, ngauss, k):
    """
    (부분 벡터화)
    각 성분의 [mu-k*sig, mu+k*sig] 윈도우를 계산하고,
    해당 구간에 포함되는 채널을 0으로 마크.
    """
    n = x.size
    mask = np.ones(n, dtype=np.uint8)
    for m in range(ngauss):
        mu  = float(theta[2 + 3*m])
        sig = float(theta[3 + 3*m])
        if sig <= 0.0:
            continue
        left  = mu - k * sig
        right = mu + k * sig
        win = (x >= left) & (x <= right)  # x 전구간 벡터 비교
        for i in range(n):
            if win[i]:
                mask[i] = 0
    return mask

@njit(cache=True, fastmath=True)
def _count_true(mask):
    return np.sum(mask == 1)

@njit(cache=True, fastmath=True)
def _count_exclusion_if_applied(mask, x, mu, sig, k):
    """
    적용 전, 현재 mask에서 (mu, sig, k) 윈도우에 의해 새로 제외될 개수만 계산.
    (벡터 연산)
    """
    left  = mu - k * sig
    right = mu + k * sig
    win = (x >= left) & (x <= right)
    # mask==1 AND win==True
    keep = (mask == 1) & win
    return np.sum(keep)

@njit(cache=True, fastmath=True)
def _apply_exclusion(mask, x, mu, sig, k):
    """
    실제 배제 적용(마스크 갱신). 반환: 새로 제외된 개수.
    """
    left  = mu - k * sig
    right = mu + k * sig
    win = (x >= left) & (x <= right)
    new_ex = 0
    for i in range(mask.size):
        if (mask[i] == 1) and win[i]:
            mask[i] = 0
            new_ex += 1
    return new_ex

@njit(cache=True, fastmath=True)
def _adaptive_linefree_mask_from_theta(
    x, theta, ngauss,
    min_bg_frac=0.25,
    k_init=5.0, k_min=2.0, shrink_factor=0.85, max_shrink_steps=6
):
    """
    1) 모든 컴포넌트를 ±k_init σ로 배제 → 배경 비율 부족 시 k를 축소
    2) 그래도 부족하면, |amp| 내림차순으로 성분을 훑으며 '선택적 배제'
       (강한 성분만 배제하고, 약한 성분은 배경에 흡수)
    반환: (mask, bg_count_min)
      - mask: uint8(0/1)
      - bg_count_min: 배경 최소 필요 개수(참고용)
    """
    N = x.size
    min_bg_cnt = int(min_bg_frac * N)
    if min_bg_cnt < 8:
        min_bg_cnt = 8

    # 1) k를 줄여가며 전 성분 배제 시도
    k_cur = k_init
    for _ in range(max_shrink_steps + 1):
        mask = _build_mask_excluding_windows(x, theta, ngauss, k_cur)
        cnt = _count_true(mask)
        if cnt >= min_bg_cnt:
            return mask, min_bg_cnt
        if k_cur <= k_min:
            break
        k_cur *= shrink_factor
        if k_cur < k_min:
            k_cur = k_min

    # 2) 선택적 배제: 전부 True에서 시작
    mask = np.ones(N, dtype=np.uint8)
    cnt  = N
    order = _argsort_desc_abs_amp_from_theta(theta, ngauss)

    for u in range(ngauss):
        m = int(order[u])
        mu  = float(theta[2 + 3*m])
        sig = float(theta[3 + 3*m])
        if sig <= 0.0:
            continue

        # 적용 전 예측: 배경이 충분히 남는다면 실제 적용
        delta = _count_exclusion_if_applied(mask, x, mu, sig, k_cur)
        if (cnt - delta) >= min_bg_cnt:
            # 실제 적용
            _ = _apply_exclusion(mask, x, mu, sig, k_cur)
            cnt -= delta
        # 아니라면 이 성분은 배제하지 않고 넘어감

    if cnt < min_bg_cnt:
        mask[:] = 1  # 최후: 전채널 사용(이후 robust 단계가 outlier 억제)

    return mask, min_bg_cnt

@njit(cache=True, fastmath=True)
def _robust_rms_from_residual_with_mask(resid, mask, min_bg_cnt, clip_sigma=3.0, max_iter=8):
    """
    resid(잔차)에서 mask==1인 값들로 robust σ를 추정.
    - 초기: MAD 기반(m, s)
    - 반복: 두쪽 σ-clipping (|r - m| < clip_sigma*s)
    - 표본 부족 시, 분위수 폴백(하위 20% 또는 30%)
    반환: robust_rms
    (가능 부분 벡터화)
    """
    # 마스크 적용
    use = (mask == 1)
    if np.sum(use) == 0:
        fb = resid.copy()
    else:
        # boolean 인덱싱 (Numba 지원)
        fb = resid[use]

    m, s = _mad_sigma_nb(fb)

    for _ in range(max_iter):
        keep = np.abs(fb - m) < (clip_sigma * s)
        cnt2 = np.sum(keep)

        if cnt2 < min_bg_cnt:
            q = 0.20 if fb.size >= 20 else 0.30
            cutoff = _quantile_nb(fb, q)
            keep2 = fb <= cutoff
            tmp_cnt = np.sum(keep2)
            if tmp_cnt < 5:
                k = int(0.3 * fb.size)
                if k < 5:
                    k = 5
                bb = fb.copy()
                bb.sort()
                if k > bb.size:
                    k = bb.size
                fb2 = bb[:k]
            else:
                fb2 = fb[keep2]
            _, s2 = _mad_sigma_nb(fb2)
            return s2

        fb2 = fb[keep]

        # 수렴 체크
        if fb2.size == fb.size:
            m2, s2 = _mad_sigma_nb(fb2)
            if (np.abs(m2 - m) < 1e-12) and (np.abs(s2 - s) < 1e-12):
                return s2
            m, s = m2, s2
            fb = fb2
            continue

        fb = fb2
        m, s = _mad_sigma_nb(fb)

    return s


@njit(cache=True, fastmath=True)
def _little_derive_rms_core(profile, x, f_min, f_max, ngauss, theta):
    """
    profile: 원시 스펙트럼 (cube[:, j, i])
    x: 채널 축 (정규화 동일 축)
    f_min, f_max: 정규화에 사용된 min/max
    ngauss: 사용 가우시안 개수
    theta: dynesty가 반환한 (정규화 스케일) 파라미터 벡터 (길이 >= 3*ngauss+2)

    변경점:
    - 기존: 모든 채널 잔차로 RMS 계산
    - 신규: 가우시안 성분 주변(±kσ) 라인 채널을 '적응형'으로 제외하여
            라인-프리 채널에서 robust(MAD + σ-clipping) RMS 추정
    (가능 부분 벡터화)
    """
    # 정규화 스펙트럼/모델 (벡터화)
    scale = (f_max - f_min)
    inv_scale = 1.0 / scale
    data_norm = (profile - f_min) * inv_scale

    model = _gaussian_sum_norm(x, theta, ngauss)

    # 잔차 (벡터화)
    resid = data_norm - model

    # 라인-프리 마스크(적응형)
    # 튠 가능한 상수(필요시 조정):
    min_bg_frac   = 0.25   # 최소 배경 비율
    k_init        = 5.0    # 초기 배제 폭(±kσ)
    k_min         = 2.0    # 최소 배제 폭
    shrink_factor = 0.85   # k 축소 비율
    max_shrink    = 6      # k 축소 최대 단계

    mask, min_bg_cnt = _adaptive_linefree_mask_from_theta(
        x, theta, int(ngauss),
        min_bg_frac=min_bg_frac,
        k_init=k_init, k_min=k_min,
        shrink_factor=shrink_factor, max_shrink_steps=max_shrink
    )

    # robust RMS (잔차 기반, 양쪽 σ-클리핑)
    clip_sigma = 3.0
    max_iter   = 8
    robust_rms = _robust_rms_from_residual_with_mask(
        resid, mask, min_bg_cnt, clip_sigma=clip_sigma, max_iter=max_iter 
    ) # normalised units
    #robust_rms_phys = robust_rms * (f_max - f_min)

    # should return rms in normalised units
    return robust_rms


def little_derive_rms(input_cube, i, j, x, f_min, f_max, ngauss, theta):
    """
    기존 코드가 호출하던 시그니처 유지. 내부는 JIT 코어 호출.
    theta는 정규화 스케일 파라미터(단위변환 전)를 넣어야 합니다.
    """
    # C-contiguous 보장(필요 시)
    prof = np.ascontiguousarray(input_cube[:, j, i], dtype=np.float64)
    x64  = np.ascontiguousarray(x, dtype=np.float64)
    th64 = np.ascontiguousarray(theta[:(3*ngauss+2)], dtype=np.float64)
    return _little_derive_rms_core(prof, x64, float(f_min), float(f_max), int(ngauss), th64)





#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# UNIT CONVERSION
def convert_units_norm_to_phys(
    gfit_results: np.ndarray,
    j: int,                     # 프로파일 index (y축 index 등)
    k: int,                     # 다중 가우스 모델 차수-1 (즉, 가우스 개수 = k+1)
    f_min: float, f_max: float, # 해당 프로파일의 플럭스 정규화 기준
    vel_min: float, vel_max: float,  # 물리 속도 범위 (항상 vel_min < vel_max 권장)
    cdelt3: float,              # 채널 축 증감 방향 (음수면 내림차순)
    max_ngauss: int             # 전체 최대 가우스 개수 (_max_ngauss)
) -> None:
    """
    gfit_results[j, k, :]의 정규화 결과를 물리 단위로 *제자리(in-place)* 변환한다.
    - 파라미터 구역: 길이 nparams = 2 + 3*(k+1)
      [model_sigma, bg, g1_x, g1_sigma, g1_peak, g2_x, g2_sigma, g2_peak, ...]
    - 에러 구역: 동일 길이 nparams, 바로 뒤에 이어짐.
      [model_sigma_e, bg_e, g1_x_e, g1_sigma_e, g1_peak_e, ...]
    - 그 뒤 부가정보들이 있으나 여기서는 배경/피크/속도/분산/에러 및 rms만 변환.

    주의: 함수는 gfit_results를 in-place로 수정하며 반환값은 없습니다.
    """
    #________________________________________________________________________________________|
    #|---------------------------------------------------------------------------------------|
    # unit conversion
    # sigma-flux --> data cube units
    gfit_results[j, k, 0] *= (f_max - f_min)  # sigma-flux to data cube units

    # background --> data cube units
    gfit_results[j, k, 1] = gfit_results[j, k, 1] * (f_max - f_min) + f_min  # background to data cube units

    # background error --> data cube units
    gfit_results[j, k, 6 + 3*k] *= (f_max - f_min)  # background error to data cube units

    # vectorization
    m_indices = np.arange(k + 1)

    velocity_indices               = 2 + 3 * m_indices
    velocity_dispersion_indices    = 3 + 3 * m_indices
    peak_flux_indices              = 4 + 3 * m_indices

    velocity_e_indices             = 7 + 3 * (m_indices + k)
    velocity_dispersion_e_indices  = 8 + 3 * (m_indices + k)
    flux_e_indices                 = 9 + 3 * (m_indices + k)

    #________________________________________________________________________________________|
    # UNIT CONVERSION
    #________________________________________________________________________________________|
    # velocity, velocity-dispersion --> km/s
    if cdelt3 > 0:  # if velocity axis is with increasing order
        gfit_results[j, k, velocity_indices] = (
            gfit_results[j, k, velocity_indices] * (vel_max - vel_min) + vel_min  # velocity
        )
    else:  # if velocity axis is with decreasing order
        gfit_results[j, k, velocity_indices] = (
            gfit_results[j, k, velocity_indices] * (vel_min - vel_max) + vel_max  # velocity
        )

    gfit_results[j, k, velocity_dispersion_indices] *= (vel_max - vel_min)  # velocity-dispersion

    #________________________________________________________________________________________|
    # peak flux --> data cube units
    # peak flux --> data cube units : (_f_max - _bg_flux) should be used for scaling as the normalised peak flux is from the bg
    #gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) + _f_min # flux
    gfit_results[j, k, peak_flux_indices] *= (f_max - f_min)  # flux

    #________________________________________________________________________________________|
    # velocity-e, velocity-dispersion-e --> km/s
    gfit_results[j, k, velocity_e_indices]            *= (vel_max - vel_min)  # velocity-e
    gfit_results[j, k, velocity_dispersion_e_indices] *= (vel_max - vel_min)  # velocity-dispersion-e

    # flux-e --> data cube units
    gfit_results[j, k, flux_e_indices] *= (f_max - f_min)  # flux-e

    # lastly put rms 
    # 위치: 2*(3*max_ngauss + 2) + k  (각 k 모델의 rms 저장 슬롯)
    gfit_results[j, k, 2 * (3 * max_ngauss + 2) + k] *= (f_max - f_min)  # rms-(k+1)gfit
#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
    


















#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# derive rms of a profile via ngfit 
def derive_rms_npoints(_inputDataCube, _cube_mask_2d, _x, _params, ngauss):

    ndim = 3*ngauss + 2
    nparams = ndim

    naxis1 = int(_params['naxis1'])
    naxis2 = int(_params['naxis2'])

    naxis1_s0 = int(_params['naxis1_s0'])
    naxis1_e0 = int(_params['naxis1_e0'])
    naxis2_s0 = int(_params['naxis2_s0'])
    naxis2_e0 = int(_params['naxis2_e0'])

    naxis1_seg = naxis1_e0 - naxis1_s0
    naxis2_seg = naxis2_e0 - naxis2_s0

    nsteps_x = int(_params['nsteps_x_rms'])
    nsteps_y = int(_params['nsteps_y_rms'])

    _rms = np.zeros(nsteps_x*nsteps_y+1, dtype=np.float32)
    _bg = np.zeros(nsteps_x*nsteps_y+1, dtype=np.float32)
    # prior arrays for the single Gaussian fit
    gfit_priors_init = np.zeros(2*5, dtype=np.float32)
    _x_boundaries = np.full(2*ngauss, fill_value=-1E11, dtype=np.float32)
    #gfit_priors_init = [sig1, bg1, x1, std1, p1, sig2, bg2, x2, std2, p2]
    gfit_priors_init = [0.0, 0.0, 0.01, 0.01, 0.01, 0.5, 0.6, 0.99, 0.6, 1.01]

    k=0
    for x in range(0, nsteps_x):
        for y in range(0, nsteps_y):

            i = int(0.5*(naxis1_seg/nsteps_x) + x*(naxis1_seg/nsteps_x)) + naxis1_s0
            j = int(0.5*(naxis2_seg/nsteps_y) + y*(naxis2_seg/nsteps_y)) + naxis2_s0

            print("[--> measure background rms at (i:%d j:%d)...]" % (i, j))

            if(_cube_mask_2d[j, i] > 0 and not np.isnan(_inputDataCube[:, j, i]).any()): # if not masked: 

                _f_max = np.max(_inputDataCube[:, j, i]) # peak flux : being used for normalization
                _f_min = np.min(_inputDataCube[:, j, i]) # lowest flux : being used for normalization
    
                if(ndim * (ndim + 1) // 2 > _params['nlive']):
                    _params['nlive'] = 1 + ndim * (ndim + 1) // 2 # optimal minimum nlive

                # run dynesty 1.1
                #sampler = NestedSampler(loglike_d, optimal_prior, ndim,
                #    vol_dec=_params['vol_dec'],
                #    vol_check=_params['vol_check'],
                #    facc=_params['facc'],
                #    nlive=_params['nlive'],
                #    sample=_params['sample'],
                #    bound=_params['bound'],
                #    #rwalk=_params['rwalk'],
                #    max_move=_params['max_move'],
                #    logl_args=[(_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])

                # run dynesty 2.0.3
                #sampler = NestedSampler(loglike_d, optimal_prior, ndim,
                #    nlive=_params['nlive'],
                #    sample=_params['sample'],
                #    bound=_params['bound'],
                #    facc=_params['facc'],
                #    fmove=_params['fmove'],
                #    max_move=_params['max_move'],
                #    logl_args=[(_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                #sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False)

                # run dynesty 2.1.15
                if _params['_dynesty_class_'] == 'static':
                    sampler = NestedSampler(loglike_d, optimal_prior, ndim,
                        nlive=_params['nlive'],
                        update_interval=_params['update_interval'],
                        sample=_params['sample'],
                        bound=_params['bound'],
                        facc=_params['facc'],
                        fmove=_params['fmove'],
                        max_move=_params['max_move'],
                        logl_args=[(_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])

                    sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=True)

                elif _params['_dynesty_class_'] == 'dynamic':
                    sampler = DynamicNestedSampler(loglike_d, optimal_prior, ndim,
                        nlive=_params['nlive'],
                        update_interval=_params['update_interval'],
                        sample=_params['sample'],
                        bound=_params['bound'],
                        facc=_params['facc'],
                        fmove=_params['fmove'],
                        max_move=_params['max_move'],
                        logl_args=[(_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                    sampler.run_nested(dlogz_init=_params['dlogz'], maxiter_init=_params['maxiter'], maxcall_init=_params['maxcall'], print_progress=True)

                _gfit_results_temp, _logz = get_dynesty_sampler_results(sampler)

                #---------------------------------------------------------
                # lower bounds : x1-3*std1, x2-3*std2, ...  | x:_gfit_results_temp[2, 5, 8, ...], std:_gfit_results_temp[3, 6, 9, ...]
                #_x_boundaries[0:ngauss] = _gfit_results_temp[2:nparams:3] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[3:nparams:3] # x - 3*std
                _x_boundaries[0:ngauss] = _gfit_results_temp[2:nparams:3] - 5*_gfit_results_temp[3:nparams:3] # x - 3*std
                #print("g:", ngauss, "upper bounds:", _x_boundaries)
    
                _x_boundaries[ngauss:2*ngauss] = _gfit_results_temp[2:nparams:3] + 5*_gfit_results_temp[3:nparams:3] # x + 3*std

                #---------------------------------------------------------
                # lower/upper bounds
                _x_boundaries_ft = np.delete(_x_boundaries, np.where(_x_boundaries < -1E3))
                _x_lower = np.sort(_x_boundaries_ft)[0]
                _x_upper = np.sort(_x_boundaries_ft)[-1]
                _x_lower = _x_lower if _x_lower > 0 else 0
                _x_upper = _x_upper if _x_upper < 1 else 1
                #print(_x_lower, _x_upper)

                #---------------------------------------------------------
                # derive the rms given the current ngfit 
                _f_ngfit = f_gaussian_model(_x, _gfit_results_temp, ngauss)
                # residual : input_flux - ngfit_flux
                _res_spect = ((_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min) - _f_ngfit)
                # rms
                #print(np.where(_x > _x_lower and _x < _x_upper))
                _index_t = np.argwhere((_x < _x_lower) | (_x > _x_upper))
                #print(_index_t)
                _res_spect_ft = np.delete(_res_spect, _index_t)

                # rms 
                _rms[k] = np.std(_res_spect_ft)*(_f_max - _f_min)
                # bg 
                _bg[k] = _gfit_results_temp[1]*(_f_max - _f_min) + _f_min # bg
                print(i, j, _rms[k], _bg[k])
                k += 1

    # median values
    # first replace 0.0 (zero) to NAN value to use numpy nanmedian function instead of using numpy median
    zero_to_nan_rms = np.where(_rms == 0.0, np.nan, _rms)
    zero_to_nan_bg = np.where(_bg == 0.0, np.nan, _bg)

    _rms_med = np.nanmedian(zero_to_nan_rms)
    _bg_med = np.nanmedian(zero_to_nan_bg)
    # update _rms_med, _bg_med in _params
    _params['_rms_med'] = _rms_med
    _params['_bg_med'] = _bg_med
    print("rms_med:_", _rms_med)
    print("bg_med:_", _bg_med)
    #-- END OF SUB-ROUTINE____________________________________________________________#



#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# derive rms of a profile using _gfit_results_temp derived from ngfit
def little_derive_rms_npoints_org(_inputDataCube, i, j, _x, _f_min, _f_max, ngauss, _gfit_results_temp):

    ndim = 3*ngauss + 2
    nparams = ndim

    _x_boundaries = np.full(2*ngauss, fill_value=-1E11, dtype=np.float32)
    #---------------------------------------------------------
    # lower bounds : x1-3*std1, x2-3*std2, ...  | x:_gfit_results_temp[2, 5, 8, ...], std:_gfit_results_temp[3, 6, 9, ...]
    #_x_boundaries[0:ngauss] = _gfit_results_temp[2:nparams:3] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[3:nparams:3] # x - 3*std
    _x_boundaries[0:ngauss] = _gfit_results_temp[2:nparams:3] - 5*_gfit_results_temp[3:nparams:3] # x - 3*std
    #print("g:", ngauss, "lower bounds:", _x_boundaries)

    #---------------------------------------------------------
    # upper bounds : x1+3*std1, x2+3*std2, ...  | x:_gfit_results_temp[2, 5, 8, ...], std:_gfit_results_temp[3, 6, 9, ...]
    #_x_boundaries[ngauss:2*ngauss] = _gfit_results_temp[2:nparams:3] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[3:nparams:3] # x + 3*std
    _x_boundaries[ngauss:2*ngauss] = _gfit_results_temp[2:nparams:3] + 5*_gfit_results_temp[3:nparams:3] # x + 3*std
    #print("g:", ngauss, "upper bounds:", _x_boundaries)

    #---------------------------------------------------------
    # lower/upper bounds
    _x_boundaries_ft = np.delete(_x_boundaries, np.where(_x_boundaries < -1E3))
    _x_lower = np.sort(_x_boundaries_ft)[0]
    _x_upper = np.sort(_x_boundaries_ft)[-1]
    _x_lower = _x_lower if _x_lower > 0 else 0
    _x_upper = _x_upper if _x_upper < 1 else 1
    #print(_x_lower, _x_upper)

    #---------------------------------------------------------
    # derive the rms given the current ngfit
    _f_ngfit = f_gaussian_model(_x, _gfit_results_temp, ngauss)
    # residual : input_flux - ngfit_flux
    _res_spect = ((_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min) - _f_ngfit)
    # rms
    #print(np.where(_x > _x_lower and _x < _x_upper))
    _index_t = np.argwhere((_x < _x_lower) | (_x > _x_upper))
    #print(_index_t)
    _res_spect_ft = np.delete(_res_spect, _index_t)

    # rms
    #_rms_ngfit = np.std(_res_spect_ft)*(_f_max - _f_min)
    _rms_ngfit = np.std(_res_spect_ft) # normalised
    # bg
    #_bg_ngfit = _gfit_results_temp[1]*(_f_max - _f_min) + _f_min # bg

    del(_x_boundaries, _x_boundaries_ft, _index_t, _res_spect_ft)
    gc.collect()

    return _rms_ngfit # resturn normalised _rms
#-- END OF SUB-ROUTINE____________________________________________________________#

