import numpy as np
from numba import njit

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



