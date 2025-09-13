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

import dynesty
from dynesty import NestedSampler
from dynesty import DynamicNestedSampler

# ---------------------------------------------- #
# Utils: median, standard deviation, quantile     #
# (Numba-compatible)                              #
# ---------------------------------------------- #
@njit(cache=True, fastmath=True)
def _median_nb(a):
    """Return the median of 1D array a by copy-sorting."""
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
    """Standard deviation of 1D array a (divide by N; not MLE)."""
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
    Return (median, robust_sigma) for 1D array a.
    robust_sigma = 1.4826 * MAD. If MAD == 0, fall back to std.
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
    q-quantile of 1D array a with linear interpolation.
    a is copy-sorted. q in [0, 1].
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
# Helper: indices that sort comps by peakflux     #
# in descending order (Numba selection sort).     #
# comps rows are [x, sigma, peakflux].            #
# ---------------------------------------------- #

@njit(cache=True, fastmath=True)
def _argsort_desc_by_peak(comps):
    n = comps.shape[0]
    idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        idx[i] = i
    # selection sort by comps[:,2] (peakflux) descending
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
# Core: from multiple seeds --> exclude +/- k*sigma adaptively #
# --> robust bg/rms estimation                               #
# comps order: [x, sigma, peakflux]                          #
# -------------------------------------------------------- #

@njit(cache=True, fastmath=True)
def robust_bg_rms_from_seeds_norm_adaptive_nb(
    f_norm, x_norm,
    comps,               # shape (ncomp, 3) = [x_center, sigma, peakflux] in normalized units
    exclude_k=5.0,       # default exclusion width (+/- k*sigma)
    k_min=2.0,           # lower bound for k when shrinking
    shrink_factor=0.85,  # k <- k * shrink_factor (recommend 0.5~0.95)
    max_shrink_steps=6,  # maximum number of shrink steps
    clip_sigma=3.0,      # sigma-clipping strength
    max_iter=8,
    min_bg_frac=0.25,    # minimum fraction of background channels
    emission_positive=True  # emission lines assumed: clip upper tail more strongly
):
    """
    Return (bg, rms) in normalized units.

    Procedure:
      1) For each component, create an exclusion mask of +/- k*sigma.
      2) If the background fraction < min_bg_frac, shrink k stepwise.
      3) If still insufficient, iterate components in descending peakflux:
         - Try excluding strong components first.
         - If exclusion reduces background below threshold, do not exclude that component
           (weaker ones are absorbed into background).
      4) Estimate bg/rms using MAD + sigma-clipping from remaining background samples.
      5) If background is still too small, fall back to lower quantiles (bottom 20~30%).
    """
    N = f_norm.size
    if N == 0:
        return 0.0, 1.0

    # 0) Minimum background sample count
    min_bg_cnt = int(min_bg_frac * N)
    if min_bg_cnt < 8:
        min_bg_cnt = 8

    # 1) Try excluding with k, shrinking as needed
    ncomp = comps.shape[0]
    k_cur = exclude_k

    for _ in range(max_shrink_steps + 1):
        # Exclude around all components
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

        # Count background samples
        cnt = 0
        for i in range(N):
            if mask[i] == 1:
                cnt += 1

        if cnt >= min_bg_cnt:
            break  # enough background -> use this mask
        # not enough -> shrink k
        if k_cur <= k_min:
            # cannot shrink further -> switch to selective exclusion
            break
        k_cur = k_cur * shrink_factor
        if k_cur < k_min:
            k_cur = k_min

    # 2) Selective exclusion if still insufficient
    # if the loop above yielded enough samples, mask is already set
    if cnt < min_bg_cnt:
        # start with all True
        for i in range(N):
            mask[i] = 1
        cnt = N

        # indices in peakflux-descending order
        order = _argsort_desc_by_peak(comps)

        # try excluding strong components first, but keep enough background
        for t in range(ncomp):
            c = order[t]
            xc = comps[c, 0]
            sc = comps[c, 1]
            if sc <= 0.0:
                continue
            left  = xc - k_cur * sc
            right = xc + k_cur * sc

            # dry-run: how many would be newly excluded?
            old_cnt = cnt
            new_excluded = 0
            for i in range(N):
                if mask[i] == 1:
                    xi = x_norm[i]
                    if (xi >= left) and (xi <= right):
                        new_excluded += 1

            new_cnt = old_cnt - new_excluded

            if new_cnt >= min_bg_cnt:
                # apply for real
                for i in range(N):
                    if mask[i] == 1:
                        xi = x_norm[i]
                        if (xi >= left) and (xi <= right):
                            mask[i] = 0
                cnt = new_cnt
            else:
                # skip excluding this component
                continue

    # 3) Extract background vector
    if cnt < min_bg_cnt:
        # still insufficient -> use all channels (robust steps will downweight outliers)
        fb = f_norm.copy()
        cnt = N
    else:
        fb = np.empty(cnt, dtype=f_norm.dtype)
        j = 0
        for i in range(N):
            if mask[i] == 1:
                fb[j] = f_norm[i]
                j += 1

    # 4) Initial robust stats (MAD)
    m, s = _mad_scale_nb(fb)

    # 5) Iterative sigma-clipping
    for _ in range(max_iter):
        if emission_positive:
            # one-sided clip: (f - m) < clip*s
            cnt2 = 0
            for i in range(cnt):
                if (fb[i] - m) < (clip_sigma * s):
                    cnt2 += 1

            if cnt2 < min_bg_cnt:
                # quantile fallback (bottom 20% or 30%)
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

            # normal apply
            fb2 = np.empty(cnt2, dtype=fb.dtype)
            jj = 0
            for i in range(cnt):
                if (fb[i] - m) < (clip_sigma * s):
                    fb2[jj] = fb[i]
                    jj += 1
        else:
            # two-sided clip: |f - m| < clip*s
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

        # simple convergence check: same length and tiny change in stats
        if fb2.size == fb.size:
            m2, s2 = _mad_scale_nb(fb2)
            if np.abs(m2 - m) < 1e-12 and np.abs(s2 - s) < 1e-12:
                return m2, s2
            m, s = m2, s2
            fb = fb2
            cnt = fb.size
            continue

        # update
        fb = fb2
        cnt = fb.size
        m, s = _mad_scale_nb(fb)

    return m, s



def robust_bg_rms_from_seed_dict_norm(
    x_norm, f_norm, seeds,
    *,
    # line-free mask parameters (adaptive)
    exclude_k=5.0,        # start with +/- 5*sigma exclusion
    k_min=2.0,            # allow shrinking down to +/- 2*sigma
    shrink_factor=0.85,   # k <- k*0.85 per step
    max_shrink_steps=6,   # up to 6 shrink steps
    min_bg_frac=0.25,     # minimum fraction of background channels
    # robust sigma-clipping
    clip_sigma=3.0,
    max_iter=8,
    emission_positive=True # for emission lines, clip upper tail only
):
    """
    x_norm, f_norm: normalized spectrum/axis (0..1)
    seeds: {'ncomp', 'components', 'bg', 'rms', ...}, where components are [x, sigma, peak] in normalized units.
    Return: (bg_norm, rms_norm)
    """
    # extract components (+ validate)
    comps = seeds.get("components", None)
    if comps is None or np.size(comps) == 0:
        comps_arr = np.zeros((0, 3), dtype=np.float64)
    else:
        comps_arr = np.ascontiguousarray(np.asarray(comps, dtype=np.float64))
        if comps_arr.ndim != 2 or comps_arr.shape[1] < 3:
            raise ValueError("seeds['components'] must be (ncomp, 3) = [x, sigma, peak].")
        # only use [x, sigma, peak]
        comps_arr = comps_arr[:, :3]

    # Numba functions expect float64 1D arrays
    f64 = np.ascontiguousarray(f_norm, dtype=np.float64).ravel()
    x64 = np.ascontiguousarray(x_norm, dtype=np.float64).ravel()

    # call the Numba core
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
    Update the already-computed (bg_norm, rms_norm) into the seed dict.
    seeds format: {"ncomp": int, "components": (n,3), "bg": float, "rms": float, "indices": ..., "debug": ...}

    Parameters
    ----------
    seeds : dict
        _gaussian_seeds in normalized units
    bg_norm : float
        normalized background (preferably 0..1)
    rms_norm : float
        normalized RMS (>= 0)
    inplace : bool
        If True, modify seeds in place. If False, return a shallow copy.
    bg_clip : (lo, hi)
        Clip range for background.
    rms_floor : float
        Lower bound to protect against too-small RMS.

    Returns
    -------
    dict
        Updated seeds dict (same object if inplace=True).
    """
    tgt = seeds if inplace else dict(seeds)

    # safety
    bg  = float(bg_norm) if np.isfinite(bg_norm) else float(tgt.get("bg", 0.0))
    rms = float(rms_norm) if np.isfinite(rms_norm) else float(tgt.get("rms", 0.1))

    # clip/floor
    bg  = float(np.clip(bg, bg_clip[0], bg_clip[1]))
    rms = float(max(rms, rms_floor))

    # keep/extend debug log
    dbg_old = tgt.get("debug", {})
    dbg = dict(dbg_old) if isinstance(dbg_old, dict) else {}
    dbg.update({
        "bg_rms_updated": True,
        "bg_old": float(seeds.get("bg", np.nan)),
        "rms_old": float(seeds.get("rms", np.nan)),
        "bg_new": bg,
        "rms_new": rms,
    })

    # update
    tgt["bg"] = bg
    tgt["rms"] = rms
    tgt["debug"] = dbg

    return tgt


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# parameters are sigma, bg, _x01, _std01, _p01, _x02, _std02, _p02...

@njit(cache=True, fastmath=True)  # no parallel
def _gaussian_sum_norm(x, theta, ngauss):
    """
    Vectorized: model = bg + sum amp * exp(-0.5*((x - mu)/sig)^2)
    """
    bg = float(theta[1])
    model = np.full(x.size, bg, dtype=np.float64)
    for m in range(ngauss):
        mu  = float(theta[2 + 3*m])
        sig = float(theta[3 + 3*m])
        amp = float(theta[4 + 3*m])
        invs = 1.0 / sig
        # compute over full x at once
        dx = (x - mu) * invs
        model += amp * np.exp(-0.5 * dx * dx)
    return model

@njit(cache=True, fastmath=True)  # no parallel
def _rms_of_residual(data_norm, model):
    """
    Vectorized: sqrt(mean((data - model)^2))
    """
    r = data_norm - model
    return np.sqrt(np.mean(r * r))

@njit(cache=True, fastmath=True)  # no parallel
def _neg_half_chi2(data_norm, model, inv_sigma2):
    """
    Vectorized: -0.5 * sum r^2 / sigma^2
    """
    r = data_norm - model
    return -0.5 * inv_sigma2 * np.sum(r * r)


# =========================
# Robust RMS helpers (Numba)
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
    1) median m
    2) MAD = median(|a - m|)
    3) robust sigma ~= 1.4826 * MAD (fallback to std if MAD==0)
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
    q-quantile of 1D array a with linear interpolation.
    a is copy-sorted. q in [0, 1].
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
    Return indices that sort components by |amp| in descending order, read from theta.
    (selection sort)
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
    Partially vectorized.
    For each component, compute [mu - k*sig, mu + k*sig] and mark channels in that interval as 0.
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
        win = (x >= left) & (x <= right)  # vector comparison over x
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
    Predict how many new samples would be excluded by applying (mu, sig, k) on current mask.
    Vector operations where possible.
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
    Apply exclusion for real (update mask). Return the number of newly excluded samples.
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
    1) Exclude all components with +/- k_init*sigma. If background is insufficient, shrink k.
    2) If still insufficient, selectively exclude components in descending |amp|.
       (exclude strong ones first; weaker ones are absorbed into background)
    Return: (mask, bg_count_min)
      - mask: uint8 array of 0/1
      - bg_count_min: minimum required background count (for reference)
    """
    N = x.size
    min_bg_cnt = int(min_bg_frac * N)
    if min_bg_cnt < 8:
        min_bg_cnt = 8

    # 1) shrink k as needed while excluding all components
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

    # 2) selective exclusion: start with all True
    mask = np.ones(N, dtype=np.uint8)
    cnt  = N
    order = _argsort_desc_abs_amp_from_theta(theta, ngauss)

    for u in range(ngauss):
        m = int(order[u])
        mu  = float(theta[2 + 3*m])
        sig = float(theta[3 + 3*m])
        if sig <= 0.0:
            continue

        # predict effect; only apply if sufficient bg remains
        delta = _count_exclusion_if_applied(mask, x, mu, sig, k_cur)
        if (cnt - delta) >= min_bg_cnt:
            _ = _apply_exclusion(mask, x, mu, sig, k_cur)
            cnt -= delta
        # else skip this component

    if cnt < min_bg_cnt:
        mask[:] = 1  # final fallback: use all channels

    return mask, min_bg_cnt

@njit(cache=True, fastmath=True)
def _robust_rms_from_residual_with_mask(resid, mask, min_bg_cnt, clip_sigma=3.0, max_iter=8):
    """
    Estimate robust sigma from residuals using mask==1 samples.
      - Init: MAD-based (m, s)
      - Iterate: two-sided sigma-clipping (|r - m| < clip_sigma*s)
      - If samples are too few, fallback to quantiles (bottom 20% or 30%)
    Return: robust_rms
    """
    # apply mask
    use = (mask == 1)
    if np.sum(use) == 0:
        fb = resid.copy()
    else:
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

        # convergence check
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
    profile: raw spectrum (cube[:, j, i])
    x: channel axis (same normalized axis)
    f_min, f_max: min/max used for normalization
    ngauss: number of Gaussian components used
    theta: parameter vector returned by dynesty in normalized scale (length >= 3*ngauss + 2)

    Change:
    - Old: RMS from all channels.
    - New: exclude line channels adaptively (+/- k*sigma around components),
           and estimate robust RMS from line-free channels (MAD + sigma-clipping).
    """
    # normalized spectrum/model (vectorized)
    scale = (f_max - f_min)
    inv_scale = 1.0 / scale
    data_norm = (profile - f_min) * inv_scale

    model = _gaussian_sum_norm(x, theta, ngauss)

    # residual (vectorized)
    resid = data_norm - model

    # line-free mask (adaptive)
    # tunable constants:
    min_bg_frac   = 0.25   # min fraction of background
    k_init        = 5.0    # initial exclusion width (+/- k*sigma)
    k_min         = 2.0    # minimum k
    shrink_factor = 0.85   # k shrink ratio
    max_shrink    = 6      # max shrink steps

    mask, min_bg_cnt = _adaptive_linefree_mask_from_theta(
        x, theta, int(ngauss),
        min_bg_frac=min_bg_frac,
        k_init=k_init, k_min=k_min,
        shrink_factor=shrink_factor, max_shrink_steps=max_shrink
    )

    # robust RMS (from residuals, two-sided sigma-clipping)
    clip_sigma = 3.0
    max_iter   = 8
    robust_rms = _robust_rms_from_residual_with_mask(
        resid, mask, min_bg_cnt, clip_sigma=clip_sigma, max_iter=max_iter 
    ) # normalized units
    # robust_rms_phys = robust_rms * (f_max - f_min)

    # return normalized rms
    return robust_rms


def little_derive_rms(input_cube, i, j, x, f_min, f_max, ngauss, theta):
    """
    Keep the original calling signature from existing code. Internally calls the JIT core.
    theta must be the normalized parameter vector (before unit conversion).
    """
    # ensure C-contiguous
    prof = np.ascontiguousarray(input_cube[:, j, i], dtype=np.float64)
    x64  = np.ascontiguousarray(x, dtype=np.float64)
    th64 = np.ascontiguousarray(theta[:(3*ngauss+2)], dtype=np.float64)
    return _little_derive_rms_core(prof, x64, float(f_min), float(f_max), int(ngauss), th64)


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# UNIT CONVERSION
def convert_units_norm_to_phys(
    gfit_results: np.ndarray,
    j: int,                     # profile index (e.g., y-axis index)
    k: int,                     # multi-Gaussian model order - 1 (i.e., number of Gaussians = k+1)
    f_min: float, f_max: float, # flux normalization bounds for this profile
    vel_min: float, vel_max: float,  # physical velocity range (recommend vel_min < vel_max)
    cdelt3: float,              # sign of channel increment (negative means descending)
    max_ngauss: int             # global maximum number of Gaussians (_max_ngauss)
) -> None:
    """
    In-place convert normalized results gfit_results[j, k, :] to physical units.
    - Parameter block length nparams = 2 + 3*(k+1)
      [model_sigma, bg, g1_x, g1_sigma, g1_peak, g2_x, g2_sigma, g2_peak, ...]
    - Error block: same length nparams immediately after.
      [model_sigma_e, bg_e, g1_x_e, g1_sigma_e, g1_peak_e, ...]
    - Additional info follows, but here we only convert background/peak/velocity/dispersion/errors and rms.

    Note: function modifies gfit_results in place and returns None.
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
    if cdelt3 > 0:  # velocity axis increasing
        gfit_results[j, k, velocity_indices] = (
            gfit_results[j, k, velocity_indices] * (vel_max - vel_min) + vel_min  # velocity
        )
    else:  # velocity axis decreasing
        gfit_results[j, k, velocity_indices] = (
            gfit_results[j, k, velocity_indices] * (vel_min - vel_max) + vel_max  # velocity
        )

    gfit_results[j, k, velocity_dispersion_indices] *= (vel_max - vel_min)  # velocity-dispersion

    #________________________________________________________________________________________|
    # peak flux --> data cube units
    # peak flux --> data cube units: (f_max - f_min) is used since normalized peaks are from bg-normalized scale
    # gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) + _f_min # flux
    gfit_results[j, k, peak_flux_indices] *= (f_max - f_min)  # flux

    #________________________________________________________________________________________|
    # velocity-e, velocity-dispersion-e --> km/s
    gfit_results[j, k, velocity_e_indices]            *= (vel_max - vel_min)  # velocity-e
    gfit_results[j, k, velocity_dispersion_e_indices] *= (vel_max - vel_min)  # velocity-dispersion-e

    # flux-e --> data cube units
    gfit_results[j, k, flux_e_indices] *= (f_max - f_min)  # flux-e

    # lastly put rms 
    # location: 2*(3*max_ngauss + 2) + k  (rms slot for each k model)
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
    # gfit_priors_init = [sig1, bg1, x1, std1, p1, sig2, bg2, x2, std2, p2]
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
                # sampler = NestedSampler(loglike_d, optimal_prior, ndim,
                #     vol_dec=_params['vol_dec'],
                #     vol_check=_params['vol_check'],
                #     facc=_params['facc'],
                #     nlive=_params['nlive'],
                #     sample=_params['sample'],
                #     bound=_params['bound'],
                #     #rwalk=_params['rwalk'],
                #     max_move=_params['max_move'],
                #     logl_args=[(_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])

                # run dynesty 2.0.3
                # sampler = NestedSampler(loglike_d, optimal_prior, ndim,
                #     nlive=_params['nlive'],
                #     sample=_params['sample'],
                #     bound=_params['bound'],
                #     facc=_params['facc'],
                #     fmove=_params['fmove'],
                #     max_move=_params['max_move'],
                #     logl_args=[(_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                # sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False)

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
                # lower bounds: x1 - 5*std1, x2 - 5*std2, ...
                # x at _gfit_results_temp[2, 5, 8, ...], std at [3, 6, 9, ...]
                _x_boundaries[0:ngauss] = _gfit_results_temp[2:nparams:3] - 5*_gfit_results_temp[3:nparams:3]
    
                _x_boundaries[ngauss:2*ngauss] = _gfit_results_temp[2:nparams:3] + 5*_gfit_results_temp[3:nparams:3] # x + 5*std

                #---------------------------------------------------------
                # lower/upper bounds
                _x_boundaries_ft = np.delete(_x_boundaries, np.where(_x_boundaries < -1E3))
                _x_lower = np.sort(_x_boundaries_ft)[0]
                _x_upper = np.sort(_x_boundaries_ft)[-1]
                _x_lower = _x_lower if _x_lower > 0 else 0
                _x_upper = _x_upper if _x_upper < 1 else 1

                #---------------------------------------------------------
                # derive the rms given the current ngfit 
                _f_ngfit = f_gaussian_model(_x, _gfit_results_temp, ngauss)
                # residual : input_flux - ngfit_flux
                _res_spect = ((_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min) - _f_ngfit)
                # indices to exclude inside [x_lower, x_upper]
                _index_t = np.argwhere((_x < _x_lower) | (_x > _x_upper))
                _res_spect_ft = np.delete(_res_spect, _index_t)

                # rms 
                _rms[k] = np.std(_res_spect_ft)*(_f_max - _f_min)
                # bg 
                _bg[k] = _gfit_results_temp[1]*(_f_max - _f_min) + _f_min # bg
                print(i, j, _rms[k], _bg[k])
                k += 1

    # medians
    # replace 0.0 with NaN to use nanmedian
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
    # lower bounds: x1 - 5*std1, x2 - 5*std2, ...
    # x at _gfit_results_temp[2, 5, 8, ...], std at [3, 6, 9, ...]
    _x_boundaries[0:ngauss] = _gfit_results_temp[2:nparams:3] - 5*_gfit_results_temp[3:nparams:3]

    #---------------------------------------------------------
    # upper bounds: x1 + 5*std1, x2 + 5*std2, ...
    _x_boundaries[ngauss:2*ngauss] = _gfit_results_temp[2:nparams:3] + 5*_gfit_results_temp[3:nparams:3]

    #---------------------------------------------------------
    # lower/upper bounds
    _x_boundaries_ft = np.delete(_x_boundaries, np.where(_x_boundaries < -1E3))
    _x_lower = np.sort(_x_boundaries_ft)[0]
    _x_upper = np.sort(_x_boundaries_ft)[-1]
    _x_lower = _x_lower if _x_lower > 0 else 0
    _x_upper = _x_upper if _x_upper < 1 else 1

    #---------------------------------------------------------
    # derive the rms given the current ngfit
    _f_ngfit = f_gaussian_model(_x, _gfit_results_temp, ngauss)
    # residual : input_flux - ngfit_flux
    _res_spect = ((_inputDataCube[:, j, i]-_f_min)/(_f_max-_f_min) - _f_ngfit)
    # indices to exclude inside [x_lower, x_upper]
    _index_t = np.argwhere((_x < _x_lower) | (_x > _x_upper))
    _res_spect_ft = np.delete(_res_spect, _index_t)

    # rms (normalized)
    _rms_ngfit = np.std(_res_spect_ft)
    # bg (if needed):
    # _bg_ngfit = _gfit_results_temp[1]*(_f_max - _f_min) + _f_min

    del(_x_boundaries, _x_boundaries_ft, _index_t, _res_spect_ft)
    gc.collect()

    return _rms_ngfit # return normalized rms

#-- END OF SUB-ROUTINE____________________________________________________________#


