#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _set_init_priors.py
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|

import sys


import os
import numpy as np

import matplotlib.pyplot as plt



# Numba-accelerated search_gaussian_seeds_matched_filter_norm
# ----------------------------------------------------
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

TWO_SQRT2_LN2 = 2.355

# ============================================================
# Matched-filter Gaussian seed finder — Numba-optimized (final)
# ============================================================
import os
import numpy as np

# ----------------- Optional: Numba import -------------------
try:
    from numba import njit, prange, set_num_threads
    from numba.typed import List as NumbaList
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

TWO_SQRT2_LN2 = 2.355




# ============================================================
# Matched-filter Gaussian seed finder — Numba-optimized (final)
# ============================================================

# ----------------- Optional: Numba import -------------------
try:
    from numba import njit, prange, set_num_threads
    from numba.typed import List as NumbaList
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

TWO_SQRT2_LN2 = 2.355


# ----------------- Thread pinning helper --------------------
def pin_threads_single():
    """
    Recommended to call once when a Ray worker (process) starts.
    - Pin BLAS/OMP/Numba threads to 1 --> avoid oversubscription.
    """
    os.environ["OMP_NUM_THREADS"]        = "1"
    os.environ["MKL_NUM_THREADS"]        = "1"
    os.environ["OPENBLAS_NUM_THREADS"]   = "1"
    os.environ["BLIS_NUM_THREADS"]       = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"]    = "1"
    if NUMBA_OK:
        try:
            set_num_threads(1)
        except Exception:
            pass


# ----------------- JIT kernels & fallbacks ------------------
if NUMBA_OK:
    @njit(cache=True)
    def _median(a):
        b = np.sort(a.copy())
        n = b.size
        if n == 0:
            return 0.0
        mid = n // 2
        if n % 2 == 1:
            return float(b[mid])
        else:
            return float(0.5 * (b[mid-1] + b[mid]))

    @njit(cache=True)
    def _mad(a, med):
        b = np.abs(a - med)
        return _median(b)

    @njit(cache=True)
    def _reflect_index(i, n):
        if n <= 1:
            return 0
        period = 2 * n - 2
        i_mod = i % period
        if i_mod < 0:
            i_mod += period
        if i_mod < n:
            return i_mod
        else:
            return period - i_mod

    @njit(cache=True)
    def _conv_same_reflect(y, g):
        # Assume symmetric kernel (Gaussian) --> no need to flip.
        n = y.size
        L = g.size
        h = (L - 1) // 2
        out = np.zeros(n, dtype=np.float64)
        for i in range(n):
            s = 0.0
            for k in range(L):
                j = i + (k - h)
                jj = _reflect_index(j, n)
                s += y[jj] * g[k]
            out[i] = s
        return out

    @njit(cache=True)
    def _parabolic_subsample_jit(y, i):
        n = y.size
        if i <= 0 or i >= n - 1:
            return 0.0, float(y[i])
        y0, y1, y2 = float(y[i-1]), float(y[i]), float(y[i+1])
        denom = 2.0 * (y0 - 2.0*y1 + y2)
        if np.abs(denom) < 1e-12:
            return 0.0, y1
        dx = 0.5 * (y0 - y2) / denom
        if dx < -0.5: dx = -0.5
        if dx >  0.5: dx =  0.5
        val = y1 - 0.25 * (y0 - y2) * dx
        return float(dx), float(val)

    @njit(cache=True)
    def _robust_bg_rms_emission_jit(x, clip_sigma=3.0, max_iter=8, min_bg_frac=0.25):
        N = x.size
        if N == 0:
            return 0.0, 1.0
        m = _median(x)
        mad = _mad(x, m)
        s = 1.4826 * mad if mad > 0 else (np.std(x) + 1e-12)

        # uint8 mask
        mask = np.ones(N, dtype=np.uint8)
        for _ in range(max_iter):
            new_mask = ((x - m) < (clip_sigma * s)).astype(np.uint8)
            new_cnt = int(np.sum(new_mask))
            if new_cnt < max(int(min_bg_frac * N), 3):
                # Fallback to lower quantiles.
                # Simplify by sorting and taking lowest k samples.
                k = max(3, int(0.3 * N))
                xs = np.sort(x.copy())
                bgc = xs[:k]
                m = _median(bgc)
                mad = _mad(bgc, m)
                s = 1.4826 * mad if mad > 0 else (np.std(bgc) + 1e-12)
                return m, s

            same = True
            for i in range(N):
                if new_mask[i] != mask[i]:
                    same = False
                    break
            mask = new_mask
            cnt = int(np.sum(mask))
            sel = np.empty(cnt, dtype=np.float64)
            c = 0
            for i in range(N):
                if mask[i]:
                    sel[c] = x[i]; c += 1
            m = _median(sel)
            mad = _mad(sel, m)
            s = 1.4826 * mad if mad > 0 else (np.std(sel) + 1e-12)
            if same:
                break
        return m, s

    # ---- Bank convolve + standardize (parallel / sequential) ----
    @njit(cache=True, parallel=True)
    def _bank_convolve_and_standardize_par(y, y_w, g_list, gnorm_list):
        K = len(g_list)
        N = y.size
        Rstd = np.empty((K, N), dtype=np.float64)
        Rraw = np.empty((K, N), dtype=np.float64)
        for k in prange(K):
            g   = g_list[k]
            gn  = gnorm_list[k]
            rs  = _conv_same_reflect(y_w, gn)
            rr  = _conv_same_reflect(y,   g)
            m   = _median(rs)
            mad = _mad(rs, m)
            s   = 1.4826*mad if mad>0 else (np.std(rs) + 1e-12)
            for i in range(N):
                Rstd[k, i] = (rs[i] - m) / s
                Rraw[k, i] = rr[i]
        return Rstd, Rraw

    @njit(cache=True, parallel=False)
    def _bank_convolve_and_standardize_seq(y, y_w, g_list, gnorm_list):
        K = len(g_list)
        N = y.size
        Rstd = np.empty((K, N), dtype=np.float64)
        Rraw = np.empty((K, N), dtype=np.float64)
        for k in range(K):
            g   = g_list[k]
            gn  = gnorm_list[k]
            rs  = _conv_same_reflect(y_w, gn)
            rr  = _conv_same_reflect(y,   g)
            m   = _median(rs)
            mad = _mad(rs, m)
            s   = 1.4826*mad if mad>0 else (np.std(rs) + 1e-12)
            for i in range(N):
                Rstd[k, i] = (rs[i] - m) / s
                Rraw[k, i] = rr[i]
        return Rstd, Rraw

else:
    # ---- NumPy fallbacks (non-Numba environment) ----
    def _parabolic_subsample_jit(y, i):
        N = len(y)
        if i <= 0 or i >= N - 1:
            return 0.0, float(y[i])
        y0, y1, y2 = float(y[i-1]), float(y[i]), float(y[i+1])
        denom = 2.0 * (y0 - 2.0*y1 + y2)
        if abs(denom) < 1e-12:
            return 0.0, y1
        dx = 0.5 * (y0 - y2) / denom
        dx = float(np.clip(dx, -0.5, 0.5))
        val = y1 - 0.25 * (y0 - y2) * dx
        return dx, float(val)

    def _robust_bg_rms_emission_jit(x, clip_sigma=3.0, max_iter=8, min_bg_frac=0.25):
        x = np.asarray(x, float)
        N = x.size
        if N == 0: return 0.0, 1.0
        m = float(np.median(x))
        mad = np.median(np.abs(x - m))
        s = float(1.4826 * mad) if mad > 0 else float(np.std(x) + 1e-12)
        mask = np.ones(N, dtype=bool)
        for _ in range(max_iter):
            new_mask = (x - m) < (clip_sigma * s)
            if new_mask.sum() < max(int(min_bg_frac * N), 3):
                k = max(3, int(0.3 * N))
                bgc = np.sort(x)[:k]
                m = float(np.median(bgc))
                mad = np.median(np.abs(bgc - m))
                s = float(1.4826 * mad) if mad > 0 else float(np.std(bgc) + 1e-12)
                return m, s
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask
            sel = x[mask]
            m = float(np.median(sel))
            mad = np.median(np.abs(sel - m))
            s = float(1.4826 * mad) if mad > 0 else float(np.std(sel) + 1e-12)
        return m, s

    def _conv_same_reflect(y, g):
        h = (len(g) - 1) // 2
        y_pad = np.pad(y, h, mode='reflect')
        r = np.convolve(y_pad, g, mode='same')
        return r[h:h+len(y)]

    def _bank_convolve_and_standardize_seq(y, y_w, g_list, gnorm_list):
        # In pure NumPy, provide the single-threaded version only.
        K = len(g_list)
        N = y.size
        Rstd = np.empty((K, N), dtype=np.float64)
        Rraw = np.empty((K, N), dtype=np.float64)
        for k in range(K):
            g   = g_list[k]
            gn  = gnorm_list[k]
            rs  = _conv_same_reflect(y_w, gn)
            rr  = _conv_same_reflect(y,   g)
            m   = float(np.median(rs))
            mad = np.median(np.abs(rs - m))
            s   = 1.4826*mad if mad>0 else (np.std(rs) + 1e-12)
            Rstd[k, :] = (rs - m) / s
            Rraw[k, :] = rr
        return Rstd, Rraw

    _bank_convolve_and_standardize_par = _bank_convolve_and_standardize_seq  # alias

#-- END OF SUB-ROUTINE____________________________________________________________#

# ----------------- Utilities (Python level) ------------------
def gaussian_kernel_bank(sigma_list_ch, k_sigma=4.0):
    bank = []
    for s in np.asarray(sigma_list_ch, dtype=float):
        s = max(1e-6, float(s))
        h = int(np.ceil(k_sigma * s))
        x = np.arange(-h, h + 1, dtype=np.float64)
        g = np.exp(-0.5 * (x / s) ** 2)
        sum_g2 = float(np.sum(g * g))
        g_norm = g / np.sqrt(sum_g2)
        bank.append((g, sum_g2, g_norm, np.sqrt(sum_g2), h, s))
    return bank

def _make_sigma_list(N, sigma_list_ch, k_sigma, *, min_sigma=0.8, max_frac=0.48):
    if sigma_list_ch is None:
        sigma_list_ch = np.array([1.0,1.5,2.0,3.0,4.0,5.0,6.0,8.0,10.0,12.0], dtype=float)
    else:
        sigma_list_ch = np.asarray(sigma_list_ch, dtype=float)
    cap = max(min_sigma, max_frac * N / max(1.0, k_sigma))
    sl = sigma_list_ch[sigma_list_ch <= cap]
    if sl.size == 0:
        M = int(np.clip(N/5, 4, 14))
        sl = np.geomspace(min_sigma, cap, M)
    sl = np.unique(np.round(sl, 6))
    return sl, cap

#-- END OF SUB-ROUTINE____________________________________________________________#




# ================== 1) Window-aware matched-filter seeding ==================
def search_gaussian_seeds_matched_filter_norm2(
    v, f, *, rms=None, bg=None,
    sigma_list_ch=None, k_sigma=4.0,
    thres_sigma=3.0, amp_sigma_thres=3.0,
    sep_channels=5, max_components=None,
    refine_center=True, detrend_local=False, detrend_halfwin=8,
    numba_threads=1,
    x_min_norm=None, x_max_norm=None,
    max_ngauss=None
):
    """
    Find initial seeds for multi-Gaussian fitting using a matched filter,
    but restrict detection to the normalized-velocity window (x_min_norm, x_max_norm).

    IMPORTANT: Returned components are in **normalized coordinates**:
        components[:, 0] = center_norm in [0,1]
        components[:, 1] = sigma_norm  (length on normalized axis)
        components[:, 2] = |amplitude| (same units as input f - bg)
    Seeds are sorted by descending |amplitude| and truncated to `max_ngauss` if provided.

    If there are no candidates in the window, this returns ncomp=0 and the bounds
    function should perform the requested fallback (use the window directly).
    """
    # -------------- threading for numba (if available) --------------
    if NUMBA_OK and (numba_threads is not None):
        try:
            set_num_threads(int(numba_threads))
        except Exception:
            pass

    v = np.asarray(v, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    N = f.size
    if N == 0:
        return dict(ncomp=0, components=np.zeros((0, 3)),
                    bg=0.0, rms=1.0, indices=[], debug={"reason": "N=0"})

    # Physical spacing (approx.)
    dv = float(np.median(np.abs(np.diff(v)))) if N > 1 else 1.0

    # ---------- robust background & rms (uses your existing helper) ----------
    if (bg is None) or (rms is None):
        bh, sh = _robust_bg_rms_emission_jit(f, clip_sigma=3.0, max_iter=8, min_bg_frac=0.25)
        if bg is None:  bg  = float(bh)
        if rms is None: rms = float(sh)
    if not np.isfinite(bg):
        bg = float(np.median(f))

    y = f - bg
    if (not np.isfinite(rms)) or (rms <= 0):
        mad = np.median(np.abs(y - np.median(y)))
        rms = float(1.4826 * mad) if mad > 0 else float(np.std(y) + 1e-12)

    # Standardized residual (for test statistic); raw residual (for amplitude)
    y_w = np.nan_to_num(y / rms, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------------- kernel bank (your existing helpers) ----------------
    sigmas, cap1 = _make_sigma_list(N, sigma_list_ch, k_sigma, max_frac=0.48)
    bank = gaussian_kernel_bank(sigmas, k_sigma=k_sigma)
    if not bank:
        return dict(ncomp=0, components=np.zeros((0, 3)), bg=float(bg), rms=float(rms),
                    indices=[], debug={"reason": "empty_bank_after_cap", "N": int(N), "cap": float(cap1)})

    if NUMBA_OK:
        g_list, gnorm_list = NumbaList(), NumbaList()
    else:
        g_list, gnorm_list = [], []
    sum_g2_arr        = np.empty(len(bank), dtype=np.float64)
    sum_g2_sqrt_arr   = np.empty(len(bank), dtype=np.float64)
    halfwidth_arr     = np.empty(len(bank), dtype=np.int64)
    sigma_ch_arr      = np.empty(len(bank), dtype=np.float64)
    for k, (g, sum_g2, g_norm, sum_g2_sqrt, h, s_ch) in enumerate(bank):
        g_list.append(g); gnorm_list.append(g_norm)
        sum_g2_arr[k]      = sum_g2
        sum_g2_sqrt_arr[k] = sum_g2_sqrt
        halfwidth_arr[k]   = h
        sigma_ch_arr[k]    = s_ch

    # ---------------- normalized axis and window mask ----------------
    v_min = float(min(v[0], v[-1]))
    v_max = float(max(v[0], v[-1]))
    span  = max(1e-30, (v_max - v_min))
    v_norm = (v - v_min) / span  # strictly increasing 0..1

    if x_min_norm is None: x_min_norm = 0.0
    if x_max_norm is None: x_max_norm = 1.0
    lo = float(min(x_min_norm, x_max_norm))
    hi = float(max(x_min_norm, x_max_norm))
    win_width = max(0.0, hi - lo)

    valid_idx = (v_norm > lo) & (v_norm < hi)  # strict interior to match your log-like mask
    #valid_idx = (v_norm > 0) & (v_norm < 1)  # strict interior to match your log-like mask
    n_valid = int(np.count_nonzero(valid_idx))
    if n_valid == 0:
        return dict(ncomp=0, components=np.zeros((0, 3)), bg=float(bg), rms=float(rms),
                    indices=[], debug={"reason": "empty_window", "win_lo": lo, "win_hi": hi})

    # ---------------- convolve & standardize (hot path) ----------------
    if NUMBA_OK and int(numba_threads) == 1:
        Rstd, Rraw = _bank_convolve_and_standardize_seq(y, y_w, g_list, gnorm_list)
    else:
        Rstd, Rraw = _bank_convolve_and_standardize_par(y, y_w, g_list, gnorm_list)

    # Best kernel index for each channel
    k_best = np.argmax(np.abs(Rstd), axis=0)
    Rbest  = Rstd[k_best, np.arange(N)]

    # First pass: threshold AND inside window
    mask = (np.abs(Rbest) >= float(thres_sigma)) & valid_idx
    cand_idx = np.flatnonzero(mask)

    debug = {
        "N": int(N), "bg": float(bg), "rms": float(rms),
        "sigma_cap1": float(cap1), "n_sigmas1": int(len(sigmas)),
        "Rbest_max_window": float(np.max(np.abs(Rbest[valid_idx]))),
        "cand_n1": int(cand_idx.size),
        "win_lo": float(lo), "win_hi": float(hi),
        "returned": "normalized",
    }

    # If no candidates, expand sigmas (still windowed)
    if cand_idx.size == 0:
        sigmas2, cap2 = _make_sigma_list(N, np.r_[sigmas, 6, 8, 10, 12, 15, 18], k_sigma, max_frac=0.90)
        bank2 = gaussian_kernel_bank(sigmas2, k_sigma=k_sigma)
        if bank2:
            if NUMBA_OK:
                g_list2, gnorm_list2 = NumbaList(), NumbaList()
            else:
                g_list2, gnorm_list2 = [], []
            sum_g2_arr2, sum_g2_sqrt_arr2 = np.empty(len(bank2)), np.empty(len(bank2))
            halfwidth_arr2, sigma_ch_arr2  = np.empty(len(bank2), dtype=np.int64), np.empty(len(bank2))
            for k, (g, sum_g2, g_norm, sum_g2_sqrt, h, s_ch) in enumerate(bank2):
                g_list2.append(g); gnorm_list2.append(g_norm)
                sum_g2_arr2[k]      = sum_g2
                sum_g2_sqrt_arr2[k] = sum_g2_sqrt
                halfwidth_arr2[k]   = h
                sigma_ch_arr2[k]    = s_ch

            if NUMBA_OK and int(numba_threads) == 1:
                R2, RR2 = _bank_convolve_and_standardize_seq(y, y_w, g_list2, gnorm_list2)
            else:
                R2, RR2 = _bank_convolve_and_standardize_par(y, y_w, g_list2, gnorm_list2)

            k_best2 = np.argmax(np.abs(R2), axis=0)
            Rbest2  = R2[k_best2, np.arange(N)]
            th2 = max(2.0, 0.6 * float(thres_sigma))
            cand_idx2 = np.flatnonzero((np.abs(Rbest2) >= th2) & valid_idx)

            debug.update({
                "sigma_cap2": float(cap2),
                "n_sigmas2": int(len(sigmas2)),
                "Rbest2_max_window": float(np.max(np.abs(Rbest2[valid_idx]))),
                "cand_n2": int(cand_idx2.size),
            })

            if cand_idx2.size > 0:
                # Switch to expanded bank
                sigmas            = sigmas2
                Rstd, Rraw        = R2, RR2
                k_best, Rbest     = k_best2, Rbest2
                cand_idx          = cand_idx2
                sum_g2_arr        = sum_g2_arr2
                sum_g2_sqrt_arr   = sum_g2_sqrt_arr2
                halfwidth_arr     = halfwidth_arr2
                sigma_ch_arr      = sigma_ch_arr2

    # Structural rescue (within window only)
    if cand_idx.size == 0:
        win_idx = np.flatnonzero(valid_idx)
        i0 = int(win_idx[np.argmax(np.abs(Rbest[win_idx]))]) if win_idx.size > 0 else -1

        if i0 >= 0 and np.isfinite(Rbest[i0]):
            kb  = int(k_best[i0])
            A_hat   = Rraw[kb, i0] / max(1e-12, float(sum_g2_arr[kb]))
            sigma_A = float(rms) / float(sum_g2_sqrt_arr[kb])
            debug.update({"rescue_i0": int(i0), "rescue_Rbest": float(Rbest[i0]),
                          "rescue_A_hat": float(A_hat), "rescue_sigma_A": float(sigma_A)})
            # Only accept if amplitude is strong enough
            if abs(A_hat) >= max(2.5, 0.8 * float(amp_sigma_thres)) * sigma_A:
                center_ch = float(i0)
                if refine_center:
                    dx, _ = _parabolic_subsample_jit(Rbest, i0)  # your helper
                    center_ch += dx
                # Convert to normalized center and sigma
                center_v   = v[0] + np.sign(v[-1] - v[0]) * center_ch * dv
                sigma_v    = float(sigma_ch_arr[kb]) * dv
                center_n   = (center_v - v_min) / span
                sigma_n    = abs(sigma_v) / span
                comps = np.array([[float(center_n), float(sigma_n), float(abs(A_hat))]], dtype=float)
                indices = [(int(np.rint(center_ch)), float(sigma_ch_arr[kb]))]
                # Trim to max_ngauss if needed (here at most 1 anyway)
                if (max_ngauss is not None) and (int(max_ngauss) < comps.shape[0]):
                    comps   = comps[:int(max_ngauss), :]
                    indices = indices[:int(max_ngauss)]
                return dict(ncomp=comps.shape[0], components=comps, bg=float(bg), rms=float(rms),
                            indices=indices, debug=debug)

        # No usable candidate -> return empty; bounds function will do fallback
        debug["reason"] = "no_candidates_in_window"
        return dict(ncomp=0, components=np.zeros((0, 3)), bg=float(bg), rms=float(rms),
                    indices=[], debug=debug)

    # ------------------- NMS + amplitude gate inside window -------------------
    order = np.argsort(-np.abs(Rbest[cand_idx]))
    cand_idx = cand_idx[order]
    kept_idx, kept_sig, kept_A = [], [], []
    taken = np.zeros(N, dtype=bool)

    for idx in cand_idx:
        lo_nms = max(0, idx - int(sep_channels))
        hi_nms = min(N, idx + int(sep_channels) + 1)
        if taken[lo_nms:hi_nms].any():
            continue

        kb  = int(k_best[idx])
        sum_g2      = float(sum_g2_arr[kb])
        sum_g2_sqrt = float(sum_g2_sqrt_arr[kb])
        s_ch        = float(sigma_ch_arr[kb])

        if detrend_local:
            h = int(halfwidth_arr[kb])
            w = int(max(h, detrend_halfwin))
            lsl = slice(max(0, idx - w), min(N, idx + w + 1))
            xv = np.arange(lsl.start, lsl.stop, dtype=float)
            yy = y[lsl]
            X = np.vstack([xv, np.ones_like(xv)]).T
            coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
            y_loc = y.copy()
            y_loc[lsl] = yy - (X @ coef)
            rr = _conv_same_reflect(y_loc, g_list[kb] if NUMBA_OK else bank[kb][0])  # your helper
            A_hat = rr[idx] / max(1e-12, sum_g2)
        else:
            # raw correlation → amplitude estimate in the same units as y
            A_hat = Rraw[kb, idx] / max(1e-12, sum_g2)

        sigma_A = float(rms) / max(1e-12, sum_g2_sqrt)
        if abs(A_hat) < float(amp_sigma_thres) * sigma_A:
            continue

        kept_idx.append(int(idx))
        kept_sig.append(float(s_ch))
        kept_A.append(float(A_hat))
        taken[lo_nms:hi_nms] = True

        if max_components is not None and len(kept_idx) >= int(max_components):
            break

    if not kept_idx:
        return dict(ncomp=0, components=np.zeros((0, 3)), bg=float(bg), rms=float(rms),
                    indices=[], debug={"reason": "all_candidates_failed_amp_gate"})

    centers_ch = np.asarray(kept_idx, dtype=np.float64)
    kept_sig   = np.asarray(kept_sig, dtype=np.float64)
    amps       = np.asarray(kept_A,   dtype=np.float64)

    if refine_center:
        for j, i0 in enumerate(centers_ch.astype(int)):
            dx, _ = _parabolic_subsample_jit(Rbest, i0)
            centers_ch[j] += dx

    # Sort by descending |amp|
    order2 = np.argsort(-np.abs(amps))
    centers_ch   = centers_ch[order2]
    kept_sig     = kept_sig[order2]
    amps_sorted  = np.abs(amps[order2])

    # Convert to normalized coordinates
    sign = np.sign(v[-1] - v[0]) if v.size > 1 else 1.0
    centers_v = v[0] + sign * centers_ch * dv
    sigmas_v  = kept_sig * abs(dv)
    centers_n = (centers_v - v_min) / span
    sigmas_n  = np.abs(sigmas_v) / span

    comps = np.stack([centers_n, sigmas_n, amps_sorted], axis=1)
    indices = list(zip(np.rint(centers_ch).astype(int).tolist(),
                       kept_sig.astype(float).tolist()))

    # Limit by max_ngauss (already |amp|-sorted)
    if max_ngauss is not None:
        k = int(max_ngauss)
        if k < comps.shape[0]:
            comps   = comps[:k, :]
            indices = indices[:k]

    return dict(ncomp=comps.shape[0], components=comps, bg=float(bg), rms=float(rms),
                indices=indices, debug=debug)








def search_gaussian_seeds_matched_filter_norm(
    v, f, *, rms=None, bg=None,
    sigma_list_ch=None, k_sigma=4.0,
    thres_sigma=3.0, amp_sigma_thres=3.0,
    sep_channels=5, max_components=None,
    refine_center=True, detrend_local=False, detrend_halfwin=8,
    numba_threads=1,
    x_min_norm=None, x_max_norm=None,
    max_ngauss=None
):
    """
    Matched-filter seed finder.
    - Convolution/standardization: (0,1) full interior 
    - Seed search: [x_min_norm, x_max_norm] 
    - If no seeds, return a single default one
    """
    # --- helpers --------------------------------------------------------------
    def _make_synthetic_seed(v, y, v_min, span, lo, hi, dv):
        """Build one fallback seed at window midpoint with half-width sigma."""
        width = max(1e-6, hi - lo)
        center_norm = 0.5 * (lo + hi)
        sigma_norm  = 0.5 * width
        x_vel   = v_min + center_norm * span
        sigma_v = sigma_norm * span
        # amplitude = max |residual| in window
        v_norm = (v - v_min) / max(1e-30, span)
        win_idx = (v_norm >= lo) & (v_norm <= hi)
        amp = float(np.max(np.abs(y[win_idx]))) if np.any(win_idx) else 0.0
        center_idx = int(np.argmin(np.abs(v - x_vel)))
        sigma_ch   = float(abs(sigma_v) / max(1e-30, dv))
        comps = np.array([[x_vel, sigma_v, abs(amp)]], dtype=float)
        indices = [(center_idx, sigma_ch)]
        return comps, indices, dict(
            reason="synthetic_seed",
            synthetic_center_norm=float(center_norm),
            synthetic_sigma_norm=float(sigma_norm),
            synthetic_amp=float(amp),
        )

    # --- thread setup ---------------------------------------------------------
    if NUMBA_OK and (numba_threads is not None):
        try:
            set_num_threads(int(numba_threads))
        except Exception:
            pass

    v = np.asarray(v, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    N = f.size

    #plt.scatter(v, f)
    #plt.plot(v, f)
    #plt.show()

    if N == 0:
        return dict(ncomp=0, components=np.zeros((0,3)),
                    bg=0.0, rms=1.0, indices=[], debug={"reason": "N=0"})

    dv = float(np.median(np.abs(np.diff(v)))) if N > 1 else 1.0
    if not np.isfinite(dv) or dv <= 0:
        dv = 1.0

    # --- robust bg/rms --------------------------------------------------------
    if (bg is None) or (rms is None):
        bh, sh = _robust_bg_rms_emission_jit(f, clip_sigma=3.0, max_iter=8, min_bg_frac=0.25)
        if bg is None:  bg  = float(bh)
        if rms is None: rms = float(sh)

    if not np.isfinite(bg):
        bg = float(np.median(f))
    y = f - bg
    if (not np.isfinite(rms)) or (rms <= 0):
        mad = np.median(np.abs(y - np.median(y)))
        rms = float(1.4826 * mad) if mad > 0 else float(np.std(y) + 1e-12)

    y_w = np.nan_to_num(y / rms, nan=0.0, posinf=0.0, neginf=0.0)

    # --- normalized axis & window --------------------------------------------
    v_min = min(v[0], v[-1])
    v_max = max(v[0], v[-1])
    span  = max(1e-30, (v_max - v_min))
    v_norm = (v - v_min) / span

    # interior for stable convolution
    valid_idx = (v_norm > 0.0) & (v_norm < 1.0)

    # selection window
    lo = 0.0 if x_min_norm is None else float(x_min_norm)
    hi = 1.0 if x_max_norm is None else float(x_max_norm)
    if lo > hi:
        lo, hi = hi, lo
    lo = float(np.clip(lo, 0.0, 1.0))
    hi = float(np.clip(hi, 0.0, 1.0))
    win_idx = (v_norm >= lo) & (v_norm <= hi)
    if not np.any(win_idx):
        # synthetic seed [lo,hi]
        win_idx = valid_idx

    # --- filter bank ----------------------------------------------------------
    sigmas, cap1 = _make_sigma_list(N, sigma_list_ch, k_sigma, max_frac=0.48)
    bank = gaussian_kernel_bank(sigmas, k_sigma=k_sigma)
    if not bank:
        # fallback: synthetic seed
        comps, indices, info = _make_synthetic_seed(v, y, v_min, span, lo, hi, dv)
        debug = {"reason": "empty_bank_after_cap",
                 "N": int(N), "cap": float(cap1), **info}
        # max_ngauss==0
        if (max_ngauss is not None) and (int(max_ngauss) <= 0):
            return dict(ncomp=0, components=np.zeros((0,3)),
                        bg=float(bg), rms=float(rms), indices=[], debug=debug)
        return dict(ncomp=1, components=comps, bg=float(bg), rms=float(rms),
                    indices=indices, debug=debug)

    if NUMBA_OK:
        g_list, gnorm_list = NumbaList(), NumbaList()
    else:
        g_list, gnorm_list = [], []
    sum_g2_arr      = np.empty(len(bank), dtype=np.float64)
    sum_g2_sqrt_arr = np.empty(len(bank), dtype=np.float64)
    halfwidth_arr   = np.empty(len(bank), dtype=np.int64)
    sigma_ch_arr    = np.empty(len(bank), dtype=np.float64)
    for k, (g, sum_g2, g_norm, sum_g2_sqrt, h, s_ch) in enumerate(bank):
        g_list.append(g); gnorm_list.append(g_norm)
        sum_g2_arr[k]      = sum_g2
        sum_g2_sqrt_arr[k] = sum_g2_sqrt
        halfwidth_arr[k]   = h
        sigma_ch_arr[k]    = s_ch

    # --- convolution over interior -------------------------------------------
    if NUMBA_OK and int(numba_threads) == 1:
        Rstd, Rraw = _bank_convolve_and_standardize_seq(y, y_w, g_list, gnorm_list)
    else:
        Rstd, Rraw = _bank_convolve_and_standardize_par(y, y_w, g_list, gnorm_list)

    k_best = np.argmax(np.abs(Rstd), axis=0)
    Rbest  = Rstd[k_best, np.arange(N)]

    # --- candidates in window -------------------------------------------------
    #mask = (np.abs(Rbest) >= float(thres_sigma)) & valid_idx & win_idx
    mask = (np.abs(Rbest) >= float(thres_sigma)) & valid_idx
    cand_idx = np.flatnonzero(mask)

    debug = {
        "N": int(N), "bg": float(bg), "rms": float(rms),
        "sigma_cap1": float(cap1), "n_sigmas1": int(len(sigmas)),
        "Rbest_max_interior": float(np.max(np.abs(Rbest[valid_idx]))) if valid_idx.any() else 0.0,
        "cand_n1": int(cand_idx.size),
        "win_lo": float(lo), "win_hi": float(hi),
    }

    # retry with expanded bank if none
    if cand_idx.size == 0:
        sigmas2, cap2 = _make_sigma_list(N, np.r_[sigmas, 6,8,10,12,15,18], k_sigma, max_frac=0.90)
        bank2 = gaussian_kernel_bank(sigmas2, k_sigma=k_sigma)
        if bank2:
            if NUMBA_OK:
                g_list2, gnorm_list2 = NumbaList(), NumbaList()
            else:
                g_list2, gnorm_list2 = [], []
            sum_g2_arr2, sum_g2_sqrt_arr2 = np.empty(len(bank2)), np.empty(len(bank2))
            halfwidth_arr2  = np.empty(len(bank2), dtype=np.int64)
            sigma_ch_arr2   = np.empty(len(bank2), dtype=np.float64)
            for k, (g, sum_g2, g_norm, sum_g2_sqrt, h, s_ch) in enumerate(bank2):
                g_list2.append(g); gnorm_list2.append(g_norm)
                sum_g2_arr2[k]      = sum_g2
                sum_g2_sqrt_arr2[k] = sum_g2_sqrt
                halfwidth_arr2[k]   = h
                sigma_ch_arr2[k]    = s_ch
            if NUMBA_OK and int(numba_threads) == 1:
                R2, RR2 = _bank_convolve_and_standardize_seq(y, y_w, g_list2, gnorm_list2)
            else:
                R2, RR2 = _bank_convolve_and_standardize_par(y, y_w, g_list2, gnorm_list2)
            k_best2 = np.argmax(np.abs(R2), axis=0)
            Rbest2  = R2[k_best2, np.arange(N)]
            th2 = max(2.0, 0.6*float(thres_sigma))
            #cand_idx2 = np.flatnonzero((np.abs(Rbest2) >= th2) & valid_idx & win_idx)
            cand_idx2 = np.flatnonzero((np.abs(Rbest2) >= th2) & valid_idx)
            debug.update({
                "sigma_cap2": float(cap2),
                "n_sigmas2": int(len(sigmas2)),
                "Rbest2_max_interior": float(np.max(np.abs(Rbest2[valid_idx]))) if valid_idx.any() else 0.0,
                "cand_n2": int(cand_idx2.size),
            })
            if cand_idx2.size > 0:
                sigmas            = sigmas2
                Rstd, Rraw        = R2, RR2
                k_best, Rbest     = k_best2, Rbest2
                cand_idx          = cand_idx2
                sum_g2_arr        = sum_g2_arr2
                sum_g2_sqrt_arr   = sum_g2_sqrt_arr2
                halfwidth_arr     = halfwidth_arr2
                sigma_ch_arr      = sigma_ch_arr2

    # still none in window -> synthetic
    if cand_idx.size == 0:
        comps, indices, info = _make_synthetic_seed(v, y, v_min, span, lo, hi, dv)
        debug.update(info)
        if (max_ngauss is not None) and (int(max_ngauss) <= 0):
            return dict(ncomp=0, components=np.zeros((0,3)),
                        bg=float(bg), rms=float(rms), indices=[], debug=debug)
        return dict(ncomp=1, components=comps, bg=float(bg), rms=float(rms),
                    indices=indices, debug=debug)

    # --- NMS + amplitude gate -------------------------------------------------
    order = np.argsort(-np.abs(Rbest[cand_idx]))
    cand_idx = cand_idx[order]
    kept_idx, kept_sig, kept_A = [], [], []
    taken = np.zeros(N, dtype=bool)

    for idx in cand_idx:
        lo_nms = max(0, idx - int(sep_channels))
        hi_nms = min(N, idx + int(sep_channels) + 1)
        if taken[lo_nms:hi_nms].any():
            continue
        kb = int(k_best[idx])
        sum_g2      = sum_g2_arr[kb]
        sum_g2_sqrt = sum_g2_sqrt_arr[kb]
        s_ch        = sigma_ch_arr[kb]
        if detrend_local:
            h = int(halfwidth_arr[kb])
            w = int(max(h, detrend_halfwin))
            lsl = slice(max(0, idx - w), min(N, idx + w + 1))
            xv = np.arange(lsl.start, lsl.stop, dtype=float)
            yy = y[lsl]
            X = np.vstack([xv, np.ones_like(xv)]).T
            coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
            y_loc = y.copy()
            y_loc[lsl] = yy - (X @ coef)
            rr = _conv_same_reflect(y_loc, g_list[kb] if NUMBA_OK else bank[kb][0])
            A_hat = rr[idx] / max(1e-12, sum_g2)
        else:
            A_hat = Rraw[kb, idx] / max(1e-12, sum_g2)
        sigma_A = float(rms) / float(sum_g2_sqrt)
        if abs(A_hat) < float(amp_sigma_thres) * sigma_A:
            continue
        kept_idx.append(int(idx))
        kept_sig.append(float(s_ch))
        kept_A.append(float(A_hat))
        taken[lo_nms:hi_nms] = True
        if max_components is not None and len(kept_idx) >= int(max_components):
            break

    # all rejected -> synthetic
    if not kept_idx:
        comps, indices, info = _make_synthetic_seed(v, y, v_min, span, lo, hi, dv)
        debug.update({"reason": "all_candidates_failed_amp_gate", **info})
        if (max_ngauss is not None) and (int(max_ngauss) <= 0):
            return dict(ncomp=0, components=np.zeros((0,3)),
                        bg=float(bg), rms=float(rms), indices=[], debug=debug)
        return dict(ncomp=1, components=comps, bg=float(bg), rms=float(rms),
                    indices=indices, debug=debug)

    centers_ch = np.asarray(kept_idx, dtype=np.float64)
    kept_sig   = np.asarray(kept_sig, dtype=np.float64)
    amps       = np.asarray(kept_A,   dtype=np.float64)

    if refine_center:
        for j, i0 in enumerate(centers_ch.astype(int)):
            dx, _ = _parabolic_subsample_jit(Rbest, i0)
            centers_ch[j] += dx

    order2 = np.argsort(-np.abs(amps))
    centers_sorted   = centers_ch[order2]
    sigmas_ch_sorted = kept_sig[order2]
    amps_sorted      = amps[order2]

    sign = np.sign(v[-1] - v[0]) if v.size > 1 else 1.0
    x_vel   = v[0] + centers_sorted * (sign * dv)
    sigma_v = sigmas_ch_sorted * dv

    comps = np.stack([x_vel, sigma_v, np.abs(amps_sorted)], axis=1)
    indices = list(zip(np.rint(centers_sorted).astype(int).tolist(),
                       sigmas_ch_sorted.astype(float).tolist()))
    debug["kept"] = len(indices)

    # restrict to top-|amp|
    if max_ngauss is not None:
        k = int(max_ngauss)
        if k <= 0:
            return dict(ncomp=0, components=np.zeros((0,3)),
                        bg=float(bg), rms=float(rms), indices=[], debug={**debug, "reason":"max_ngauss<=0"})
        if k < comps.shape[0]:
            comps = comps[:k, :]
            indices = indices[:k]
            debug["kept_after_max_ngauss"] = k

    return dict(ncomp=comps.shape[0], components=comps, bg=float(bg), rms=float(rms),
                indices=indices, debug=debug)
















# use the full spectral window like the log-like windowing : should be used with loglike_d
def search_gaussian_seeds_matched_filter_norm0(
    v, f, *, rms=None, bg=None,
    sigma_list_ch=None, k_sigma=4.0,
    thres_sigma=3.0, amp_sigma_thres=3.0,
    sep_channels=5, max_components=None,
    refine_center=True, detrend_local=False, detrend_halfwin=8,
    numba_threads=1,
    x_min_norm=None, x_max_norm=None,
    max_ngauss=None
):
    """
    Find initial seeds for multi-Gaussian fitting using a matched filter,
    but restrict detection to the normalized velocity window (x_min_norm, x_max_norm).

    Parameters
    ----------
    v : array-like
        Physical velocity axis (monotonic).
    f : array-like
        Spectrum values (same length as v).
    ...
    x_min_norm, x_max_norm : float or None
        Normalized bounds in [0, 1]. If None, defaults to full range (0, 1).

    max_ngauss : int or None
        If given, return at most this many Gaussian seeds with the largest amplitudes.

    Returns
    -------
    dict
        (ncomp, components, bg, rms, indices, debug).
    """

    # --- (original preamble unchanged) ----------------------------------------
    if NUMBA_OK and (numba_threads is not None):
        try:
            set_num_threads(int(numba_threads))
        except Exception:
            pass

    v = np.asarray(v, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    N = f.size
    if N == 0:
        return dict(ncomp=0, components=np.zeros((0,3)),
                    bg=0.0, rms=1.0, indices=[], debug={"reason":"N=0"})

    dv = float(np.median(np.abs(np.diff(v)))) if N > 1 else 1.0

    # Robust background and rms (unchanged)
    if (bg is None) or (rms is None):
        bh, sh = _robust_bg_rms_emission_jit(f, clip_sigma=3.0, max_iter=8, min_bg_frac=0.25)
        if bg is None:  bg  = float(bh)
        if rms is None: rms = float(sh)

    if not np.isfinite(bg):  bg = float(np.median(f))
    y = f - bg
    if (not np.isfinite(rms)) or (rms <= 0):
        mad = np.median(np.abs(y - np.median(y)))
        rms = float(1.4826 * mad) if mad > 0 else float(np.std(y) + 1e-12)
    y_w = np.nan_to_num(y / rms, nan=0.0, posinf=0.0, neginf=0.0)

    # sigma list & kernel bank (unchanged)
    sigmas, cap1 = _make_sigma_list(N, sigma_list_ch, k_sigma, max_frac=0.48)
    bank = gaussian_kernel_bank(sigmas, k_sigma=k_sigma)
    if not bank:
        return dict(ncomp=0, components=np.zeros((0,3)), bg=float(bg), rms=float(rms),
                    indices=[], debug={"reason":"empty_bank_after_cap","N":int(N),"cap":float(cap1)})

    # Prepare lists/arrays (unchanged)
    if NUMBA_OK:
        g_list     = NumbaList()
        gnorm_list = NumbaList()
    else:
        g_list, gnorm_list = [], []

    sum_g2_arr        = np.empty(len(bank), dtype=np.float64)
    sum_g2_sqrt_arr   = np.empty(len(bank), dtype=np.float64)
    halfwidth_arr     = np.empty(len(bank), dtype=np.int64)
    sigma_ch_arr      = np.empty(len(bank), dtype=np.float64)
    for k, (g, sum_g2, g_norm, sum_g2_sqrt, h, s_ch) in enumerate(bank):
        g_list.append(g)
        gnorm_list.append(g_norm)
        sum_g2_arr[k]      = sum_g2
        sum_g2_sqrt_arr[k] = sum_g2_sqrt
        halfwidth_arr[k]   = h
        sigma_ch_arr[k]    = s_ch

    # --- New: build the valid window mask on normalized velocity -------------
    # Normalize v to [0,1] using its physical min/max (handles dv sign).
    v_min = np.minimum(v[0], v[-1])
    v_max = np.maximum(v[0], v[-1])
    span  = max(1e-30, (v_max - v_min))
    v_norm = (v - v_min) / span

    # Defaults: full range
    if x_min_norm is None: x_min_norm = 0.0
    if x_max_norm is None: x_max_norm = 1.0
    lo = x_min_norm if x_min_norm <= x_max_norm else x_max_norm
    hi = x_max_norm if x_max_norm >= x_min_norm else x_min_norm

    # Strict interior to mirror the log-like windowing
    # Strict interior to mirror the log-like windowing : should be used with loglike_d_x_window
    #valid_idx = (v_norm > lo) & (v_norm < hi)
    valid_idx = (v_norm > 0) & (v_norm < 1)
    n_valid = int(np.count_nonzero(valid_idx))
    if n_valid == 0:
        return dict(ncomp=0, components=np.zeros((0,3)), bg=float(bg), rms=float(rms),
                    indices=[], debug={"reason":"empty_window","lo":float(lo),"hi":float(hi)})

    # --- Hot path: convolve bank and standardize -----------------
    if NUMBA_OK and int(numba_threads) == 1:
        Rstd, Rraw = _bank_convolve_and_standardize_seq(y, y_w, g_list, gnorm_list)
    else:
        Rstd, Rraw = _bank_convolve_and_standardize_par(y, y_w, g_list, gnorm_list)

    # Choose best sigma per channel
    k_best = np.argmax(np.abs(Rstd), axis=0)
    Rbest  = Rstd[k_best, np.arange(N)]

    # First-pass candidates: restrict to window
    mask = (np.abs(Rbest) >= float(thres_sigma)) & valid_idx
    cand_idx = np.flatnonzero(mask)

    debug = {
        "N": int(N), "bg": float(bg), "rms": float(rms),
        "sigma_cap1": float(cap1), "n_sigmas1": int(len(sigmas)),
        "Rbest_max_window": float(np.max(np.abs(Rbest[valid_idx]))),
        "cand_n1": int(cand_idx.size),
        "win_lo": float(lo), "win_hi": float(hi),
    }

    # If no candidates, expand sigma bank and retry (still within window)
    if cand_idx.size == 0:
        sigmas2, cap2 = _make_sigma_list(N, np.r_[sigmas, 6,8,10,12,15,18], k_sigma, max_frac=0.90)
        bank2 = gaussian_kernel_bank(sigmas2, k_sigma=k_sigma)
        if bank2:
            if NUMBA_OK:
                g_list2     = NumbaList()
                gnorm_list2 = NumbaList()
            else:
                g_list2, gnorm_list2 = [], []
            sum_g2_arr2        = np.empty(len(bank2), dtype=np.float64)
            sum_g2_sqrt_arr2   = np.empty(len(bank2), dtype=np.float64)
            halfwidth_arr2     = np.empty(len(bank2), dtype=np.int64)
            sigma_ch_arr2      = np.empty(len(bank2), dtype=np.float64)
            for k, (g, sum_g2, g_norm, sum_g2_sqrt, h, s_ch) in enumerate(bank2):
                g_list2.append(g); gnorm_list2.append(g_norm)
                sum_g2_arr2[k]      = sum_g2
                sum_g2_sqrt_arr2[k] = sum_g2_sqrt
                halfwidth_arr2[k]   = h
                sigma_ch_arr2[k]    = s_ch

            if NUMBA_OK and int(numba_threads) == 1:
                R2, RR2 = _bank_convolve_and_standardize_seq(y, y_w, g_list2, gnorm_list2)
            else:
                R2, RR2 = _bank_convolve_and_standardize_par(y, y_w, g_list2, gnorm_list2)

            k_best2 = np.argmax(np.abs(R2), axis=0)
            Rbest2  = R2[k_best2, np.arange(N)]
            th2 = max(2.0, 0.6*float(thres_sigma))
            cand_idx2 = np.flatnonzero((np.abs(Rbest2) >= th2) & valid_idx)

            debug.update({
                "sigma_cap2": float(cap2),
                "n_sigmas2": int(len(sigmas2)),
                "Rbest2_max_window": float(np.max(np.abs(Rbest2[valid_idx]))),
                "cand_n2": int(cand_idx2.size),
            })

            if cand_idx2.size > 0:
                # Switch to the new bank
                sigmas            = sigmas2
                Rstd, Rraw        = R2, RR2
                k_best, Rbest     = k_best2, Rbest2
                cand_idx          = cand_idx2
                sum_g2_arr        = sum_g2_arr2
                sum_g2_sqrt_arr   = sum_g2_sqrt_arr2
                halfwidth_arr     = halfwidth_arr2
                sigma_ch_arr      = sigma_ch_arr2

    # Structural rescue when still no candidates: choose within window only
    if cand_idx.size == 0:
        win_idx = np.flatnonzero(valid_idx)
        if win_idx.size > 0:
            i_rel = int(np.argmax(np.abs(Rbest[win_idx])))
            i0 = int(win_idx[i_rel])
        else:
            i0 = -1

        if i0 >= 0 and np.isfinite(Rbest[i0]):
            kb  = int(k_best[i0])
            A_hat   = Rraw[kb, i0] / max(1e-12, sum_g2_arr[kb])
            sigma_A = float(rms) / float(sum_g2_sqrt_arr[kb])
            debug.update({"rescue_i0": i0, "rescue_Rbest": float(Rbest[i0]),
                          "rescue_A_hat": float(A_hat),
                          "rescue_sigma_A": float(sigma_A)})
            if abs(A_hat) >= max(2.5, 0.8*float(amp_sigma_thres)) * sigma_A:
                sign = np.sign(v[-1] - v[0]) if v.size > 1 else 1.0
                center_ch = float(i0)
                if refine_center:
                    dx, _ = _parabolic_subsample_jit(Rbest, i0)
                    center_ch += dx
                x_vel   = v[0] + center_ch * (sign * dv)
                sigma_v = float(sigma_ch_arr[kb]) * dv
                comps = np.array([[abs(float(A_hat)), float(x_vel), float(sigma_v)]], dtype=float)
                indices = [(int(np.rint(center_ch)), float(sigma_ch_arr[kb]))]
                return dict(ncomp=1, components=comps, bg=float(bg), rms=float(rms),
                            indices=indices, debug=debug)

        debug["reason"] = "no_candidates_after_rescue"
        return dict(ncomp=0, components=np.zeros((0,3)), bg=float(bg), rms=float(rms),
                    indices=[], debug=debug)

    # Normal path: NMS + amplitude gate (unchanged except cand_idx already windowed)
    order = np.argsort(-np.abs(Rbest[cand_idx]))
    cand_idx = cand_idx[order]
    kept_idx, kept_sig, kept_A = [], [], []
    taken = np.zeros(N, dtype=bool)

    for idx in cand_idx:
        lo_nms = max(0, idx - int(sep_channels))
        hi_nms = min(N, idx + int(sep_channels) + 1)
        if taken[lo_nms:hi_nms].any():
            continue

        kb  = int(k_best[idx])
        sum_g2       = sum_g2_arr[kb]
        sum_g2_sqrt  = sum_g2_sqrt_arr[kb]
        s_ch         = sigma_ch_arr[kb]

        if detrend_local:
            h = int(halfwidth_arr[kb])
            w = int(max(h, detrend_halfwin))
            lsl = slice(max(0, idx - w), min(N, idx + w + 1))
            xv = np.arange(lsl.start, lsl.stop, dtype=float)
            yy = y[lsl]
            X = np.vstack([xv, np.ones_like(xv)]).T
            coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
            y_loc = y.copy()
            y_loc[lsl] = yy - (X @ coef)
            rr = _conv_same_reflect(y_loc, g_list[kb] if NUMBA_OK else bank[kb][0])
            A_hat = rr[idx] / max(1e-12, sum_g2)
        else:
            A_hat = Rraw[kb, idx] / max(1e-12, sum_g2)

        sigma_A = float(rms) / float(sum_g2_sqrt)
        if abs(A_hat) < float(amp_sigma_thres) * sigma_A:
            continue

        kept_idx.append(int(idx))
        kept_sig.append(float(s_ch))
        kept_A.append(float(A_hat))
        taken[lo_nms:hi_nms] = True

        if max_components is not None and len(kept_idx) >= int(max_components):
            break

    if not kept_idx:
        debug["reason"] = "all_candidates_failed_amp_gate"
        return dict(ncomp=0, components=np.zeros((0,3)), bg=float(bg), rms=float(rms),
                    indices=[], debug=debug)

    centers_ch = np.asarray(kept_idx, dtype=np.float64)
    kept_sig   = np.asarray(kept_sig, dtype=np.float64)
    amps       = np.asarray(kept_A,   dtype=np.float64)

    if refine_center:
        for j, i0 in enumerate(centers_ch.astype(int)):
            dx, _ = _parabolic_subsample_jit(Rbest, i0)
            centers_ch[j] += dx

    order2 = np.argsort(-np.abs(amps))
    centers_sorted   = centers_ch[order2]
    sigmas_ch_sorted = kept_sig[order2]
    amps_sorted      = amps[order2]

    sign = np.sign(v[-1] - v[0]) if v.size > 1 else 1.0
    x_vel   = v[0] + centers_sorted * (sign * dv)
    sigma_v = sigmas_ch_sorted * dv

    # comps = [center(vel), sigma(vel), |amplitude|]  (amp 내림차순)
    comps = np.stack([x_vel, sigma_v, np.abs(amps_sorted)], axis=1)
    indices = list(zip(np.rint(centers_sorted).astype(int).tolist(),
                       sigmas_ch_sorted.astype(float).tolist()))
    debug["kept"] = len(indices)

    # ---: limit by max_ngauss (top-|amp| already) ---
    if max_ngauss is not None:
        k = int(max_ngauss)
        if k < comps.shape[0]:
            comps = comps[:k, :]
            indices = indices[:k]
            debug["kept_after_max_ngauss"] = k

    return dict(ncomp=comps.shape[0], components=comps, bg=float(bg), rms=float(rms),
                indices=indices, debug=debug)

#-- END OF SUB-ROUTINE____________________________________________________________#







# Strict interior to mirror the log-like windowing : should be used with loglike_d_x_window
def search_gaussian_seeds_matched_filter_norm_x_window(
    v, f, *, rms=None, bg=None,
    sigma_list_ch=None, k_sigma=4.0,
    thres_sigma=3.0, amp_sigma_thres=3.0,
    sep_channels=5, max_components=None,
    refine_center=True, detrend_local=False, detrend_halfwin=8,
    numba_threads=1,
    x_min_norm=None, x_max_norm=None,
    max_ngauss=None
):
    """
    Find initial seeds for multi-Gaussian fitting using a matched filter,
    but restrict detection to the normalized velocity window (x_min_norm, x_max_norm).

    Parameters
    ----------
    v : array-like
        Physical velocity axis (monotonic).
    f : array-like
        Spectrum values (same length as v).
    ...
    x_min_norm, x_max_norm : float or None
        Normalized bounds in [0, 1]. If None, defaults to full range (0, 1).

    max_ngauss : int or None
        If given, return at most this many Gaussian seeds with the largest amplitudes.

    Returns
    -------
    dict
        (ncomp, components, bg, rms, indices, debug).
    """

    # --- (original preamble unchanged) ----------------------------------------
    if NUMBA_OK and (numba_threads is not None):
        try:
            set_num_threads(int(numba_threads))
        except Exception:
            pass

    v = np.asarray(v, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    N = f.size
    if N == 0:
        return dict(ncomp=0, components=np.zeros((0,3)),
                    bg=0.0, rms=1.0, indices=[], debug={"reason":"N=0"})

    dv = float(np.median(np.abs(np.diff(v)))) if N > 1 else 1.0

    # Robust background and rms (unchanged)
    if (bg is None) or (rms is None):
        bh, sh = _robust_bg_rms_emission_jit(f, clip_sigma=3.0, max_iter=8, min_bg_frac=0.25)
        if bg is None:  bg  = float(bh)
        if rms is None: rms = float(sh)

    if not np.isfinite(bg):  bg = float(np.median(f))
    y = f - bg
    if (not np.isfinite(rms)) or (rms <= 0):
        mad = np.median(np.abs(y - np.median(y)))
        rms = float(1.4826 * mad) if mad > 0 else float(np.std(y) + 1e-12)
    y_w = np.nan_to_num(y / rms, nan=0.0, posinf=0.0, neginf=0.0)

    # sigma list & kernel bank (unchanged)
    sigmas, cap1 = _make_sigma_list(N, sigma_list_ch, k_sigma, max_frac=0.48)
    bank = gaussian_kernel_bank(sigmas, k_sigma=k_sigma)
    if not bank:
        return dict(ncomp=0, components=np.zeros((0,3)), bg=float(bg), rms=float(rms),
                    indices=[], debug={"reason":"empty_bank_after_cap","N":int(N),"cap":float(cap1)})

    # Prepare lists/arrays (unchanged)
    if NUMBA_OK:
        g_list     = NumbaList()
        gnorm_list = NumbaList()
    else:
        g_list, gnorm_list = [], []

    sum_g2_arr        = np.empty(len(bank), dtype=np.float64)
    sum_g2_sqrt_arr   = np.empty(len(bank), dtype=np.float64)
    halfwidth_arr     = np.empty(len(bank), dtype=np.int64)
    sigma_ch_arr      = np.empty(len(bank), dtype=np.float64)
    for k, (g, sum_g2, g_norm, sum_g2_sqrt, h, s_ch) in enumerate(bank):
        g_list.append(g)
        gnorm_list.append(g_norm)
        sum_g2_arr[k]      = sum_g2
        sum_g2_sqrt_arr[k] = sum_g2_sqrt
        halfwidth_arr[k]   = h
        sigma_ch_arr[k]    = s_ch

    # --- New: build the valid window mask on normalized velocity -------------
    # Normalize v to [0,1] using its physical min/max (handles dv sign).
    v_min = np.minimum(v[0], v[-1])
    v_max = np.maximum(v[0], v[-1])
    span  = max(1e-30, (v_max - v_min))
    v_norm = (v - v_min) / span

    # Defaults: full range
    if x_min_norm is None: x_min_norm = 0.0
    if x_max_norm is None: x_max_norm = 1.0
    lo = x_min_norm if x_min_norm <= x_max_norm else x_max_norm
    hi = x_max_norm if x_max_norm >= x_min_norm else x_min_norm

    # Strict interior to mirror the log-like windowing
    # Strict interior to mirror the log-like windowing : should be used with loglike_d_x_window
    valid_idx = (v_norm > lo) & (v_norm < hi)
    n_valid = int(np.count_nonzero(valid_idx))
    if n_valid == 0:
        return dict(ncomp=0, components=np.zeros((0,3)), bg=float(bg), rms=float(rms),
                    indices=[], debug={"reason":"empty_window","lo":float(lo),"hi":float(hi)})

    # --- Hot path: convolve bank and standardize (unchanged) -----------------
    if NUMBA_OK and int(numba_threads) == 1:
        Rstd, Rraw = _bank_convolve_and_standardize_seq(y, y_w, g_list, gnorm_list)
    else:
        Rstd, Rraw = _bank_convolve_and_standardize_par(y, y_w, g_list, gnorm_list)

    # Choose best sigma per channel
    k_best = np.argmax(np.abs(Rstd), axis=0)
    Rbest  = Rstd[k_best, np.arange(N)]

    # First-pass candidates: restrict to window
    mask = (np.abs(Rbest) >= float(thres_sigma)) & valid_idx
    cand_idx = np.flatnonzero(mask)

    debug = {
        "N": int(N), "bg": float(bg), "rms": float(rms),
        "sigma_cap1": float(cap1), "n_sigmas1": int(len(sigmas)),
        "Rbest_max_window": float(np.max(np.abs(Rbest[valid_idx]))),
        "cand_n1": int(cand_idx.size),
        "win_lo": float(lo), "win_hi": float(hi),
    }

    # If no candidates, expand sigma bank and retry (still within window)
    if cand_idx.size == 0:
        sigmas2, cap2 = _make_sigma_list(N, np.r_[sigmas, 6,8,10,12,15,18], k_sigma, max_frac=0.90)
        bank2 = gaussian_kernel_bank(sigmas2, k_sigma=k_sigma)
        if bank2:
            if NUMBA_OK:
                g_list2     = NumbaList()
                gnorm_list2 = NumbaList()
            else:
                g_list2, gnorm_list2 = [], []
            sum_g2_arr2        = np.empty(len(bank2), dtype=np.float64)
            sum_g2_sqrt_arr2   = np.empty(len(bank2), dtype=np.float64)
            halfwidth_arr2     = np.empty(len(bank2), dtype=np.int64)
            sigma_ch_arr2      = np.empty(len(bank2), dtype=np.float64)
            for k, (g, sum_g2, g_norm, sum_g2_sqrt, h, s_ch) in enumerate(bank2):
                g_list2.append(g); gnorm_list2.append(g_norm)
                sum_g2_arr2[k]      = sum_g2
                sum_g2_sqrt_arr2[k] = sum_g2_sqrt
                halfwidth_arr2[k]   = h
                sigma_ch_arr2[k]    = s_ch

            if NUMBA_OK and int(numba_threads) == 1:
                R2, RR2 = _bank_convolve_and_standardize_seq(y, y_w, g_list2, gnorm_list2)
            else:
                R2, RR2 = _bank_convolve_and_standardize_par(y, y_w, g_list2, gnorm_list2)

            k_best2 = np.argmax(np.abs(R2), axis=0)
            Rbest2  = R2[k_best2, np.arange(N)]
            th2 = max(2.0, 0.6*float(thres_sigma))
            cand_idx2 = np.flatnonzero((np.abs(Rbest2) >= th2) & valid_idx)

            debug.update({
                "sigma_cap2": float(cap2),
                "n_sigmas2": int(len(sigmas2)),
                "Rbest2_max_window": float(np.max(np.abs(Rbest2[valid_idx]))),
                "cand_n2": int(cand_idx2.size),
            })

            if cand_idx2.size > 0:
                # Switch to the new bank
                sigmas            = sigmas2
                Rstd, Rraw        = R2, RR2
                k_best, Rbest     = k_best2, Rbest2
                cand_idx          = cand_idx2
                sum_g2_arr        = sum_g2_arr2
                sum_g2_sqrt_arr   = sum_g2_sqrt_arr2
                halfwidth_arr     = halfwidth_arr2
                sigma_ch_arr      = sigma_ch_arr2

    # Structural rescue when still no candidates: choose within window only
    if cand_idx.size == 0:
        win_idx = np.flatnonzero(valid_idx)
        if win_idx.size > 0:
            i_rel = int(np.argmax(np.abs(Rbest[win_idx])))
            i0 = int(win_idx[i_rel])
        else:
            i0 = -1

        if i0 >= 0 and np.isfinite(Rbest[i0]):
            kb  = int(k_best[i0])
            A_hat   = Rraw[kb, i0] / max(1e-12, sum_g2_arr[kb])
            sigma_A = float(rms) / float(sum_g2_sqrt_arr[kb])
            debug.update({"rescue_i0": i0, "rescue_Rbest": float(Rbest[i0]),
                          "rescue_A_hat": float(A_hat),
                          "rescue_sigma_A": float(sigma_A)})
            if abs(A_hat) >= max(2.5, 0.8*float(amp_sigma_thres)) * sigma_A:
                sign = np.sign(v[-1] - v[0]) if v.size > 1 else 1.0
                center_ch = float(i0)
                if refine_center:
                    dx, _ = _parabolic_subsample_jit(Rbest, i0)
                    center_ch += dx
                x_vel   = v[0] + center_ch * (sign * dv)
                sigma_v = float(sigma_ch_arr[kb]) * dv
                comps = np.array([[abs(float(A_hat)), float(x_vel), float(sigma_v)]], dtype=float)
                indices = [(int(np.rint(center_ch)), float(sigma_ch_arr[kb]))]
                return dict(ncomp=1, components=comps, bg=float(bg), rms=float(rms),
                            indices=indices, debug=debug)

        debug["reason"] = "no_candidates_after_rescue"
        return dict(ncomp=0, components=np.zeros((0,3)), bg=float(bg), rms=float(rms),
                    indices=[], debug=debug)

    # Normal path: NMS + amplitude gate (unchanged except cand_idx already windowed)
    order = np.argsort(-np.abs(Rbest[cand_idx]))
    cand_idx = cand_idx[order]
    kept_idx, kept_sig, kept_A = [], [], []
    taken = np.zeros(N, dtype=bool)

    for idx in cand_idx:
        lo_nms = max(0, idx - int(sep_channels))
        hi_nms = min(N, idx + int(sep_channels) + 1)
        if taken[lo_nms:hi_nms].any():
            continue

        kb  = int(k_best[idx])
        sum_g2       = sum_g2_arr[kb]
        sum_g2_sqrt  = sum_g2_sqrt_arr[kb]
        s_ch         = sigma_ch_arr[kb]

        if detrend_local:
            h = int(halfwidth_arr[kb])
            w = int(max(h, detrend_halfwin))
            lsl = slice(max(0, idx - w), min(N, idx + w + 1))
            xv = np.arange(lsl.start, lsl.stop, dtype=float)
            yy = y[lsl]
            X = np.vstack([xv, np.ones_like(xv)]).T
            coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
            y_loc = y.copy()
            y_loc[lsl] = yy - (X @ coef)
            rr = _conv_same_reflect(y_loc, g_list[kb] if NUMBA_OK else bank[kb][0])
            A_hat = rr[idx] / max(1e-12, sum_g2)
        else:
            A_hat = Rraw[kb, idx] / max(1e-12, sum_g2)

        sigma_A = float(rms) / float(sum_g2_sqrt)
        if abs(A_hat) < float(amp_sigma_thres) * sigma_A:
            continue

        kept_idx.append(int(idx))
        kept_sig.append(float(s_ch))
        kept_A.append(float(A_hat))
        taken[lo_nms:hi_nms] = True

        if max_components is not None and len(kept_idx) >= int(max_components):
            break

    if not kept_idx:
        debug["reason"] = "all_candidates_failed_amp_gate"
        return dict(ncomp=0, components=np.zeros((0,3)), bg=float(bg), rms=float(rms),
                    indices=[], debug=debug)

    centers_ch = np.asarray(kept_idx, dtype=np.float64)
    kept_sig   = np.asarray(kept_sig, dtype=np.float64)
    amps       = np.asarray(kept_A,   dtype=np.float64)

    if refine_center:
        for j, i0 in enumerate(centers_ch.astype(int)):
            dx, _ = _parabolic_subsample_jit(Rbest, i0)
            centers_ch[j] += dx

    order2 = np.argsort(-np.abs(amps))
    centers_sorted   = centers_ch[order2]
    sigmas_ch_sorted = kept_sig[order2]
    amps_sorted      = amps[order2]

    sign = np.sign(v[-1] - v[0]) if v.size > 1 else 1.0
    x_vel   = v[0] + centers_sorted * (sign * dv)
    sigma_v = sigmas_ch_sorted * dv

    # comps = [center(vel), sigma(vel), |amplitude|]  (amp 내림차순)
    comps = np.stack([x_vel, sigma_v, np.abs(amps_sorted)], axis=1)
    indices = list(zip(np.rint(centers_sorted).astype(int).tolist(),
                       sigmas_ch_sorted.astype(float).tolist()))
    debug["kept"] = len(indices)

    # ---: limit by max_ngauss (top-|amp| already) ---
    if max_ngauss is not None:
        k = int(max_ngauss)
        if k < comps.shape[0]:
            comps = comps[:k, :]
            indices = indices[:k]
            debug["kept_after_max_ngauss"] = k

    return dict(ncomp=comps.shape[0], components=comps, bg=float(bg), rms=float(rms),
                indices=indices, debug=debug)

#-- END OF SUB-ROUTINE____________________________________________________________#


# --------------- Main function (overhead-minimized) ----------
def search_gaussian_seeds_matched_filter_norm_non_mask(
    v, f, *, rms=None, bg=None,
    sigma_list_ch=None, k_sigma=4.0,
    thres_sigma=3.0, amp_sigma_thres=3.0,
    sep_channels=5, max_components=None,
    refine_center=True, detrend_local=False, detrend_halfwin=8,
    numba_threads=1  # Default 1: with Ray, avoid oversubscription + remove parallel overhead.
):
    
    # USE RMS to set model_sigma boundary
    # Numba threads setup (optional)
    if NUMBA_OK and (numba_threads is not None):
        try:
            set_num_threads(int(numba_threads))
        except Exception:
            pass

    v = np.asarray(v, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    N = f.size
    if N == 0:
        return dict(ncomp=0, components=np.zeros((0,3)),
                    bg=0.0, rms=1.0, indices=[], debug={"reason":"N=0"})

    dv = float(np.median(np.abs(np.diff(v)))) if N > 1 else 1.0

    # Robust background and rms
    if (bg is None) or (rms is None):
        bh, sh = _robust_bg_rms_emission_jit(f, clip_sigma=3.0, max_iter=8, min_bg_frac=0.25)
        if bg is None:  bg  = float(bh)
        if rms is None: rms = float(sh)

    if not np.isfinite(bg):  bg = float(np.median(f))
    y = f - bg
    if (not np.isfinite(rms)) or (rms <= 0):
        mad = np.median(np.abs(y - np.median(y)))
        rms = float(1.4826 * mad) if mad > 0 else float(np.std(y) + 1e-12)
    y_w = np.nan_to_num(y / rms, nan=0.0, posinf=0.0, neginf=0.0)

    # sigma list & kernel bank
    sigmas, cap1 = _make_sigma_list(N, sigma_list_ch, k_sigma, max_frac=0.48)
    bank = gaussian_kernel_bank(sigmas, k_sigma=k_sigma)
    if not bank:
        return dict(ncomp=0, components=np.zeros((0,3)), bg=float(bg), rms=float(rms),
                    indices=[], debug={"reason":"empty_bank_after_cap","N":int(N),"cap":float(cap1)})

    # Wrap into typed lists (Numba parallel/single both use this)
    if NUMBA_OK:
        g_list     = NumbaList()
        gnorm_list = NumbaList()
    else:
        g_list, gnorm_list = [], []

    sum_g2_arr        = np.empty(len(bank), dtype=np.float64)
    sum_g2_sqrt_arr   = np.empty(len(bank), dtype=np.float64)
    halfwidth_arr     = np.empty(len(bank), dtype=np.int64)
    sigma_ch_arr      = np.empty(len(bank), dtype=np.float64)
    for k, (g, sum_g2, g_norm, sum_g2_sqrt, h, s_ch) in enumerate(bank):
        g_list.append(g)
        gnorm_list.append(g_norm)
        sum_g2_arr[k]      = sum_g2
        sum_g2_sqrt_arr[k] = sum_g2_sqrt
        halfwidth_arr[k]   = h
        sigma_ch_arr[k]    = s_ch

    # --- Hot path: convolve entire bank + robust standardization ---
    if NUMBA_OK and int(numba_threads) == 1:
        Rstd, Rraw = _bank_convolve_and_standardize_seq(y, y_w, g_list, gnorm_list)
    else:
        # Parallel (Numba) or NumPy fallback
        Rstd, Rraw = _bank_convolve_and_standardize_par(y, y_w, g_list, gnorm_list)

    # Choose best sigma per channel
    k_best = np.argmax(np.abs(Rstd), axis=0)
    Rbest  = Rstd[k_best, np.arange(N)]

    # First-pass candidates
    mask = np.abs(Rbest) >= float(thres_sigma)
    cand_idx = np.flatnonzero(mask)

    debug = {
        "N": int(N), "bg": float(bg), "rms": float(rms),
        "sigma_cap1": float(cap1), "n_sigmas1": int(len(sigmas)),
        "Rbest_max": float(np.max(np.abs(Rbest))) if N>0 else 0.0,
        "cand_n1": int(cand_idx.size),
    }

    # If no candidates, expand sigma bank and retry
    if cand_idx.size == 0:
        sigmas2, cap2 = _make_sigma_list(N, np.r_[sigmas, 6,8,10,12,15,18], k_sigma, max_frac=0.90)
        bank2 = gaussian_kernel_bank(sigmas2, k_sigma=k_sigma)
        if bank2:
            if NUMBA_OK:
                g_list2     = NumbaList()
                gnorm_list2 = NumbaList()
            else:
                g_list2, gnorm_list2 = [], []
            sum_g2_arr2        = np.empty(len(bank2), dtype=np.float64)
            sum_g2_sqrt_arr2   = np.empty(len(bank2), dtype=np.float64)
            halfwidth_arr2     = np.empty(len(bank2), dtype=np.int64)
            sigma_ch_arr2      = np.empty(len(bank2), dtype=np.float64)
            for k, (g, sum_g2, g_norm, sum_g2_sqrt, h, s_ch) in enumerate(bank2):
                g_list2.append(g); gnorm_list2.append(g_norm)
                sum_g2_arr2[k]      = sum_g2
                sum_g2_sqrt_arr2[k] = sum_g2_sqrt
                halfwidth_arr2[k]   = h
                sigma_ch_arr2[k]    = s_ch

            if NUMBA_OK and int(numba_threads) == 1:
                R2, RR2 = _bank_convolve_and_standardize_seq(y, y_w, g_list2, gnorm_list2)
            else:
                R2, RR2 = _bank_convolve_and_standardize_par(y, y_w, g_list2, gnorm_list2)

            k_best2 = np.argmax(np.abs(R2), axis=0)
            Rbest2  = R2[k_best2, np.arange(N)]
            th2 = max(2.0, 0.6*float(thres_sigma))
            cand_idx2 = np.flatnonzero(np.abs(Rbest2) >= th2)

            debug.update({
                "sigma_cap2": float(cap2),
                "n_sigmas2": int(len(sigmas2)),
                "Rbest2_max": float(np.max(np.abs(Rbest2))),
                "cand_n2": int(cand_idx2.size),
            })

            if cand_idx2.size > 0:
                # Switch to the new bank
                sigmas            = sigmas2
                Rstd, Rraw        = R2, RR2
                k_best, Rbest     = k_best2, Rbest2
                cand_idx          = cand_idx2
                sum_g2_arr        = sum_g2_arr2
                sum_g2_sqrt_arr   = sum_g2_sqrt_arr2
                halfwidth_arr     = halfwidth_arr2
                sigma_ch_arr      = sigma_ch_arr2

    # Structural rescue when still no candidates
    if cand_idx.size == 0:
        i0 = int(np.argmax(np.abs(Rbest))) if N>0 else -1
        if i0 >= 0 and np.isfinite(Rbest[i0]):
            kb  = int(k_best[i0])
            A_hat   = Rraw[kb, i0] / max(1e-12, sum_g2_arr[kb])
            sigma_A = float(rms) / float(sum_g2_sqrt_arr[kb])
            debug.update({"rescue_i0": i0, "rescue_Rbest": float(Rbest[i0]),
                          "rescue_A_hat": float(A_hat),
                          "rescue_sigma_A": float(sigma_A)})
            if abs(A_hat) >= max(2.5, 0.8*float(amp_sigma_thres)) * sigma_A:
                sign = np.sign(v[-1] - v[0]) if v.size > 1 else 1.0
                center_ch = float(i0)
                if refine_center:
                    dx, _ = _parabolic_subsample_jit(Rbest, i0)
                    center_ch += dx
                x_vel   = v[0] + center_ch * (sign * dv)
                sigma_v = float(sigma_ch_arr[kb]) * dv
                comps = np.array([[abs(float(A_hat)), float(x_vel), float(sigma_v)]], dtype=float)
                indices = [(int(np.rint(center_ch)), float(sigma_ch_arr[kb]))]
                return dict(ncomp=1, components=comps, bg=float(bg), rms=float(rms),
                            indices=indices, debug=debug)
        debug["reason"] = "no_candidates_after_rescue"
        return dict(ncomp=0, components=np.zeros((0,3)), bg=float(bg), rms=float(rms),
                    indices=[], debug=debug)

    # Normal path: NMS + amplitude gate
    order = np.argsort(-np.abs(Rbest[cand_idx]))
    cand_idx = cand_idx[order]
    kept_idx, kept_sig, kept_A = [], [], []
    taken = np.zeros(N, dtype=bool)

    for idx in cand_idx:
        lo = max(0, idx - int(sep_channels))
        hi = min(N, idx + int(sep_channels) + 1)
        if taken[lo:hi].any():
            continue

        kb  = int(k_best[idx])
        sum_g2       = sum_g2_arr[kb]
        sum_g2_sqrt  = sum_g2_sqrt_arr[kb]
        s_ch         = sigma_ch_arr[kb]

        if detrend_local:
            h = int(halfwidth_arr[kb])
            w = int(max(h, detrend_halfwin))
            lsl = slice(max(0, idx - w), min(N, idx + w + 1))
            xv = np.arange(lsl.start, lsl.stop, dtype=float)
            yy = y[lsl]
            X = np.vstack([xv, np.ones_like(xv)]).T
            coef, *_ = np.linalg.lstsq(X, yy, rcond=None)  # BLAS may be used --> pin threads recommended
            y_loc = y.copy()
            y_loc[lsl] = yy - (X @ coef)
            rr = _conv_same_reflect(y_loc, g_list[kb] if NUMBA_OK else bank[kb][0])
            A_hat = rr[idx] / max(1e-12, sum_g2)
        else:
            A_hat = Rraw[kb, idx] / max(1e-12, sum_g2)

        sigma_A = float(rms) / float(sum_g2_sqrt)
        if abs(A_hat) < float(amp_sigma_thres) * sigma_A:
            continue

        kept_idx.append(int(idx))
        kept_sig.append(float(s_ch))
        kept_A.append(float(A_hat))
        taken[lo:hi] = True

        if max_components is not None and len(kept_idx) >= int(max_components):
            break

    if not kept_idx:
        debug["reason"] = "all_candidates_failed_amp_gate"
        return dict(ncomp=0, components=np.zeros((0,3)), bg=float(bg), rms=float(rms),
                    indices=[], debug=debug)

    centers_ch = np.asarray(kept_idx, dtype=np.float64)
    kept_sig   = np.asarray(kept_sig, dtype=np.float64)
    amps       = np.asarray(kept_A,   dtype=np.float64)

    if refine_center:
        for j, i0 in enumerate(centers_ch.astype(int)):
            dx, _ = _parabolic_subsample_jit(Rbest, i0)
            centers_ch[j] += dx

    order2 = np.argsort(-np.abs(amps))
    centers_sorted   = centers_ch[order2]
    sigmas_ch_sorted = kept_sig[order2]
    amps_sorted      = amps[order2]

    sign = np.sign(v[-1] - v[0]) if v.size > 1 else 1.0
    x_vel   = v[0] + centers_sorted * (sign * dv)
    sigma_v = sigmas_ch_sorted * dv

    comps = np.stack([x_vel, sigma_v, np.abs(amps_sorted)], axis=1)
    indices = list(zip(np.rint(centers_sorted).astype(int).tolist(),
                       sigmas_ch_sorted.astype(float).tolist()))
    debug["kept"] = len(indices)

    return dict(ncomp=comps.shape[0], components=comps, bg=float(bg), rms=float(rms),
                indices=indices, debug=debug)
#-- END OF SUB-ROUTINE____________________________________________________________#



def _clamp_01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _clamp_window(x_lo: float, x_hi: float, w_lo: float | None, w_hi: float | None,
                  min_span: float = 2e-2):
    """
    Clamp [x_lo, x_hi] to a given window [w_lo, w_hi] in [0,1].
    If clamping degenerates the interval, re-center with a small minimal span.
    """
    if (w_lo is None) or (w_hi is None):
        lo = _clamp_01(x_lo)
        hi = _clamp_01(x_hi)
        if hi < lo:
            lo, hi = hi, lo
        if (hi - lo) < min_span:
            mid = 0.5 * (lo + hi)
            lo = max(0.0, mid - 0.5 * min_span)
            hi = min(1.0, mid + 0.5 * min_span)
        return lo, hi

    lo_w = _clamp_01(min(w_lo, w_hi))
    hi_w = _clamp_01(max(w_lo, w_hi))
    lo = max(_clamp_01(x_lo), lo_w)
    hi = min(_clamp_01(x_hi), hi_w)

    if hi < lo:
        #mid = 0.5 * (lo_w + hi_w)
        #lo = max(0.0, mid - 0.5 * min_span)
        #hi = min(1.0, mid + 0.5 * min_span)

        lo = w_lo
        hi = w_hi

    if (hi - lo) < min_span:
        #mid = 0.5 * (lo + hi)
        #lo = max(0.0, mid - 0.5 * min_span)
        #hi = min(1.0, mid + 0.5 * min_span)

        lo = w_lo
        hi = w_hi

    return lo, hi





# ======= 2) Bounds from seeds, clamped to window with requested fallback =======
def set_sgfit_bounds_from_matched_filter_seeds_norm(
    _gaussian_seeds,
    *,
    # (1) model residual sigma bounds
    model_sigma_bounds=(0.0, 0.7),
    # (2) background upper bound = bg + k_bg * rms  (clipped to 1.0 for safety)
    k_bg=3.0,
    # (3) center bounds = [min(center) - k_x * sigma, max(center) + k_x * sigma]
    k_x=5.0,
    # (4) sigma bounds = [k_sig_lo * s_min, k_sig_hi * s_max]
    sigma_scale_bounds=(0.1, 3.0),
    # (5) peak bounds = [k_p_lo * A_min, k_p_hi * A_max]
    peak_scale_bounds=(0.3, 2.0),
    # Stabilization (minimum width)
    min_x_span=1e-5,
    min_sigma=1e-4,
    # Clipping upper bound (normalized axis is 0..1)
    clip_sigma_hi=0.999,
    # Physical axis options for computing x-bounds and sigma scaling
    use_phys_for_x_bounds=True,  # If True, compute x-bounds in physical axis and map back to normalized
    v_min=None, v_max=None,      # Physical velocity range
    cdelt3=None,                 # FITS CDELT3 (sign respected)
    v0_anchor=None, v1_anchor=None,  # If set, override cdelt3 when mapping
    # New: normalized window to clamp final x-bounds and cap sigma_hi
    x_min_norm=None, x_max_norm=None
):
    """
    Build scalar bounds (single set) for a subsequent multi-Gaussian fit.

    INPUT:
      _gaussian_seeds: dict from search_gaussian_seeds_matched_filter_norm
          components[:,0]=center_norm, [:,1]=sigma_norm, [:,2]=|amp|
          'bg' and 'rms' are passed through.

    OUTPUT (length 10 list):
      [model_sigma_lo, bg_lo, x_lo, s_lo, p_lo,
       model_sigma_hi, bg_hi, x_hi, s_hi, p_hi]

    Fallback (no seeds inside window):
      x_lo = x_min_norm (or 0), x_hi = x_max_norm (or 1)
      s_lo = min_sigma
      s_hi = min(clip_sigma_hi, max(min_sigma*1.05, window_width))
      p_lo = 0.0, p_hi = 1.0
    """
    msig_lo_seed, msig_hi_seed = map(float, model_sigma_bounds)

    # Pull bg/rms from seeds (raw units). We clip bg_hi to 1.0 only for safety if your pipeline expects 0..1.
    bg  = float(_gaussian_seeds.get('bg', 0.0))
    rms = float(_gaussian_seeds.get('rms', 0.1))
    if not np.isfinite(rms) or (rms <= 0):
        rms = 0.1

    comps = _gaussian_seeds.get('components', None)
    ncomp = int(_gaussian_seeds.get('ncomp', 0))

    # Background bounds
    bg_lo = 0.0
    bg_hi = max(bg_lo + 1e-6, min(1.0, bg + k_bg * rms))

    # Window setup
    w_lo = 0.0 if (x_min_norm is None) else float(x_min_norm)
    w_hi = 1.0 if (x_max_norm is None) else float(x_max_norm)
    if w_hi < w_lo:
        w_lo, w_hi = w_hi, w_lo
    window_width = max(0.0, w_hi - w_lo)

    # Defaults in case we have no seeds
    x_lo = 0.0
    x_hi = 1.0
    #s_lo = max(min_sigma, 1e-3)
    s_lo = 0.01
    s_hi = 0.9
    p_lo = 0.0
    p_hi = 1.0

    if ncomp > 0 and comps is not None and getattr(comps, "size", 0):
        comps = np.asarray(comps, dtype=float)
        # comps: [center_norm, sigma_norm, |amp|]
        centers_n = comps[:, 0]
        sigs_n    = comps[:, 1]
        amps      = comps[:, 2]

        # ----- x bounds (either via physical axis, or on normalized) -----
        if use_phys_for_x_bounds:
            if (v_min is None) or (v_max is None):
                raise ValueError("use_phys_for_x_bounds=True requires v_min and v_max.")
            v0, v1 = _resolve_anchors(float(v_min), float(v_max), cdelt3, v0_anchor, v1_anchor)
            v_scale = (v1 - v0)   # keep sign
            if abs(v_scale) < 1e-20:
                v_scale = 1e-12

            centers_p = v0 + v_scale * centers_n
            sigs_p    = abs(v_scale) * sigs_n

            # min/max in physical values
            idx_min = int(np.argmin(centers_p))
            idx_max = int(np.argmax(centers_p))
            xl_p, xl_sig_p = centers_p[idx_min], sigs_p[idx_min]
            xh_p, xh_sig_p = centers_p[idx_max], sigs_p[idx_max]

            x_lo_p = xl_p - float(k_x) * xl_sig_p
            x_hi_p = xh_p + float(k_x) * xh_sig_p

            # Map physical → normalized (respect sign)
            x_lo = (x_lo_p - v0) / v_scale
            x_hi = (x_hi_p - v0) / v_scale
            x_lo = float(np.clip(x_lo, 0.0, 1.0))
            x_hi = float(np.clip(x_hi, 0.0, 1.0))
            if x_lo > x_hi:
                x_lo, x_hi = x_hi, x_lo

        else:
            # Work directly on normalized axis
            idx_min = int(np.argmin(centers_n))
            idx_max = int(np.argmax(centers_n))
            xl_n, xl_sig_n = centers_n[idx_min], sigs_n[idx_min]
            xh_n, xh_sig_n = centers_n[idx_max], sigs_n[idx_max]
            x_lo = float(np.clip(xl_n - float(k_x) * xl_sig_n, 0.0, 1.0))
            x_hi = float(np.clip(xh_n + float(k_x) * xh_sig_n, 0.0, 1.0))
            if (x_hi - x_lo) < min_x_span:
                mid = 0.5 * (x_lo + x_hi)
                x_lo = max(0.0, mid - 0.02)
                x_hi = min(1.0, mid + 0.02)

        # Clamp x-bounds to the requested window
        x_lo, x_hi = _clamp_window(x_lo, x_hi, w_lo, w_hi, min_span=max(min_x_span, 2e-2))

        # to avoid narrow priors for x
        if np.fabs(x_hi - x_lo) < 0.3:
            x_lo = 0.01
            x_hi = 0.99

        # ----- sigma bounds -----
        k_sig_lo, k_sig_hi = map(float, sigma_scale_bounds)
        if use_phys_for_x_bounds:
            s_min_p = float(np.min(abs(v1 - v0) * sigs_n))
            s_max_p = float(np.max(abs(v1 - v0) * sigs_n))
            # Convert back to normalized
            v_scale_abs = abs(v1 - v0) if abs(v1 - v0) > 0 else 1.0
            s_lo = max(min_sigma, (k_sig_lo * s_min_p) / v_scale_abs)
            s_hi = max(s_lo * 1.05, (k_sig_hi * s_max_p) / v_scale_abs)
        else:
            s_min_n = float(np.min(sigs_n))
            s_max_n = float(np.max(sigs_n))
            s_lo = max(min_sigma, k_sig_lo * s_min_n)
            s_hi = max(s_lo * 1.05, k_sig_hi * s_max_n)

        # Cap sigma upper bound
        s_hi = float(min(clip_sigma_hi, s_hi))
        if window_width > 0:
            # Respect user's request: do not exceed the window width
            s_hi = float(min(s_hi, window_width))

        # to avoid narrow priors for sigma
        if np.fabs(s_hi - s_lo) < 0.2:
            s_lo = 0.01
            s_hi = 0.2


        # ----- peak bounds (amplitude) -----
        a_min = float(np.min(amps))
        a_max = float(np.max(amps))
        k_p_lo, k_p_hi = map(float, peak_scale_bounds)
        p_lo = float(np.clip(k_p_lo * a_min, 0.0, 1.0))
        p_hi = float(np.clip(k_p_hi * a_max, 0.0, 1.0))
        if (p_hi - p_lo) < min_x_span:
            mid = 0.5 * (p_lo + p_hi)
            p_lo = max(0.0, mid - 0.05)
            p_hi = min(1.0, mid + 0.05)

        # to avoid narrow priors for amp
        if np.fabs(p_hi - p_lo) < 0.2:
            p_lo = 0.001
            p_hi = 0.999


        # Model residual sigma bounds
        msig_lo = float(msig_lo_seed)
        msig_hi = float(msig_hi_seed)

        # to avoid narrow priors for model sigma
        if np.fabs(msig_hi - msig_lo) < 0.2:
            msig_lo = 0.001
            msig_hi = 0.9

    else:
        # ------------------ Fallback: no seeds in the window ------------------
        # x in the window, sigma max = window width, amplitude [0,1]
        x_lo = 0.0 if (x_min_norm is None) else float(x_min_norm)
        x_hi = 1.0 if (x_max_norm is None) else float(x_max_norm)
        if x_hi < x_lo:
            x_lo, x_hi = x_hi, x_lo
        x_lo, x_hi = _clamp_window(x_lo, x_hi, x_lo, x_hi, min_span=max(min_x_span, 2e-2))

        window_width = max(0.0, x_hi - x_lo)
        s_lo = max(min_sigma, 1e-3)
        # Respect request: sigma upper bound = window width (capped)
        s_hi = float(min(clip_sigma_hi, max(s_lo * 1.05, window_width)))

        x_lo = 0.01
        x_hi = 0.99
        s_lo = 0.01
        s_hi = 0.6
        p_lo = 0.001
        p_hi = 0.999
        msig_lo = 0.001
        msig_hi = 0.9

    return [msig_lo, bg_lo, x_lo, s_lo, p_lo,
            msig_hi, bg_hi, x_hi, s_hi, p_hi]






def set_sgfit_bounds_from_matched_filter_seeds_norm1(
    _gaussian_seeds,
    *,
    # (1) model residual sigma bounds
    model_sigma_bounds=(0.0, 0.7),
    # (2) background upper bound = bg + k_bg * rms
    k_bg=3.0,
    # (3) center bounds = [x - k_x * s, x + k_x * s]
    k_x=5.0,
    # (4) sigma bounds = [k_sig_lo * s, k_sig_hi * s]
    sigma_scale_bounds=(0.1, 3.0),
    # (5) peak bounds = [k_p_lo * A, k_p_hi * A]
    peak_scale_bounds=(0.3, 2.0),
    # Stabilization (minimum width)
    min_x_span=1e-5,
    min_sigma=1e-4,
    # Clipping upper bound (normalized axis is 0..1)
    clip_sigma_hi=0.999,
    # ---- Consider physical axis options ----
    use_phys_for_x_bounds=True,  # If True, compute x-bounds in physical axis and map back to normalized
    v_min=None, v_max=None,      # Physical velocity range
    cdelt3=None,                 # FITS CDELT3 (sign respected)
    v0_anchor=None, v1_anchor=None,  # If set, override cdelt3 when mapping
    # ---- NEW: restrict final x bounds to a normalized window ----
    x_min_norm=None, x_max_norm=None
):
    """
    _gaussian_seeds: output dict from search_gaussian_seeds_matched_filter_norm
         components[:,0]=center (normalized), [:,1]=sigma (normalized), [:,2]=amp (normalized)

    Return (list, length 10):
      [model_sigma_lo, bg_lo, x_lo, s_lo, p_lo,
       model_sigma_hi, bg_hi, x_hi, s_hi, p_hi]
    """
    msig_lo, msig_hi = map(float, model_sigma_bounds)

    bg  = float(_gaussian_seeds.get('bg', 0.0))
    rms = float(_gaussian_seeds.get('rms', 0.1))
    if rms < 0.1:
        rms = 0.1

    ncomp = int(_gaussian_seeds.get('ncomp', 0))
    comps = _gaussian_seeds.get('components', None)

    if ncomp > 0 and getattr(comps, "size", 0):
        comps = np.asarray(comps, dtype=float)
        xs_n   = comps[:, 0]  # normalized centers
        sigs_n = comps[:, 1]  # normalized sigmas (length)
        amps   = comps[:, 2]  # normalized peaks

        # (2) background bounds
        bg_lo = 0.0
        bg_hi = max(bg_lo + 1e-6, min(1.0, bg + k_bg * rms))

        # ----- compute x bounds -----
        if use_phys_for_x_bounds:
            if (v_min is None) or (v_max is None):
                raise ValueError("When use_phys_for_x_bounds=True, you must provide v_min and v_max.")
            v0, v1 = _resolve_anchors(v_min, v_max, cdelt3, v0_anchor, v1_anchor)
            v_scale = (v1 - v0)
            if abs(v_scale) < 1e-20:
                v_scale = 1e-12

            xs_p   = v0 + v_scale * xs_n     # physical centers
            sigs_p = abs(v_scale) * sigs_n   # physical sigmas

            idx_min = int(np.argmin(xs_p))
            idx_max = int(np.argmax(xs_p))
            xl_p, xl_sig_p = xs_p[idx_min], sigs_p[idx_min]
            xh_p, xh_sig_p = xs_p[idx_max], sigs_p[idx_max]

            x_lo = (xl_p - k_x * xl_sig_p - v0) / v_scale
            x_hi = (xh_p + k_x * xh_sig_p - v0) / v_scale

            x_lo = float(np.clip(x_lo, 0.0, 1.0))
            x_hi = float(np.clip(x_hi, 0.0, 1.0))
            if x_lo > x_hi:
                x_lo, x_hi = x_hi, x_lo
            if (x_hi - x_lo) < min_x_span:
                mid = 0.5 * (x_lo + x_hi)
                x_lo = max(0.0, mid - 0.02)
                x_hi = min(1.0, mid + 0.02)
        else:
            idx_min = int(np.argmin(xs_n))
            idx_max = int(np.argmax(xs_n))
            xl_n, xl_sig_n = xs_n[idx_min], sigs_n[idx_min]
            xh_n, xh_sig_n = xs_n[idx_max], sigs_n[idx_max]
            x_lo = float(np.clip(xl_n - k_x * xl_sig_n, 0.0, 1.0))
            x_hi = float(np.clip(xh_n + k_x * xh_sig_n, 0.0, 1.0))
            if (x_hi - x_lo) < min_x_span:
                mid = 0.5 * (x_lo + x_hi)
                x_lo = max(0.0, mid - 0.02)
                x_hi = min(1.0, mid + 0.02)

        # -----: clamp x bounds to the provided normalized window -----
        if (x_min_norm is not None) or (x_max_norm is not None):
            win_lo = 0.0 if x_min_norm is None else float(x_min_norm)
            win_hi = 1.0 if x_max_norm is None else float(x_max_norm)
            if win_lo > win_hi:
                win_lo, win_hi = win_hi, win_lo
            # ensure window within [0,1]
            win_lo = float(np.clip(win_lo, 0.0, 1.0))
            win_hi = float(np.clip(win_hi, 0.0, 1.0))

            x_lo = max(x_lo, win_lo)
            x_hi = min(x_hi, win_hi)

            # keep at least min_x_span, staying inside the window
            if (x_hi - x_lo) < min_x_span:
                mid  = 0.5 * (x_lo + x_hi)
                half = 0.5 * min_x_span
                x_lo = max(win_lo, mid - half)
                x_hi = min(win_hi, mid + half)
                # if still too narrow due to tiny window, pin to window edges
                if (x_hi - x_lo) < min_x_span:
                    x_lo = win_lo
                    x_hi = min(win_hi, win_lo + min_x_span)

        # ----- sigma bounds -----
        k_sig_lo, k_sig_hi = map(float, sigma_scale_bounds)
        if use_phys_for_x_bounds:
            s_min_p = float(np.min(sigs_p))
            s_max_p = float(np.max(sigs_p))
            s_lo = max(min_sigma, (k_sig_lo * s_min_p) / abs(v_scale))
            s_hi = max(s_lo * 1.05, (k_sig_hi * s_max_p) / abs(v_scale))
        else:
            s_min_n = float(np.min(sigs_n))
            s_max_n = float(np.max(sigs_n))
            s_lo = max(min_sigma, k_sig_lo * s_min_n)
            s_hi = max(s_lo * 1.05, k_sig_hi * s_max_n)
        s_hi = min(clip_sigma_hi, s_hi)

        # ----- peak bounds -----
        a_min = float(np.min(amps))
        a_max = float(np.max(amps))
        k_p_lo, k_p_hi = map(float, peak_scale_bounds)
        p_lo = float(np.clip(k_p_lo * a_min, 0.0, 1.0))
        p_hi = float(np.clip(k_p_hi * a_max, 0.0, 3.0))
        if (p_hi - p_lo) < min_x_span:
            mid = 0.5 * (p_lo + p_hi)
            p_lo = max(0.0, mid - 0.05)
            p_hi = min(1.0, mid + 0.05)

    else:
        # No seeds: broad defaults
        bg_lo = 0.0
        bg_hi = min(1.0, bg + k_bg * rms if np.isfinite(rms) else 0.5)
        # Default window
        x_lo_default, x_hi_default = 0.0, 1.0
        if (x_min_norm is not None) or (x_max_norm is not None):
            win_lo = 0.0 if x_min_norm is None else float(x_min_norm)
            win_hi = 1.0 if x_max_norm is None else float(x_max_norm)
            if win_lo > win_hi:
                win_lo, win_hi = win_hi, win_lo
            x_lo_default = float(np.clip(win_lo, 0.0, 1.0))
            x_hi_default = float(np.clip(win_hi, 0.0, 1.0))
            if (x_hi_default - x_lo_default) < min_x_span:
                x_hi_default = min(1.0, x_lo_default + min_x_span)
        x_lo, x_hi = x_lo_default, x_hi_default
        s_lo, s_hi = max(min_sigma, 1e-3), 0.3
        p_lo, p_hi = 0.0, 1.0

    return [msig_lo, bg_lo, x_lo, s_lo, p_lo,
            msig_hi, bg_hi, x_hi, s_hi, p_hi]

#-- END OF SUB-ROUTINE____________________________________________________________#







def set_sgfit_bounds_from_matched_filter_seeds_norm_non_mask(
    _gaussian_seeds,
    *,
    # (1) model residual sigma bounds
    model_sigma_bounds=(0.0, 0.7),
    # (2) background upper bound = bg + k_bg * rms
    k_bg=3.0,
    # (3) center bounds = [x - k_x * s, x + k_x * s]
    k_x=5.0,
    # (4) sigma bounds = [k_sig_lo * s, k_sig_hi * s]
    sigma_scale_bounds=(0.1, 3.0),
    # (5) peak bounds = [k_p_lo * A, k_p_hi * A]
    peak_scale_bounds=(0.3, 2.0),
    # Stabilization (minimum width)
    min_x_span=1e-5,
    min_sigma=1e-4,
    # Clipping upper bound (normalized axis is 0..1)
    clip_sigma_hi=0.999,
    # ---- Consider physical axis options ----
    use_phys_for_x_bounds=True,  # If True, compute x-bounds in physical axis and map back to normalized
    v_min=None, v_max=None,      # Physical velocity range
    cdelt3=None,                 # FITS CDELT3 (sign respected)
    v0_anchor=None, v1_anchor=None  # If set, override cdelt3 when mapping
):
    """
    _gaussian_seeds: output dict from search_gaussian_seeds_matched_filter_norm
         components[:,0]=amp (normalized flux), [:,1]=center (normalized velocity), [:,2]=sigma (normalized velocity)
         bg, rms are assumed to be in 0..1 normalized units.

    Return (list, length 10):
      [model_sigma_lo, bg_lo, x_lo, s_lo, p_lo,
       model_sigma_hi, bg_hi, x_hi, s_hi, p_hi]
    """
    # (1) model residual sigma bounds
    msig_lo, msig_hi = map(float, model_sigma_bounds)

    # bg/rms (normalized)
    bg  = float(_gaussian_seeds.get('bg', 0.0))
    rms = float(_gaussian_seeds.get('rms', 0.1))
    if rms < 0.1: # avoid too small rms just in case
        rms = 0.1 # set a large rms

    ncomp = int(_gaussian_seeds.get('ncomp', 0))
    comps = _gaussian_seeds.get('components', None)

    if ncomp > 0 and getattr(comps, "size", 0):
        comps = np.asarray(comps, dtype=float)
        xs_n = comps[:, 0]   # normalized center in [0,1]
        sigs_n = comps[:, 1] # normalized sigma (length on x_norm)
        amps = comps[:, 2] # normalized peak flux in [0,1]

        # (2) background: lo=0.0 fixed as requested, hi=bg + k_bg*rms (clipped)
        bg_lo = 0.0
        bg_hi = max(bg_lo + 1e-6, min(1.0, bg + k_bg * rms))

        # ----- compute x bounds -----
        if use_phys_for_x_bounds:
            # Compute bounds in physical axis: [min(center) - k_x*sigma, max(center) + k_x*sigma]
            # then map back to normalized axis.
            if (v_min is None) or (v_max is None):
                raise ValueError("When use_phys_for_x_bounds=True, you must provide v_min and v_max.")
            v0, v1 = _resolve_anchors(v_min, v_max, cdelt3, v0_anchor, v1_anchor)
            v_scale = (v1 - v0)   # keep sign (negative if descending)
            if abs(v_scale) < 1e-20:
                # Guard against degenerate axis
                v_scale = 1e-12

            xs_p = v0 + v_scale * xs_n          # physical centers
            sigs_p = abs(v_scale) * sigs_n      # physical sigmas (length scale)

            # min/max in physical values
            idx_min = int(np.argmin(xs_p))
            idx_max = int(np.argmax(xs_p))
            xl_p, xl_sig_p = xs_p[idx_min], sigs_p[idx_min]
            xh_p, xh_sig_p = xs_p[idx_max], sigs_p[idx_max]

            x_lo_p = xl_p - k_x * xl_sig_p
            x_hi_p = xh_p + k_x * xh_sig_p

            # Map physical --> normalized (respect sign)
            x_lo = (x_lo_p - v0) / v_scale
            x_hi = (x_hi_p - v0) / v_scale
            # Clip to normalized range
            x_lo = float(np.clip(x_lo, 0.0, 1.0))
            x_hi = float(np.clip(x_hi, 0.0, 1.0))
            # Ensure order for readability
            if x_lo > x_hi:
                x_lo, x_hi = x_hi, x_lo
            if (x_hi - x_lo) < min_x_span:
                mid = 0.5 * (x_lo + x_hi)
                x_lo = max(0.0, mid - 0.02)
                x_hi = min(1.0, mid + 0.02)
        else:
            # Legacy: compute directly on normalized axis (ignore cdelt3 sign)
            idx_min = int(np.argmin(xs_n))
            idx_max = int(np.argmax(xs_n))
            xl_n, xl_sig_n = xs_n[idx_min], sigs_n[idx_min]
            xh_n, xh_sig_n = xs_n[idx_max], sigs_n[idx_max]
            x_lo = float(np.clip(xl_n - k_x * xl_sig_n, 0.0, 1.0))
            x_hi = float(np.clip(xh_n + k_x * xh_sig_n, 0.0, 1.0))
            if (x_hi - x_lo) < min_x_span:
                mid = 0.5 * (x_lo + x_hi)
                x_lo = max(0.0, mid - 0.02)
                x_hi = min(1.0, mid + 0.02)

        # ----- sigma bounds -----
        k_sig_lo, k_sig_hi = map(float, sigma_scale_bounds)
        if use_phys_for_x_bounds:
            s_min_p = float(np.min(sigs_p))
            s_max_p = float(np.max(sigs_p))
            # Convert physical scale back to normalized sigma
            s_lo = max(min_sigma, (k_sig_lo * s_min_p) / abs(v_scale))
            s_hi = max(s_lo * 1.05, (k_sig_hi * s_max_p) / abs(v_scale))
        else:
            s_min_n = float(np.min(sigs_n))
            s_max_n = float(np.max(sigs_n))
            s_lo = max(min_sigma, k_sig_lo * s_min_n)
            s_hi = max(s_lo * 1.05, k_sig_hi * s_max_n)
        s_hi = min(clip_sigma_hi, s_hi)

        # ----- peak bounds -----
        a_min = float(np.min(amps))
        a_max = float(np.max(amps))
        k_p_lo, k_p_hi = map(float, peak_scale_bounds)
        p_lo = float(np.clip(k_p_lo * a_min, 0.0, 1.0))
        p_hi = float(np.clip(k_p_hi * a_max, 0.0, 3.0)) # allow up to 3.0 just in case
        if (p_hi - p_lo) < min_x_span:
            mid = 0.5 * (p_lo + p_hi)
            p_lo = max(0.0, mid - 0.05)
            p_hi = min(1.0, mid + 0.05)

    else:
        # No seeds: provide broad default bounds
        bg_lo = 0.0
        bg_hi = min(1.0, bg + k_bg * rms if np.isfinite(rms) else 0.5)
        x_lo, x_hi = 0.0, 1.0
        s_lo, s_hi = max(min_sigma, 1e-3), 0.3
        p_lo, p_hi = 0.0, 1.0

    return [msig_lo, bg_lo, x_lo, s_lo, p_lo,
            msig_hi, bg_hi, x_hi, s_hi, p_hi]

#-- END OF SUB-ROUTINE____________________________________________________________#




def _resolve_anchors(v_min, v_max, cdelt3=None, v0_anchor=None, v1_anchor=None):
    """
    Decide mapping for x_norm: x_norm=0 --> v0_anchor, x_norm=1 --> v1_anchor.
    - If v0_anchor/v1_anchor are provided, use them directly.
    - Else infer by sign of cdelt3:
        cdelt3>=0: v0=v_min, v1=v_max
        cdelt3< 0: v0=v_max, v1=v_min
    - If cdelt3 is None, use v0=v_min, v1=v_max.
    """
    if (v0_anchor is not None) and (v1_anchor is not None):
        return float(v0_anchor), float(v1_anchor)
    if cdelt3 is None:
        return float(v_min), float(v_max)
    return (float(v_min), float(v_max)) if (cdelt3 >= 0) else (float(v_max), float(v_min))

#-- END OF SUB-ROUTINE____________________________________________________________#





def gaussian_seeds_norm_to_phys(_gaussian_seeds, f_min, f_max, v_min, v_max, *,
                     cdelt3=None, v0_anchor=None, v1_anchor=None):
    """
    Convert normalized 0..1 _gaussian_seeds (from search_gaussian_seeds_matched_filter_norm)
    to physical units.
    - bg/rms: apply flux scaling (background includes offset).
    - components: [amp, center, sigma] --> [flux, velocity, velocity_sigma].
    - Supports CDELT3<0 (descending): define linear mapping by anchors.
    """
    v0, v1 = _resolve_anchors(v_min, v_max, cdelt3, v0_anchor, v1_anchor)
    f_scale = (f_max - f_min) if (f_max > f_min) else 1.0
    v_scale = (v1 - v0)  # keep sign (negative if descending)

    bg_n  = float(_gaussian_seeds.get('bg', 0.0))
    rms_n = float(_gaussian_seeds.get('rms', 0.0))
    bg_phys  = bg_n  * f_scale + f_min
    rms_phys = rms_n * f_scale

    comps_n = np.asarray(_gaussian_seeds.get('components', np.zeros((0,3))), dtype=float)
    comps_p = comps_n.copy()
    if comps_p.size:
        # amp: only scale, center: v = v0 + v_scale*x_norm, sigma: |v_scale|*sigma_norm
        comps_p[:, 0] = v0 + v_scale * comps_n[:, 0]
        comps_p[:, 1] = abs(v_scale) * comps_n[:, 1]
        comps_p[:, 2] = f_scale      * comps_n[:, 2]

    gaussian_seeds_phys = dict(_gaussian_seeds)
    gaussian_seeds_phys['bg_phys'] = bg_phys
    gaussian_seeds_phys['rms_phys'] = rms_phys
    gaussian_seeds_phys['components_phys'] = comps_p
    gaussian_seeds_phys['anchors'] = (v0, v1)
    return gaussian_seeds_phys

#-- END OF SUB-ROUTINE____________________________________________________________#


def print_priors_both(
    priors_norm, f_min, f_max, v_min, v_max, *,
    cdelt3=None, v0_anchor=None, v1_anchor=None,
    unit_flux="Jy", unit_vel="km/s"
):
    """
    priors_norm: [msig_lo, bg_lo, x_lo, s_lo, p_lo, msig_hi, bg_hi, x_hi, s_hi, p_hi] (normalized 0..1)
    f_min,f_max : physical flux range
    v_min,v_max : physical velocity range
    cdelt3     : FITS CDELT3 (respect sign). If negative, handle descending axis.
    v0_anchor, v1_anchor: if provided, x_norm=0/1 map to these physical velocities (ignore cdelt3).

    Prints normalized bounds and physical bounds side by side.
    """
    priors_phys = priors_norm_to_phys(
        priors_norm, f_min, f_max, v_min, v_max,
        cdelt3=cdelt3, v0_anchor=v0_anchor, v1_anchor=v1_anchor
    )

    names = ["model_sigma", "background", "gauss_x", "gauss_sigma", "gauss_peakflux"]
    print("[Norm] vs [Phys]")
    for k in range(5):
        lo_n = priors_norm[k]
        hi_n = priors_norm[k+5]
        lo_p = priors_phys[k]
        hi_p = priors_phys[k+5]
        u = unit_vel if names[k] in ("gauss_x", "gauss_sigma") else unit_flux
        print(f"  {names[k]:>13}: [{lo_n:8.5f}, {hi_n:8.5f}]  |  [{lo_p:10.5g}, {hi_p:10.5g}] {u}")

#-- END OF SUB-ROUTINE____________________________________________________________#



def priors_norm_to_phys(priors, f_min, f_max, v_min, v_max, *,
                        cdelt3=None, v0_anchor=None, v1_anchor=None):
    """
    [msig_lo, bg_lo, x_lo, s_lo, p_lo, msig_hi, bg_hi, x_hi, s_hi, p_hi] (normalized)
    --> convert to a list in the same order but in physical units.
    - x_lo/x_hi: linear mapping by anchors; ensure [smaller, larger] order for readability.
    - sigma uses absolute length scale abs(v1-v0).
    """
    msig_lo, bg_lo, x_lo, s_lo, p_lo, msig_hi, bg_hi, x_hi, s_hi, p_hi = map(float, priors)
    v0, v1 = _resolve_anchors(v_min, v_max, cdelt3, v0_anchor, v1_anchor)
    f_scale = (f_max - f_min) if (f_max > f_min) else 1.0
    v_scale = (v1 - v0)

    # model sigma (residual) scales with flux
    msig_lo_p = msig_lo * f_scale
    msig_hi_p = msig_hi * f_scale

    # background includes offset
    bg_lo_p = bg_lo * f_scale + f_min
    bg_hi_p = bg_hi * f_scale + f_min

    # center: linear mapping and then ensure order
    x_lo_p = v0 + v_scale * x_lo
    x_hi_p = v0 + v_scale * x_hi
    if x_lo_p > x_hi_p:
        x_lo_p, x_hi_p = x_hi_p, x_lo_p

    # sigma: length scale
    s_lo_p = abs(v_scale) * s_lo
    s_hi_p = abs(v_scale) * s_hi

    # peak flux: scale only
    p_lo_p = p_lo * f_scale
    p_hi_p = p_hi * f_scale

    return [msig_lo_p, bg_lo_p, x_lo_p, s_lo_p, p_lo_p,
            msig_hi_p, bg_hi_p, x_hi_p, s_hi_p, p_hi_p]

#-- END OF SUB-ROUTINE____________________________________________________________#



def print_gaussian_seeds_matched_filter(_gaussian_seeds, f_min, f_max, v_min, v_max, *,
                   cdelt3=None, v0_anchor=None, v1_anchor=None,
                   unit_flux="Jy/beam", unit_vel="km/s", show_fwhm=True):
    """
    Print normalized _gaussian_seeds and the converted physical units side by side.
    Handles CDELT3<0 (descending) by mapping to correct physical coordinates.
    """
    out_p = gaussian_seeds_norm_to_phys(_gaussian_seeds, f_min, f_max, v_min, v_max,
                             cdelt3=cdelt3, v0_anchor=v0_anchor, v1_anchor=v1_anchor)
    bg_n  = float(_gaussian_seeds.get('bg', 0.0));   rms_n = float(_gaussian_seeds.get('rms', 0.0))
    bg_p  = float(out_p['bg_phys']);     rms_p = float(out_p['rms_phys'])

    print("=== BG / RMS ===")
    print(f"  bg  : {bg_n:10.6g} (norm) | {bg_p:10.6g} {unit_flux}")
    print(f"  rms : {rms_n:10.6g} (norm) | {rms_p:10.6g} {unit_flux}")

    comps_n = np.asarray(_gaussian_seeds.get('components', np.zeros((0,3))), dtype=float)
    comps_p = np.asarray(out_p.get('components_phys', np.zeros((0,3))), dtype=float)

    print("\n=== Components (amp, center, sigma) ===")
    for idx in range(comps_n.shape[0]):
        x_n, s_n, a_n = comps_n[idx]
        x_p, s_p, a_p = comps_p[idx]
        print(f"  comp[{idx:02d}]  amp: {a_n:8.5f} | {a_p:10.6g} {unit_flux}   "
              f"cen: {x_n:8.5f} | {x_p:10.6g} {unit_vel}   "
              f"sig: {s_n:8.5f} | {s_p:10.6g} {unit_vel}")
        if show_fwhm:
            fwhm_n = 2.355 * s_n
            fwhm_p = 2.355 * s_p
            print(f"           FWHM: {fwhm_n:8.5f} (norm) | {fwhm_p:10.6g} {unit_vel}")

#-- END OF SUB-ROUTINE____________________________________________________________#



def set_init_priors_multiple_gaussians_non_mask(
    M,
    seed_bounds,
    *,
    seed_out=None,      # search_gaussian_seeds_matched_filter_norm out(dict); use bg/rms (normalized)
    prev_fit=None,      # output of prev_fit_from_results_slice(...)
    # Tighter bounds around previous components if available
    k_x=3.0, # x - k_x*sigma ~ x + k_x*sigma
    sigma_scale_bounds=(0.5, 2.0),
    peak_scale_bounds=(0.5, 1.5),
    # bg width: bg +/- k_bg * rms(seed)
    k_bg=3.0,
    k_msig=3.0,
    # Stabilization / clipping
    min_x_span=1e-5,
    min_sigma=1e-4,
    clip_sigma_hi=0.999,
    peak_upper_cap=1.0,
):
    """
    Build an initial priors vector for fitting M Gaussian components.
    - seed_bounds: output (length 10) of make_single_gauss_bounds_from_seed_norm()
      [msig_lo, bg_lo, x_lo, s_lo, p_lo, msig_hi, bg_hi, x_hi, s_hi, p_hi]
      --> replicate for new components (g_{n_prev+1}..g_M) when no previous component exists.
    - prev_fit: if available, tighten bounds around prior parameters of each component.

    Return: gfit_priors_init (length = 4 + 6*M)
      [msig_lo, bg_lo, g1_x_lo, g1_s_lo, g1_p_lo, ..., gM_x_lo, gM_s_lo, gM_p_lo,
       msig_hi, bg_hi, g1_x_hi, g1_s_hi, g1_p_hi, ..., gM_x_hi, gM_s_hi, gM_p_hi]
    """
    M = int(M)
    msig_lo_seed, bg_lo_seed, x_lo_seed, s_lo_seed, p_lo_seed, \
    msig_hi_seed, bg_hi_seed, x_hi_seed, s_hi_seed, p_hi_seed = map(float, seed_bounds)


    # bg bounds: if prev_fit exists, use bg_center +/- k_bg * rms(seed); otherwise use seed's broad bounds
    if prev_fit is not None and ("bg" in prev_fit) and (seed_out is not None):
        bg_center = float(prev_fit["bg"])
        rms_seed  = float(prev_fit.get("model_sigma", 0.1))

        msig_lo = 0.0
        msig_hi = k_msig * rms_seed

        bg_lo = max(0.0, bg_center - k_bg * rms_seed)
        bg_hi = min(1.0, bg_center + k_bg * rms_seed)

    else:
        msig_lo = float(msig_lo_seed)
        msig_hi = float(msig_hi_seed)
        bg_lo = float(bg_lo_seed)
        bg_hi = float(bg_hi_seed)


    # number of previous components
    n_prev = 0
    prev_components = None
    if prev_fit is not None and ("components" in prev_fit) and (prev_fit["components"] is not None):
        prev_components = np.asarray(prev_fit["components"], dtype=float)
        if prev_components.ndim == 2 and prev_components.shape[1] >= 3:
            n_prev = min(M, prev_components.shape[0])

    k_sig_lo, k_sig_hi = map(float, sigma_scale_bounds)
    k_p_lo,  k_p_hi    = map(float, peak_scale_bounds)

    # accumulate
    lo_parts = [msig_lo, bg_lo]
    hi_parts = [msig_hi, bg_hi]

    # 1) For existing components from previous fit: tight bounds
    for m in range(n_prev):
        x_c, s, amp = map(float, prev_components[m, :3])

        # x bounds
        x_lo = float(np.clip(x_c - k_x * s, 0.0, 1.0))
        x_hi = float(np.clip(x_c + k_x * s, 0.0, 1.0))
        if (x_hi - x_lo) < min_x_span:
            mid = 0.5 * (x_lo + x_hi)
            x_lo = max(0.0, mid - 0.02)
            x_hi = min(1.0, mid + 0.02)

        # sigma bounds
        s_lo = max(min_sigma, k_sig_lo * s)
        s_hi = min(clip_sigma_hi, max(s_lo * 1.05, k_sig_hi * s))

        # peak bounds
        p_lo = float(np.clip(k_p_lo * amp, 0.0, peak_upper_cap))
        p_hi = float(np.clip(k_p_hi * amp, 0.0, peak_upper_cap))
        if (p_hi - p_lo) < min_x_span:
            mid = 0.5 * (p_lo + p_hi)
            p_lo = max(0.0, mid - 0.05)
            p_hi = min(peak_upper_cap, mid + 0.05)

        lo_parts += [x_lo, s_lo, p_lo]
        hi_parts += [x_hi, s_hi, p_hi]

    # 2) For remaining components: replicate broad seed-based bounds
    for m in range(n_prev, M):
        lo_parts += [x_lo_seed, s_lo_seed, p_lo_seed]
        hi_parts += [x_hi_seed, s_hi_seed, p_hi_seed]


    return np.asarray(lo_parts + hi_parts, dtype=np.float32)  # convert to np.array
#-- END OF SUB-ROUTINE____________________________________________________________#



def set_init_priors_multiple_gaussians(
    M,
    seed_bounds,
    *,
    seed_out=None,      # search_gaussian_seeds_matched_filter_norm out(dict); use bg/rms (normalized)
    prev_fit=None,      # output of prev_fit_from_results_slice(...)
    # Tighter bounds around previous components if available
    k_x=3.0, # x - k_x*sigma ~ x + k_x*sigma
    sigma_scale_bounds=(0.5, 2.0),
    peak_scale_bounds=(0.5, 1.5),
    # bg width: bg +/- k_bg * rms(seed)
    k_bg=3.0,
    k_msig=3.0,
    # Stabilization / clipping
    min_x_span=1e-5,
    min_sigma=1e-4,
    clip_sigma_hi=0.999,
    peak_upper_cap=1.0,
    # ---- NEW: clamp x bounds to normalized window ----
    x_min_norm=None,
    x_max_norm=None,
):
    """
    Build an initial priors vector for fitting M Gaussian components.

    The resulting x-bounds for all components are clamped to the normalized window
    [x_min_norm, x_max_norm] (if provided).

    Return: gfit_priors_init (length = 4 + 6*M)
      [msig_lo, bg_lo, g1_x_lo, g1_s_lo, g1_p_lo, ..., gM_x_lo, gM_s_lo, gM_p_lo,
       msig_hi, bg_hi, g1_x_hi, g1_s_hi, g1_p_hi, ..., gM_x_hi, gM_s_hi, gM_p_hi]
    """
    def _clamp_window(x_lo, x_hi, w_lo, w_hi, min_span):
        # If no window is specified, return as-is
        if w_lo is None and w_hi is None:
            return float(x_lo), float(x_hi)
        wl = 0.0 if w_lo is None else float(w_lo)
        wh = 1.0 if w_hi is None else float(w_hi)
        if wl > wh:
            wl, wh = wh, wl
        # keep inside [0,1]
        wl = float(np.clip(wl, 0.0, 1.0))
        wh = float(np.clip(wh, 0.0, 1.0))
        # clamp to window
        lo = max(float(x_lo), wl)
        hi = min(float(x_hi), wh)
        # ensure minimum span inside the window
        if (hi - lo) < float(min_span):
            mid = 0.5 * (lo + hi)
            half = 0.5 * float(min_span)
            lo = max(wl, mid - half)
            hi = min(wh, mid + half)
            # if window itself is too small, pin to edges
            if (hi - lo) < float(min_span):
                lo = wl
                hi = min(wh, wl + float(min_span))
        return float(lo), float(hi)

    M = int(M)
    msig_lo_seed, bg_lo_seed, x_lo_seed, s_lo_seed, p_lo_seed, \
    msig_hi_seed, bg_hi_seed, x_hi_seed, s_hi_seed, p_hi_seed = map(float, seed_bounds)

    # Pre-clamp seed x-bounds to the window
    x_lo_seed, x_hi_seed = _clamp_window(x_lo_seed, x_hi_seed, x_min_norm, x_max_norm, min_x_span)

    # bg/model-sigma bounds
    if prev_fit is not None and ("bg" in prev_fit) and (seed_out is not None):
        bg_center = float(prev_fit["bg"])
        rms_seed  = float(prev_fit.get("model_sigma", 0.1))

        msig_lo = 0.0
        msig_hi = k_msig * rms_seed

        bg_lo = max(0.0, bg_center - k_bg * rms_seed)
        bg_hi = min(1.0, bg_center + k_bg * rms_seed)
    else:
        msig_lo = float(msig_lo_seed)
        msig_hi = float(msig_hi_seed)
        bg_lo   = float(bg_lo_seed)
        bg_hi   = float(bg_hi_seed)

    # previous components
    n_prev = 0
    prev_components = None
    if prev_fit is not None and ("components" in prev_fit) and (prev_fit["components"] is not None):
        prev_components = np.asarray(prev_fit["components"], dtype=float)
        if prev_components.ndim == 2 and prev_components.shape[1] >= 3:
            n_prev = min(M, prev_components.shape[0])

    k_sig_lo, k_sig_hi = map(float, sigma_scale_bounds)
    k_p_lo,  k_p_hi    = map(float, peak_scale_bounds)

    # accumulate bounds
    lo_parts = [msig_lo, bg_lo]
    hi_parts = [msig_hi, bg_hi]

    # 1) Tight bounds around previous components
    for m in range(n_prev):
        x_c, s, amp = map(float, prev_components[m, :3])

        # x bounds from previous fit
        x_lo = float(np.clip(x_c - k_x * s, 0.0, 1.0))
        x_hi = float(np.clip(x_c + k_x * s, 0.0, 1.0))
        if (x_hi - x_lo) < min_x_span:
            mid = 0.5 * (x_lo + x_hi)
            x_lo = max(0.0, mid - 0.02)
            x_hi = min(1.0, mid + 0.02)

        # --- NEW: clamp to window ---
        x_lo, x_hi = _clamp_window(x_lo, x_hi, x_min_norm, x_max_norm, min_x_span)

        # sigma bounds
        s_lo = max(min_sigma, k_sig_lo * s)
        s_hi = min(clip_sigma_hi, max(s_lo * 1.05, k_sig_hi * s))

        # peak bounds
        p_lo = float(np.clip(k_p_lo * amp, 0.0, peak_upper_cap))
        p_hi = float(np.clip(k_p_hi * amp, 0.0, peak_upper_cap))
        if (p_hi - p_lo) < min_x_span:
            mid = 0.5 * (p_lo + p_hi)
            p_lo = max(0.0, mid - 0.05)
            p_hi = min(peak_upper_cap, mid + 0.05)

        lo_parts += [x_lo, s_lo, p_lo]
        hi_parts += [x_hi, s_hi, p_hi]

    # 2) For remaining components: replicate (clamped) seed-based bounds
    p_lo_seed = p_lo # copy the previous' amp_lo to make the fit fair
    p_hi_seed = p_lo # copy the previous' amp_hi to make the fit fair

    for m in range(n_prev, M):
        lo_parts += [x_lo_seed, s_lo_seed, p_lo_seed]
        hi_parts += [x_hi_seed, s_hi_seed, p_hi_seed]

    return np.asarray(lo_parts + hi_parts, dtype=np.float32)
#-- END OF SUB-ROUTINE____________________________________________________________#



def prev_fit_from_results_slice(gfit_results_slice, n_prev):
    """
    Read previous-step parameters from a slice gfit_results[j, k_prev] and convert to prev_fit dict.
    - n_prev: number of Gaussians in the previous step (k_prev = n_prev, typically 1..M-1)
    - Extract params/errors from gfit_results_slice[:2*nparams_prev]
      nparams_prev = 2 + 3*n_prev
      Parameter order: [model_sigma, bg, g1_x, g1_sigma, g1_peak, g2_x, g2_sigma, g2_peak, ...]
      (errors follow, but not used here)
    Return:
      {
        "model_sigma": float,
        "bg": float,
        "components": np.ndarray (n_prev, 3) with columns [x, sigma, peakflux] in 0..1
      }
    """
    n_prev = int(n_prev)
    if n_prev <= 0:
        return {"model_sigma": 0.0, "bg": 0.0, "components": np.zeros((0,3), dtype=float)}

    nparams_prev = 2 + 3*n_prev
    vec = np.asarray(gfit_results_slice[:2*nparams_prev], dtype=float)
    if vec.size < 2*nparams_prev:
        # If the vector is unexpectedly short: return empty prev_fit
        return {"model_sigma": 0.0, "bg": 0.0, "components": np.zeros((0,3), dtype=float)}

    pars = vec[:nparams_prev]  # [msig, bg, g1_x, g1_s, g1_p, g2_x, ...]
    # errs = vec[nparams_prev:2*nparams_prev]  # not used here

    model_sigma = float(np.clip(pars[0], 0.0, 1.0)) # model sigma
    bg          = float(np.clip(pars[1], 0.0, 1.0)) # bg

    comps = np.zeros((n_prev, 3), dtype=float)
    for m in range(n_prev):
        x = float(pars[2 + 3*m + 0])  # x
        s = float(pars[2 + 3*m + 1])  # s
        p = float(pars[2 + 3*m + 2])  # peak (amp)
        # prev_fit format is [x, sigma, peak_flux]
        comps[m, 0] = np.clip(x, 0.0, 1.0)
        comps[m, 1] = max(0.0, s)  # sigma only clipped at lower bound (upper handled when making bounds)
        comps[m, 2] = np.clip(p, 0.0, 1.0)

    return {"model_sigma": model_sigma, "bg": bg, "components": comps}

#-- END OF SUB-ROUTINE____________________________________________________________#
