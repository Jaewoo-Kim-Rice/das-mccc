"""Network multi-channel cross-correlation (MCCC) on an aligned DAS gather.

One MCCC pass measures, for every channel, the lag that maximises the absolute
correlation with about fifty partner channels drawn from a normal distribution
within +-corr_len channels, then solves the sparse least-squares problem

    min_tau || lamb * (tau_i - tau_j - lag_ij) ||^2 + || smoothness * (tau_{c+1} - tau_c) ||^2

and smooths tau with a moving average. ``iterate_align`` repeats the pass, applying tau
to the gather and median-filtering it along the fibre between passes.

Sign convention: a positive tau[c] means channel c currently arrives *later* than its
partners by tau samples (the aligned trace has to be advanced by tau).

Every quantity is in samples or channels; the per-pair lag bound
``max(pair_slope * |i - j|, pair_min_shift)`` is a bound on the *residual* moveout
slope relative to the initial curve, not on the absolute moveout.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from .ops import moving_avg, spatial_median, tau_shift
from .signal import NUMBA_AVAILABLE, jit, limited_cc, normal_distribution, prange

log = logging.getLogger("dasmccc")


def partner_pairs(
    n_channels: int, corr_len: int, n_partners: int = 50, partner_std: float = 20.0
) -> tuple[np.ndarray, np.ndarray]:
    """Channel pairs (i, j) with j > i to correlate.

    For each channel i, ``n_partners`` candidates are drawn as quantiles of a normal
    distribution (std ``partner_std`` channels) truncated to
    [i - corr_len, min(n_channels - 1, i + corr_len)]; candidates outside the fibre or
    with j <= i are dropped, so each channel keeps roughly n_partners / 2 partners ahead
    of it (the pairs behind it come from the earlier channels).
    """
    pairs_i, pairs_j = [], []
    for i in range(n_channels):
        cand = normal_distribution(
            i - corr_len, min(n_channels - 1, i + corr_len), n_partners, partner_std
        )
        j = cand[(cand >= 0) & (cand < n_channels) & (cand > i)]
        if j.size:
            pairs_i.append(np.full(j.size, i, dtype=np.int64))
            pairs_j.append(j.astype(np.int64))
    if not pairs_i:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(pairs_i), np.concatenate(pairs_j)


@jit(nopython=True, parallel=True, cache=True)
def _pair_lags_numba(data, pairs_i, pairs_j, pair_max_shift, lags):  # pragma: no cover - compiled
    n_samples = data.shape[1]
    for k in prange(len(pairs_i)):
        tr_i = data[pairs_i[k], :]
        tr_j = data[pairs_j[k], :]
        ms = pair_max_shift[k]
        best_shift = 0
        best_abs = 0.0
        for s in range(2 * ms + 1):
            shift = s - ms
            total = 0.0
            if shift < 0:
                for idx in range(n_samples + shift):
                    total += tr_i[idx] * tr_j[idx - shift]
            elif shift > 0:
                for idx in range(n_samples - shift):
                    total += tr_i[idx + shift] * tr_j[idx]
            else:
                for idx in range(n_samples):
                    total += tr_i[idx] * tr_j[idx]
            if abs(total) > best_abs:
                best_abs = abs(total)
                best_shift = shift
        lags[k] = best_shift


def pairwise_lags(
    data: np.ndarray,
    pairs_i: np.ndarray,
    pairs_j: np.ndarray,
    pair_slope: float = 0.2,
    pair_min_shift: int = 3,
    use_numba: bool | None = None,
) -> np.ndarray:
    """Lag (samples) maximising |cross-correlation| of each pair, searched within
    +-max(pair_slope * (j - i), pair_min_shift) samples. Sign-agnostic, so a polarity
    flip between channels does not break the alignment.
    """
    if use_numba is None:
        use_numba = NUMBA_AVAILABLE
    if use_numba and not NUMBA_AVAILABLE:
        raise RuntimeError("use_numba=True requested but numba is not installed")
    pair_max_shift = np.maximum((pairs_j - pairs_i) * pair_slope, pair_min_shift).astype(np.int64)
    lags = np.zeros(len(pairs_i), dtype=np.float64)
    if len(pairs_i) == 0:
        return lags
    data_c = np.ascontiguousarray(data, dtype=np.float64)
    if use_numba:
        _pair_lags_numba(data_c, pairs_i, pairs_j, pair_max_shift, lags)
    else:
        for k in range(len(pairs_i)):
            ms = int(pair_max_shift[k])
            corr = limited_cc(data_c[pairs_i[k]], data_c[pairs_j[k]], ms, use_numba=False)
            lags[k] = np.argmax(np.abs(corr)) - ms
    return lags


def solve_tau(
    n_channels: int,
    pairs_i: np.ndarray,
    pairs_j: np.ndarray,
    lags: np.ndarray,
    lamb: float = 1.0,
    smoothness: float = 0.0,
    reference_dt: np.ndarray | None = None,
    tau_avg: int = 100,
) -> np.ndarray:
    """Least-squares tau (n_channels,) from pairwise lags, with an optional first-difference
    smoothness term and a final moving average of ``tau_avg`` channels.

    ``reference_dt`` (n_channels - 1,) makes the smoothness term target that adjacent-channel
    difference instead of zero (follow a theoretical moveout while correlating).
    """
    n_pairs = len(pairs_i)
    rows = np.repeat(np.arange(n_pairs), 2)
    cols = np.column_stack([pairs_i, pairs_j]).ravel()
    vals = np.tile([1.0, -1.0], n_pairs)
    diff = sparse.csr_matrix((lamb * vals, (rows, cols)), shape=(n_pairs, n_channels))
    b = lamb * np.asarray(lags, float)
    if smoothness > 0:
        m = n_channels - 1
        d_rows = np.repeat(np.arange(m), 2)
        d_cols = np.column_stack([np.arange(m), np.arange(1, n_channels)]).ravel()
        d_vals = np.tile([-1.0, 1.0], m)
        d_mat = sparse.csr_matrix((smoothness * d_vals, (d_rows, d_cols)), shape=(m, n_channels))
        if reference_dt is None:
            b_smooth = np.zeros(m)
        else:
            reference_dt = np.asarray(reference_dt, float)
            if reference_dt.shape != (m,):
                raise ValueError(f"reference_dt must have shape ({m},), got {reference_dt.shape}")
            b_smooth = reference_dt
        diff = sparse.vstack([diff, d_mat]).tocsr()
        b = np.concatenate([b, smoothness * b_smooth])
    tau = lsqr(diff, b, atol=1e-10, btol=1e-10)[0]
    if tau_avg > n_channels:
        log.warning(
            "tau_avg %d longer than the %d refined channels; averaging over all of them",
            tau_avg,
            n_channels,
        )
        tau_avg = n_channels
    return moving_avg(tau, tau_avg)


def mccc(
    data: np.ndarray,
    corr_len: int,
    n_partners: int = 50,
    partner_std: float = 20.0,
    pair_slope: float = 0.2,
    pair_min_shift: int = 3,
    lamb: float = 1.0,
    smoothness: float = 0.0,
    reference_dt: np.ndarray | None = None,
    tau_avg: int = 100,
    use_numba: bool | None = None,
) -> np.ndarray:
    """One MCCC pass on an aligned gather (n_channels, n_samples); returns tau (n_channels,)."""
    pairs_i, pairs_j = partner_pairs(data.shape[0], corr_len, n_partners, partner_std)
    lags = pairwise_lags(data, pairs_i, pairs_j, pair_slope, pair_min_shift, use_numba)
    return solve_tau(data.shape[0], pairs_i, pairs_j, lags, lamb, smoothness, reference_dt, tau_avg)


def iterate_align(
    aligned: np.ndarray,
    corr_len: int,
    n_iter: int = 4,
    medfilt_channels: int = 25,
    medfilt_iters: tuple[int, ...] = (1, 2, 3),
    use_numba: bool | None = None,
    **mccc_kwargs,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Iterated MCCC on a pre-aligned, windowed gather.

    Pass i (1-based) correlates within ``corr_len // i`` channels, applies tau, and median
    filters the gather along the fibre when i is in ``medfilt_iters``. Returns
    (aligned, total_tau, taus); ``total_tau`` (n_channels,) is the summed tau, so the
    refined arrival of channel c is ``round(initial_pick[c]) + total_tau[c]`` in the
    original sample axis (see ``pipeline.refine_curve``).
    """
    taus = []
    total = np.zeros(aligned.shape[0])
    for i in range(1, n_iter + 1):
        tau = mccc(aligned, corr_len // i, use_numba=use_numba, **mccc_kwargs)
        taus.append(tau)
        total += tau
        aligned = tau_shift(aligned, tau)
        if i in medfilt_iters:
            aligned = spatial_median(aligned, medfilt_channels)
    return aligned, total, taus
