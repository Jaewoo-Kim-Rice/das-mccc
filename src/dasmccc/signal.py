"""Low-level signal helpers: wavelet, robust statistics, bounded cross-correlation.

Numba is optional. When it is missing the bounded cross-correlation falls back to a
pure-numpy loop that returns the same values but runs one to two orders of magnitude
slower; a warning is logged once at import so the slow path is never silent.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import truncnorm

log = logging.getLogger("dasmccc")

try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without numba installed
    NUMBA_AVAILABLE = False
    prange = range

    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    log.warning(
        "numba is not installed: dasmccc pairwise correlation runs in pure numpy "
        "(same results, much slower). Install the 'numba' extra for the fast path."
    )


def mmad(data: np.ndarray) -> np.ndarray:
    """Modified z-score 0.6745 * (x - median) / MAD, NaN-aware.

    Returns zeros when the MAD is zero (constant input).
    """
    data = np.asarray(data, float)
    median_val = np.nanmedian(data)
    mad_val = np.nanmedian(np.abs(data - median_val))
    if mad_val == 0:
        return np.zeros_like(data)
    return 0.6745 * (data - median_val) / mad_val


def ricker(center_freq: float, n_samples: int, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Ricker (Mexican hat) wavelet of ``n_samples`` centred on zero; returns (t, y)."""
    t = np.arange(-n_samples / 2, n_samples / 2) / sample_rate
    a = (np.pi**2) * (center_freq**2) * (t**2)
    return t, (1.0 - 2.0 * a) * np.exp(-a)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


_TRUNCNORM_TEMPLATES: dict[tuple[float, float, int], np.ndarray] = {}


def _truncnorm_template(a: float, b: float, n: int) -> np.ndarray:
    """Quantiles of a standard normal truncated to [a, b], mapped onto [0, 1]; cached."""
    key = (round(a, 2), round(b, 2), n)
    if key not in _TRUNCNORM_TEMPLATES:
        a_r, b_r, _ = key
        raw = truncnorm.ppf(np.linspace(0, 1, n), a_r, b_r, loc=0, scale=1)
        _TRUNCNORM_TEMPLATES[key] = (raw - a_r) / (b_r - a_r)
    return _TRUNCNORM_TEMPLATES[key]


def normal_distribution(start: int, end: int, n: int, std: float = 20.0) -> np.ndarray:
    """``n`` strictly increasing integers in [start, end] drawn as quantiles of a
    normal distribution (std in the same units as start/end) truncated to that range.

    Used to pick correlation partners: dense near the channel itself, sparse at the
    edges of the neighbourhood. The first and last points always land on ``start``
    and ``end``; ties after rounding are pushed up by one so the result is strictly
    increasing (it may therefore exceed ``end`` by a few units).
    """
    mean = (start + end) / 2
    a = (start - mean) / std
    b = (end - mean) / std
    points = start + (end - start) * _truncnorm_template(a, b, n)
    points_int = np.round(points).astype(int)
    for i in range(1, len(points_int)):
        if points_int[i] <= points_int[i - 1]:
            points_int[i] = points_int[i - 1] + 1
    return points_int


@jit(nopython=True, cache=True)
def _limited_cc_numba(tr, other, max_shift):  # pragma: no cover - compiled
    len_tr = len(tr)
    n_shifts = 2 * max_shift + 1
    corr_values = np.zeros(n_shifts)
    for i in range(n_shifts):
        shift = i - max_shift
        total = 0.0
        if shift < 0:
            for k in range(len_tr + shift):
                total += tr[k] * other[k - shift]
        elif shift > 0:
            for k in range(len_tr - shift):
                total += tr[k + shift] * other[k]
        else:
            for k in range(len_tr):
                total += tr[k] * other[k]
        corr_values[i] = total
    return corr_values


def _limited_cc_numpy(tr, other, max_shift):
    len_tr = len(tr)
    corr_values = np.zeros(2 * max_shift + 1)
    for i, shift in enumerate(range(-max_shift, max_shift + 1)):
        if shift < 0:
            corr_values[i] = np.dot(tr[: len_tr + shift], other[-shift:])
        elif shift > 0:
            corr_values[i] = np.dot(tr[shift:], other[: len_tr - shift])
        else:
            corr_values[i] = np.dot(tr, other)
    return corr_values


def limited_cc(
    tr: np.ndarray, other: np.ndarray, max_shift: int, use_numba: bool | None = None
) -> np.ndarray:
    """Unnormalised cross-correlation of two equal-length traces for lags in
    [-max_shift, max_shift]; index k corresponds to lag k - max_shift.

    A positive lag means ``tr`` is shifted forward relative to ``other``.
    """
    if use_numba is None:
        use_numba = NUMBA_AVAILABLE
    if use_numba and not NUMBA_AVAILABLE:
        raise RuntimeError("use_numba=True requested but numba is not installed")
    if use_numba:
        return _limited_cc_numba(
            np.ascontiguousarray(tr, dtype=np.float64),
            np.ascontiguousarray(other, dtype=np.float64),
            int(max_shift),
        )
    return _limited_cc_numpy(np.asarray(tr, float), np.asarray(other, float), int(max_shift))
