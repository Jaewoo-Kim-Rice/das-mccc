"""Array operations on a (n_channels, n_samples) gather: integer time shifts,
tau smoothing and the spatial (channel-axis) median filter."""

from __future__ import annotations

import numpy as np
from scipy.signal import medfilt

PAD_VALUE = 1e-14  # samples rolled in from outside the trace (tiny, not exactly zero)


def _shift_trace(trace: np.ndarray, shift: int) -> np.ndarray:
    """Delay ``trace`` by ``shift`` samples (negative = advance), padding with PAD_VALUE."""
    if shift > 0:
        return np.concatenate([np.full(shift, PAD_VALUE), trace[:-shift]])
    if shift < 0:
        return np.concatenate([trace[-shift:], np.full(-shift, PAD_VALUE)])
    return trace.copy()


def shift_arr(data: np.ndarray, picks: np.ndarray, base_time: int) -> tuple[np.ndarray, np.ndarray]:
    """Shift every channel so that its pick (sample index, rounded) lands on ``base_time``.

    Returns (shifted, shifts) with ``shifts[c] = round(base_time - picks[c])`` in samples.
    """
    picks = np.asarray(picks, float)
    if picks.shape != (data.shape[0],):
        raise ValueError(f"picks must have shape ({data.shape[0]},), got {picks.shape}")
    if not np.all(np.isfinite(picks)):
        raise ValueError("picks must be finite for every channel")
    shifts = np.round(base_time - picks).astype(int)
    shifted = np.stack([_shift_trace(data[c], int(shifts[c])) for c in range(data.shape[0])])
    return shifted, shifts


def tau_shift(data: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Advance every channel by ``round(tau[c])`` samples (tau > 0 moves the trace earlier)."""
    tau = np.asarray(tau, float)
    if tau.shape != (data.shape[0],):
        raise ValueError(f"tau must have shape ({data.shape[0]},), got {tau.shape}")
    return np.stack([_shift_trace(data[c], int(round(-tau[c]))) for c in range(data.shape[0])])


def moving_avg(x: np.ndarray, win: int) -> np.ndarray:
    """Centred moving average of length ``win`` with reflected edges; same length as ``x``."""
    if win < 2:
        return np.asarray(x, float).copy()
    if win > len(x):
        raise ValueError(f"moving_avg window {win} longer than the array ({len(x)})")
    pad_left = win // 2
    pad_right = win - 1 - pad_left
    padded = np.pad(np.asarray(x, float), (pad_left, pad_right), mode="reflect")
    return np.convolve(padded, np.ones(win) / win, mode="valid")


def spatial_median(data: np.ndarray, n_channels: int) -> np.ndarray:
    """Median filter along the channel axis with an odd kernel of ``n_channels``
    (zero-padded at the fibre ends, as scipy.signal.medfilt does)."""
    if n_channels % 2 == 0:
        raise ValueError("spatial median kernel must be odd")
    return medfilt(data, kernel_size=(n_channels, 1))
