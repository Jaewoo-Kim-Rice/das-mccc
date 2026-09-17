"""Per-channel polarity, SNR and amplitude QC on an aligned gather (Ricker matching)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .signal import limited_cc, mmad, ricker, rms


@dataclass(frozen=True)
class PolarityConfig:
    """Ricker matching around the alignment sample.

    fs, ricker_hz : sampling rate and Ricker centre frequency; only their ratio matters
                    (50 Hz at 1 kHz = one main lobe of about 20 samples).
    half_win      : half width (samples) of the signal window around the alignment sample;
                    the Ricker template has 2 * half_win samples.
    max_lag       : lag search (samples) of the template inside the signal window.
    snr_thresh    : rms(signal window) / rms(rest of the trace) below which polarity is 0.
    mmad_thresh   : modified z-score of the window rms above which a channel is an
                    amplitude outlier (near-field or bad channel) and its polarity is 0.
    """

    fs: float = 1000.0
    ricker_hz: float = 50.0
    half_win: int = 30
    max_lag: int = 10
    snr_thresh: float = 3.0
    mmad_thresh: float = 4.0


@dataclass
class PolarityResult:
    polarity: np.ndarray  # (n_channels,) in {-1, 0, +1}; 0 = undetermined
    snr: np.ndarray  # (n_channels,) NaN where the trace is not available
    amplitude: np.ndarray  # (n_channels,) window rms
    lag: np.ndarray  # (n_channels,) best Ricker lag in samples (NaN where undetermined)
    outlier: np.ndarray  # (n_channels,) bool, amplitude outlier by MMAD


def ricker_polarity(aligned: np.ndarray, cfg: PolarityConfig | None = None) -> PolarityResult:
    """Polarity of every channel of an aligned gather (n_channels, n_samples).

    The sign is that of the Ricker cross-correlation at its absolute maximum within
    +-max_lag of the alignment sample. Rows that are not finite (channels outside the
    refined range) get polarity 0, NaN snr/amplitude/lag and outlier False.
    """
    cfg = cfg or PolarityConfig()
    n_ch, n = aligned.shape
    centre = n // 2
    lo, hi = centre - cfg.half_win, centre + cfg.half_win
    if lo < 0 or hi > n:
        raise ValueError(f"half_win {cfg.half_win} does not fit in {n} samples")
    template = ricker(cfg.ricker_hz, 2 * cfg.half_win, cfg.fs)[1]
    snr = np.full(n_ch, np.nan)
    amp = np.full(n_ch, np.nan)
    lag = np.full(n_ch, np.nan)
    pol = np.zeros(n_ch)
    rest = np.ones(n, bool)
    rest[lo:hi] = False
    for c in range(n_ch):
        tr = aligned[c]
        if not np.all(np.isfinite(tr)):
            continue
        amp[c] = rms(tr[lo:hi])
        noise = rms(tr[rest])
        snr[c] = amp[c] / noise if noise > 0 else np.inf
        corr = limited_cc(template, tr[lo:hi], cfg.max_lag)
        k = int(np.argmax(np.abs(corr)))
        lag[c] = k - cfg.max_lag
        pol[c] = np.sign(corr[k])
    outlier = np.zeros(n_ch, bool)
    finite = np.isfinite(amp)
    if finite.any():
        outlier[finite] = mmad(amp[finite]) > cfg.mmad_thresh
    weak = finite & (snr < cfg.snr_thresh)
    pol[weak | outlier] = 0
    lag[weak | outlier] = np.nan
    return PolarityResult(pol, snr, amp, lag, outlier)


@dataclass
class RickerWindows:
    """Per-channel Ricker match on an aligned gather (the das-focmec ``diff_corr_ric`` step).

    dt        : matched arrival sample per channel (n_samples // 2 - lag); NaN below SNR.
    polarity  : sign of the correlation at its absolute maximum; NaN below SNR or outlier.
    amplitude : rms of the trace inside the matched window; NaN below SNR or outlier.
    snr       : rms inside / rms outside the matched window.
    window    : (n_channels, 2) sample window around the match; [0, 0] for outliers.
    """

    dt: np.ndarray
    polarity: np.ndarray
    amplitude: np.ndarray
    snr: np.ndarray
    window: np.ndarray


def ricker_windows(
    aligned: np.ndarray,
    max_lag: int,
    snr_thresh: float = 5.0,
    mmad_thresh: float = 4.0,
    ricker_hz: float = 50.0,
    fs: float = 1000.0,
    half_win: int = 25,
) -> RickerWindows:
    """Match a full-length Ricker template to every aligned trace within +-max_lag samples
    and derive a per-channel window (+-half_win around the match), SNR and polarity; channels
    whose window rms is an MMAD outlier (near-field, bad channel) get window [0, 0] and NaN
    polarity / amplitude.

    Ported verbatim from das-focmec ``diff_corr_ric`` (including its window clipping rule)
    so that the focal-mechanism pipeline reproduces bit for bit.
    """
    n_ch, n = aligned.shape
    template = ricker(ricker_hz, n, fs)[1]
    half = n // 2
    dts = np.full(n_ch, np.nan)
    pol = np.full(n_ch, np.nan)
    amp = np.full(n_ch, np.nan)
    snr = np.full(n_ch, np.nan)
    wins = np.zeros((n_ch, 2), dtype=int)
    for c in range(n_ch):
        tr = aligned[c]
        corr = limited_cc(template, tr, max_lag)
        k = int(np.argmax(np.abs(corr)))
        w0 = max_lag + half - min(half, k) - half_win
        w1 = max_lag + half - max(-half, k) + half_win
        wins[c] = (w0, w1)
        rest = np.ones(n, bool)
        rest[w0:w1] = False
        with np.errstate(divide="ignore", invalid="ignore"):
            a = rms(tr[w0:w1])
            s = a / rms(tr[rest])
        snr[c] = s
        if s >= snr_thresh:
            dts[c] = k - max_lag
            pol[c] = np.sign(corr[k])
            amp[c] = a
    outlier = mmad(amp) > mmad_thresh
    wins[outlier] = 0
    amp[outlier] = np.nan
    pol[outlier] = np.nan
    return RickerWindows(half - dts, pol, amp, snr, wins)
