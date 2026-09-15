"""Curve refinement: initial curve in, refined curve plus QC out.

Everything is in samples (time axis) and channels (space axis); there is no notion of
site, file or sampling rate except the ratio needed by the Ricker template.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .anchor import first_lobe, stack_peak
from .core import iterate_align
from .ops import shift_arr
from .polarity import PolarityConfig, PolarityResult, ricker_polarity

log = logging.getLogger("dasmccc")


@dataclass(frozen=True)
class RefineConfig:
    """Knobs of one refinement, in samples / channels.

    window          : length of the time window cut around the initial curve after
                      alignment (+-window/2). Nothing outside it is ever seen.
    corr_len        : channel neighbourhood half width for correlation partners; pass i of
                      the iteration uses corr_len // i.
    n_partners,
    partner_std     : partners per channel, drawn as quantiles of a normal distribution
                      (std in channels) truncated to the neighbourhood (see core).
    pair_slope,
    pair_min_shift  : per-pair lag bound max(pair_slope * gap, pair_min_shift) samples.
                      0.2 samples/channel bounds the slope of the *residual* moveout with
                      respect to the initial curve (at 1 kHz and 2 m spacing: 0.1 ms/m);
                      adjacent channels may disagree by at most pair_min_shift samples.
    lamb, smoothness: weights of the data term and of ||d tau / d channel||^2.
    n_iter          : MCCC passes.
    tau_avg         : moving-average length (channels) applied to tau after each solve.
    medfilt_channels,
    medfilt_iters   : spatial median filter of the aligned gather after the listed passes.
    pre_mask        : keep only +-pre_mask samples around the alignment sample before the
                      first pass (None = off).
    stack_polarity  : False (default) stacks the aligned traces as they are, which is what
                      the human-referenced anchor result was measured with. True multiplies
                      each trace by its Ricker polarity first (channels with polarity 0
                      left out) so a genuine reversal along the fibre does not cancel the
                      stack; on the CAPE 2025 reads this made the anchor worse (see docs).
    anchor          : "first_lobe", "stack_peak" or None (keep the initial curve's level).
    anchor_min_frac,
    anchor_guard    : first_lobe parameters (see anchor.first_lobe).
    coherence_half  : half window (samples) for the trace-vs-stack coherence.
    polarity        : Ricker polarity / SNR / MMAD settings.
    """

    window: int = 200
    corr_len: int = 200
    n_partners: int = 50
    partner_std: float = 20.0
    pair_slope: float = 0.2
    pair_min_shift: int = 3
    lamb: float = 1.0
    smoothness: float = 50.0
    n_iter: int = 4
    tau_avg: int = 100
    medfilt_channels: int = 25
    medfilt_iters: tuple[int, ...] = (1, 2, 3)
    pre_mask: int | None = 100
    stack_polarity: bool = False
    anchor: str | None = "first_lobe"
    anchor_min_frac: float = 0.3
    anchor_guard: int | None = 40
    coherence_half: int = 30
    polarity: PolarityConfig = field(default_factory=PolarityConfig)


DIRECT = RefineConfig()
"""Direct P / S waves (the das-focmec ev_1090 settings)."""

SECONDARY = RefineConfig(window=120, corr_len=100, n_iter=3, pre_mask=50)
"""Secondary phases (SP conversions, reflections): narrower window, shorter neighbourhood."""


@dataclass
class RefineResult:
    """All arrays are indexed by channel of the input waveform.

    curve         : refined arrival (samples, float); NaN where the input curve was NaN.
                    Equals the relative level plus ``anchor_offset`` when that is finite.
    shifts        : curve - input curve.
    aligned       : (n_channels, window) gather after the last pass; NaN rows outside the
                    refined channel range. Sample window // 2 is the alignment sample.
    stack         : mean of the finite rows of ``aligned`` (polarity-corrected if asked).
    coherence     : |normalised correlation| of each aligned trace with the stack within
                    +-coherence_half samples of the alignment sample.
    polarity, snr : from the Ricker step (0 / NaN where undetermined).
    kept          : input finite and not an amplitude outlier and snr >= threshold.
    anchor_offset : samples added to the relative level (0 when anchor is None, NaN when
                    the anchor rule refused).
    taus          : tau of each pass on the refined channel range.
    """

    curve: np.ndarray
    shifts: np.ndarray
    aligned: np.ndarray
    stack: np.ndarray
    coherence: np.ndarray
    polarity: np.ndarray
    snr: np.ndarray
    kept: np.ndarray
    anchor_offset: float
    taus: list[np.ndarray]
    qc: PolarityResult
    channel_range: tuple[int, int]

    @property
    def curve_relative(self) -> np.ndarray:
        """Refined curve at the initial curve's level (no anchor)."""
        off = 0.0 if not np.isfinite(self.anchor_offset) else self.anchor_offset
        return self.curve - off


def _coherence(aligned: np.ndarray, stack: np.ndarray, centre: int, half: int) -> np.ndarray:
    w = slice(centre - half, centre + half)
    s = stack[w] - stack[w].mean()
    t = aligned[:, w] - aligned[:, w].mean(axis=1, keepdims=True)
    num = t @ s
    den = np.linalg.norm(t, axis=1) * np.linalg.norm(s) + 1e-12
    return np.abs(num / den)


def refine_curve(
    waveform: np.ndarray,
    curve: np.ndarray,
    cfg: RefineConfig = DIRECT,
    mask: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> RefineResult:
    """Refine one arrival curve on a gather.

    waveform : (n_channels, n_samples) float, already filtered as desired.
    curve    : (n_channels,) initial arrival in samples; NaN where the phase is not
               picked. The refined range is [first finite, last finite]; interior NaNs are
               linearly interpolated for the alignment only and stay NaN in the output.
               The fractional part of the curve is dropped (integer alignment).
    mask     : optional (n_channels, n_samples) bool, True where the waveform is zeroed
               first (used to hide already refined phases).

    Steps: align every channel on its pick, cut +-window/2, optional pre-mask, iterate
    MCCC with the spatial median filter, then anchor the level on the aligned stack and run
    the Ricker polarity / SNR / MMAD QC on the aligned gather.
    """
    waveform = np.asarray(waveform, float)
    curve = np.asarray(curve, float)
    if waveform.ndim != 2:
        raise ValueError("waveform must be (n_channels, n_samples)")
    n_ch, n = waveform.shape
    if curve.shape != (n_ch,):
        raise ValueError(f"curve must have shape ({n_ch},), got {curve.shape}")
    finite = np.isfinite(curve)
    if finite.sum() < 2:
        raise ValueError("curve needs at least two finite picks")
    half = cfg.window // 2
    if 2 * half > n:
        raise ValueError(f"window {cfg.window} longer than the waveform ({n} samples)")
    if cfg.pre_mask is not None and cfg.pre_mask > half:
        raise ValueError("pre_mask must not exceed window // 2")
    if mask is not None:
        mask = np.asarray(mask, bool)
        if mask.shape != waveform.shape:
            raise ValueError("mask must have the waveform's shape")
        waveform = np.where(mask, 0.0, waveform)

    idx = np.flatnonzero(finite)
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    sub = waveform[lo:hi]
    c = curve[lo:hi].copy()
    v = np.isfinite(c)
    c[~v] = np.interp(np.flatnonzero(~v), np.flatnonzero(v), c[v])

    base = n // 2
    shifted, _ = shift_arr(sub, c, base)
    shifted = shifted[:, base - half : base + half]
    if cfg.pre_mask is not None:
        keep = np.zeros_like(shifted)
        keep[:, half - cfg.pre_mask : half + cfg.pre_mask] = shifted[
            :, half - cfg.pre_mask : half + cfg.pre_mask
        ]
        shifted = keep

    aligned, total_tau, taus = iterate_align(
        shifted,
        cfg.corr_len,
        n_iter=cfg.n_iter,
        medfilt_channels=cfg.medfilt_channels,
        medfilt_iters=cfg.medfilt_iters,
        use_numba=use_numba,
        n_partners=cfg.n_partners,
        partner_std=cfg.partner_std,
        pair_slope=cfg.pair_slope,
        pair_min_shift=cfg.pair_min_shift,
        lamb=cfg.lamb,
        smoothness=cfg.smoothness,
        tau_avg=cfg.tau_avg,
    )
    relative = np.round(c) + total_tau

    qc_sub = ricker_polarity(aligned, cfg.polarity)
    signed = qc_sub.polarity != 0
    if cfg.stack_polarity and signed.any():
        stack = (aligned[signed] * qc_sub.polarity[signed, None]).mean(axis=0)
    else:
        if cfg.stack_polarity:
            log.warning("no channel has a determined polarity; using the plain stack")
        stack = aligned.mean(axis=0)
    if cfg.anchor is None:
        offset = 0.0
    elif cfg.anchor == "first_lobe":
        offset = first_lobe(stack, half, cfg.anchor_min_frac, cfg.anchor_guard)
    elif cfg.anchor == "stack_peak":
        offset = stack_peak(stack, half)
    else:
        raise ValueError(f"unknown anchor rule {cfg.anchor!r}")
    refined = relative + (offset if np.isfinite(offset) else 0.0)

    coh = _coherence(aligned, stack, half, cfg.coherence_half)

    def full(values, fill):
        out = np.full(n_ch, fill, dtype=float)
        out[lo:hi] = values
        return out

    out_curve = full(refined, np.nan)
    out_curve[~finite] = np.nan
    aligned_full = np.full((n_ch, 2 * half), np.nan)
    aligned_full[lo:hi] = aligned
    polarity = full(qc_sub.polarity, 0.0)
    snr = full(qc_sub.snr, np.nan)
    outlier = np.zeros(n_ch, bool)
    outlier[lo:hi] = qc_sub.outlier
    kept = finite & ~outlier & (np.nan_to_num(snr, nan=-np.inf) >= cfg.polarity.snr_thresh)
    return RefineResult(
        curve=out_curve,
        shifts=out_curve - curve,
        aligned=aligned_full,
        stack=stack,
        coherence=full(coh, np.nan),
        polarity=polarity,
        snr=snr,
        kept=kept,
        anchor_offset=float(offset),
        taus=taus,
        qc=qc_sub,
        channel_range=(lo, hi),
    )


def refine_phases(
    waveform: np.ndarray,
    curves: dict[str, np.ndarray],
    order: tuple[str, ...] = ("S", "P", "SP", "REFL"),
    cfg_by_tag: dict[str, RefineConfig] | None = None,
    mask_half: int = 45,
    guard_channels: int = 50,
    use_numba: bool | None = None,
    tags: dict[str, str] | None = None,
) -> dict[str, RefineResult]:
    """Refine several curves of one gather, strongest phase first.

    ``curves`` maps a key to an initial curve. By default the key is the phase tag; when
    one gather holds several curves of the same tag (two reflections, an SP candidate per
    interface) pass ``tags`` mapping every key to its tag and use any keys you like.
    Curves are processed by the position of their tag in ``order`` (tags absent from
    ``curves`` are skipped; a curve whose tag is not in ``order`` is an error) and, within
    one tag, in the order of ``curves``. The result is keyed like ``curves`` in processing
    order.

    Before refining a curve, every curve already refined is hidden by zeroing +-mask_half
    samples around its refined curve at the initial curve's level (``curve_relative``),
    except within ``guard_channels`` of the junction channel where the new curve's initial
    curve comes within ``mask_half`` samples of that refined curve: blanking the junction
    would let the child drift there.

    ``cfg_by_tag`` defaults to DIRECT for "P" and "S" and SECONDARY for every other tag;
    entries given override or extend that.
    """
    tag_of = {k: k for k in curves} if tags is None else dict(tags)
    missing = set(curves) - set(tag_of)
    if missing:
        raise ValueError(f"curves {sorted(missing)} have no entry in tags")
    unknown = {k: tag_of[k] for k in curves if tag_of[k] not in order}
    if unknown:
        raise ValueError(f"curves {unknown} have tags outside order {order}")
    cfgs = {"P": DIRECT, "S": DIRECT}
    if cfg_by_tag:
        cfgs.update(cfg_by_tag)
    keys = sorted(curves, key=lambda k: order.index(tag_of[k]))  # stable within a tag
    results: dict[str, RefineResult] = {}
    for key in keys:
        tag = tag_of[key]
        child0 = np.asarray(curves[key], float)
        mask = np.zeros(waveform.shape, bool)
        for done in results.values():
            parent = done.curve_relative
            both = np.isfinite(child0) & np.isfinite(parent)
            guarded = np.zeros(waveform.shape[0], bool)
            if both.any():
                d = np.abs(child0 - parent)
                d[~both] = np.inf
                jc = int(np.argmin(d))
                if d[jc] <= mask_half:
                    guarded[max(0, jc - guard_channels) : jc + guard_channels + 1] = True
            for ch in np.flatnonzero(np.isfinite(parent) & ~guarded):
                p = int(round(parent[ch]))
                mask[ch, max(0, p - mask_half) : p + mask_half] = True
        cfg = cfgs.get(tag, SECONDARY)
        name = "DIRECT" if cfg is DIRECT else "config"
        log.info("refine_phases: %s (%s) with %s", key, tag, name)
        results[key] = refine_curve(waveform, child0, cfg, mask=mask, use_numba=use_numba)
    return results
