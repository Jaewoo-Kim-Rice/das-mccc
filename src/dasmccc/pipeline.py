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
                      first pass (None = off). DIRECT: 40, so that another arrival more than
                      ~50 samples away (an S behind a P) stays outside what is correlated;
                      the price is that an initial curve more than 40 samples off is not
                      recovered (its coherence says so). Was 100 (= the whole window) before
                      0.2.0, with exclude_near doing that job by dropping channels.
    pre_mask_taper  : cosine edge of the pre-mask, in samples: the weight is 1 within
                      +-(pre_mask - pre_mask_taper) and falls to 0 at +-pre_mask. 0 is the
                      hard cut. A hard edge at the same sample on every channel is a feature
                      the correlation can lock onto (a bias towards zero lag, i.e. towards
                      the initial curve); with a narrow pre-mask, use a taper (DIRECT: 10).
    stack           : "norm" (default; every aligned trace divided by its rms before the
                      mean, so strong channels do not own the stack), "mean" (plain mean) or
                      "median" (channel-wise median).
    stack_polarity  : False (default) stacks the aligned traces as they are. True multiplies
                      each trace by its Ricker polarity first (channels with polarity 0
                      left out) so a genuine reversal along the fibre does not cancel the
                      stack; on the CAPE 2025 reads this made the anchor worse (see docs).
    anchor          : "first_lobe", "stack_peak", "parent" or None (keep the initial curve's
                      level). "parent" is only meaningful inside refine_phases: the curve
                      takes the anchor offset of the already refined curve it leaves (the one
                      it comes closest to, within mask_half), so the junction is preserved;
                      with no such parent the level is kept and a warning is logged.
    anchor_min_frac,
    anchor_guard,
    anchor_contiguous,
    anchor_window   : first_lobe parameters (see anchor.first_lobe).
    exclude_near    : refine_phases only. Channels where this curve's initial curve comes
                      within exclude_near samples of a curve refined before it are not
                      refined at all: another arrival inside the correlation window
                      corrupts the alignment and the stack (a P within 170 ms of the S:
                      with a +-100 window the S wavelet's leading lobes reach well beyond
                      the S curve). The remaining runs of at least ``min_run`` channels
                      are refined separately, each with its own anchor, and the excluded
                      channels are bridged so the result stays continuous (see ``bridge``
                      and RefineResult.runs). None (default since 0.2.0): the narrow
                      tapered pre-mask keeps the other arrival out instead, and every
                      channel is refined; on 616 CAPE 2025 reads this beat exclusion at 170
                      on every P and S figure. SECONDARY never excludes: a converted or
                      reflected phase meets its parent at the junction by construction.
    bridge          : how excluded channels are filled. "shift" (default): the initial
                      curve's shape carried at the refined level, the shift (refined minus
                      initial at the run ends) interpolated linearly across a gap and held
                      before the first / after the last run. "initial": the initial curve
                      as drawn, joined to the runs by tapering the run-end shift to zero
                      over 50 channels. On the CAPE 2025 P reads "shift" scored better
                      against the human picks (the initial picker's level convention
                      differs from the refined one); "initial" is right where the picker
                      happened to sit on the onset lobe.
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
    pre_mask: int | None = 40
    pre_mask_taper: int = 10
    stack: str = "norm"
    stack_polarity: bool = False
    anchor: str | None = "first_lobe"
    anchor_min_frac: float = 0.4
    anchor_guard: int | None = 40
    anchor_contiguous: bool = True
    anchor_window: tuple[int, int] | None = (-30, 10)
    exclude_near: int | None = None
    min_run: int = 60
    bridge: str = "shift"
    coherence_half: int = 30
    polarity: PolarityConfig = field(default_factory=PolarityConfig)

    def __post_init__(self):
        """Refuse settings that cannot run: the per-pair lag bound must fit inside the
        correlation window, else the overlap of two traces is empty and the correlation fails
        deep inside numpy with an unrelated message."""
        if self.window < 4:
            raise ValueError(f"window {self.window} samples is too short (need at least 4)")
        max_lag = max(self.pair_slope * self.corr_len, self.pair_min_shift)
        if max_lag >= self.window // 2:
            raise ValueError(
                f"the per-pair lag bound max(pair_slope * corr_len, pair_min_shift) = {max_lag:g} "
                f"samples must be below window / 2 = {self.window // 2}; at this sampling the "
                f"window ({self.window} samples) is too short for these lags"
            )
        if self.pre_mask is not None and self.pre_mask > self.window // 2:
            raise ValueError(
                f"pre_mask {self.pre_mask} must not exceed window // 2 = {self.window // 2}"
            )
        if self.pre_mask_taper < 0 or (
            self.pre_mask is not None and self.pre_mask_taper > self.pre_mask
        ):
            raise ValueError(
                f"pre_mask_taper {self.pre_mask_taper} must lie in [0, pre_mask = {self.pre_mask}]"
            )


DIRECT = RefineConfig()
"""Direct P / S waves (the das-focmec ev_1090 settings)."""

SECONDARY = RefineConfig(
    window=120, corr_len=100, n_iter=3, pre_mask=50, anchor="parent", exclude_near=None
)
"""Secondary phases (SP conversions, reflections): narrower window, shorter neighbourhood, and
the level inherited from the parent curve (their own stacks are too weak for a first-lobe rule;
an independent anchor pulled the junction apart by 15 ms on the CAPE 2025 reads)."""


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
                    the anchor rule refused or no parent was found).
    parent        : key of the refined curve whose anchor this one inherited (refine_phases
                    with anchor "parent"), else None.
    runs          : refine_phases with exclude_near: the [first, last] channel runs that were
                    refined (separately). Channels of the initial curve outside them are
                    bridged (RefineConfig.bridge) so the curve is continuous; ``refined``
                    marks the channels the MCCC actually refined (coherence, snr, polarity
                    are NaN / 0 elsewhere). None otherwise.
    anchor_offsets: the anchor offset of each run (``anchor_offset`` is the longest run's).
    refined       : (n_channels,) bool, see ``runs``; None when nothing was excluded.
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
    parent: str | None = None
    runs: list[tuple[int, int]] | None = None
    anchor_offsets: list[float] | None = None
    refined: np.ndarray | None = None

    @property
    def curve_relative(self) -> np.ndarray:
        """Refined curve at the initial curve's level (no anchor)."""
        off = 0.0 if not np.isfinite(self.anchor_offset) else self.anchor_offset
        return self.curve - off


def _stack(aligned: np.ndarray, cfg: RefineConfig, polarity: np.ndarray) -> np.ndarray:
    tr = aligned
    if cfg.stack_polarity:
        signed = polarity != 0
        if signed.any():
            tr = aligned[signed] * polarity[signed, None]
        else:
            log.warning("no channel has a determined polarity; stacking without polarity")
    if cfg.stack == "mean":
        return tr.mean(axis=0)
    if cfg.stack == "median":
        return np.median(tr, axis=0)
    if cfg.stack == "norm":
        rms = np.sqrt((tr**2).mean(axis=1, keepdims=True))
        rms[rms == 0] = 1.0
        return (tr / rms).mean(axis=0)
    raise ValueError(f"unknown stack kind {cfg.stack!r}")


def _coherence(aligned: np.ndarray, stack: np.ndarray, centre: int, half: int) -> np.ndarray:
    w = slice(centre - half, centre + half)
    s = stack[w] - stack[w].mean()
    t = aligned[:, w] - aligned[:, w].mean(axis=1, keepdims=True)
    num = t @ s
    den = np.linalg.norm(t, axis=1) * np.linalg.norm(s) + 1e-12
    return np.abs(num / den)


def pre_mask_weights(n: int, pre_mask: int, taper: int = 0) -> np.ndarray:
    """(n,) weights of the pre-mask on a window whose alignment sample is n // 2: 1 within
    +-(pre_mask - taper), a half-cosine from 1 to 0 over the last ``taper`` samples on each
    side, 0 beyond +-pre_mask. taper = 0 is the hard cut (1 on [half - pre_mask, half +
    pre_mask), 0 elsewhere)."""
    half = n // 2
    if not (0 <= taper <= pre_mask <= half):
        raise ValueError(f"need 0 <= taper {taper} <= pre_mask {pre_mask} <= n // 2 {half}")
    w = np.zeros(n)
    w[half - pre_mask : half + pre_mask] = 1.0
    if taper > 0:
        ramp = 0.5 * (1.0 + np.cos(np.pi * np.arange(1, taper + 1) / (taper + 1)))  # 1 -> 0
        w[half - pre_mask : half - pre_mask + taper] = ramp[::-1]  # rising edge
        w[half + pre_mask - taper : half + pre_mask] = ramp  # falling edge
    return w


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
        shifted = shifted * pre_mask_weights(2 * half, cfg.pre_mask, cfg.pre_mask_taper)

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
    stack = _stack(aligned, cfg, qc_sub.polarity)
    if cfg.anchor is None or cfg.anchor == "parent":
        offset = 0.0  # "parent" is resolved by refine_phases
    elif cfg.anchor == "first_lobe":
        offset = first_lobe(
            stack,
            half,
            cfg.anchor_min_frac,
            cfg.anchor_guard,
            cfg.anchor_contiguous,
            cfg.anchor_window,
        )
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
    on_excluded: str = "raise",
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

    A curve whose config sets ``exclude_near`` is not refined where its initial curve runs
    within that many samples of a curve refined before it (see RefineConfig). When nothing
    of it survives, ``on_excluded`` decides: "raise" (default) raises ValueError, "skip"
    logs a warning and leaves the key out of the result.
    """
    if on_excluded not in ("raise", "skip"):
        raise ValueError(f"on_excluded must be 'raise' or 'skip', got {on_excluded!r}")
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
        if cfg.exclude_near is not None and results:
            try:
                res = _refine_runs(waveform, child0, cfg, mask, results, use_numba)
            except NothingToRefine as e:
                if on_excluded == "raise":
                    raise ValueError(f"{key} ({tag}): {e}") from e
                log.warning("%s (%s) skipped: %s", key, tag, e)
                continue
        else:
            res = refine_curve(waveform, child0, cfg, mask=mask, use_numba=use_numba)
        if cfg.anchor == "parent":
            parent_key = _closest_parent(child0, results, mask_half)
            if parent_key is None:
                log.warning(
                    "%s (%s): no refined parent within %d samples; level kept", key, tag, mask_half
                )
                res.anchor_offset = float("nan")
            else:
                off = results[parent_key].anchor_offset
                off = 0.0 if not np.isfinite(off) else off
                res.curve = res.curve + off
                res.shifts = res.shifts + off
                res.anchor_offset = float(off)
                res.parent = parent_key
        results[key] = res
    return results


class NothingToRefine(ValueError):
    """Every channel of a curve lies within ``exclude_near`` of an already refined curve."""


def _refine_runs(
    waveform: np.ndarray,
    child0: np.ndarray,
    cfg: RefineConfig,
    mask: np.ndarray,
    done: dict[str, RefineResult],
    use_numba: bool | None,
) -> RefineResult:
    """Refine the runs of ``child0`` that stay at least ``cfg.exclude_near`` samples away from
    every curve in ``done``; each run of at least ``cfg.min_run`` channels is refined on its
    own and the per-channel results are assembled (NaN elsewhere)."""
    keep = np.isfinite(child0)
    for r in done.values():
        parent = r.curve_relative
        both = keep & np.isfinite(parent)
        keep &= ~(both & (np.abs(np.nan_to_num(child0 - parent, nan=np.inf)) < cfg.exclude_near))
    idx = np.flatnonzero(keep)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if idx.size else []
    runs = [run for run in runs if len(run) >= cfg.min_run]
    n_ex = int(np.isfinite(child0).sum() - sum(len(r) for r in runs))
    if not runs:
        raise NothingToRefine(
            f"no run of {cfg.min_run} channels stays {cfg.exclude_near} samples away from the "
            "curves refined before this one"
        )
    if n_ex:
        log.info(
            "exclude_near %d: %d channels not refined, %d runs", cfg.exclude_near, n_ex, len(runs)
        )
    parts = []
    for run in runs:
        c = np.full(child0.shape, np.nan)
        c[run] = child0[run]
        parts.append((run, refine_curve(waveform, c, cfg, mask=mask, use_numba=use_numba)))
    longest = max(parts, key=lambda p: len(p[0]))[1]
    out = RefineResult(
        curve=np.full(child0.shape, np.nan),
        shifts=np.full(child0.shape, np.nan),
        aligned=longest.aligned,
        stack=longest.stack,
        coherence=np.full(child0.shape, np.nan),
        polarity=np.zeros(child0.shape),
        snr=np.full(child0.shape, np.nan),
        kept=np.zeros(child0.shape, bool),
        anchor_offset=longest.anchor_offset,
        taus=longest.taus,
        qc=longest.qc,
        channel_range=(int(runs[0][0]), int(runs[-1][-1]) + 1),
        runs=[(int(r[0]), int(r[-1])) for r in runs],
        anchor_offsets=[p[1].anchor_offset for p in parts],
    )
    for run, r in parts:
        for name in ("curve", "shifts", "coherence", "polarity", "snr", "kept"):
            getattr(out, name)[run] = getattr(r, name)[run]
    refined = np.isfinite(out.curve)
    _bridge(out, child0, [p[0] for p in parts], cfg.bridge)
    out.refined = refined
    return out


def _bridge(
    out: RefineResult,
    child0: np.ndarray,
    runs: list[np.ndarray],
    mode: str = "shift",
    edge: int = 10,
    taper: int = 50,
) -> None:
    """Fill the channels of ``child0`` that were not refined (see RefineConfig.bridge)."""
    if mode not in ("shift", "initial"):
        raise ValueError(f"bridge must be 'shift' or 'initial', got {mode!r}")
    shift = out.curve - child0
    n = len(child0)
    ends = [
        (
            int(r[0]),
            int(r[-1]),
            float(np.nanmedian(shift[r[:edge]])),
            float(np.nanmedian(shift[r[-edge:]])),
        )
        for r in runs
    ]
    fill_shift = np.zeros(n)
    if mode == "shift":
        for k, (a, b, s_in, s_out) in enumerate(ends):
            if k == 0:
                fill_shift[:a] = s_in
            if k == len(ends) - 1:
                fill_shift[b + 1 :] = s_out
            else:
                a2, s2 = ends[k + 1][0], ends[k + 1][2]
                gap = np.arange(b + 1, a2)
                fill_shift[gap] = s_out + (s2 - s_out) * (gap - b) / (a2 - b)
    else:
        for a, b, s_in, s_out in ends:
            before = np.arange(max(0, a - taper), a)
            fill_shift[before] += s_in * (1 - (a - before) / taper)
            after = np.arange(b + 1, min(n, b + 1 + taper))
            fill_shift[after] += s_out * (1 - (after - b) / taper)
    fill = np.isfinite(child0) & ~np.isfinite(out.curve)
    out.curve[fill] = child0[fill] + fill_shift[fill]
    out.shifts[fill] = fill_shift[fill]


def _closest_parent(
    child0: np.ndarray, done: dict[str, RefineResult], mask_half: int
) -> str | None:
    """Key of the refined curve the child's initial curve comes closest to (within mask_half)."""
    best = None
    for key, r in done.items():
        parent = r.curve_relative
        both = np.isfinite(child0) & np.isfinite(parent)
        if not both.any():
            continue
        d = float(np.min(np.abs(child0[both] - parent[both])))
        if d <= mask_half and (best is None or d < best[0]):
            best = (d, key)
    return None if best is None else best[1]
