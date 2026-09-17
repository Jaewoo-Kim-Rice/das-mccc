# Algorithm and API

## Refining one curve (`refine_curve`)

Input: a gather `waveform (n_channels, n_samples)` and an initial arrival `curve
(n_channels,)` in samples (NaN where absent). Everything below is in samples and channels.

```
[1] range      channels [first finite, last finite]; interior NaNs interpolated for the
               alignment only (their output stays NaN)
[2] align      each channel shifted by round(base - curve[c]) so its pick sits at the
               centre; window of +-window/2 cut around it; optional pre_mask keeps only
               +-pre_mask samples (100 of the 200-sample window by default: nothing
               beyond 100 ms of the initial curve is ever correlated)
[3] passes     for i = 1 .. n_iter:
      (a) partners : for each channel, n_partners (50) candidates drawn as quantiles of a
                     normal distribution (std partner_std = 20 channels) truncated to
                     +-corr_len // i channels; only pairs j > i are kept
      (b) lags     : for each pair the lag maximising |cross-correlation| within
                     +-max(pair_slope * gap, pair_min_shift) samples (0.2 samples per
                     channel of gap, at least 3). Sign-agnostic: a polarity reversal
                     between channels does not break the alignment.
      (c) solve    : sparse least squares
                     min ||lamb (tau_i - tau_j - lag_ij)||^2 + ||smoothness (tau_{c+1} - tau_c)||^2
                     then a moving average of tau_avg (100) channels
      (d) apply    : each channel advanced by round(tau); after the passes listed in
                     medfilt_iters (1, 2, 3) the gather is median filtered along the fibre
                     with medfilt_channels (25)
[4] relative   curve_relative = round(curve) + sum of tau (the level of the initial curve
               is preserved: MCCC measures only relative delays)
[5] QC         Ricker polarity / SNR / MMAD on the aligned gather (below)
[6] stack      channel-normalised mean of the aligned traces (`stack`: "norm", "mean" or
               "median"; stack_polarity=True multiplies each trace by its polarity first)
[7] anchor     one offset measured on the stack, added to every channel (below)
[8] coherence  |normalised correlation| of each aligned trace with the stack within
               +-coherence_half (30) samples of the centre
```

What the knobs mean physically:

* `pair_slope` bounds the slope of the *residual* moveout with respect to the initial
  curve, not the moveout itself. 0.2 samples/channel at 1 kHz and 2 m channel spacing is
  0.1 ms/m; the initial curve must already be within that of the true slope over the
  correlation neighbourhood. `pair_min_shift` lets adjacent channels disagree by 3
  samples regardless.
* `corr_len // i` shrinks the partner neighbourhood pass by pass (200, 100, 66, 50
  channels). Because the partner distribution has a fixed std of 20 channels, this
  changes only the far tail of the partners; the time window never shrinks.
* `smoothness` 50 with `lamb` 1 hardly changes the result on the reference reads (the
  distance-scaled lag bound and the spatial median filter do the work); it matters when
  a stretch of channels has no coherent signal.
* The last pass is not followed by the median filter (default `medfilt_iters` (1, 2, 3)
  with `n_iter` 4), so `aligned` and the QC see the real traces after alignment. Note
  that the earlier median filters have already replaced isolated bad channels by their
  neighbours; an amplitude outlier narrower than `medfilt_channels` is not visible to the
  MMAD test.

## Settings in physical units

Every knob is in samples and channels; `DIRECT` and `SECONDARY` were calibrated at 1 kHz on
fibres with 2 m channel spacing (das-focmec ev_1090, CAPE 2025 reads) and have not been
validated at other rates or spacings. A caller with a different `fs` (Hz) and channel
spacing `dx` (m) converts as follows and should log the resulting configuration:

| knob | DIRECT | physical meaning at 1 kHz, 2 m | conversion |
|---|---|---|---|
| `window`, `pre_mask` | 200, 100 | 200 ms, 100 ms | ms x fs / 1000 |
| `corr_len` | 200 ch | 400 m along the fibre | m / dx |
| `partner_std` | 20 ch | 40 m | m / dx |
| `pair_slope` | 0.2 sample/ch | 0.1 ms/m residual slope | ms/m x dx x fs / 1000 |
| `pair_min_shift` | 3 samples | 3 ms | ms x fs / 1000 |
| `tau_avg` | 100 ch | 200 m | m / dx |
| `medfilt_channels` | 25 ch | 50 m | m / dx, rounded to odd |
| `anchor_guard`, `coherence_half` | 40, 30 samples | 40 ms, 30 ms | ms x fs / 1000 |
| `anchor_window` | (-30, 10) samples | (-30, 10) ms | ms x fs / 1000 |
| `exclude_near` | 170 samples (None for SECONDARY) | 170 ms | ms x fs / 1000 |
| `polarity.half_win`, `max_lag` | 30, 10 samples | 30 ms, 10 ms | ms x fs / 1000 |
| `polarity.fs`, `ricker_hz` | 1000, 50 | 50 Hz Ricker | set `fs`; keep `ricker_hz` |
| `smoothness`, `lamb`, `n_iter`, `n_partners` | 50, 1, 4, 50 | dimensionless | unchanged |

`smoothness` weights `(tau[c+1] - tau[c])^2` in samples^2 per channel, so strictly it scales
with `(fs / 1000)^2 / dx`; at the tested settings it hardly changes the result (see above)
and is left as is.

## Anchoring (`anchor.py`)

Network MCCC returns relative delays only, so the refined curve sits where the initial
curve was, i.e. on whatever lobe the initial picker traced. `first_lobe(stack, centre)`
measures one offset on the aligned stack so that the curve is moved to a reproducible
feature of the wavelet:

```
[1] stack     channel-normalised mean of the aligned traces (`stack="norm"`; "mean" and
              "median" available)
[2] window    only centre + (-30, 10) samples are searched (`anchor_window`): the initial
              picker traced a lobe of the arrival, so the onset lies at most one wavelet
              before it and hardly after it
[3] peak      the |stack| maximum inside the window is the reference amplitude
[4] walk      from that lobe backwards, lobe by lobe, while each local maximum of |stack|
              keeps >= anchor_min_frac (0.4) of the reference (`anchor_contiguous`); the
              earliest lobe of that run is the first lobe. A precursor separated from the
              wavelet by a weaker lobe is not the onset.
[5] guard     |offset| > anchor_guard (40) -> warning, NaN, relative level kept
```

The rule was tuned on 616 CAPE 2025 reads with human P and S picks (das-phase-agent
research record, `docs/11_anchor_tuning.md`). Bias = refined curve minus human pick per
read; what matters is its spread across reads, not its value:

| rule (on reads whose shape matches the human curve within 5 ms) | P: MAD / gross > 15 ms / refused (363) | S: MAD / gross / refused (358) |
|---|---|---|
| relative level only (`anchor=None`) | 4.4 / 39 / 0 | 7.0 / 65 / 0 |
| original: plain stack, no window, earliest 30 % lobe | 1.6 / 24 / 16 | 5.4 / 48 / 15 |
| default: normalised stack, window (-30, 10), contiguous 40 % | 1.6 / 10 / 0 | 4.8 / 40 / 0 |

Per-fibre offsets differ (S: Gold -4.6, 16B +2.8, Delano +2.7 ms; P about -5 ms on all
three); with per-fibre constants removed the S spread is 4.0 ms MAD. The offset between the
first-lobe centre and a human onset is a convention: calibrate it per site and fibre against
a few human picks when onsets are needed. Onset-style rules (zero crossing or envelope
threshold before the first lobe, AIC) were not better on these reads.

`stack_polarity=True` multiplies each trace by its Ricker sign before stacking; it fixes the
synthetic case of a reversal along the fibre but was worse on the real reads (at S-wave SNRs
the sign estimate is noisy). Off by default. `stack_peak` is provided for comparison.

`anchor="parent"` (the `SECONDARY` default) is resolved in `refine_phases`: the curve takes
the anchor offset of the refined curve its initial curve comes closest to (within
`mask_half`), and `RefineResult.parent` names it. Secondary phases leave their parent at a
junction, so the same convention must apply to both; an independent first-lobe anchor on
their weak stacks pulled the junction apart by 15 ms median on the CAPE reads, inheriting
keeps it at the relative-alignment value (8 ms).

## Polarity and QC (`polarity.py`)

On each aligned trace a Ricker template (`ricker_hz` 50 at `fs` 1000, 2 * `half_win` = 60
samples) is correlated within +-`max_lag` (10) samples of the centre; the polarity is the
sign of the correlation at its absolute maximum. SNR is rms(centre +- half_win) over
rms(rest of the window); polarity is 0 below `snr_thresh` (3). The window rms of all
channels goes through a modified z-score (MMAD); channels above `mmad_thresh` (4) are
amplitude outliers (near-field or bad channels), get polarity 0 and `kept` False.

`kept` = input finite and not an outlier and SNR above threshold. The refined `curve`
itself is never blanked by the QC; apply `kept` as you see fit.

## Several phases (`refine_phases`)

`curves` maps a key to an initial curve; the key is the phase tag unless `tags` maps each
key to its tag, which is how several curves of one tag (two reflections, an SP per
interface) are refined in one call. Curves are refined by the position of their tag in
`order` (default S, P, SP, REFL), the strongest first, and within a tag in input order.
Before a curve is refined every already refined curve is hidden: +-`mask_half` (45) samples around
its `curve_relative` are zeroed, except within `guard_channels` (50) of the junction
channel where the new phase's initial curve comes within `mask_half` of the parent.
Blanking the junction would let the child drift there (measured: junction |dt| 0 to
11 ms without the guard). Direct phases (P, S) use `DIRECT`, others `SECONDARY` unless
`cfg_by_tag` says otherwise. Re-imposing the junction as a constraint on the child is not
implemented.

**Exclusion of interfering channels (`exclude_near`).** A direct curve (`DIRECT`,
`exclude_near` 170 samples) is not refined where its initial curve comes within that many
samples of a curve refined before it: with the S inside the P correlation window (+-100)
the pairwise correlations lock on the S and the stack is S energy, and no mask width fixes
it (tested 45 to 75 samples, relative or anchored centre). The runs of at least `min_run`
(60) channels that stay clear are refined separately, each with its own anchor. The
excluded channels are not dropped: they follow the initial curve as drawn, joined to the
refined runs by tapering the run-end shift to zero over 50 channels, so the curve stays
continuous and `RefineResult.refined` marks what was measured. On 423 CAPE 2025 P reads
(exclusion at 100 / 120 samples) the bias spread went from 1.84 to 1.63 / 1.70 ms MAD on
the refined channels and the reads with a shape error above 5 ms from 59 to 41 / 39 (48 /
44 when the bridged channels are scored too: they keep whatever the initial picker did
there). More than half of the reads lose no channel; the 90th percentile of excluded
channels is 32 / 45 %. If no run survives, `refine_phases` raises (`on_excluded="raise"`,
default) or logs a warning and leaves the curve out of the result (`on_excluded="skip"`).
Secondary phases (`SECONDARY`, `exclude_near` None) are never excluded: they meet their
parent at the junction by construction.

## Polarity and QC (`polarity.py`)

On each aligned trace a Ricker template (`ricker_hz` 50 at `fs` 1000, 2 * `half_win` = 60
samples) is correlated within +-`max_lag` (10) samples of the centre; the polarity is the
sign of the correlation at its absolute maximum. SNR is rms(centre +- half_win) over
rms(rest of the window); polarity is 0 below `snr_thresh` (3). The window rms of all
channels goes through a modified z-score (MMAD); channels above `mmad_thresh` (4) are
amplitude outliers (near-field or bad channels), get polarity 0 and `kept` False.

`kept` = input finite and not an outlier and SNR above threshold. The refined `curve`
itself is never blanked by the QC; apply `kept` as you see fit.

## Several phases (`refine_phases`)

`curves` maps a key to an initial curve; the key is the phase tag unless `tags` maps each
key to its tag, which is how several curves of one tag (two reflections, an SP per
interface) are refined in one call. Curves are refined by the position of their tag in
`order` (default S, P, SP, REFL), the strongest first, and within a tag in input order.
Before a curve is refined every already refined curve is hidden: +-`mask_half` (45) samples around
its `curve_relative` are zeroed, except within `guard_channels` (50) of the junction
channel where the new phase's initial curve comes within `mask_half` of the parent.
Blanking the junction would let the child drift there (measured: junction |dt| 0 to
11 ms without the guard). Direct phases (P, S) use `DIRECT`, others `SECONDARY` unless
`cfg_by_tag` says otherwise. Re-imposing the junction as a constraint on the child is not
implemented.

**Exclusion of interfering channels (`exclude_near`).** A direct curve (`DIRECT`,
`exclude_near` 100 = the correlation half window) is not refined where its initial curve
comes within that many samples of a curve refined before it: with the S inside the P window
the pairwise correlations lock on the S and the stack is S energy, and no mask width fixes
it (tested 45 to 75 samples, relative or anchored centre). The remaining runs of at least
`min_run` (60) channels are refined separately, each with its own anchor; the assembled
result has NaN on the excluded channels, `RefineResult.runs` lists the runs and
`anchor_offsets` their anchors. On 423 CAPE 2025 P reads this took the bias spread from
1.84 to 1.63 ms MAD and the reads with a shape error above 5 ms from 59 to 41, touching no
channel on more than half of the reads (90th percentile of excluded channels: 32 %). If no
run survives, `refine_phases` raises (`on_excluded="raise"`, default) or logs a warning
and leaves the curve out of the result (`on_excluded="skip"`). Secondary phases (`SECONDARY`, `exclude_near` None) are
never excluded: they meet their parent at the junction by construction.

## das-focmec compatibility (`dasmccc.legacy`)

`legacy.ultra_mccc_iterative` and `legacy.diff_corr_ric` keep the signatures and return
values of the das-focmec originals (intermediate arrays, (n, 2) pick and tau layouts,
`(dts, polarities, amps, snrs, wins)` tuple) on top of the building blocks above, bit for
bit. `legacy.diff_corr_ric` is the tuple form of `polarity.ricker_windows`, the full-length
Ricker match that yields the per-channel masking windows of the focal-mechanism pipeline.
The `max_shift` argument is accepted and ignored, as it always was; pass `None` to avoid
the deprecation warning.

## API

| name | in | out |
|---|---|---|
| `refine_curve(waveform, curve, cfg=DIRECT, mask=None, use_numba=None)` | gather, curve | `RefineResult` |
| `refine_phases(waveform, curves, order, cfg_by_tag, mask_half, guard_channels, tags)` | gather, `{key: curve}` (+ `{key: tag}`) | `{key: RefineResult}` |
| `RefineConfig`, `DIRECT`, `SECONDARY` | | frozen dataclass |
| `RefineResult` | | `curve, shifts, aligned, stack, coherence, polarity, snr, kept, anchor_offset, taus, qc, channel_range, parent, runs, anchor_offsets, refined`, property `curve_relative` |
| `first_lobe(stack, centre, min_frac, guard, contiguous, window)`, `stack_peak(stack, centre)` | 1-D stack | offset (samples) |
| `ricker_polarity(aligned, PolarityConfig)` | aligned gather | `PolarityResult(polarity, snr, amplitude, lag, outlier)` |
| `partner_pairs`, `pairwise_lags`, `solve_tau`, `mccc`, `iterate_align` | | the building blocks of one pass and of the iteration |

Sign conventions: `tau > 0` means the channel arrives later than its partners (the
aligned trace is advanced by tau); the refined arrival is `round(curve) + sum(tau)`.
`limited_cc(tr, other, m)[k]` is the correlation at lag `k - m`, positive when `tr` is the
delayed one.
