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
[6] stack      plain mean of the aligned traces (stack_polarity=True multiplies each trace
               by its polarity first; see the anchoring section before using it)
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
| `polarity.half_win`, `max_lag` | 30, 10 samples | 30 ms, 10 ms | ms x fs / 1000 |
| `polarity.fs`, `ricker_hz` | 1000, 50 | 50 Hz Ricker | set `fs`; keep `ricker_hz` |
| `smoothness`, `lamb`, `n_iter`, `n_partners` | 50, 1, 4, 50 | dimensionless | unchanged |

`smoothness` weights `(tau[c+1] - tau[c])^2` in samples^2 per channel, so strictly it scales
with `(fs / 1000)^2 / dx`; at the tested settings it hardly changes the result (see above)
and is left as is.

## Anchoring (`anchor.py`)

Network MCCC returns relative delays only, so the refined curve sits where the initial
curve was, i.e. on whatever lobe the initial picker traced. `first_lobe(stack, centre)`
takes the earliest local maximum of |stack| before the |stack| peak with at least
`min_frac` (0.3) of the peak amplitude and returns its offset from the alignment sample.
Against human onset picks on the CAPE 2025 reads this offset was the most constant of the
rules tested:

| rule | P bias median / spread (ms) | S bias median / spread (ms) |
|---|---|---|
| none (initial level) | +10 / 3.3 | +10 / 7.6 |
| stack peak | +11 / 2.4 | +15 / 10.0 |
| first lobe | -4 / 1.3 | +0.5 / 3.4 |

The stack is the plain mean of the aligned traces. A polarity-corrected stack (each trace
multiplied by its Ricker sign, `stack_polarity=True`) fixes the synthetic case of a
reversal along the fibre but was worse on the real reads (P -3.6 / 1.5, S -2.5 / 4.6, the
off case -9 instead of -2 ms): at S-wave SNRs the sign estimate is noisy and the flips
damage the stack more than the reversals do. It stays an option, off by default.

`anchor_guard` (40 samples) refuses an offset larger than that: a warning is logged,
`anchor_offset` is NaN and the curve keeps the relative level. The guard cannot catch a
stack whose peak already sits on a later arrival; the initial curve's level is the only
protection there. `stack_peak` is provided for comparison and `anchor=None` keeps the
initial level.

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

## API

| name | in | out |
|---|---|---|
| `refine_curve(waveform, curve, cfg=DIRECT, mask=None, use_numba=None)` | gather, curve | `RefineResult` |
| `refine_phases(waveform, curves, order, cfg_by_tag, mask_half, guard_channels, tags)` | gather, `{key: curve}` (+ `{key: tag}`) | `{key: RefineResult}` |
| `RefineConfig`, `DIRECT`, `SECONDARY` | | frozen dataclass |
| `RefineResult` | | `curve, shifts, aligned, stack, coherence, polarity, snr, kept, anchor_offset, taus, qc, channel_range`, property `curve_relative` |
| `first_lobe(stack, centre, min_frac, guard)`, `stack_peak(stack, centre)` | 1-D stack | offset (samples) |
| `ricker_polarity(aligned, PolarityConfig)` | aligned gather | `PolarityResult(polarity, snr, amplitude, lag, outlier)` |
| `partner_pairs`, `pairwise_lags`, `solve_tau`, `mccc`, `iterate_align` | | the building blocks of one pass and of the iteration |

Sign conventions: `tau > 0` means the channel arrives later than its partners (the
aligned trace is advanced by tau); the refined arrival is `round(curve) + sum(tau)`.
`limited_cc(tr, other, m)[k]` is the correlation at lag `k - m`, positive when `tr` is the
delayed one.
