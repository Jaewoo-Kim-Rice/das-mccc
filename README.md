# das-mccc

Refine a DAS arrival curve with an iterative network multi-channel cross-correlation
(MCCC), anchor its absolute level on the aligned stack, and return per-channel polarity
and quality measures. Arrays in, arrays out: no site, file or path concepts.

The core is the refiner of [das-focmec](https://github.com/Jaewoo-Kim-Rice/das-focmec)
(`das_focmec.processing.mccc_core.ultra_mccc_iterative`), extracted with its history so it
can be used by any picker. Started from a rough curve (a VLM trace, a bracket, a
theoretical moveout) it recovers the shape of the arrival to the precision of a human
curve: on 24 CAPE 2025 reads the shape MAD against human picks went from 2.2 to 1.95 ms
(P) and 4.5 to 4.15 ms (S), and a curve started on the wrong lobe went from 5.6 to 2.65 ms.

## Install

```
pip install -e .            # numpy, scipy
pip install -e '.[numba]'   # fast pairwise correlation (strongly recommended)
pip install -e '.[dev]'     # pytest, ruff
```

Without numba the pairwise correlation runs in pure numpy: identical results, one to two
orders of magnitude slower, and a warning is logged at import.

## Use

```python
import numpy as np
from dasmccc import refine_curve, refine_phases, DIRECT, SECONDARY

# waveform: (n_channels, n_samples) float, filtered as you like
# curve   : (n_channels,) arrival in samples, NaN where the phase is not picked
res = refine_curve(waveform, curve, DIRECT)

res.curve          # refined arrival (samples), NaN where the input was NaN
res.curve_relative # same shape, at the initial curve's level (no anchor)
res.anchor_offset  # samples added by the anchor rule (NaN if the rule refused)
res.polarity       # -1 / 0 / +1 per channel
res.snr, res.coherence, res.kept
res.aligned, res.stack   # the aligned window and its stack, for plots

# several phases of one gather, strongest first; refined phases are masked for the next, and
# a direct curve is not refined where it runs within 170 samples of one (res.runs, res.refined)
out = refine_phases(waveform, {"S": s_curve, "P": p_curve, "SP": sp_curve})
# several curves of one tag: any keys, plus a key -> tag map
out = refine_phases(waveform, {"S": s_curve, "R1": r1, "R2": r2},
                    tags={"S": "S", "R1": "REFL", "R2": "REFL"})
```

Settings live in `RefineConfig` (everything in samples and channels). `DIRECT` is the
das-focmec configuration for direct waves (window 200, corr_len 200, smoothness 50, four
passes, pre-mask 100); `SECONDARY` narrows it for conversions and reflections (window
120, corr_len 100, three passes, pre-mask 50). Both were calibrated at 1 kHz and 2 m channel
spacing; `docs/algorithm.md` gives the conversion to other rates and spacings, what each
knob does, and the anchoring and masking rules.

## For das-focmec

`dasmccc.legacy` exposes `ultra_mccc_iterative` and `diff_corr_ric` with the das-focmec
signatures and identical results, so `das_focmec.processing.workflows` only changes its
import line.

## What it does not do

* It does not re-pick. The initial curve decides which arrival and roughly which lobe is
  refined; MCCC measures relative delays within `pair_slope` samples per channel of it.
* The anchor moves the whole curve by one offset measured on the stack (secondary phases
  inherit their parent's). The first-lobe rule sits about 5 ms before the human pick for P
  and within a few ms of it for S on the CAPE 2025 fibres, with a per-fibre constant;
  calibrate it per site against a few human picks when onsets are needed.
* Sub-sample precision: the alignment is integer; tau is a float but the returned curve
  inherits the integer initial alignment plus the smoothed tau.

## Development

```
PYTHONPATH=src pytest -q
ruff check src tests && ruff format --check src tests
```
