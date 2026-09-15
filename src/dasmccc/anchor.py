"""Absolute anchoring of a relatively aligned gather.

Network MCCC fixes only relative delays; the level of the refined curve is whatever the
initial curve's level was (the lobe the initial picker traced). The rules here measure one
offset on the aligned stack so that the curve is moved to a reproducible feature of the
wavelet. All offsets are in samples relative to ``centre`` (the alignment sample).
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("dasmccc")


def stack_peak(stack: np.ndarray, centre: int) -> float:
    """Offset of the |stack| maximum. Reproducible on strong reads, but the peak lobe is
    not the same lobe on every read (spread of tens of ms against human onsets)."""
    return float(int(np.argmax(np.abs(stack))) - centre)


def first_lobe(
    stack: np.ndarray, centre: int, min_frac: float = 0.3, guard: int | None = 40
) -> float:
    """Offset of the centre of the first lobe of the stack: the earliest local maximum of
    |stack| before the |stack| peak whose amplitude is at least ``min_frac`` of the peak
    (the peak itself when no such lobe exists).

    Against human onset picks this rule gave the most constant offset of the rules tested
    (P about -4 ms, S about +0.5 ms at 1 kHz on the CAPE 2025 fibres, spread 1 to 3 ms).

    ``guard`` bounds |offset|: when the chosen lobe lies farther than ``guard`` samples from
    the alignment sample the rule is judged unreliable, a warning is logged and NaN is
    returned so the caller keeps the relative level. The guard does not catch the case
    where the stack peak itself sits on a later arrival; only the initial curve's level
    protects against that.
    """
    a = np.abs(np.asarray(stack, float))
    pk = int(np.argmax(a))
    first = pk
    for i in range(1, pk):
        if a[i] >= a[i - 1] and a[i] >= a[i + 1] and a[i] >= min_frac * a[pk]:
            first = i
            break
    offset = float(first - centre)
    if guard is not None and abs(offset) > guard:
        log.warning(
            "first_lobe anchor %+.0f samples exceeds guard %d; anchor not applied", offset, guard
        )
        return float("nan")
    return offset
