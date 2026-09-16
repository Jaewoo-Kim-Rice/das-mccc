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
    stack: np.ndarray,
    centre: int,
    min_frac: float = 0.4,
    guard: int | None = 40,
    contiguous: bool = True,
    window: tuple[int, int] | None = (-30, 10),
) -> float:
    """Offset of the centre of the first lobe of the stack.

    Candidates are the local maxima of |stack|. ``window`` = (lo, hi) restricts the search to
    ``centre + lo .. centre + hi`` (samples): the prior that the initial picker traced a lobe
    of the arrival, so the onset lies at most one wavelet before it and hardly after it. The
    reference amplitude is the |stack| peak inside the window. With ``contiguous`` the rule
    walks back from that peak lobe by lobe while each lobe keeps at least ``min_frac`` of the
    peak and returns the earliest lobe of that run (a precursor separated by a weaker lobe is
    not the onset); without it the earliest candidate above ``min_frac`` anywhere before the
    peak is taken (the original rule). The peak itself is returned when no earlier lobe
    qualifies.

    Tuned on 616 CAPE 2025 reads with human picks (das-phase-agent research record,
    `docs/11_anchor_tuning.md`): window (-30, 10), contiguous, min_frac 0.4 on the
    channel-normalised stack removed every refusal and halved the gross anchor errors
    against the original rule (window None, contiguous False, min_frac 0.3, plain stack).

    ``guard`` bounds |offset|: a larger offset is judged unreliable, a warning is logged and
    NaN is returned so the caller keeps the relative level.
    """
    a = np.abs(np.asarray(stack, float))
    n = len(a)
    if window is None:
        lo, hi = 0, n
    else:
        lo, hi = max(0, centre + int(window[0])), min(n, centre + int(window[1]) + 1)
        if hi - lo < 3:
            raise ValueError(f"anchor window {window} leaves no samples around centre {centre}")
    pk = lo + int(np.argmax(a[lo:hi]))
    ext = [i for i in range(max(1, lo), pk) if a[i] >= a[i - 1] and a[i] >= a[i + 1]]
    first = pk
    if contiguous:
        for i in reversed(ext):
            if a[i] >= min_frac * a[pk]:
                first = i
            else:
                break
    else:
        ok = [i for i in ext if a[i] >= min_frac * a[pk]]
        if ok:
            first = ok[0]
    offset = float(first - centre)
    if guard is not None and abs(offset) > guard:
        log.warning(
            "first_lobe anchor %+.0f samples exceeds guard %d; anchor not applied", offset, guard
        )
        return float("nan")
    return offset
