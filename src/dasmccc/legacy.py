"""das-focmec compatible entry points.

``das_focmec.processing.workflows.mccc_pipeline`` was written against
``ultra_mccc_iterative`` and ``diff_corr_ric`` from its own ``mccc_core``; these wrappers
keep those signatures and return values exactly (intermediate arrays, legacy (n, 2) pick
and tau layouts) on top of the dasmccc building blocks, so that the focal-mechanism
pipeline reproduces bit for bit while new code uses ``dasmccc.refine_curve``.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from .core import mccc
from .ops import shift_arr, spatial_median, tau_shift
from .polarity import ricker_windows

log = logging.getLogger("dasmccc")


def ultra_mccc_iterative(
    das_arr: np.ndarray,
    pick: np.ndarray,
    corr_len: int,
    max_shift: int | None,
    lamb: float,
    n_iterations: int = 3,
    shrinked_window_length: int = 300,
    medfilt_iterations=(1, 2, 3),
    smoothness: float = 0.0,
    reference_dt: np.ndarray | None = None,
    pre_mccc_mask_half_width: int | None = None,
):
    """Legacy iterative MCCC.

    pick : (n_channels, 2) array [channel index, pick sample] as in das-focmec.
    max_shift : accepted for compatibility and ignored, as it always was: the per-pair lag
        bound is max(0.2 * channel gap, 3) samples.

    Returns ``(intermediates, total_shift, [first_shifts, base_time], tau_list)`` where
    ``intermediates = [das_arr, shifted, after pass 1, ...]``, the refined pick is
    ``base_time - total_shift`` and each tau is an (n_channels, 2) [index, tau] array.
    """
    if max_shift is not None:
        warnings.warn(
            "ultra_mccc_iterative: max_shift has no effect (per-pair bound is "
            "max(0.2 * gap, 3) samples); pass None to silence",
            DeprecationWarning,
            stacklevel=2,
        )
    half = shrinked_window_length // 2
    base_time = das_arr.shape[1] // 2
    pick = np.asarray(pick, float)
    if pick.shape != (das_arr.shape[0], 2):
        raise ValueError(f"pick must be (n_channels, 2), got {pick.shape}")
    log.info("legacy ultra_mccc_iterative: initial shift")
    shifted, first_shifts = shift_arr(das_arr, pick[:, 1], base_time)
    shifted = shifted[:, base_time - half : base_time + half]
    if pre_mccc_mask_half_width is not None:
        centre = shifted.shape[1] // 2
        keep = np.zeros_like(shifted)
        sl = slice(centre - pre_mccc_mask_half_width, centre + pre_mccc_mask_half_width)
        keep[:, sl] = shifted[:, sl]
        shifted = keep
        log.info("  pre-MCCC mask +-%d samples", pre_mccc_mask_half_width)
    current = shifted
    intermediates = [das_arr, shifted]
    tau_list = []
    total_shift = first_shifts.astype(np.float64).copy()
    for i in range(1, n_iterations + 1):
        log.info("  MCCC pass %d", i)
        tau = mccc(
            current, corr_len // i, lamb=lamb, smoothness=smoothness, reference_dt=reference_dt
        )
        tau_list.append(np.column_stack([np.arange(tau.size), tau]))
        current = tau_shift(current, tau)
        if i in medfilt_iterations:
            current = spatial_median(current, 25)
        intermediates.append(current)
        total_shift -= tau
    return intermediates, total_shift, [first_shifts, base_time], tau_list


def diff_corr_ric(shifted_arr, max_shift, snr_thresh=5, mmad_thresh=4.0, ricker_freq=50):
    """Legacy tuple form of :func:`dasmccc.polarity.ricker_windows`:
    ``(dts, polarities, amps, snrs, wins)``."""
    r = ricker_windows(shifted_arr, max_shift, snr_thresh, mmad_thresh, ricker_freq)
    return r.dt, r.polarity, r.amplitude, r.snr, r.window
