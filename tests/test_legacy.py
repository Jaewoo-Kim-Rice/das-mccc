import warnings

import numpy as np
import pytest
from conftest import N_CH, initial_curve

from dasmccc import RefineConfig, refine_curve, ricker_windows
from dasmccc.legacy import diff_corr_ric, ultra_mccc_iterative


def test_legacy_matches_refine_curve_without_anchor(gather):
    curve = initial_curve()
    pick = np.column_stack([np.arange(N_CH), np.round(curve)]).astype(float)
    with pytest.warns(DeprecationWarning):
        arrs, total_shift, (first_shifts, base_time), taus = ultra_mccc_iterative(
            gather,
            pick,
            corr_len=200,
            max_shift=10,
            lamb=1,
            n_iterations=4,
            shrinked_window_length=200,
            smoothness=50.0,
            pre_mccc_mask_half_width=100,
        )
    res = refine_curve(gather, curve, RefineConfig(anchor=None))
    assert len(arrs) == 6 and arrs[0] is gather
    assert np.array_equal(arrs[-1], res.aligned)
    assert np.allclose(base_time - total_shift, res.curve)
    assert taus[0].shape == (N_CH, 2)
    assert np.array_equal(first_shifts, np.round(base_time - curve).astype(int))


def test_legacy_max_shift_none_is_silent(gather):
    pick = np.column_stack([np.arange(N_CH), np.round(initial_curve())]).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ultra_mccc_iterative(gather, pick, 100, None, 1, n_iterations=1, shrinked_window_length=100)


def test_ricker_windows_shapes_and_rules(gather):
    res = refine_curve(gather, initial_curve(), RefineConfig(anchor=None))
    r = ricker_windows(res.aligned, max_lag=37, snr_thresh=1.5, mmad_thresh=3.5, ricker_hz=50)
    assert r.window.shape == (N_CH, 2) and r.dt.shape == (N_CH,)
    good = np.isfinite(r.polarity)
    assert good.mean() > 0.9
    assert set(np.unique(r.polarity[good])) <= {-1.0, 1.0}
    # the match sits near the alignment sample (window centre 100, wavelet peak 10 earlier)
    assert abs(np.nanmedian(r.dt) - 90) < 3
    dts, pol, amp, snr, wins = diff_corr_ric(res.aligned, 37, snr_thresh=1.5, mmad_thresh=3.5)
    assert np.array_equal(wins, r.window) and np.array_equal(dts, r.dt, equal_nan=True)


def test_ricker_windows_keeps_the_legacy_nan_snr_rule():
    # an all-zero block wider than the spatial median kernel reaches the QC untouched:
    # 0 / 0 gives a NaN snr, which the legacy rule does not reject (amp 0.0, sign kept)
    rng = np.random.default_rng(0)
    aligned = rng.standard_normal((60, 200))
    aligned[20:50] = 0.0
    r = ricker_windows(aligned, max_lag=37, snr_thresh=1.5, mmad_thresh=3.5)
    assert np.isnan(r.snr[20:50]).all()
    assert (r.amplitude[20:50] == 0.0).all()
    assert np.isfinite(r.polarity[20:50]).all()
