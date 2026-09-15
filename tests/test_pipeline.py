import numpy as np
import pytest
from conftest import N_CH, RICKER_TROUGH, TRUE, initial_curve, make_gather

from dasmccc import (
    DIRECT,
    SECONDARY,
    RefineConfig,
    first_lobe,
    refine_curve,
    refine_phases,
    stack_peak,
)


def shape_mad(curve):
    r = curve - TRUE
    r = r[np.isfinite(r)]
    return np.median(np.abs(r - np.median(r))), np.median(r)


def test_refine_recovers_the_shape_and_keeps_the_level_without_anchor(gather):
    curve = initial_curve()
    res = refine_curve(gather, curve, RefineConfig(anchor=None))
    mad, bias = shape_mad(res.curve)
    mad0, _ = shape_mad(curve)
    assert mad0 > 2.0 and mad < 0.6  # the smooth error is removed
    assert abs(bias - 10.0) < 1.0  # relative alignment keeps the initial curve's level
    assert res.anchor_offset == 0.0
    assert np.isfinite(res.curve).all()


def test_first_lobe_anchor_lands_on_the_leading_trough(gather):
    res = refine_curve(gather, initial_curve(), DIRECT)
    _, bias = shape_mad(res.curve)
    assert abs(bias + RICKER_TROUGH) < 1.5  # trough is 7.8 samples before the peak
    assert np.isclose(res.curve_relative[10] - res.curve[10], -res.anchor_offset)
    assert np.nanmedian(res.coherence) > 0.9


def test_polarity_corrected_stack_survives_a_reversal_along_the_fibre():
    x = make_gather(flip_from=150)
    plain = refine_curve(x, initial_curve(), DIRECT)
    assert np.nanmedian(plain.coherence) < 0.5  # the plain stack cancels
    corrected = refine_curve(x, initial_curve(), RefineConfig(stack_polarity=True))
    assert np.nanmedian(corrected.coherence) > 0.9
    _, bias = shape_mad(corrected.curve)
    assert abs(bias + RICKER_TROUGH) < 1.5


def test_polarity_follows_the_flip_and_outliers_are_dropped():
    # the outlier block must be wider than the 25-channel spatial median filter to survive it
    x = make_gather(flip_from=150, outliers=range(40, 70))
    res = refine_curve(x, initial_curve(), DIRECT)
    assert (
        res.polarity[:30] == 1
    ).all()  # channels next to the block are smeared by the median filter
    assert (res.polarity[160:] == -1).mean() > 0.97
    assert not res.kept[45:65].any()
    assert (res.polarity[45:65] == 0).all()
    assert res.kept[80:].mean() > 0.95


def test_nan_channels_stay_nan_and_the_range_is_respected(gather):
    curve = initial_curve()
    curve[:20] = np.nan
    curve[100:105] = np.nan
    curve[-30:] = np.nan
    res = refine_curve(gather, curve, DIRECT)
    assert np.isnan(res.curve[:20]).all() and np.isnan(res.curve[-30:]).all()
    assert np.isnan(res.curve[100:105]).all()
    assert np.isfinite(res.curve[20:100]).all()
    assert res.channel_range == (20, N_CH - 30)
    assert np.isnan(res.aligned[:20]).all() and np.isfinite(res.aligned[20]).all()
    assert not res.kept[100:105].any()


def test_input_validation(gather):
    with pytest.raises(ValueError):
        refine_curve(gather, np.full(N_CH, np.nan))
    with pytest.raises(ValueError):
        refine_curve(gather, initial_curve()[:-1])
    with pytest.raises(ValueError):
        refine_curve(gather, initial_curve(), RefineConfig(window=5000))
    with pytest.raises(ValueError):
        refine_curve(gather, initial_curve(), RefineConfig(anchor="bogus"))


def test_first_lobe_rule_and_guard():
    t = np.arange(200)
    stack = np.exp(-((t - 120) ** 2) / 20.0) - 0.5 * np.exp(-((t - 100) ** 2) / 20.0)
    assert first_lobe(stack, 100) == 0.0  # lobe at 100 is 50 % of the peak at 120
    assert first_lobe(stack, 100, min_frac=0.6) == 20.0  # too small: peak itself
    assert stack_peak(stack, 100) == 20.0
    assert first_lobe(stack, 80) == 20.0
    assert np.isnan(first_lobe(stack, 80, guard=10))  # |20| > 10 -> refused


def test_numpy_fallback_path_gives_the_same_curve(gather):
    curve = initial_curve()
    a = refine_curve(gather, curve, DIRECT, use_numba=None)
    b = refine_curve(gather, curve, DIRECT, use_numba=False)
    assert np.allclose(a.curve, b.curve, equal_nan=True)


def test_refine_phases_masks_the_parent_outside_the_junction_guard(gather):
    s0 = initial_curve()
    sp0 = TRUE + 80.0 - 0.6 * np.arange(N_CH)  # crosses S near channel 133
    sp0[:60] = np.nan
    with pytest.raises(ValueError):
        refine_phases(gather, {"S": s0, "X": sp0})
    out = refine_phases(gather, {"S": s0, "SP": sp0}, cfg_by_tag={"SP": SECONDARY})
    assert list(out) == ["S", "SP"]
    # S refined as usual; SP sees the gather with S masked except around the junction
    assert shape_mad(out["S"].curve)[0] < 0.6
    assert out["SP"].channel_range == (60, N_CH)
