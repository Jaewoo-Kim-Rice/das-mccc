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
    legacy = dict(contiguous=False, window=None)
    assert first_lobe(stack, 100, 0.3, 40, **legacy) == 0.0  # lobe at 100 is 50 % of the peak
    assert first_lobe(stack, 100, 0.6, 40, **legacy) == 20.0  # too small: peak itself
    assert stack_peak(stack, 100) == 20.0
    assert first_lobe(stack, 80, 0.3, 40, **legacy) == 20.0
    assert np.isnan(first_lobe(stack, 80, 0.3, 10, **legacy))  # |20| > 10 -> refused
    # default window (-30, 10): from centre 100 the peak at 120 is outside, the lobe at 100 is
    # the in-window peak; from centre 115 the peak is inside and the 50 % lobe qualifies
    assert first_lobe(stack, 100) == 0.0
    assert first_lobe(stack, 115) == -15.0
    assert first_lobe(stack, 115, min_frac=0.6) == 5.0
    # contiguity: a 50 % lobe at 60 behind a 20 % lobe at 80 is not the onset
    stack2 = stack - 0.5 * np.exp(-((t - 60) ** 2) / 20.0) + 0.2 * np.exp(-((t - 80) ** 2) / 20.0)
    assert first_lobe(stack2, 115, window=(-60, 10), guard=None) == -15.0
    assert first_lobe(stack2, 115, window=(-60, 10), guard=None, contiguous=False) == -55.0
    with pytest.raises(ValueError):
        first_lobe(stack, 199, window=(0, 1))


def test_stack_kinds_and_short_curves(gather):
    from dasmccc.pipeline import _stack

    curve = initial_curve()
    for kind in ("mean", "median", "norm"):
        res = refine_curve(gather, curve, RefineConfig(stack=kind))
        assert abs(shape_mad(res.curve)[1] + RICKER_TROUGH) < 1.5
    with pytest.raises(ValueError):
        _stack(np.ones((3, 5)), RefineConfig(stack="bogus"), np.ones(3))
    # 71 channels < tau_avg 100: averaged over all channels instead of failing
    short = curve.copy()
    short[71:] = np.nan
    res = refine_curve(gather, short, DIRECT)
    assert res.channel_range == (0, 71) and np.isfinite(res.curve[:71]).all()


def test_refine_phases_parent_anchor_preserves_the_junction(gather):
    s0 = initial_curve()
    sp0 = TRUE + 10.0 + 0.6 * (150 - np.arange(N_CH))  # leaves S at channel 150, earlier above
    sp0[150:] = np.nan
    out = refine_phases(gather, {"S": s0, "SP": sp0})
    assert out["SP"].parent == "S"
    assert out["SP"].anchor_offset == out["S"].anchor_offset != 0.0
    d_rel = out["SP"].curve_relative[149] - out["S"].curve_relative[149]
    d_abs = out["SP"].curve[149] - out["S"].curve[149]
    assert abs(d_abs - d_rel) < 1e-9
    # no parent within mask_half: level kept, offset NaN, warning
    far = sp0 - 200.0
    alone = refine_phases(gather, {"S": s0, "SP": far})
    assert np.isnan(alone["SP"].anchor_offset) and alone["SP"].parent is None


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


def test_refine_phases_several_curves_per_tag_via_tags(gather):
    s0 = initial_curve()
    r1 = TRUE + 80.0 - 0.6 * np.arange(N_CH)
    r1[:60] = np.nan
    r2 = TRUE + 60.0 + 0.4 * np.arange(N_CH)
    r2[-50:] = np.nan
    curves = {"refl_b": r2, "direct": s0, "refl_a": r1}
    with pytest.raises(ValueError):
        refine_phases(gather, curves)  # keys are not tags
    with pytest.raises(ValueError):
        refine_phases(gather, curves, tags={"direct": "S", "refl_a": "REFL"})  # refl_b missing
    tags = {"direct": "S", "refl_a": "REFL", "refl_b": "REFL"}
    out = refine_phases(gather, curves, tags=tags)
    assert list(out) == ["direct", "refl_b", "refl_a"]  # S first, then REFL in input order
    assert shape_mad(out["direct"].curve)[0] < 0.6
    assert out["refl_a"].channel_range == (60, N_CH)
    assert out["refl_b"].channel_range == (0, N_CH - 50)


def test_refine_phases_excludes_channels_too_close_to_a_refined_curve(gather):
    s0 = initial_curve()
    # a "P" 60 samples before S over channels 100-199, 250 samples before elsewhere: the middle
    # is inside exclude_near (170) and must not be refined; the two outer runs are
    p0 = TRUE - 250.0
    p0[100:200] = TRUE[100:200] - 60.0
    out = refine_phases(gather, {"S": s0, "P": p0})
    p = out["P"]
    assert p.runs == [(0, 99), (200, N_CH - 1)]
    assert not p.refined[100:200].any() and p.refined[:100].all() and p.refined[200:].all()
    assert np.isnan(p.coherence[100:200]).all() and (p.polarity[100:200] == 0).all()
    assert np.isfinite(p.curve).all()  # bridged: continuous
    # the bridge keeps the initial shape and joins the runs: shift interpolated across the gap
    sh = p.curve - p0
    assert (
        abs(sh[100] - np.median(sh[90:100])) < 1.5 and abs(sh[199] - np.median(sh[200:210])) < 1.5
    )
    # default bridge "shift": the gap carries the interpolated run-end shift
    assert abs(sh[150] - 0.5 * (np.median(sh[90:100]) + np.median(sh[200:210]))) < 1.5
    out2 = refine_phases(
        gather, {"S": s0, "P": p0}, cfg_by_tag={"P": RefineConfig(bridge="initial")}
    )
    assert abs((out2["P"].curve - p0)[150]) < 1e-9  # mid-gap, beyond the tapers: the curve as drawn
    with pytest.raises(ValueError):
        refine_phases(gather, {"S": s0, "P": p0}, cfg_by_tag={"P": RefineConfig(bridge="bogus")})
    assert len(p.anchor_offsets) == 2 and p.anchor_offset in p.anchor_offsets
    # secondaries are never excluded (they meet their parent by construction)
    sp0 = TRUE + 10.0 + 0.6 * (150 - np.arange(N_CH))
    sp0[150:] = np.nan
    assert refine_phases(gather, {"S": s0, "SP": sp0})["SP"].runs is None
    with pytest.raises(ValueError):
        refine_phases(gather, {"S": s0, "P": TRUE - 30.0})  # everything too close
    out = refine_phases(gather, {"S": s0, "P": TRUE - 30.0}, on_excluded="skip")
    assert list(out) == ["S"]
