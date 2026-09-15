import numpy as np
import pytest

from dasmccc import NUMBA_AVAILABLE, mccc, pairwise_lags, partner_pairs, solve_tau
from dasmccc.ops import moving_avg, shift_arr, tau_shift
from dasmccc.signal import limited_cc, normal_distribution


def test_partner_pairs_are_forward_and_within_reach():
    pi, pj = partner_pairs(120, 60)
    assert np.all(pj > pi)
    assert pj.max() < 120
    # every channel but the last has partners ahead of it
    assert set(pi) == set(range(119))


def test_normal_distribution_strictly_increasing_and_spans_range():
    p = normal_distribution(-200, 200, 50)
    assert np.all(np.diff(p) > 0)
    assert p[0] == -200 and p[-1] >= 200


def test_limited_cc_recovers_a_known_lag():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(500)
    b = np.roll(a, 4)  # b is a delayed by 4 samples
    corr = limited_cc(a, b, 10, use_numba=False)
    assert np.argmax(corr) - 10 == -4
    if NUMBA_AVAILABLE:
        assert np.allclose(corr, limited_cc(a, b, 10, use_numba=True))


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
def test_numpy_fallback_matches_numba(gather):
    sub = gather[:80, 350:650]
    pi, pj = partner_pairs(80, 40)
    assert np.array_equal(
        pairwise_lags(sub, pi, pj, use_numba=True), pairwise_lags(sub, pi, pj, use_numba=False)
    )


def test_use_numba_true_without_numba_raises(monkeypatch):
    import dasmccc.core as core

    monkeypatch.setattr(core, "NUMBA_AVAILABLE", False)
    with pytest.raises(RuntimeError):
        pairwise_lags(np.zeros((2, 10)), np.array([0]), np.array([1]), use_numba=True)


def test_solve_tau_recovers_a_smooth_delay():
    n = 200
    true_tau = 3 * np.sin(np.arange(n) / 30.0)
    pi, pj = partner_pairs(n, 60)
    lags = np.round(true_tau[pi] - true_tau[pj])
    tau = solve_tau(n, pi, pj, lags, lamb=1.0, smoothness=0.0, tau_avg=1)
    tau -= tau.mean()
    assert np.abs(tau - (true_tau - true_tau.mean())).max() < 0.6


def test_solve_tau_reference_dt_shape_is_checked():
    with pytest.raises(ValueError):
        solve_tau(
            5,
            np.array([0]),
            np.array([1]),
            np.array([1.0]),
            smoothness=1.0,
            reference_dt=np.zeros(3),
        )


def test_mccc_pass_measures_a_smooth_delay_on_an_aligned_gather(gather):
    from conftest import TRUE

    shift = np.round(4 * np.sin(np.arange(100) / 20.0)).astype(int)
    sub = np.stack(
        [
            np.roll(gather[c, int(round(TRUE[c])) - 200 : int(round(TRUE[c])) + 200], shift[c])
            for c in range(100)
        ]
    )
    tau = mccc(sub, 60, smoothness=50.0, tau_avg=10)
    # positive tau = channel arrives late = needs advancing
    assert np.corrcoef(tau, shift)[0, 1] > 0.9


def test_shift_and_tau_shift_are_inverse():
    x = np.arange(20.0)[None, :].repeat(3, 0)
    shifted, shifts = shift_arr(x, np.array([5.0, 10.0, 12.0]), base_time=10)
    assert list(shifts) == [5, 0, -2]
    back = tau_shift(shifted, shifts.astype(float))
    assert np.array_equal(back[1], x[1])
    assert np.array_equal(back[0][:15], x[0][:15])
    assert np.array_equal(back[2][2:], x[2][2:])


def test_moving_avg_uses_its_window():
    x = np.zeros(50)
    x[25] = 1.0
    assert np.isclose(moving_avg(x, 5).max(), 0.2)
    assert np.isclose(moving_avg(x, 10).max(), 0.1)
    assert moving_avg(x, 31).shape == x.shape
