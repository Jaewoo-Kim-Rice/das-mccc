import numpy as np
import pytest

from dasmccc.signal import ricker

FS = 1000.0
N_CH, N = 300, 1000
SLOPE = 0.15  # samples per channel of the true moveout
TRUE = 400 + SLOPE * np.arange(N_CH)
RICKER_TROUGH = np.sqrt(1.5) / (np.pi * 50.0) * FS  # leading trough of a 50 Hz Ricker, samples


def make_gather(seed=3, noise=0.3, flip_from=None, outliers=()):
    """Linear moveout, 50 Hz Ricker, polarity flipped from channel ``flip_from`` on,
    Gaussian noise, and optional channels with a 20x amplitude (near-field outliers)."""
    rng = np.random.default_rng(seed)
    w = ricker(50.0, 60, FS)[1]
    x = rng.standard_normal((N_CH, N)) * noise
    for c in range(N_CH):
        k = int(round(TRUE[c]))
        gain = 20.0 if c in outliers else 1.0
        sign = -1.0 if flip_from is not None and c >= flip_from else 1.0
        x[c, k - 30 : k + 30] += w * sign * gain
    return x


def initial_curve(late=10.0, wiggle=4.0):
    """The true curve placed ``late`` samples later (a later lobe) with a smooth error."""
    return TRUE + late + wiggle * np.sin(np.arange(N_CH) / 40.0)


@pytest.fixture(scope="session")
def gather():
    return make_gather()
