"""dasmccc: iterative network MCCC refinement of DAS arrival curves.

from dasmccc import refine_curve, DIRECT
res = refine_curve(waveform, curve, DIRECT)   # arrays in, RefineResult out
"""

__version__ = "0.2.0"

from .anchor import first_lobe, stack_peak
from .core import iterate_align, mccc, pairwise_lags, partner_pairs, solve_tau
from .pipeline import (
    DIRECT,
    SECONDARY,
    NothingToRefine,
    RefineConfig,
    RefineResult,
    refine_curve,
    refine_phases,
)
from .polarity import PolarityConfig, PolarityResult, RickerWindows, ricker_polarity, ricker_windows
from .signal import NUMBA_AVAILABLE

__all__ = [
    "DIRECT",
    "__version__",
    "NUMBA_AVAILABLE",
    "NothingToRefine",
    "SECONDARY",
    "PolarityConfig",
    "PolarityResult",
    "RefineConfig",
    "RefineResult",
    "RickerWindows",
    "first_lobe",
    "iterate_align",
    "mccc",
    "pairwise_lags",
    "partner_pairs",
    "refine_curve",
    "refine_phases",
    "ricker_polarity",
    "ricker_windows",
    "solve_tau",
    "stack_peak",
]
