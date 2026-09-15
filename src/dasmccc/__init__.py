"""dasmccc: iterative network MCCC refinement of DAS arrival curves.

from dasmccc import refine_curve, DIRECT
res = refine_curve(waveform, curve, DIRECT)   # arrays in, RefineResult out
"""

from .anchor import first_lobe, stack_peak
from .core import iterate_align, mccc, pairwise_lags, partner_pairs, solve_tau
from .pipeline import DIRECT, SECONDARY, RefineConfig, RefineResult, refine_curve, refine_phases
from .polarity import PolarityConfig, PolarityResult, ricker_polarity
from .signal import NUMBA_AVAILABLE

__all__ = [
    "DIRECT",
    "NUMBA_AVAILABLE",
    "SECONDARY",
    "PolarityConfig",
    "PolarityResult",
    "RefineConfig",
    "RefineResult",
    "first_lobe",
    "iterate_align",
    "mccc",
    "pairwise_lags",
    "partner_pairs",
    "refine_curve",
    "refine_phases",
    "ricker_polarity",
    "solve_tau",
    "stack_peak",
]
