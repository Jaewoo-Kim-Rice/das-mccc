"""
Signal processing utilities for DAS data.

This module provides:
- picking: Phase picking algorithms
- filters: Preprocessing filters
- mccc: Multi-channel cross-correlation
"""

# Picking functions
# Filtering functions
from .filters import (
    apply_preprocessing_pipeline,
    butterworth_bandpass,
    butterworth_lowpass,
    normalize_channels,
    remove_bad_channels,
    remove_trend,
)

# MCCC functions
from .mccc import (
    calculate_coherency,
    cross_correlate_channels,
    refine_picks_with_mccc,
)
from .picking import (
    align_waveforms,
    calculate_sp_ratio,
    extract_polarity,
    polynomial_regression_ransac,
    smooth_picks_median,
)

# Travel time functions
from .travel_times import (
    calculate_travel_times,
    get_theoretical_picks,
    load_event_from_csv,
    travel_times_to_samples,
)

__all__ = [
    # Picking
    "polynomial_regression_ransac",
    "smooth_picks_median",
    "align_waveforms",
    "calculate_sp_ratio",
    "extract_polarity",
    # Filters
    "butterworth_lowpass",
    "butterworth_bandpass",
    "remove_trend",
    "normalize_channels",
    "remove_bad_channels",
    "apply_preprocessing_pipeline",
    # MCCC
    "cross_correlate_channels",
    "refine_picks_with_mccc",
    "calculate_coherency",
    # Travel times
    "calculate_travel_times",
    "get_theoretical_picks",
    "load_event_from_csv",
    "travel_times_to_samples",
]
