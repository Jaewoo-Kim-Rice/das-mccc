"""
Signal processing utilities for DAS data.

This module provides low-level signal processing functions
including wavelets, statistics, and cross-correlation.
"""

import numpy as np
from functools import lru_cache
from scipy.stats import truncnorm

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


def MMAD(data):
    """
    Computes the modified Z-score for a 1D numpy array using the median and Median Absolute Deviation (MAD),
    while ignoring NaN values.
    
    Parameters:
      - data: 1D numpy array (may contain NaN)
      
    Returns:
      - modified_z_scores: 1D numpy array of modified Z-scores computed as 0.6745 * (data - median) / MAD.
        If MAD is zero, an array of zeros is returned.
    """
    # Compute the median ignoring NaN values
    median_val = np.nanmedian(data)
    
    # Compute the Median Absolute Deviation (MAD) ignoring NaN values
    mad_val = np.nanmedian(np.abs(data - median_val))
    
    # If MAD is zero, avoid division by zero by returning zeros
    if mad_val == 0:
        return np.zeros_like(data)
    
    # Compute the modified Z-score using a scaling factor of 0.6745
    modified_z_scores = 0.6745 * (data - median_val) / mad_val    
    return modified_z_scores

def ricker(center_freq, n_samples, sample_rate):
    t = np.arange(-n_samples / 2, n_samples / 2) / sample_rate
    y = (1.0 - 2.0 * (np.pi**2) * (center_freq**2) * (t**2)) * np.exp(-(np.pi**2) * (center_freq**2) * (t**2))
    return t, y
    

@lru_cache(maxsize=4096)
def _normal_distribution_cached(corr_start, corr_end, N, std):
    """Cached computation of normal distribution sampling points (original slow version)."""
    mean = (corr_start + corr_end) / 2
    a = (corr_start - mean) / std  # lower bound
    b = (corr_end - mean) / std    # upper bound
    p = np.linspace(0, 1, N)
    points = truncnorm.ppf(p, a, b, loc=mean, scale=std)
    points_int = np.round(points).astype(int)
    for i in range(1, len(points_int)):
        if points_int[i] <= points_int[i-1]:
            points_int[i] = points_int[i-1] + 1
    # Return as tuple for caching (immutable)
    return tuple(points_int)


# Pre-computed templates for fast normal distribution sampling
# Key: (a_rounded, b_rounded, N) -> normalized template
_TRUNCNORM_TEMPLATES = {}

def _get_truncnorm_template_for_ab(a, b, N=50):
    """Get or compute the normalized truncnorm template for given a, b bounds."""
    # Round a, b to reduce number of unique templates (2 decimal places for accuracy)
    a_r = round(a, 2)
    b_r = round(b, 2)
    key = (a_r, b_r, N)

    if key not in _TRUNCNORM_TEMPLATES:
        p = np.linspace(0, 1, N)
        raw = truncnorm.ppf(p, a_r, b_r, loc=0, scale=1)
        # Normalize to [0, 1]: map [a, b] -> [0, 1]
        _TRUNCNORM_TEMPLATES[key] = (raw - a_r) / (b_r - a_r)

    return _TRUNCNORM_TEMPLATES[key]


def _normal_distribution_fast(corr_start, corr_end, N, std=20):
    """
    Fast normal distribution sampling using pre-computed templates.

    Uses cached templates based on the normalized bounds (a, b) to match
    original truncnorm behavior while avoiding repeated ppf calls.
    """
    mean = (corr_start + corr_end) / 2
    a = (corr_start - mean) / std
    b = (corr_end - mean) / std

    template = _get_truncnorm_template_for_ab(a, b, N)

    # Scale template from [0, 1] to [corr_start, corr_end]
    points = corr_start + (corr_end - corr_start) * template
    points_int = np.round(points).astype(int)

    # Ensure strictly increasing
    for i in range(1, len(points_int)):
        if points_int[i] <= points_int[i-1]:
            points_int[i] = points_int[i-1] + 1

    return points_int


@jit(nopython=True, cache=True)
def _normal_distribution_fast_core(corr_start, corr_end, N):
    """
    Fast approximation of truncated normal sampling.

    Produces points concentrated near the center (like truncnorm) but much faster.
    Uses power transform: |2p-1|^k with k>1 stretches ends, concentrating points at center.
    """
    mean = (corr_start + corr_end) / 2.0
    half_range = (corr_end - corr_start) / 2.0

    # Power transform: k > 1 concentrates points at center (like truncnorm)
    # Higher k = more concentration at center
    k = 2.5  # Tuned to approximate truncnorm shape
    points_int = np.empty(N, dtype=np.int64)
    for i in range(N):
        p = i / (N - 1) if N > 1 else 0.5
        # Map [0,1] -> [-1,1] with center concentration
        sign = 1.0 if p >= 0.5 else -1.0
        t = sign * (abs(2 * p - 1) ** k)
        points_int[i] = int(np.round(mean + half_range * t))

    # Ensure strictly increasing (same post-processing as original)
    for i in range(1, N):
        if points_int[i] <= points_int[i-1]:
            points_int[i] = points_int[i-1] + 1

    return points_int


def normal_distribution(corr_start, corr_end, N, std=20):
    """Sample N points from truncated normal distribution between corr_start and corr_end."""
    # Use fast template-based version (same results, ~2-3x faster)
    return _normal_distribution_fast(corr_start, corr_end, N, std)

def cc_right(arr):
    ccs=[]
    for i in range(arr.shape[0]-1):
        tr = arr[i, :]
        tr_right = arr[i+1,:]
        cc = np.correlate(tr, tr_right)
        ccs.append(cc)
    return np.array(ccs)

def rms(x):
    return np.sqrt(np.mean(x**2))   
    

def get_SNR(arr, picks, width, phase='p'):
    snr = []
    for i, tr in enumerate(arr):
        pick = round(picks[i])
        signal = rms(tr[pick - width: pick + width])
        if phase == 'p':
            noise = rms(tr[pick-width*6:pick-width])
        elif phase == 's':
            noise = rms(tr[pick+width:pick+width*6])
        snr.append(signal/noise)
    return snr
        
        

@jit(nopython=True, cache=True)
def _limited_cc_core(tr, shifted, max_shift):
    """Numba-optimized core cross-correlation computation."""
    len_tr = len(tr)
    n_shifts = 2 * max_shift + 1
    corr_values = np.zeros(n_shifts)

    for i in range(n_shifts):
        shift = i - max_shift
        if shift < 0:
            # Negative shift: tr[:len_tr+shift] dot shifted[-shift:]
            total = 0.0
            for k in range(len_tr + shift):
                total += tr[k] * shifted[k - shift]
            corr_values[i] = total
        elif shift > 0:
            # Positive shift: tr[shift:] dot shifted[:len_tr-shift]
            total = 0.0
            for k in range(len_tr - shift):
                total += tr[k + shift] * shifted[k]
            corr_values[i] = total
        else:
            # Zero shift: tr dot shifted
            total = 0.0
            for k in range(len_tr):
                total += tr[k] * shifted[k]
            corr_values[i] = total

    return corr_values


def limited_cc(tr, shifted, max_shift):
    """
    Compute limited cross-correlation between two traces.

    Uses Numba JIT compilation if available for ~10x speedup.
    """
    if NUMBA_AVAILABLE:
        return _limited_cc_core(tr, shifted, max_shift)
    else:
        # Fallback to numpy implementation
        len_tr = len(tr)
        shifts = np.arange(-max_shift, max_shift + 1)
        corr_values = np.zeros(len(shifts))

        for i, shift in enumerate(shifts):
            if shift < 0:
                corr_values[i] = np.dot(tr[:len_tr+shift], shifted[-shift:])
            elif shift > 0:
                corr_values[i] = np.dot(tr[shift:], shifted[:len_tr-shift])
            else:
                corr_values[i] = np.dot(tr, shifted)

        return corr_values
