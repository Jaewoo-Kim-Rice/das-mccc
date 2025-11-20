"""
Phase picking utilities for DAS data.

This module provides functions for detecting P and S wave arrivals
from DAS strain rate data using various methods.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.signal import medfilt
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def polynomial_regression_ransac(
    data: np.ndarray,
    degree: int = 5,
    fit_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    random_state: int = 42,
) -> np.ndarray:
    """
    Perform polynomial regression with RANSAC on pick data.

    This function fits a polynomial curve to noisy pick data using
    RANSAC to handle outliers, then extrapolates predictions over
    the entire data range.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (n, 2) where column 0 is channel indices
        and column 1 is pick times (may contain NaNs)
    degree : int
        Degree of polynomial basis functions (default: 5)
    fit_range : tuple, optional
        (min_channel, max_channel) to restrict fitting range.
        Use None for endpoints to indicate no limit (default: None)
    random_state : int
        Random state for RANSAC (default: 42)

    Returns
    -------
    np.ndarray
        Array of shape (n, 2) with [channel_index, fitted_pick_time]

    Examples
    --------
    >>> picks = np.array([[0, 100], [1, 105], [2, np.nan], [3, 115]])
    >>> fitted = polynomial_regression_ransac(picks, degree=2)
    >>> print(fitted.shape)
    (4, 2)

    Notes
    -----
    NaN values in the input are automatically excluded from fitting
    but predictions are made for all channels.
    """
    # Mask out NaN values
    mask_valid = ~np.isnan(data[:, 1])

    # Restrict to specified channel range if provided
    if fit_range is not None:
        ch_min, ch_max = fit_range
        in_range_mask = np.ones(len(data), dtype=bool)
        if ch_min is not None:
            in_range_mask &= data[:, 0] >= ch_min
        if ch_max is not None:
            in_range_mask &= data[:, 0] <= ch_max
        mask_train = mask_valid & in_range_mask
    else:
        mask_train = mask_valid

    # Prepare training data
    X_train = data[mask_train, 0][:, None]  # Channel indices
    Y_train = data[mask_train, 1]  # Pick times

    # Build polynomial regression pipeline with RANSAC
    model = make_pipeline(
        PolynomialFeatures(degree),
        RANSACRegressor(random_state=random_state),
    )

    # Fit model
    model.fit(X_train, Y_train)

    # Predict for all channels
    X_all = data[:, 0][:, None]
    Y_pred = model.predict(X_all)

    return np.column_stack([data[:, 0], Y_pred])


def smooth_picks_median(
    picks: np.ndarray,
    kernel_size: int = 31,
) -> np.ndarray:
    """
    Apply median filtering to smooth pick times.

    Parameters
    ----------
    picks : np.ndarray
        Array of shape (n_channels, 2) with [channel_index, pick_time]
    kernel_size : int
        Median filter kernel size (default: 31)

    Returns
    -------
    np.ndarray
        Smoothed picks with same shape as input

    Examples
    --------
    >>> picks = np.array([[0, 100], [1, 150], [2, 105], [3, 110]])
    >>> smoothed = smooth_picks_median(picks, kernel_size=3)
    """
    smoothed = picks.copy()
    smoothed[:, 1] = medfilt(picks[:, 1], kernel_size=kernel_size)
    return smoothed


def align_waveforms(
    das_array: np.ndarray,
    picks: np.ndarray,
    base_time: float = 2500.0,
    fill_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align DAS waveforms based on pick times.

    Shifts each channel's waveform so that the pick time aligns
    with a common reference time (base_time).

    Parameters
    ----------
    das_array : np.ndarray
        DAS data of shape (n_channels, n_samples)
    picks : np.ndarray
        Pick times of shape (n_channels, 2) where column 1 is pick time
    base_time : float
        Reference time in samples (default: 2500.0)
    fill_value : float
        Value to fill shifted regions (default: 0.0)

    Returns
    -------
    aligned_array : np.ndarray
        Time-aligned DAS array (n_channels, n_samples)
    shifts : np.ndarray
        Applied shifts for each channel (n_channels,)

    Examples
    --------
    >>> das_data = np.random.randn(100, 5000)
    >>> picks = np.array([[i, 2500 + i*2] for i in range(100)])
    >>> aligned, shifts = align_waveforms(das_data, picks)
    >>> aligned.shape
    (100, 5000)

    Notes
    -----
    Positive shift moves waveform to the right (delays it),
    negative shift moves it to the left (advances it).
    """
    n_channels, n_samples = das_array.shape
    aligned_array = np.zeros_like(das_array)
    shifts = np.zeros(n_channels)

    for ch_idx in range(n_channels):
        trace = das_array[ch_idx, :]
        pick_time = picks[ch_idx, 1]

        # Calculate shift needed to align pick to base_time
        shift = int(round(base_time - pick_time))
        shifts[ch_idx] = shift

        # Apply shift
        if shift > 0:
            # Delay: pad at beginning, truncate at end
            aligned_array[ch_idx, :] = np.concatenate([np.full(shift, fill_value), trace[:-shift]])
        elif shift < 0:
            # Advance: truncate at beginning, pad at end
            aligned_array[ch_idx, :] = np.concatenate([trace[-shift:], np.full(-shift, fill_value)])
        else:
            # No shift
            aligned_array[ch_idx, :] = trace

    return aligned_array, shifts


def calculate_sp_ratio(
    das_array: np.ndarray,
    p_picks: np.ndarray,
    s_picks: np.ndarray,
    p_window: Tuple[int, int] = (-50, 50),
    s_window: Tuple[int, int] = (-50, 50),
) -> np.ndarray:
    """
    Calculate S/P amplitude ratio for each channel.

    Parameters
    ----------
    das_array : np.ndarray
        DAS data of shape (n_channels, n_samples)
    p_picks : np.ndarray
        P-wave pick times of shape (n_channels, 2)
    s_picks : np.ndarray
        S-wave pick times of shape (n_channels, 2)
    p_window : tuple
        Time window (before, after) P-pick for amplitude measurement
    s_window : tuple
        Time window (before, after) S-pick for amplitude measurement

    Returns
    -------
    np.ndarray
        S/P amplitude ratio for each channel

    Examples
    --------
    >>> das_data = np.random.randn(100, 5000)
    >>> p_picks = np.array([[i, 1000] for i in range(100)])
    >>> s_picks = np.array([[i, 3000] for i in range(100)])
    >>> sp_ratio = calculate_sp_ratio(das_data, p_picks, s_picks)
    >>> len(sp_ratio)
    100
    """
    n_channels = das_array.shape[0]
    sp_ratios = np.zeros(n_channels)

    for ch_idx in range(n_channels):
        # Get pick times
        p_time = int(p_picks[ch_idx, 1])
        s_time = int(s_picks[ch_idx, 1])

        # Extract windows
        p_start = max(0, p_time + p_window[0])
        p_end = min(das_array.shape[1], p_time + p_window[1])
        s_start = max(0, s_time + s_window[0])
        s_end = min(das_array.shape[1], s_time + s_window[1])

        # Calculate RMS amplitudes
        p_amp = np.sqrt(np.mean(das_array[ch_idx, p_start:p_end] ** 2))
        s_amp = np.sqrt(np.mean(das_array[ch_idx, s_start:s_end] ** 2))

        # Calculate ratio (avoid division by zero)
        if p_amp > 1e-12:
            sp_ratios[ch_idx] = s_amp / p_amp
        else:
            sp_ratios[ch_idx] = np.nan

    return sp_ratios


def extract_polarity(
    das_array: np.ndarray,
    picks: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """
    Extract first motion polarity from DAS data.

    Parameters
    ----------
    das_array : np.ndarray
        DAS data of shape (n_channels, n_samples)
    picks : np.ndarray
        Pick times of shape (n_channels, 2)
    window : int
        Number of samples after pick to average for polarity (default: 10)

    Returns
    -------
    np.ndarray
        Polarity for each channel (+1 or -1)

    Examples
    --------
    >>> das_data = np.random.randn(100, 5000)
    >>> picks = np.array([[i, 2500] for i in range(100)])
    >>> polarities = extract_polarity(das_data, picks)
    >>> set(polarities)
    {-1.0, 1.0}
    """
    n_channels = das_array.shape[0]
    polarities = np.zeros(n_channels)

    for ch_idx in range(n_channels):
        pick_time = int(picks[ch_idx, 1])

        # Extract window after pick
        start = pick_time
        end = min(das_array.shape[1], pick_time + window)

        # Average amplitude in window
        avg_amp = np.mean(das_array[ch_idx, start:end])

        # Determine polarity
        polarities[ch_idx] = 1.0 if avg_amp > 0 else -1.0

    return polarities
