"""
Multi-channel cross-correlation (MCCC) for phase alignment.

This module provides functions for refining phase arrivals using
cross-correlation across multiple DAS channels.
"""

import numpy as np
from scipy.signal import correlate


def cross_correlate_channels(
    das_array: np.ndarray,
    reference_channel: int,
    max_lag: int = 50,
) -> np.ndarray:
    """
    Cross-correlate all channels with a reference channel.

    Parameters
    ----------
    das_array : np.ndarray
        DAS data of shape (n_channels, n_samples)
    reference_channel : int
        Index of reference channel
    max_lag : int
        Maximum lag to consider in samples (default: 50)

    Returns
    -------
    np.ndarray
        Time lags for maximum correlation for each channel

    Examples
    --------
    >>> data = np.random.randn(100, 5000)
    >>> lags = cross_correlate_channels(data, reference_channel=50)
    >>> len(lags)
    100
    """
    n_channels, n_samples = das_array.shape
    lags = np.zeros(n_channels)

    ref_trace = das_array[reference_channel, :]

    for ch_idx in range(n_channels):
        trace = das_array[ch_idx, :]

        # Compute cross-correlation
        cc = correlate(trace, ref_trace, mode="same")

        # Find lag of maximum correlation within max_lag
        center = len(cc) // 2
        search_start = max(0, center - max_lag)
        search_end = min(len(cc), center + max_lag + 1)

        search_region = cc[search_start:search_end]
        max_idx = np.argmax(np.abs(search_region))

        # Convert to lag relative to center
        lags[ch_idx] = (search_start + max_idx) - center

    return lags


def refine_picks_with_mccc(
    das_array: np.ndarray,
    initial_picks: np.ndarray,
    window_length: int = 100,
    max_shift: int = 25,
) -> np.ndarray:
    """
    Refine initial picks using multi-channel cross-correlation.

    This function uses cross-correlation between adjacent channels
    to refine pick times and ensure consistency.

    Parameters
    ----------
    das_array : np.ndarray
        DAS data of shape (n_channels, n_samples)
    initial_picks : np.ndarray
        Initial pick times of shape (n_channels, 2)
    window_length : int
        Length of window for cross-correlation (default: 100)
    max_shift : int
        Maximum allowed shift from initial pick (default: 25)

    Returns
    -------
    np.ndarray
        Refined picks of shape (n_channels, 2)

    Examples
    --------
    >>> data = np.random.randn(100, 5000)
    >>> picks = np.array([[i, 2500 + i] for i in range(100)])
    >>> refined = refine_picks_with_mccc(data, picks)
    >>> refined.shape
    (100, 2)

    Notes
    -----
    This is a simplified implementation. A full MCCC implementation
    would involve iterative refinement and coherency weighting.
    """
    n_channels = das_array.shape[0]
    refined_picks = initial_picks.copy()

    for ch_idx in range(1, n_channels):
        # Get windows around picks for current and previous channel
        prev_pick = int(initial_picks[ch_idx - 1, 1])
        curr_pick = int(initial_picks[ch_idx, 1])

        # Extract windows
        half_win = window_length // 2
        prev_start = max(0, prev_pick - half_win)
        prev_end = min(das_array.shape[1], prev_pick + half_win)
        curr_start = max(0, curr_pick - half_win)
        curr_end = min(das_array.shape[1], curr_pick + half_win)

        prev_window = das_array[ch_idx - 1, prev_start:prev_end]
        curr_window = das_array[ch_idx, curr_start:curr_end]

        # Cross-correlate
        cc = correlate(curr_window, prev_window, mode="same")

        # Find best lag
        center = len(cc) // 2
        search_start = max(0, center - max_shift)
        search_end = min(len(cc), center + max_shift + 1)

        search_region = cc[search_start:search_end]
        best_idx = np.argmax(np.abs(search_region))
        lag = (search_start + best_idx) - center

        # Apply refinement
        refined_picks[ch_idx, 1] = curr_pick + lag

    return refined_picks


def calculate_coherency(
    das_array: np.ndarray,
    picks: np.ndarray,
    window_length: int = 100,
) -> np.ndarray:
    """
    Calculate waveform coherency around picks.

    Measures how similar each channel's waveform is to its neighbors
    around the pick time. Higher coherency indicates better pick quality.

    Parameters
    ----------
    das_array : np.ndarray
        DAS data of shape (n_channels, n_samples)
    picks : np.ndarray
        Pick times of shape (n_channels, 2)
    window_length : int
        Window length for coherency calculation (default: 100)

    Returns
    -------
    np.ndarray
        Coherency values for each channel (0 to 1)

    Examples
    --------
    >>> data = np.random.randn(100, 5000)
    >>> picks = np.array([[i, 2500] for i in range(100)])
    >>> coherency = calculate_coherency(data, picks)
    >>> len(coherency)
    100
    """
    n_channels = das_array.shape[0]
    coherency = np.zeros(n_channels)

    for ch_idx in range(1, n_channels - 1):
        pick_time = int(picks[ch_idx, 1])
        half_win = window_length // 2

        # Extract windows
        start = max(0, pick_time - half_win)
        end = min(das_array.shape[1], pick_time + half_win)

        curr_window = das_array[ch_idx, start:end]
        prev_window = das_array[ch_idx - 1, start:end]
        next_window = das_array[ch_idx + 1, start:end]

        # Calculate normalized cross-correlation with neighbors
        cc_prev = np.corrcoef(curr_window, prev_window)[0, 1]
        cc_next = np.corrcoef(curr_window, next_window)[0, 1]

        # Average coherency
        coherency[ch_idx] = (abs(cc_prev) + abs(cc_next)) / 2.0

    # Handle edge channels
    coherency[0] = coherency[1]
    coherency[-1] = coherency[-2]

    return coherency
