"""
Filtering utilities for DAS data preprocessing.

This module provides filtering functions for cleaning and preparing
DAS strain rate data for analysis.
"""

from typing import Literal, Optional

import numpy as np
from scipy.signal import butter, detrend, filtfilt


def butterworth_lowpass(
    data: np.ndarray,
    cutoff: float,
    sampling_rate: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply Butterworth lowpass filter to data.

    Parameters
    ----------
    data : np.ndarray
        Input data (can be 1D or 2D)
    cutoff : float
        Cutoff frequency in Hz
    sampling_rate : float
        Sampling rate in Hz
    order : int
        Filter order (default: 4)

    Returns
    -------
    np.ndarray
        Filtered data with same shape as input

    Examples
    --------
    >>> data = np.random.randn(100, 5000)
    >>> filtered = butterworth_lowpass(data, cutoff=50.0, sampling_rate=10000.0)
    >>> filtered.shape
    (100, 5000)
    """
    nyquist = sampling_rate / 2.0
    normalized_cutoff = cutoff / nyquist

    # Design filter
    b, a = butter(order, normalized_cutoff, btype="low")

    # Apply filter
    if data.ndim == 1:
        return filtfilt(b, a, data)
    elif data.ndim == 2:
        # Filter each channel
        return np.array([filtfilt(b, a, data[i, :]) for i in range(data.shape[0])])
    else:
        raise ValueError(f"Data must be 1D or 2D, got shape {data.shape}")


def butterworth_bandpass(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    sampling_rate: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to data.

    Parameters
    ----------
    data : np.ndarray
        Input data (can be 1D or 2D)
    lowcut : float
        Low cutoff frequency in Hz
    highcut : float
        High cutoff frequency in Hz
    sampling_rate : float
        Sampling rate in Hz
    order : int
        Filter order (default: 4)

    Returns
    -------
    np.ndarray
        Filtered data with same shape as input

    Examples
    --------
    >>> data = np.random.randn(100, 5000)
    >>> filtered = butterworth_bandpass(data, 1.0, 50.0, sampling_rate=10000.0)
    """
    nyquist = sampling_rate / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design filter
    b, a = butter(order, [low, high], btype="band")

    # Apply filter
    if data.ndim == 1:
        return filtfilt(b, a, data)
    elif data.ndim == 2:
        return np.array([filtfilt(b, a, data[i, :]) for i in range(data.shape[0])])
    else:
        raise ValueError(f"Data must be 1D or 2D, got shape {data.shape}")


def remove_trend(
    data: np.ndarray,
    type: Literal["linear", "constant"] = "linear",
) -> np.ndarray:
    """
    Remove trend from data.

    Parameters
    ----------
    data : np.ndarray
        Input data (can be 1D or 2D)
    type : {'linear', 'constant'}
        Type of detrending:
        - 'linear': remove linear trend
        - 'constant': remove mean

    Returns
    -------
    np.ndarray
        Detrended data

    Examples
    --------
    >>> data = np.random.randn(100, 5000) + np.linspace(0, 10, 5000)
    >>> detrended = remove_trend(data, type='linear')
    """
    if data.ndim == 1:
        return detrend(data, type=type)
    elif data.ndim == 2:
        # Detrend along time axis (axis=1)
        return detrend(data, axis=1, type=type)
    else:
        raise ValueError(f"Data must be 1D or 2D, got shape {data.shape}")


def normalize_channels(
    data: np.ndarray,
    method: Literal["standard", "minmax", "rms"] = "standard",
    axis: int = 1,
) -> np.ndarray:
    """
    Normalize each channel independently.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_channels, n_samples)
    method : {'standard', 'minmax', 'rms'}
        Normalization method:
        - 'standard': zero mean, unit variance
        - 'minmax': scale to [-1, 1]
        - 'rms': divide by RMS amplitude
    axis : int
        Axis along which to normalize (default: 1, time axis)

    Returns
    -------
    np.ndarray
        Normalized data

    Examples
    --------
    >>> data = np.random.randn(100, 5000) * np.random.rand(100, 1)
    >>> normalized = normalize_channels(data, method='standard')
    """
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got shape {data.shape}")

    normalized = data.copy()

    if method == "standard":
        # Zero mean, unit variance
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        std[std < 1e-12] = 1.0  # Avoid division by zero
        normalized = (data - mean) / std

    elif method == "minmax":
        # Scale to [-1, 1]
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        range_val = max_val - min_val
        range_val[range_val < 1e-12] = 1.0
        normalized = 2.0 * (data - min_val) / range_val - 1.0

    elif method == "rms":
        # Divide by RMS
        rms = np.sqrt(np.mean(data**2, axis=axis, keepdims=True))
        rms[rms < 1e-12] = 1.0
        normalized = data / rms

    else:
        raise ValueError(f"Unknown method: {method}")

    return normalized


def remove_bad_channels(
    data: np.ndarray,
    threshold: float = 3.0,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Remove or zero out channels with anomalous RMS amplitudes.

    Identifies channels whose RMS amplitude is more than `threshold`
    standard deviations away from the median RMS across all channels.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_channels, n_samples)
    threshold : float
        Number of standard deviations for outlier detection (default: 3.0)
    fill_value : float
        Value to fill bad channels (default: 0.0)

    Returns
    -------
    np.ndarray
        Data with bad channels replaced by fill_value

    Examples
    --------
    >>> data = np.random.randn(100, 5000)
    >>> data[5, :] *= 100  # Create bad channel
    >>> cleaned = remove_bad_channels(data, threshold=3.0)
    """
    # Calculate RMS for each channel
    rms = np.sqrt(np.mean(data**2, axis=1))

    # Find outliers
    median_rms = np.median(rms)
    std_rms = np.std(rms)

    bad_channels = np.abs(rms - median_rms) > threshold * std_rms

    # Replace bad channels
    cleaned = data.copy()
    cleaned[bad_channels, :] = fill_value

    return cleaned


def apply_preprocessing_pipeline(
    data: np.ndarray,
    sampling_rate: float,
    lowpass_freq: Optional[float] = None,
    bandpass_freq: Optional[tuple] = None,
    detrend_type: Literal["linear", "constant", None] = "linear",
    normalize_method: Literal["standard", "minmax", "rms", None] = None,
    remove_bad: bool = False,
) -> np.ndarray:
    """
    Apply standard preprocessing pipeline to DAS data.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_channels, n_samples)
    sampling_rate : float
        Sampling rate in Hz
    lowpass_freq : float, optional
        Lowpass filter cutoff frequency
    bandpass_freq : tuple, optional
        Bandpass filter (lowcut, highcut) frequencies
    detrend_type : {'linear', 'constant', None}
        Detrending method
    normalize_method : {'standard', 'minmax', 'rms', None}
        Normalization method
    remove_bad : bool
        Whether to remove bad channels

    Returns
    -------
    np.ndarray
        Preprocessed data

    Examples
    --------
    >>> data = np.random.randn(100, 5000)
    >>> processed = apply_preprocessing_pipeline(
    ...     data,
    ...     sampling_rate=10000.0,
    ...     lowpass_freq=50.0,
    ...     detrend_type='linear',
    ...     normalize_method='standard',
    ... )
    """
    processed = data.copy()

    # Remove bad channels first
    if remove_bad:
        processed = remove_bad_channels(processed)

    # Detrend
    if detrend_type is not None:
        processed = remove_trend(processed, type=detrend_type)

    # Filter
    if bandpass_freq is not None:
        lowcut, highcut = bandpass_freq
        processed = butterworth_bandpass(processed, lowcut, highcut, sampling_rate)
    elif lowpass_freq is not None:
        processed = butterworth_lowpass(processed, lowpass_freq, sampling_rate)

    # Normalize
    if normalize_method is not None:
        processed = normalize_channels(processed, method=normalize_method)

    return processed
