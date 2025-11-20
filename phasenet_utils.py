"""
Utility functions for PhaseNet-based picking.

This module contains helper functions for PhaseNet processing,
originally from BPMF.utils but adapted for das_focmec.
"""

import numpy as np
from scipy.signal import find_peaks


def normalize_batch(seismogram, normalization_window_sample=3000, overlap=0.50):
    """Apply Z-score normalization in running windows.

    Following Zhu et al. 2019, this function applied Z-score
    normalization in running windows with length `normalization_window_sample`.

    Parameters
    -----------
    seismogram : numpy.ndarray
        Three-component seismograms. `seismogram` has shape
        (num_traces, num_channels=3, num_time_samples).
    normalization_window_sample : integer, optional
        The window length, in samples, over which normalization is applied.
        Default is 3000 (like in Zhu et al. 2019).

    Returns
    --------
    normalized_seismogram : numpy.ndarray
        Normalized seismogram with same shape as `seismogram`.
    """
    from scipy.interpolate import interp1d

    #shift = normalization_window_sample // 2
    shift = int((1. - overlap) * normalization_window_sample)
    num_stations, num_channels, num_time_samples = seismogram.shape

    # std in sliding windows
    seismogram_pad = np.pad(
        seismogram, ((0, 0), (0, 0), (shift, shift)), mode="reflect"
    )
    # time = np.arange(0, num_time_samples, shift, dtype=np.int32)
    seismogram_view = np.lib.stride_tricks.sliding_window_view(
        seismogram_pad, normalization_window_sample, axis=-1
    )[:, :, ::shift, :]
    sliding_std = np.std(seismogram_view, axis=-1)
    sliding_mean = np.mean(seismogram_view, axis=-1)

    # time at centers of sliding windows
    num_sliding_windows = seismogram_view.shape[2]
    time = np.linspace(shift, num_time_samples - shift, num_sliding_windows)

    sliding_std[:, :, -1], sliding_mean[:, :, -1] = (
        sliding_std[:, :, -2],
        sliding_mean[:, :, -2],
    )
    sliding_std[:, :, 0], sliding_mean[:, :, 0] = (
        sliding_std[:, :, 1],
        sliding_mean[:, :, 1],
    )
    sliding_std[sliding_std == 0] = 1

    # normalize data with sliding std and mean
    t_interp = np.arange(num_time_samples)
    std_interp = np.stack(
            tuple(np.interp(
                t_interp, time, sld_std, left=sld_std[0], right=sld_std[-1]
                )
                for sld_std in sliding_std.reshape(-1, sliding_std.shape[-1])
                ),
            axis=0
            ).reshape(
                    sliding_std.shape[:-1] + (len(t_interp),)
                    )
    mean_interp = np.stack(
            tuple(np.interp(
                t_interp, time, m_std, left=m_std[0], right=m_std[-1]
                )
                for m_std in sliding_mean.reshape(-1, sliding_mean.shape[-1])
                ),
            axis=0
            ).reshape(
                    sliding_mean.shape[:-1] + (len(t_interp),)
                    )

    seismogram = (seismogram - mean_interp) / std_interp

    return seismogram


def find_picks(
        phase_probability, threshold, return_peaks=False, **kwargs
        ):
    """Find phase picks from time series of phase probability.

    Phase picks are given by the peaks exceeding `threshold` in
    the time series `phase_probability`. The probability neighborhood
    around each peak is used as the pdf of a given pick measurement.

    Note: Additional key-word arguments are passed to
    'scipy.signal.find_peaks'.

    Parameters
    ----------
    phase_probability : array-like
        Probability time series of observing the arrival of a
        given seismic phase.
    threshold : float
        Value above which peaks are taken to be candidate phase picks.

    Returns
    -------
    peaks_value : `numpy.ndarray`
        Probability value of the selected peaks.
    peaks_mean : `numpy.ndarray`
        Expected pick timing, in samples.
    peaks_std : `numpy.ndarray`
        Uncertainty on pick timing, in samples.
    """
    # set the width kwarg to 1 if not defined
    # so that `properties` has peak width info
    kwargs.setdefault("width", 1)
    peak_indexes, peak_properties = find_peaks(
            phase_probability, height=threshold, **kwargs
            )
    peaks_value, peaks_mean, peaks_std = [], [], []
    for i in range(len(peak_indexes)):
        idx1 = int(peak_properties["left_ips"][i])
        idx2 = int(peak_properties["right_ips"][i])
        samples = np.arange(idx1, idx2+1)
        prob = phase_probability[samples]

        mean = np.sum(samples * prob) / prob.sum()
        std = np.sqrt(np.sum((samples - mean)**2) / prob.sum())

        peaks_mean.append(mean)
        peaks_std.append(std)
        peaks_value.append(phase_probability[peak_indexes[i]])

    if return_peaks:
        return peak_indexes, np.asarray(peaks_value), np.asarray(peaks_mean), np.asarray(peaks_std)

    return np.asarray(peaks_value), np.asarray(peaks_mean), np.asarray(peaks_std)
