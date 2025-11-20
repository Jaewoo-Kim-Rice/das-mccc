"""
Array manipulation utilities for DAS data.

This module provides functions for converting between ObsPy Stream
and numpy arrays, as well as time-shifting operations.
"""

import numpy as np
from obspy import Stream, Trace, UTCDateTime


def stream2array(
    st,
    starttime=None,
    endtime=None,
    sampling_rate=None,
    join: str = "overlap",     # "overlap" or "union"
    fill_value=np.nan,
    sort: bool = True,
    time_axis: str = "seconds" # "seconds" (since t0), or "none"
):
    """
    Convert an ObsPy Stream to a 2D numpy array with a common time base.

    Parameters
    ----------
    st : obspy.Stream
        Input stream.
    starttime, endtime : UTCDateTime or str or None
        Time window. If None, computed from traces.
        - overlap: [max(start_i), min(end_i)]
        - union  : [min(start_i), max(end_i)]
    sampling_rate : float or None
        Target sampling rate (Hz). If None, all traces must already share the same rate.
        If given, traces are resampled via Lanczos interpolation.
    join : {"overlap","union"}
        Use only overlapping window, or pad to union with fill_value.
    fill_value : float
        Value to fill gaps or out-of-coverage samples (default: np.nan).
    sort : bool
        Sort by network/station/location/channel for stable channel order.
    time_axis : {"seconds","none"}
        If "seconds", also return a 1D array of time in seconds since t0.

    Returns
    -------
    data : (n_traces, n_samples) ndarray
        Aligned data matrix.
    times : (n_samples,) ndarray or None
        Time vector in seconds relative to t0 (start of master window), or None.
    ids : list[str]
        Channel IDs in row order (e.g., "XX.ABC..EHZ").
    """
    st = st.copy()
    if len(st) == 0:
        raise ValueError("Empty Stream")

    if sort:
        st.sort(keys=["network", "station", "location", "channel"])

    # 1) Unify sampling rate
    if sampling_rate is None:
        fs_set = {float(f"{tr.stats.sampling_rate:.8f}") for tr in st}
        if len(fs_set) != 1:
            raise ValueError("Traces have different sampling rates. Provide sampling_rate to resample.")
        fs = fs_set.pop()
    else:
        fs = float(sampling_rate)
        for tr in st:
            # Lanczos interpolation is good for seismic signals
            tr.interpolate(fs, method="lanczos", a=8)

    # 2) Determine master window [t0, t1]
    starts = [tr.stats.starttime for tr in st]
    ends   = [tr.stats.endtime   for tr in st]

    if starttime is not None:
        t0 = UTCDateTime(starttime)
    else:
        t0 = max(starts) if join == "overlap" else min(starts)

    if endtime is not None:
        t1 = UTCDateTime(endtime)
    else:
        t1 = min(ends)   if join == "overlap" else max(ends)

    if t1 <= t0:
        raise ValueError("Non-positive time window. Check join/starttime/endtime.")

    # 3) Allocate output matrix
    n = int(round((t1 - t0) * fs)) + 1  # inclusive end
    data = np.full((len(st), n), fill_value, dtype=np.float64)

    # 4) Place each trace onto the master grid
    for i, tr in enumerate(st):
        # indices where this trace would land on the master grid
        s0 = int(round((tr.stats.starttime - t0) * fs))
        e0 = s0 + tr.stats.npts

        # intersection with [0, n)
        seg_start = max(0, s0)
        seg_end   = min(n, e0)
        if seg_end <= seg_start:
            continue  # no overlap

        # slice the trace data accordingly
        tr_s = seg_start - s0
        tr_e = tr_s + (seg_end - seg_start)
        data[i, seg_start:seg_end] = tr.data[tr_s:tr_e]

    ids = [tr.id for tr in st]
    times = np.arange(n, dtype=np.float64) / fs if time_axis == "seconds" else None
    return data, times, ids


def array2stream(data, sampling_rate=1000.0, starttime=None):
    """
    Convert a 2D NumPy array (channel, time) into an ObsPy Stream object.

    Parameters:
    ----------
    data : np.ndarray
        2D array with shape (channels, samples)
    sampling_rate : float
        Sampling rate in Hz
    starttime : obspy.UTCDateTime
        Start time for traces (optional)

    Returns:
    -------
    st : obspy.Stream
        ObsPy stream object with channels as individual traces.
    """
    if starttime is None:
        starttime = UTCDateTime(0)

    st = Stream()
    for ch_idx, channel_data in enumerate(data):
        trace = Trace(data=channel_data.astype(np.float32))
        trace.stats.station = f'CH{ch_idx:04d}'
        trace.stats.channel = 'DAS'
        trace.stats.sampling_rate = sampling_rate
        trace.stats.starttime = starttime
        st.append(trace)

    return st


def shift_arr(sub_das, S_result, reg=True, start_idx=0, base_time=2500):
    """
    Shift DAS array based on pick times.

    Parameters
    ----------
    sub_das : np.ndarray
        DAS data array (n_channels, n_samples)
    S_result : np.ndarray
        Pick results (n_picks, 2) with [channel_index, pick_time]
    reg : bool
        Unused parameter (kept for compatibility)
    start_idx : int
        Starting index in S_result
    base_time : float
        Reference time for alignment (default: 2500)

    Returns
    -------
    shifted_arr : np.ndarray
        Time-shifted array
    zero : int
        Always returns 0 (for compatibility)
    shifts : np.ndarray
        Applied shifts for each channel
    """
    shifted_arr = []
    shifts = []
    for ch_idx in range(sub_das.shape[0]):
        trace = sub_das[ch_idx, :]
        pick_time = S_result[start_idx+ch_idx, 1]
        shift = pick_time
        shift = round(base_time - shift)
        if shift > 0:
            new_trace = np.concatenate([np.ones(shift) * 1e-14, trace[:-shift]])
        elif shift < 0:
            new_trace = np.concatenate([trace[-shift:], np.ones(-shift) * 1e-14])
        else:
            new_trace = trace.copy()
        shifts.append(shift)
        shifted_arr.append(new_trace)
    shifted_arr = np.array(shifted_arr)
    return shifted_arr, 0, np.array(shifts)


def shift_arr_special(sub_das, S_result, reg=True, start_idx=0, base_time=2500):
    """
    Shift DAS array with RANSAC-fitted pick times and outlier detection.

    Parameters
    ----------
    sub_das : np.ndarray
        DAS data array (n_channels, n_samples)
    S_result : np.ndarray
        Pick results (n_picks, 2) with [channel_index, pick_time]
    reg : bool
        If True, apply RANSAC regression to picks
    start_idx : int
        Starting index in S_result
    base_time : float
        Reference time for alignment (default: 2500)

    Returns
    -------
    shifted_arr : np.ndarray
        Time-shifted array
    fitted_S : np.ndarray
        Fitted pick times
    shifts : np.ndarray
        Applied shifts for each channel
    """
    # Import here to avoid circular dependency
    from .regression import pick_regression_ransac

    if reg:
        fitted_S = pick_regression_ransac(S_result[start_idx:])
    else:
        fitted_S = S_result
    shifted_arr = []
    shifts = []
    for ch_idx in range(sub_das.shape[0]):
        trace = sub_das[ch_idx, :]
        pick_time = S_result[start_idx+ch_idx, 1]
        fitted_time = fitted_S[ch_idx, 1]
        if np.isnan(pick_time):
            shift = fitted_time
        elif abs(pick_time - fitted_time) < 50:
            shift = fitted_time
        else:
            shift = pick_time
        shift = round(base_time - shift)
        if shift > 0:
            new_trace = np.concatenate([np.ones(shift) * 1e-14, trace[:-shift]])
        elif shift < 0:
            new_trace = np.concatenate([trace[-shift:], np.ones(-shift) * 1e-14])
        else:
            new_trace = trace.copy()
        shifts.append(shift)
        shifted_arr.append(new_trace)
    shifted_arr = np.array(shifted_arr)
    return shifted_arr, fitted_S, np.array(shifts)


def tau_shift(das, tau):
    """
    Shift DAS array based on tau values.

    Parameters
    ----------
    das : np.ndarray
        DAS data array (n_channels, n_samples)
    tau : np.ndarray
        Tau shifts (n_channels, 2) with [channel_index, tau_value]

    Returns
    -------
    shifted_arr : np.ndarray
        Time-shifted array
    """
    shifted_arr = []
    for ch_idx in range(das.shape[0]):
        trace = das[ch_idx, :]
        shift = round(-tau[ch_idx, 1])

        if shift > 0:
            new_trace = np.concatenate([np.ones(shift) * 1e-14, trace[:-shift]])
        elif shift < 0:
            new_trace = np.concatenate([trace[-shift:], np.ones(-shift) * 1e-14])
        else:
            new_trace = trace.copy()
        shifted_arr.append(new_trace)
    shifted_arr = np.array(shifted_arr)
    return shifted_arr
