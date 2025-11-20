import os, random
SEED = 42
os.environ['PYTHONHASHSEED']          = str(SEED)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8" 

import numpy as np
import torch
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic        = True
torch.backends.cudnn.benchmark            = False
torch.backends.cudnn.enabled              = False   # kills all cuDNN kernels
torch.use_deterministic_algorithms(True)
import seisbench.models as sbm
import matplotlib.pyplot as plt
from . import phasenet_utils as utils
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import truncnorm
from scipy.signal import medfilt
from scipy.stats import zscore
import h5py
from obspy import Stream, Trace
from obspy.signal.trigger import classic_sta_lta, trigger_onset
import math
import copy
import sys
import re
from pathlib import Path


device_torch = "cuda"
ml_detector = sbm.PhaseNet(in_channels=1).from_pretrained('original', version_str = 'latest')
ml_detector.eval()
ml_detector.to("cuda")
ml_detector_3d = sbm.PhaseNet(in_channels=3).from_pretrained('original', version_str = 'latest')
ml_detector_3d.eval()
ml_detector_3d.to("cuda")


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
        from obspy import UTCDateTime
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

#Define Ricker wavelet
def ricker(center_freq, n_samples, sample_rate):
    t = np.arange(-n_samples / 2, n_samples / 2) / sample_rate
    y = (1.0 - 2.0 * (np.pi**2) * (center_freq**2) * (t**2)) * np.exp(-(np.pi**2) * (center_freq**2) * (t**2))
    return t, y
    
def normal_distribution(corr_start, corr_end, N, std=20):
    mean = (corr_start+corr_end)/2
    a = (corr_start - mean) / std  # lower bound
    b = (corr_end - mean) / std    # upper bound
    p = np.linspace(0, 1, N)
    points = truncnorm.ppf(p, a, b, loc=mean, scale=std)
    points_int = np.round(points).astype(int)
    for i in range(1, len(points_int)):
        if points_int[i] <= points_int[i-1]:
            points_int[i] = points_int[i-1] + 1
    return points_int
def get_sign(arr, snr_thresh = 3, cc=True):
    ric = ricker(50, 60, 1000)[1]
    center = arr.shape[1]//2
    snrs=[]
    signs = []
    amps = []
    for i in range(arr.shape[0]):
        tr = arr[i, :]
        signal_win = [center-30, center+30]
        amp = rms(tr[signal_win[0]:signal_win[1]])
        mask = np.ones(len(tr), dtype=bool)
        mask[signal_win[0]:signal_win[1]] = False
        noise = rms(tr[mask])
        snr = amp/noise
        # sign = np.sign(tr[center])
        if cc:
            corr = limited_cc(ric, tr[signal_win[0]:signal_win[1]], 10)
            corr_argmax = np.argmax(abs(corr))
            sign = np.sign(corr[corr_argmax])
        else:
            sign = np.sign(tr[center])
        # sign = np.sign(np.correlate(ric, tr[signal_win[0]:signal_win[1]]))[0]
        if snr < snr_thresh:
            sign = 0
            
        snrs.append(snr)
        signs.append(sign)
        amps.append(amp)
    return np.array(snrs), np.array(signs), np.array(amps)
        
def cc_right(arr):
    ccs=[]
    for i in range(arr.shape[0]-1):
        tr = arr[i, :]
        tr_right = arr[i+1,:]
        cc = np.correlate(tr, tr_right)
        ccs.append(cc)
    return np.array(ccs)
def diff_corr_ric(shifted_arr, max_shift, snr_thresh = 5, mmad_thresh = 4.0, ricker_freq = 50):
    
    ric = ricker(ricker_freq, shifted_arr.shape[1], 1000)[1]
    dts, polarities, amps, snrs = [],[],[],[]
    wins = []
    for i in range(shifted_arr.shape[0]):
        tr = shifted_arr[i, :]
        corr = limited_cc(ric, tr, max_shift)
        corr_argmax = np.argmax(abs(corr))
        argmax_win = [min(len(tr)//2, corr_argmax),max(-len(tr)//2, corr_argmax)]
        # print('before_', argmax_win[0], argmax_win[1])
        argmax_win = [max_shift + len(tr)//2 - i for i in argmax_win]
        argmax_win[0] -= 25
        argmax_win[1] += 25
        # print('winlength', argmax_win[0], argmax_win[1], argmax_win[1]-argmax_win[0])
        wins.append(argmax_win)
        dt = np.argmax(abs(corr)) - max_shift # if negative, it's arriving late
        polarity = np.sign(corr[corr_argmax])
        center_idx = shifted_arr.shape[1]//2 - dt
        amp = rms(tr[argmax_win[0]:argmax_win[1]])
        # print(argmax_win)
        mask = np.ones(len(tr), dtype=bool)
        mask[argmax_win[0]:argmax_win[1]] = False
        noise = rms(tr[mask])
        snr = amp/noise
        if snr <snr_thresh:
            polarity = np.nan
            amp = np.nan
            dt = np.nan
        snrs.append(snr)
        dts.append(dt)
        polarities.append(polarity)
        amps.append(amp)
    polarities = np.array(polarities)
    # filtering near-field affected channels
    amps = np.array(amps)
    mmad = MMAD(amps)
    wins = np.array(wins)
    #filtering nearfield signals by mmad values
    wins[mmad>mmad_thresh] = np.array([0,0])
    amps[mmad> mmad_thresh] = np.nan
    polarities[mmad> mmad_thresh] = np.nan

    
    dts = len(tr)//2 - np.array(dts)
    # amps[np.where(amp_zscore>3)[0]] = 0
    
    return dts, polarities, amps, snrs, wins

def mask_noise(arr, win):
    new_arr = arr.copy()
    std = np.std(arr)/10
    mean = 0
    for i in range(new_arr.shape[0]):
        tr = new_arr[i]
        start = win[i,0]
        end = win[i,1]
        median = np.median(tr)
        mask = np.ones(len(tr), dtype=bool)
        mask[start:end] = False
        size = len(tr) - (end-start)
        tr[mask] = np.random.normal(loc=mean, scale=std, size=size)
        
    return new_arr
def get_diff_corr(shifted_arr, corr_len, max_shift):
    taus = []
    Diff= []
    indices = []
    Corrs =[]
    for i in range(shifted_arr.shape[0]):
        tr = shifted_arr[i, :]
        corr_start = i-corr_len
        corr_end = min(shifted_arr.shape[0]-1, i+corr_len)
        target_idx = normal_distribution(corr_start, corr_end, 50)
        target_idx = target_idx[(target_idx>=0) & (target_idx<shifted_arr.shape[0])]
        tau = []
        for j in target_idx:
            if i < j:
                max_shift = int(max((j-i)*0.2, 3)) # max_shift depends on channel diff
                row = np.zeros(shifted_arr.shape[0])
                row[i]= 1; row[j] = -1
                Diff.append(row)
                corr = limited_cc(tr, shifted_arr[j, :], max_shift)
                Corrs.append(corr)
                dt = np.argmax(abs(corr)) - max_shift
                taus.append(dt)
                indices.append([i,j])
    taus= np.array(taus)
    Diff= np.array(Diff)
    return Diff, taus
    
def moving_avg(tau, conv_n=100):
    conv_n = 100
    pad_left = conv_n // 2        
    pad_right = conv_n // 2 - 1      
    padded = np.pad(tau, (pad_left, pad_right), mode='reflect')
    tau_p = np.convolve(padded, np.ones(conv_n)/conv_n, mode='valid')
    return tau_p


def moving_median(arr, width=50):
    """
    Computes the moving median of a 1D array using a sliding window of specified width.
    If more than half of the values in the window are NaN, the median for that window is set to NaN.
    
    Parameters:
      - arr: 1D numpy array (or any sequence convertible to a numpy array)
      - width: Window size for computing the median (default is 50)
      
    Returns:
      - A numpy array of the same length as arr containing the moving median values.
    """
    arr = np.asarray(arr)  # Ensure input is a numpy array
    n = len(arr)
    medians = np.empty(n)
    
    for i in range(n):
        # Define window boundaries for a centered window; adjust for boundaries
        start = max(0, i - width // 2)
        end = min(n, i + width // 2 + 1)
        window = arr[start:end]
        
        # Count number of NaN values in the window
        num_nan = np.sum(np.isnan(window))
        
        # If more than half of the values are NaN, set median to NaN
        if num_nan > (len(window) / 2):
            medians[i] = np.nan
        else:
            medians[i] = np.nanmedian(window)
    
    return medians

def MCCC(shifted_arr, corr_len, max_shift, lamb, avg_win=30, pad=False):

    #Padding
    if pad:
        pad_size = 300
        padding_arr = shifted_arr[-pad_size:]
        shifted_arr = np.concatenate((shifted_arr, padding_arr[::-1]))
    Diff, taus = get_diff_corr(shifted_arr, corr_len=corr_len, max_shift=max_shift)
    # Inversion
    M = np.vstack([lamb*Diff])
    b  = np.concatenate([lamb*taus])
    tau, _, _, _ = np.linalg.lstsq(M, b, rcond=None)


    #moving average to get smooth tau
    tau = moving_avg(tau, avg_win)
    # median filter to get smooth tau
    if pad:
        # #removing pad
        tau = tau[:-pad_size]
    tau = np.array([range(tau.shape[0]), tau]).T
    return tau


def ultra_mccc_iterative(das_arr, pick, corr_len, max_shift, lamb, n_iterations=3, shrinked_window_length= 300, medfilt_iterations=[1, 2, 3]):
    half_win_len = shrinked_window_length//2
    # Initial setup: calculate base_time and perform the initial shift operation
    # base_time = int(das_arr.shape[1] * 5 / 6)
    base_time = int(das_arr.shape[1] / 2)

    print('Starting initial shift')
    shifted_arr, fitted_arv, first_shifts = shift_arr(das_arr, pick, base_time=base_time)
    # Select the region of interest (e.g., 150 samples around base_time)
    shifted_arr = shifted_arr[:, base_time - half_win_len: base_time + half_win_len]

    # Set up the initial array for the iterative process
    current_arr = shifted_arr
    tau_list = []  # List to store tau values from each iteration
    intermediate_results = [das_arr, shifted_arr]  # Store initial results (original array and first shift result)

    # Perform iterative MCCC inversion
    for i in range(1, n_iterations + 1):
        corr_scale = i
        print(f'Starting MCCC iteration {i}')
        tau = MCCC(current_arr, corr_len//corr_scale, max_shift//corr_scale, lamb)
        tau_list.append(tau)
        # Apply tau_shift to adjust the array based on the computed tau
        current_arr = tau_shift(current_arr, tau)
        # Apply median filtering on specified iterations to reduce noise
        if i in medfilt_iterations:
            current_arr = medfilt(current_arr, kernel_size=(25, 1))
        # Save the intermediate result after this iteration
        intermediate_results.append(current_arr)

    # Calculate the total shift by summing the initial first_shifts with all tau shifts (second column)
    # Convert first_shifts to float to ensure proper addition with float tau values
    total_shift = first_shifts.astype(np.float64).copy()
    for tau in tau_list:
        total_shift -= tau[:, 1]  # Accumulate the shifts from the second column of each tau

    # Return the list of intermediate results, the total shift, initial information, and the list of tau values
    return intermediate_results, total_shift, [first_shifts, base_time], tau_list

def ultra_mccc(das_arr, pick, corr_len, max_shift, lamb):
    base_time = int(das_arr.shape[1]*5/6)
    # First MCCC inversion
    print('starting first MCCC')
    shifted_arr, fitted_arv, first_shifts = shift_arr(das_arr, pick, base_time = base_time)
    shifted_arr = shifted_arr[:, base_time-300: base_time+300]

    
    tau_0 = MCCC(shifted_arr, corr_len, max_shift, lamb)
    reshifted_arr = tau_shift(shifted_arr, tau_0)
    
    # median filter for denoise
    medfilted_arr = medfilt(reshifted_arr, kernel_size = (25,1))

    # Second MCCC inversion 
    print('starting second MCCC')
    tau_1 = MCCC(medfilted_arr, corr_len, max_shift, lamb)
    final_arr = tau_shift(medfilted_arr, tau_1)
    #third MCCC inversion 
    print('starting third MCCC')
    tau_2 = MCCC(final_arr, corr_len, max_shift, lamb)
    final_final = tau_shift(final_arr, tau_2)
    final_final = medfilt(final_final, kernel_size=(25,1))

    total_shift = tau_0[:,1] + tau_1[:,1] + tau_2[:,1] + first_shifts
    return [das_arr, shifted_arr, reshifted_arr, medfilted_arr, final_arr, final_final], total_shift, [first_shifts, base_time], [tau_0, tau_1, tau_2]

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
        
        
def limited_cc(tr, shifted, max_shift):
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
def runPNDAS(arr,begin_time, sample_rate = 1000, gauge_length = 10):
    def _remove_duplicate(pick, chn_length):
        filtered = pick.sort_values('phase_score', ascending=False).drop_duplicates('channel_index')
        full_index = pd.Series(range(chn_length), name='channel_index')
        merged = pd.merge(full_index.to_frame(), filtered[['channel_index', 'phase_index']], 
                          on='channel_index', how='left')
        result_array = merged.to_numpy()
        return result_array
        
    tmp_file_name = './tmp_waveform.h5'
    with h5py.File(tmp_file_name, "w") as file:
        ds = file.create_dataset("data", data = arr)
        ds.attrs["dt_s"] = 1/sample_rate
        ds.attrs["dx_m"] = gauge_length
        if hasattr(begin_time, "isoformat"):
            ds.attrs["begin_time"] = begin_time.isoformat()
        else:
            ds.attrs["begin_time"] = begin_time
    with open("tmp_files.txt", "w") as f:
        f.write(tmp_file_name)
    EQNetpath = '/home/jk103/02.Reagle_microseismic/Reagle_microseismic/EQNet'
    base_cmd = f"{EQNetpath}/predict.py --model phasenet_das --data_list=tmp_files.txt  --data_path {tmp_file_name} --result_path ./tmp_results --format=h5  --batch_size 1 --workers 0"
    cmd = f"{sys.executable} {base_cmd}"
    os.system(cmd)
    picks = pd.read_csv("./tmp_results/picks_phasenet_das/tmp_waveform.csv")
    p_pick = _remove_duplicate(picks[picks.phase_type =='P'], arr.shape[0])
    s_pick = _remove_duplicate(picks[picks.phase_type =='S'], arr.shape[0])
    return p_pick, s_pick
        
    
    
def runPN(arr, n_component = 1): # 1c arr = (sensor, trace); 3c arr = (sensor, component (N E Z), trace)

    waveform_features = np.zeros([arr.shape[0], 2, arr.shape[-1]])
    for i in range(len(arr)):
        with torch.no_grad():
            if n_component ==1:
                PN_model = ml_detector
                input_arr = np.vstack([arr[i:i+1]] * 3).reshape(1,3,-1)
            elif n_component ==3:
                PN_model = ml_detector_3d
                input_arr = arr[i:i+1]
            
            _x = torch.from_numpy(utils.normalize_batch(input_arr)).float().to(device_torch)
            PN_probas = PN_model(_x)
            waveform_features[i] = PN_probas[:, 1:, :].cpu().detach().numpy().astype(np.float32)
    return waveform_features


    
def get_picks(das_arr, threshold_P=0.2, threshold_S=0.2):
	waveform_features = runPN(das_arr)
	waveform_features.shape

	sum_prob = waveform_features.sum(axis=1)
	picks = pd.DataFrame(
	    index=np.arange(waveform_features.shape[0]),
	    columns=["P_probas", "P_picks", "P_unc", "S_probas", "S_picks", "S_unc"],
	)
	for sta in range(waveform_features.shape[0]):
	    picks.loc[sta, ["P_probas", "P_picks", "P_unc"]] = utils.find_picks(
	        waveform_features[sta, 0, :], threshold_P
	    )
	    picks.loc[sta, ["S_probas", "S_picks", "S_unc"]] = utils.find_picks(
	        waveform_features[sta, 1, :], threshold_S
	    )
	P_result = np.array([(i, p[0] if p.size > 0 else np.nan) for i, p in enumerate(picks.P_picks)])
	S_result = np.array([(i, p[0] if p.size > 0 else np.nan) for i, p in enumerate(picks.S_picks)])
	return P_result, S_result, sum_prob

def get_stalta_triggers(das_arr, sampling_rate, sta, lta, thresh_on, thresh_off):
    """
    Detects all event onsets in a DAS array using the STA/LTA algorithm.

    Parameters
    ----------
    das_arr : np.ndarray
        2D array of DAS data (channels, samples).
    sampling_rate : float
        Sampling rate of the data in Hz.
    sta : float
        Short-term average window length in seconds.
    lta : float
        Long-term average window length in seconds.
    thresh_on : float
        STA/LTA ratio threshold for trigger activation.
    thresh_off : float
        STA/LTA ratio threshold for trigger deactivation.

    Returns
    -------
    np.ndarray
        A 2D array of shape (n_picks, 2) where each row is [channel_index, pick_time].
        Picks are sorted by channel index, then by time.
    """
    all_picks = []
    for i, trace_data in enumerate(das_arr):
        trace = Trace(data=trace_data)
        trace.stats.sampling_rate = sampling_rate
        cft = classic_sta_lta(trace.data, int(sta * sampling_rate), int(lta * sampling_rate))
        on_off = trigger_onset(cft, thresh_on, thresh_off)
        
        if len(on_off) > 0:
            for onset in on_off[:, 0]:
                all_picks.append([i, onset])
            
    return np.array(all_picks)
    
def pick_regression_ransac(data, degree=5, fit_range=None):
    """
    Perform polynomial regression with RANSAC and extrapolate predictions over the entire data range.
    
    Parameters
    ----------
    data : np.ndarray, shape (n, 2)
        Input array where column 0 is X values and column 1 is Y values (may contain NaNs).
    degree : int, default=5
        Degree of the polynomial basis functions.
    fit_range : tuple (xmin, xmax) or (xmin, None), optional
        If provided, only data points with xmin <= X <= xmax are used for fitting.
        If None, all non-NaN points are used.

    Returns
    -------
    np.ndarray, shape (n, 2)
        Array of [X, Y_pred], where Y_pred is the model’s prediction for each X in `data`.
    """
    # Mask out any rows where Y is NaN
    mask_valid = ~np.isnan(data[:, 1])
    
    # Restrict to the specified X-range if fit_range is given
    if fit_range is not None:
        xmin, xmax = fit_range
        in_range = (data[:, 0] >= xmin) & (
            data[:, 0] <= xmax if xmax is not None else True
        )
        mask_train = mask_valid & in_range
    else:
        mask_train = mask_valid
    
    # Prepare training arrays
    X_train = data[mask_train, 0][:, None]
    Y_train = data[mask_train, 1]
    # Build a pipeline: polynomial feature expansion + RANSAC regressor
    model = make_pipeline(
        PolynomialFeatures(degree),
        RANSACRegressor(random_state=0)
    )
    # Fit the model
    model.fit(X_train, Y_train)
    
    # Predict over the full X-range
    X_all = data[:, 0][:, None]
    Y_pred = model.predict(X_all)
    
    # Return combined array of X and predicted Y
    return np.column_stack([data[:, 0], Y_pred])




def shift_arr(sub_das, S_result, reg=True, start_idx = 0, base_time= 2500):

    shifted_arr = []
    shifts = []
    for ch_idx in range(sub_das.shape[0]):
    # for ch_idx in range(1500,2900):
    
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
    return shifted_arr, 0 , np.array(shifts)


def shift_arr_special(sub_das, S_result, reg=True, start_idx = 0, base_time= 2500):
    if reg:
        fitted_S = pick_regression_ransac(S_result[start_idx:])
    else:
        fitted_S = S_result
    shifted_arr = []
    shifts = []
    for ch_idx in range(sub_das.shape[0]):
    # for ch_idx in range(1500,2900):
    
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
    shifted_arr=[]
    for ch_idx in range(das.shape[0]):
        trace = das[ch_idx,:]
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

def sign_filter_by_amp(arr, thresh_percentile = 10):
    thresh = np.percentile(arr, thresh_percentile)
    mask = np.ones_like(arr)
    mask[arr<thresh] = 0 
    return mask
    
    

def masking_picks(
    s_picks: np.ndarray,
    sp_picks: np.ndarray,
    tol_channel: float,
    tol_pick: float,
    fill_value: float=0
) -> np.ndarray:
    """
    Remove S-wave picks that overlap with any SP pick within given tolerances,
    while preserving the shape of the S picks array.
    
    Each row in s_picks and sp_picks is assumed to be [channel_index, pick_value].
    For each S-wave pick, if there is any SP pick such that the absolute difference 
    in channel indices is <= tol_channel and the absolute difference in pick values 
    is <= tol_pick, that S-wave pick is replaced with [fill_value, fill_value].
    
    Parameters
    ----------
    s_picks : np.ndarray
        A 2D array of shape (n, 2) containing S-wave picks as [channel_index, pick_value].
    sp_picks : np.ndarray
        A 2D array of shape (m, 2) containing SP picks as [channel_index, pick_value].
    tol_channel : float
        Tolerance for the difference in channel index.
    tol_pick : float
        Tolerance for the difference in pick value.
    fill_value : float, optional
        Value used to fill the removed picks. Default is np.nan.
        
    Returns
    -------
    filtered_s_picks : np.ndarray
        A 2D array of the same shape as s_picks. 
        S-wave picks that overlap with any SP pick are replaced with [fill_value, fill_value].
    """
    # Create a copy of s_picks as float to allow fill_value (like np.nan) assignment
    filtered_s_picks = s_picks.copy().astype(float)
    
    # Loop over each S-wave pick
    for i, (channel_s, value_s) in enumerate(s_picks):
        # Check against every SP pick
        for channel_sp, value_sp in sp_picks:
            # If the S pick is within tolerance of the SP pick in both channel and value
            if (abs(channel_s - channel_sp) <= tol_channel) and (abs(value_s - value_sp) <= tol_pick):
                # Replace the S pick with fill values
                filtered_s_picks[i,1] = fill_value
                break  # No need to check other SP picks for this S pick
                
    return filtered_s_picks

def mask_points_with_noise(
    data: np.ndarray,
    points: np.ndarray,
    patch_size: list = [5,5],
    method: str = 'zero',
    fill_value: float = np.nan,
    random_seed: int = None
) -> np.ndarray:
    """
    Masks specific points (and their surrounding patches) in a 2D array by replacing
    their values with zeros, a constant value, or locally generated noise.

    Parameters
    ----------
    data : np.ndarray
        A 2D array of shape (H, W) representing the original data.
    points : np.ndarray
        A 2D array of shape (n, 2) where each row is (row_index, col_index).
        These points indicate the centers of patches to mask.
    patch_size : int, optional
        The half-size of the square patch to mask around each point.
        For example, patch_size=[5,10] will mask the region [r-1 : r+1, c-5 : c+10].
        Default is [5,5].
    method : str, optional
        The masking method. Possible options:
          - 'zero': fill the patch with zeros
          - 'constant': fill the patch with the value specified by fill_value
          - 'local_noise': fill the patch with random Gaussian noise whose mean
                          and standard deviation are derived from the patch
        Default is 'zero'.
    fill_value : float, optional
        If method='constant', this value will be used to fill the patch.
        Default is 0.0.
    random_seed : int, optional
        If provided, sets the random seed for reproducibility when method='local_noise'.
        Default is None (no fixed seed).

    Returns
    -------
    masked_data : np.ndarray
        A 2D array of the same shape as `data`, with specified patches masked.
    """
    # Create a copy to avoid modifying the original data
    masked_data = data.copy()
    std = np.std(data)
    if random_seed is not None:
        np.random.seed(random_seed)

    # Get the shape of the input data
    nrows, ncols = data.shape

    # Iterate through each point
    for (r, c) in points:
        if np.isnan(r) | np.isnan(c):
            continue
        # Ensure the row and column indices are integers
        r = int(r)
        c = int(c)

        # Compute patch boundaries
        r_min = max(0, r - patch_size[0]//5)
        r_max = min(nrows, r + patch_size[0]//5 + 1)
        c_min = max(0, c - patch_size[0])
        c_max = min(ncols, c + patch_size[1] + 1)

        if method == 'zero':
            # Fill the patch with zeros
            masked_data[r_min:r_max, c_min:c_max] = 0.0

        elif method == 'constant':
            # Fill the patch with a constant value
            masked_data[r_min:r_max, c_min:c_max] = fill_value

        elif method == 'local_noise':
            # Replace the patch with local Gaussian noise
            # Compute the median and std from the original data patch
            patch = data[r_min:r_max, c_min:c_max]
            local_mean = np.median(data[r_min:r_max, c_min-patch_size[0]*5:c_max+patch_size[0]*5])
            local_std = std

            # Generate noise with the local mean and std
            noise = np.random.normal(loc=local_mean, scale=local_std, size=patch.shape)
            masked_data[r_min:r_max, c_min:c_max] = noise

        else:
            raise ValueError(f"Unknown masking method: {method}")

    return masked_data
        
def _ordinal(n):
    """
    Convert an integer n to its English ordinal word (1 -> 'first', 2 -> 'second', etc.).
    Falls back to numeric+suffix for n > 10.
    """
    ordinals = {
        1: 'first', 2: 'second', 3: 'third', 4: 'fourth',
        5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth',
        9: 'ninth', 10: 'tenth'
    }
    return ordinals.get(n, f"{n}th")

def plot_MCCC_results(arrs, initial_pick, max_cols=3, line_at=150):
    """
    Plot a series of 2D arrays in a flexible grid layout.

    Parameters
    ----------
    arrs : list of 2D numpy arrays
        The data arrays to display.
    max_cols : int, default=3
        Maximum number of columns in the grid.
    line_at : int or float, default=150
        Horizontal line position (in index units) drawn on all but
        the first subplot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the subplots.
    axs : numpy.ndarray of Axes
        The flattened array of subplot axes.
    """
    N = len(arrs)

    # Define the first two fixed titles
    fixed_titles = ['Raw data', 'flatten from arrival picks']

    # Generate ordinal MCCC titles for the remaining plots
    mccc_titles = [f"{_ordinal(i)} MCCC" for i in range(1, N-1)]

    # Combine and truncate to match the number of arrays
    titles = (fixed_titles + mccc_titles)[:N]

    # Determine grid size
    ncols = min(max_cols, N)
    nrows = math.ceil(N / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axs = axs.flatten()  # Flatten in case of multiple rows/columns

    # Loop over each array and corresponding axis
    for i, ax in enumerate(axs[:N]):
        data = arrs[i]
        # Compute robust max for color scaling (99th percentile)
        vmax = np.percentile(data, 99)
        # Display the transposed array with symmetric color limits
        ax.imshow(data.T, vmin=-vmax, vmax=vmax, aspect='auto')
        if i==0:
            ax.plot(initial_pick[:,0], initial_pick[:,1])
        # Set numbered title
        ax.set_title(f'({i+1}) {titles[i]}', fontsize=12)
        ax.set_ylabel('Time (ms)', fontsize=10)
        # Draw a horizontal dashed line on all but the first plot
        if i != 0:
            ax.axhline(line_at, color='r', linestyle='--')

    # Remove any unused subplots
    for j in range(N, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()
    return fig, axs   

def mccc_pipeline(
    das_arr,
    s_masked_arr,
    P_result,
    shrinked_window_length = 400,
    corr_len_initial=200,
    max_shift_initial=10,
    n_iter_initial=4,
    lamb_initial=1,
    corr_len_secondary=200,
    max_shift_secondary=20,
    n_iter_secondary=2,
    snr_thresh=3.0,
    ricker_freq=50,
    ricker_max_shift = 100,
    medfilt_kernel_arr=(25, 1),
    medfilt_kernel_dt=301,
    rolling_window=500,
    median_filter_signs=151,
    cc_for_sign = True,
):
    """
    Run full MCCC workflow with plots and return final amplitudes and sign arrays.

    Parameters
    ----------
    das_arr : ndarray
        Original DAS data for final P arrival plotting.
    s_masked_arr : ndarray
        Masked DAS data for initial MCCC.
    P_result : ndarray
        Initial P picks for MCCC.
    Other parameters : optional
        Workflow hyperparameters (see code).

    Returns
    -------
    amps : ndarray
        Amplitude ratios from get_sign.
    signs : ndarray
        Filtered sign array.
    """
    half_win_len = shrinked_window_length//2
    # Initial iterative MCCC
    arrs, total_shift, (first_shifts, base_time), _ = ultra_mccc_iterative(
        s_masked_arr,
        P_result,
        shrinked_window_length = shrinked_window_length,
        corr_len=corr_len_initial,
        max_shift=max_shift_initial,
        n_iterations=n_iter_initial,
        lamb=lamb_initial
    )
    # plot_MCCC_results(arrs, initial_pick = P_result, line_at = half_win_len)

    # Ricker fitting time adjustment
    dt, _, amp, _, wins = diff_corr_ric(
        arrs[-1],
        max_shift=ricker_max_shift,
        snr_thresh=snr_thresh,
        mmad_thresh=3.5,
        ricker_freq=ricker_freq
    )
    dt = wins.mean(axis=1)
    # print(np.array(wins).max())
    # print(arrs[-1], wins)
    filt_arr = mask_noise(arrs[-1], wins)
    filt_arr = medfilt(filt_arr, kernel_size=medfilt_kernel_arr)
    dt = medfilt(dt, medfilt_kernel_dt)
    dt = pd.Series(dt).fillna(pd.Series(dt).rolling(
        window=rolling_window,
        center=True,
        min_periods=rolling_window // 10
    ).mean())
    dt = np.vstack((np.arange(dt.shape[0]), dt.values)).T

    # Secondary MCCC with filtered data
    filt_arrs, filt_total_shift, (_, filt_base_time), _ = ultra_mccc_iterative(
        filt_arr,
        dt,
        shrinked_window_length = shrinked_window_length//2,
        corr_len=corr_len_secondary,
        max_shift=max_shift_secondary,
        lamb=1,
        n_iterations=n_iter_secondary
    )
    # plot_MCCC_results(filt_arrs, initial_pick = dt,line_at = half_win_len/2)

    # Extract phase amplitudes and signs
    _, signs, amps = get_sign(filt_arrs[-1], snr_thresh=snr_thresh, cc=cc_for_sign)
    signs = medfilt(signs, median_filter_signs)
    signs = signs * sign_filter_by_amp(amps)

    # Final P arrival picks
    final_P_arrivals = base_time - (total_shift + filt_total_shift)
    
    
    
    
    # ── data for plot ──
    arrivals_plot = np.where(np.abs(signs) == 1, final_P_arrivals, np.nan) - 2
    left_overlay   = (shrinked_window_length // 4) - 50 * signs
    plot_payload = {
        "filt_last":     filt_arrs[-1],   # Left imshow background (used with T)
        "left_overlay":  left_overlay,    # Left red line
        "arrivals_plot": arrivals_plot,   # Right red line
    }
    
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # sharey=True if needed
    
    # # --- Left: Phase Sign Results ---
    # vmin1, vmax1 = -np.percentile(filt_arrs[-1], 99), np.percentile(filt_arrs[-1], 99)
    # ax1.imshow(filt_arrs[-1].T, aspect='auto', vmin=vmin1, vmax=vmax1)
    # ax1.plot(shrinked_window_length//4 - 50 * signs, color='red')
    # ax1.set_title('Phase Sign Results')
    # ax1.set_xlabel('Channel index')
    # ax1.set_ylabel('Time (ms)')
    
    # # --- Right: Final Arrival Picks ---
    # vmin2, vmax2 = -np.percentile(das_arr, 99), np.percentile(das_arr, 99)
    # ax2.imshow(das_arr.T, aspect='auto', vmin=vmin2, vmax=vmax2)
    # ax2.plot(arrivals_plot, linestyle='-', color='red', label='arrival picks', lw=1.0)
    # ax2.legend()
    # ax2.set_ylim([np.nanmax(arrivals_plot) + 100, np.nanmin(arrivals_plot) - 100])
    # ax2.set_xlabel('Channel index')
    # ax2.set_ylabel('Time (ms)')
    # ax2.set_title('Final Arrival Picks')
    
    # plt.tight_layout()
    # plt.show()

    return amps, signs, final_P_arrivals, plot_payload

def plot_payload_grid(payloads, save_path=None, das_arrs=None, percentile=99, suptitle=None):
    """
    payloads: [{"filt_last": 2D, "left_overlay": 1D, "arrivals_plot": 1D}, ...]
    das_arrs: None | 2D ndarray (common) | list of 2D ndarray (per payload)
    """
    rows = len(payloads)
    fig, axes = plt.subplots(rows, 2, figsize=(16, 5*rows))
    if rows == 1:
        axes = np.array([axes])  # Unify shape to (1,2)

    for r, pay in enumerate(payloads):
        # --- Left panel ---
        v1 = np.nanpercentile(pay["filt_last"], percentile)
        ax = axes[r, 0]
        print(v1)
        ax.imshow(pay["filt_last"].T, aspect='auto', vmin=-v1, vmax=v1)
        ax.plot(pay["left_overlay"], color='red')
        ax.set_title(f'Phase Sign Results #{r+1}')
        ax.set_xlabel('Channel index'); ax.set_ylabel('Time (ms)')

        # --- Right panel ---
        if isinstance(das_arrs, (list, tuple)):
            bg = das_arrs[r]
        elif das_arrs is None:
            bg = pay["filt_last"]           # fallback: reuse left background
        else:
            bg = das_arrs                   # common background

        v2 = np.nanpercentile(bg, percentile)
        ax = axes[r, 1]
        ax.imshow(bg.T, aspect='auto', vmin=-v2, vmax=v2)
        ax.plot(pay["arrivals_plot"], color='red', lw=1.0, label='arrival picks')
        ax.legend()
        arr = pay["arrivals_plot"]
        # if np.isfinite(arr).any():
        ax.set_ylim([np.nanmax(arr) + 100, np.nanmin(arr) - 100])
        ax.set_title(f'Final Arrival Picks #{r+1}')
        ax.set_xlabel('Channel index'); ax.set_ylabel('Time (ms)')

    if suptitle:
        fig.suptitle(suptitle, y=0.995)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return save_path, None  # <- return save path
    else:
        plt.show()
        return fig, axes


def masking_sp(
    das_arr,
    P_result,
    final_P_arrivals,
    granite_idx,
    begin_time,
    p_sp_time_gap = 180,
    patch_size_primary=(100, 20),
    patch_size_secondary=(100, 31),
    method='local_noise',
    mask_pick_window=10,
    mask_pick_thresh=10,
    ransac_degree=2,
    ransac_fit_range=(500, None)
):
    """
    Apply noise-based masking to P picks, run PNDAS, mask and merge picks,
    then smooth with RANSAC regression.

    Parameters
    ----------
    das_arr : ndarray
        Original DAS data array.
    P_result : ndarray, shape (n_channels, 2)
        Initial P pick times per channel and associated metric.
    final_P_arrivals : ndarray, shape (n_channels,)
        Final P arrival times from MCCC pipeline.
    granite_idx : int
        Channel index threshold separating two depth regions.
    patch_size_primary : tuple of int
        Window size (rows, cols) for primary noise masking.
    patch_size_secondary : tuple of int
        Window size for secondary noise masking on shifted picks.
    method : str
        Masking method for mask_points_with_noise.
    begin_time : datetime or float
        Start time parameter for runPNDAS.
    mask_pick_window : int
        Window size parameter for masking_picks.
    mask_pick_thresh : int
        Threshold parameter for masking_picks.
    ransac_degree : int
        Polynomial degree for pick_regression_ransac smoothing.
    ransac_fit_range : tuple
        Slice range for regression fitting (start, end).

    Returns
    -------
    ndarray
        Masked and RANSAC-smoothed pick array, shape (n_channels, 2).
    """
    # Stack initial and final picks
    perfect_p = np.vstack((P_result[:, 0], final_P_arrivals)).T

    # Primary noise masking
    perfect_p_masked = mask_points_with_noise(
        das_arr,
        perfect_p,
        patch_size=patch_size_primary,
        method=method
    )
    # Shift top region picks for secondary masking
    sp = np.vstack((
        perfect_p[:granite_idx, 0],
        perfect_p[:granite_idx, 1] + p_sp_time_gap
    )).T
    
    # Secondary noise masking
    sp_masked = mask_points_with_noise(
        perfect_p_masked,
        sp,
        patch_size=patch_size_secondary,
        method=method
    )

    # Run PNDAS to get S picks
    _, S_result = runPNDAS(
        sp_masked,
        begin_time=begin_time
    )

    # Mask picks and merge two regions
    sp_primary_picks = masking_picks(
        S_result[:granite_idx],
        sp,
        mask_pick_window,
        mask_pick_thresh
    )
    merged_picks = np.vstack((
        sp_primary_picks,
        S_result[granite_idx:]
    ))

    # Smooth picks via RANSAC regression
    smoothed_picks = pick_regression_ransac(
        merged_picks,
        degree=ransac_degree,
        fit_range=ransac_fit_range
    )

    return sp_masked, smoothed_picks, perfect_p_masked


def pick_fit_arrivals(das_arr, begin_time, 
                      plot=True, use_PNDAS=True, p_fit_range = (0,None), s_fit_range=(500,None),
                      P_fit_degree = 2, S_fit_degree = 2):
    if use_PNDAS:
        P_result, _, sum_prob = get_picks(das_arr, threshold_P=0.2,threshold_S=0.2)
        _, S_result = runPNDAS(das_arr, begin_time = begin_time)
    else:
        P_result, S_result, sum_prob = get_picks(das_arr, threshold_P=0.2,threshold_S=0.2)
    P_result[:,1] = medfilt(P_result[:,1], kernel_size=31)
    S_result[:,1] = medfilt(S_result[:,1], kernel_size=31)
    # S_result[:,1] = das_utils.pick_regression_ransac(S_result[:], degree=3)[:,1]
    # P_result[:,1] = pick_regression_ransac(P_result[:], degree=3)[:,1]
    
    fit_S = pick_regression_ransac(S_result[:], degree=S_fit_degree, fit_range=s_fit_range)
    fit_P = pick_regression_ransac(P_result[:], degree=P_fit_degree, fit_range=p_fit_range)
    s_masked_arr = mask_points_with_noise(das_arr, fit_S, patch_size=[15,300], method='local_noise')
    p_masked_arr = mask_points_with_noise(das_arr, fit_P, patch_size=[0,0], method='local_noise')
    if plot:
        vmin, vmax = -np.percentile(das_arr, 99), np.percentile(das_arr, 99)
        plt.imshow(das_arr.T, vmin=vmin, vmax=vmax, aspect='auto')
        plt.scatter(P_result[:,0], P_result[:,1],s=0.1, c='blue', alpha=1, label='initial p arrival pick')
        plt.scatter(S_result[:,0], S_result[:,1],s=0.1, c='red',alpha=0.4 , label='initial s arrival pick')
        
        plt.ylabel('Time (ms)')
        plt.xlabel('Channel index')
        
        plt.plot(fit_P[:,0], fit_P[:,1], c='blue', alpha=0.3, label='poly-fitted p')
        plt.plot(fit_S[:,0], fit_S[:,1], c='red',  alpha=0.3, label='poly-fitted s')
        plt.legend()
        plt.ylim([fit_S[:,1].max() + 200, fit_P[:,1].min() - 200])

        plt.show()

    return p_masked_arr, s_masked_arr, fit_S, fit_P, P_result, S_result





