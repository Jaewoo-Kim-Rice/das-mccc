"""
Multi-channel cross-correlation (MCCC) core algorithms.

This module provides the core MCCC algorithms for phase alignment
and polarity determination in DAS data.
"""

import numpy as np
import pandas as pd
from scipy.signal import medfilt

from .signal_utils import ricker, MMAD, rms, normal_distribution, limited_cc
from .filtering import moving_avg
from .array_ops import shift_arr, tau_shift


def MCCC(shifted_arr, corr_len, max_shift, lamb, avg_win=30, pad=False, smoothness=0.0, reference_dt=None):
    """
    Multi-channel cross-correlation with optional smoothness regularization.

    Parameters
    ----------
    shifted_arr : ndarray
        Shifted waveform array
    corr_len : int
        Correlation window length
    max_shift : int
        Maximum allowed shift
    lamb : float
        Regularization parameter for cross-correlation
    avg_win : int
        Moving average window size
    pad : bool
        Whether to pad the array
    smoothness : float
        Smoothness regularization parameter. Higher values enforce smoother
        shifts between adjacent channels. Typical values: 0.1-10.0
        Based on physical constraint: adjacent channels (5m apart) should not
        differ by more than ~1ms at 5000 m/s max velocity.
    reference_dt : ndarray, optional
        Expected time difference between adjacent channels from theoretical curve.
        Shape: (n_channels-1,). If provided, smoothness regularization targets
        this curve instead of zero, preventing cycle skipping by constraining
        the solution to follow the expected moveout pattern.

        When reference_dt is provided:
          - Regularization becomes: ||smoothness * (D*tau - reference_dt)||²
          - This allows MCCC to follow the theoretical moveout while still
            using cross-correlation for fine-tuning

        When reference_dt is None (default):
          - Regularization is: ||smoothness * D*tau||²
          - This enforces adjacent channels to have similar shifts (zero difference)

    Returns
    -------
    tau : ndarray
        Time shifts for each channel, shape (n_channels, 2)
    """
    #Padding
    if pad:
        pad_size = 300
        padding_arr = shifted_arr[-pad_size:]
        shifted_arr = np.concatenate((shifted_arr, padding_arr[::-1]))
    Diff, taus = get_diff_corr(shifted_arr, corr_len=corr_len, max_shift=max_shift)

    # Get number of channels
    n_channels = Diff.shape[1]

    # Build first-difference matrix for smoothness regularization
    if smoothness > 0:
        # D matrix: penalizes differences between adjacent channels
        D = np.zeros((n_channels - 1, n_channels))
        for i in range(n_channels - 1):
            D[i, i] = -1
            D[i, i + 1] = 1

        # Determine target for smoothness constraint
        if reference_dt is not None:
            # Use theoretical moveout as target
            # Ensure reference_dt has correct shape
            if len(reference_dt) != n_channels - 1:
                print(f"  Warning: reference_dt length ({len(reference_dt)}) != n_channels-1 ({n_channels-1}), ignoring")
                b_smooth = np.zeros(n_channels - 1)
            else:
                b_smooth = reference_dt
        else:
            # Original behavior: target zero difference
            b_smooth = np.zeros(n_channels - 1)

        # Regularized inversion: min ||lamb*Diff*tau - lamb*taus||² + ||smoothness*(D*tau - b_smooth)||²
        M = np.vstack([lamb * Diff, smoothness * D])
        b = np.concatenate([lamb * taus, smoothness * b_smooth])
    else:
        # Original inversion without smoothness
        M = np.vstack([lamb * Diff])
        b = np.concatenate([lamb * taus])

    tau, _, _, _ = np.linalg.lstsq(M, b, rcond=None)

    #moving average to get smooth tau
    tau = moving_avg(tau, avg_win)
    # median filter to get smooth tau
    if pad:
        # #removing pad
        tau = tau[:-pad_size]
    tau = np.array([range(tau.shape[0]), tau]).T
    return tau




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



def ultra_mccc_iterative(das_arr, pick, corr_len, max_shift, lamb, n_iterations=3, shrinked_window_length= 300, medfilt_iterations=[1, 2, 3], smoothness=0.0, reference_dt=None, pre_mccc_mask_half_width=None):
    """
    Iterative MCCC with optional smoothness regularization.

    Parameters
    ----------
    smoothness : float
        Smoothness regularization parameter for MCCC. Higher values enforce
        smoother shifts between adjacent channels. Default: 0.0 (no smoothness).
        Typical values: 1.0-10.0 for enforcing physical constraints.
    reference_dt : ndarray, optional
        Expected time difference between adjacent channels from theoretical curve.
        Shape: (n_channels-1,). If provided, MCCC will constrain the solution to
        follow this moveout pattern, preventing cycle skipping.
    pre_mccc_mask_half_width : int, optional
        If provided, apply aggressive masking after initial shift but before MCCC.
        Only keeps center +/- pre_mccc_mask_half_width samples, zeros out the rest.
        Example: pre_mccc_mask_half_width=100 keeps only 100 samples on each side of center.
    """
    half_win_len = shrinked_window_length//2
    # Initial setup: calculate base_time and perform the initial shift operation
    # base_time = int(das_arr.shape[1] * 5 / 6)
    base_time = int(das_arr.shape[1] / 2)

    print('Starting initial shift')
    shifted_arr, fitted_arv, first_shifts = shift_arr(das_arr, pick, base_time=base_time)
    # Select the region of interest (e.g., 150 samples around base_time)
    shifted_arr = shifted_arr[:, base_time - half_win_len: base_time + half_win_len]

    # Apply aggressive pre-MCCC masking if requested
    if pre_mccc_mask_half_width is not None:
        center = shifted_arr.shape[1] // 2
        mask_start = center - pre_mccc_mask_half_width
        mask_end = center + pre_mccc_mask_half_width
        # Zero out everything outside the mask window
        masked_arr = np.zeros_like(shifted_arr)
        masked_arr[:, mask_start:mask_end] = shifted_arr[:, mask_start:mask_end]
        shifted_arr = masked_arr
        print(f'  Applied pre-MCCC mask: center +/- {pre_mccc_mask_half_width} samples (keeping {mask_start}:{mask_end})')

    # Set up the initial array for the iterative process
    current_arr = shifted_arr
    tau_list = []  # List to store tau values from each iteration
    intermediate_results = [das_arr, shifted_arr]  # Store initial results (original array and first shift result)

    # Perform iterative MCCC inversion
    for i in range(1, n_iterations + 1):
        corr_scale = i
        print(f'Starting MCCC iteration {i}')
        tau = MCCC(current_arr, corr_len//corr_scale, max_shift//corr_scale, lamb, smoothness=smoothness, reference_dt=reference_dt)
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


