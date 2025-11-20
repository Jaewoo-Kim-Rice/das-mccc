"""
Signal processing utilities for DAS data.

This module provides low-level signal processing functions
including wavelets, statistics, and cross-correlation.
"""

import numpy as np
from scipy.stats import truncnorm


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
