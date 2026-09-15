"""
Filtering utilities for DAS data.

This module provides temporal filtering functions
for smoothing and cleaning DAS signals.
"""

import numpy as np
import pandas as pd


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


def sign_filter_by_amp(arr, thresh_percentile = 10):
    thresh = np.percentile(arr, thresh_percentile)
    mask = np.ones_like(arr)
    mask[arr<thresh] = 0 
    return mask
    
    

