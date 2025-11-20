"""
Masking utilities for DAS data.

This module provides functions for masking noisy channels
and data points in DAS arrays.
"""

import numpy as np
from scipy.signal import medfilt


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
        
