"""
High-level workflows for DAS processing.

This module provides complete workflows for pick generation,
MCCC refinement, and masking operations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt

from .phasenet import get_picks, runPNDAS
from .regression import pick_regression_ransac
from .masking import mask_points_with_noise, mask_noise, masking_picks
from .mccc_core import ultra_mccc_iterative, diff_corr_ric, get_sign
from .filtering import sign_filter_by_amp


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
    smoothness=0.0,
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
        lamb=lamb_initial,
        smoothness=smoothness
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
        n_iterations=n_iter_secondary,
        smoothness=smoothness
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



