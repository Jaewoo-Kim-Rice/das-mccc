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


def interpolate_pick_outliers(picks, smooth_window=101, threshold=8.0):
    """
    Detect and interpolate outlier picks by comparing to smoothed trend.

    This function detects "staircase" patterns (----___----) by:
    1. Computing a heavily smoothed version of picks (expected trend)
    2. Finding channels that deviate significantly from the smooth trend

    Parameters
    ----------
    picks : ndarray
        1D array of pick times for each channel.
    smooth_window : int
        Window size for median smoothing to compute expected trend.
        Should be large enough to smooth over staircase gaps (default: 101).
    threshold : float
        Maximum allowed deviation from smoothed trend (in samples).
        Picks deviating more than this are considered outliers.

    Returns
    -------
    interpolated_picks : ndarray
        Pick times with outliers replaced by interpolated values.
    outlier_mask : ndarray
        Boolean mask where True indicates an outlier that was interpolated.
    """
    picks = np.array(picks, dtype=float)
    n_channels = len(picks)

    # Step 1: Create smoothed trend using large median filter
    # This will smooth over staircase jumps
    smooth_trend = medfilt(picks, kernel_size=smooth_window)

    # Step 2: Find deviation from smooth trend
    deviation = np.abs(picks - smooth_trend)

    # Step 3: Mark outliers where deviation exceeds threshold
    outlier_mask = deviation > threshold

    # Also mark NaN values as outliers
    outlier_mask = outlier_mask | np.isnan(picks)

    # Step 4: Interpolate outliers using the smooth trend
    interpolated_picks = picks.copy()
    interpolated_picks[outlier_mask] = smooth_trend[outlier_mask]

    # For any remaining NaN (edges where medfilt might fail), use neighbor interp
    still_nan = np.isnan(interpolated_picks)
    if np.any(still_nan):
        valid_idx = np.where(~still_nan)[0]
        nan_idx = np.where(still_nan)[0]
        if len(valid_idx) > 1:
            interpolated_picks[nan_idx] = np.interp(
                nan_idx, valid_idx, interpolated_picks[valid_idx]
            )

    return interpolated_picks, outlier_mask


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
    interpolate_outliers=True,
    outlier_smooth_window=101,
    outlier_threshold=8.0,
    use_global_window=True,
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

    # Masking strategy: global window vs per-channel windows
    valid_wins_mask = ~((wins[:, 0] == 0) & (wins[:, 1] == 0))

    if use_global_window and np.any(valid_wins_mask):
        # Use global window range instead of per-channel windows
        # This prevents wins=[0,0] channels from being completely corrupted
        global_win_min = int(np.min(wins[valid_wins_mask, 0]))
        global_win_max = int(np.max(wins[valid_wins_mask, 1]))
        global_win_width = global_win_max - global_win_min
        print(f"    Global window: [{global_win_min}, {global_win_max}] (width: {global_win_width})")
        print(f"    Per-channel wins range: [{wins[:,0].min():.0f}-{wins[:,0].max():.0f}] to [{wins[:,1].min():.0f}-{wins[:,1].max():.0f}]")
        # Create uniform wins array with global window for all channels
        uniform_wins = np.column_stack([
            np.full(len(wins), global_win_min),
            np.full(len(wins), global_win_max)
        ])
        filt_arr = mask_noise(arrs[-1], uniform_wins)
        masking_wins = uniform_wins  # For diagnostic
    else:
        # Use per-channel windows (original behavior)
        print(f"    Per-channel windows: wins range [{wins[:,0].min():.0f}-{wins[:,0].max():.0f}] to [{wins[:,1].min():.0f}-{wins[:,1].max():.0f}]")
        filt_arr = mask_noise(arrs[-1], wins)
        masking_wins = wins  # For diagnostic
    filt_arr = medfilt(filt_arr, kernel_size=medfilt_kernel_arr)

    # Secondary MCCC setup: use smoothed Ricker window centers
    dt = wins.mean(axis=1)
    dt = medfilt(dt, medfilt_kernel_dt)
    dt = pd.Series(dt).fillna(pd.Series(dt).rolling(
        window=rolling_window,
        center=True,
        min_periods=rolling_window // 10
    ).mean())
    center_pick = np.vstack((np.arange(len(dt)), dt.values)).T
    secondary_win = shrinked_window_length // 2

    effective_max_shift = min(max_shift_secondary, 10)  # Reduced from 20 to 10
    print(f"    Secondary MCCC: window={secondary_win}, max_shift={effective_max_shift}")
    filt_arrs, filt_total_shift, (_, filt_base_time), _ = ultra_mccc_iterative(
        filt_arr,
        center_pick,
        shrinked_window_length=secondary_win,
        corr_len=corr_len_secondary,
        max_shift=effective_max_shift,
        lamb=1,
        n_iterations=n_iter_secondary,
        smoothness=smoothness
    )

    # Stage 5: Ricker correlation on aligned data (refine after secondary MCCC)
    # Use higher mmad_thresh (or disable) since global masking changes amplitude distribution
    dt2, signs2, amp2, snrs2, wins2 = diff_corr_ric(
        filt_arrs[-1],
        max_shift=ricker_max_shift // 2,  # Smaller search range since already aligned
        snr_thresh=snr_thresh,
        mmad_thresh=10.0,  # Relaxed MMAD filtering (3.5 is too strict after global masking)
        ricker_freq=ricker_freq
    )

    # Propagate Stage 2 invalid channels to Stage 5
    # Channels that were filtered in Stage 2 should remain invalid
    stage2_invalid_mask = ~valid_wins_mask  # Channels with wins=[0,0] in Stage 2
    signs2[stage2_invalid_mask] = np.nan
    amp2[stage2_invalid_mask] = np.nan
    wins2[stage2_invalid_mask] = np.array([0, 0])

    n_zero_wins = np.sum((wins2[:, 0] == 0) & (wins2[:, 1] == 0))
    n_low_snr = np.sum(np.array(snrs2) < snr_thresh)
    n_nan_signs = np.sum(np.isnan(signs2))
    n_stage2_filtered = np.sum(stage2_invalid_mask)
    print(f"    Stage 5 Ricker: wins=[0,0]={n_zero_wins} (stage2={n_stage2_filtered}), low_snr={n_low_snr}, nan_signs={n_nan_signs}")

    # Use Stage 5 Ricker results for sign extraction
    signs = np.nan_to_num(signs2, nan=0).astype(int)
    amps = np.nan_to_num(amp2, nan=0)
    signs = medfilt(signs, median_filter_signs)
    signs = signs * sign_filter_by_amp(amps)

    # Final P arrival picks
    final_P_arrivals = base_time - (total_shift + filt_total_shift)

    # Detect and exclude outlier picks
    n_outliers = 0
    if interpolate_outliers:
        _, outlier_mask = interpolate_pick_outliers(
            final_P_arrivals,
            smooth_window=outlier_smooth_window,
            threshold=outlier_threshold
        )
        n_outliers = np.sum(outlier_mask)
        if n_outliers > 0:
            # Set signs to 0 for outlier channels (exclude from analysis)
            signs[outlier_mask] = 0
            print(f"    Excluded {n_outliers} outlier picks ({100*n_outliers/len(final_P_arrivals):.1f}%)")

    # ── data for plot ──
    arrivals_plot = np.where(np.abs(signs) == 1, final_P_arrivals, np.nan) - 2
    left_overlay   = (shrinked_window_length // 4) - 50 * signs
    plot_payload = {
        "filt_last":     filt_arrs[-1],   # Left imshow background (used with T)
        "left_overlay":  left_overlay,    # Left red line
        "arrivals_plot": arrivals_plot,   # Right red line
    }

    # ── diagnostic data for MCCC stages ──
    diagnostic_payload = {
        "stage1_after_mccc1": arrs[-1],           # After initial MCCC
        "stage2_wins": wins,                       # Ricker correlation windows (per-channel)
        "stage3_after_mask": filt_arr,             # After mask_noise + medfilt
        "stage3_masking_wins": masking_wins,       # Actual windows used for masking (global or per-channel)
        "stage4_after_mccc2": filt_arrs[-1],       # After secondary MCCC
        "stage5_wins": wins2,                      # Ricker windows after secondary MCCC
        "total_shift": total_shift,                # Accumulated shifts from MCCC1
        "filt_total_shift": filt_total_shift,      # Shifts from MCCC2
        "final_picks": final_P_arrivals,           # Final pick times
        "signs": signs,                            # Polarity signs
        "base_time": base_time,
        "half_win_len": half_win_len,
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

    return amps, signs, final_P_arrivals, plot_payload, diagnostic_payload



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



