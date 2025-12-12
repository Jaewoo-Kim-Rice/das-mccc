"""
Travel time calculation module for theoretical phase picking.

This module provides functions to calculate theoretical P and S wave travel times
using a 1D velocity model and convert them to sample indices for DAS data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union
from pathlib import Path

try:
    from pyrocko import cake
except ImportError:
    cake = None

from pyproj import Proj


def latlon_to_utm(lat: float, lon: float, zone: int = 12) -> Tuple[float, float]:
    """
    Convert latitude/longitude to UTM coordinates.

    Parameters
    ----------
    lat : float
        Latitude in degrees
    lon : float
        Longitude in degrees
    zone : int
        UTM zone (default: 12 for Utah FORGE)

    Returns
    -------
    easting : float
        UTM easting in meters
    northing : float
        UTM northing in meters
    """
    proj = Proj(proj='utm', zone=zone, ellps='WGS84')
    easting, northing = proj(lon, lat)
    return easting, northing


def calculate_travel_times(
    event_lat: float,
    event_lon: float,
    event_depth: float,
    channel_coords: np.ndarray,
    model_path: str,
    utm_zone: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate theoretical P and S travel times for all channels.

    Parameters
    ----------
    event_lat : float
        Event latitude in degrees
    event_lon : float
        Event longitude in degrees
    event_depth : float
        Event depth in meters (positive downward)
    channel_coords : np.ndarray
        Channel coordinates array of shape (n_channels, 3) with (easting, northing, depth)
        Depth should be positive downward (e.g., 1500m below surface = 1500)
    model_path : str
        Path to 1D velocity model file (.nd format for PyRocko cake)
    utm_zone : int
        UTM zone for coordinate conversion (default: 12)

    Returns
    -------
    travel_times_p : np.ndarray
        P-wave travel times in seconds for each channel
    travel_times_s : np.ndarray
        S-wave travel times in seconds for each channel
    """
    if cake is None:
        raise ImportError("PyRocko is required for travel time calculation. "
                         "Install with: pip install pyrocko")

    # Convert event location to UTM
    event_x, event_y = latlon_to_utm(event_lat, event_lon, utm_zone)

    # Load velocity model
    model = cake.load_model(model_path)

    n_channels = len(channel_coords)
    travel_times_p = np.zeros(n_channels)
    travel_times_s = np.zeros(n_channels)

    # Calculate travel times for each channel
    for i in range(n_channels):
        rec_x, rec_y, rec_depth = channel_coords[i]

        # Ensure receiver depth is non-negative (PyRocko requires z >= 0)
        rec_depth = max(0.0, rec_depth)

        # Calculate horizontal distance
        dx = rec_x - event_x
        dy = rec_y - event_y
        distance = np.sqrt(dx**2 + dy**2)

        # P-wave travel time
        phases_p = cake.PhaseDef("p")
        rays_p = model.arrivals(
            phases=[phases_p],
            distances=[distance * cake.m2d],
            zstart=event_depth,
            zstop=rec_depth,
        )

        if rays_p:
            travel_times_p[i] = rays_p[0].t
        else:
            travel_times_p[i] = np.nan

        # S-wave travel time
        phases_s = cake.PhaseDef("s")
        rays_s = model.arrivals(
            phases=[phases_s],
            distances=[distance * cake.m2d],
            zstart=event_depth,
            zstop=rec_depth,
        )

        if rays_s:
            travel_times_s[i] = rays_s[0].t
        else:
            travel_times_s[i] = np.nan

    return travel_times_p, travel_times_s


def travel_times_to_samples(
    travel_times: np.ndarray,
    sampling_rate: float,
    origin_sample: int = 0
) -> np.ndarray:
    """
    Convert travel times in seconds to sample indices.

    Parameters
    ----------
    travel_times : np.ndarray
        Travel times in seconds
    sampling_rate : float
        Sampling rate in Hz
    origin_sample : int
        Sample index corresponding to origin time (default: 0)

    Returns
    -------
    sample_indices : np.ndarray
        Sample indices for each travel time (NaN values are interpolated)
    """
    # Convert to samples (keeping as float for now)
    samples_float = origin_sample + (travel_times * sampling_rate)

    # Handle NaN values by interpolation from neighboring valid values
    nan_mask = np.isnan(samples_float)
    if nan_mask.any():
        valid_mask = ~nan_mask
        if valid_mask.any():
            # Interpolate NaN values from valid neighbors
            indices = np.arange(len(samples_float))
            samples_float[nan_mask] = np.interp(
                indices[nan_mask],
                indices[valid_mask],
                samples_float[valid_mask]
            )

    sample_indices = samples_float.astype(int)
    return sample_indices


def get_theoretical_picks(
    event_lat: float,
    event_lon: float,
    event_depth: float,
    channel_coords: np.ndarray,
    model_path: str,
    sampling_rate: float = 1000.0,
    origin_sample: int = 1000,
    utm_zone: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get theoretical P and S pick times as sample indices.

    This is the main function to generate theoretical picks for MCCC initialization.

    Parameters
    ----------
    event_lat : float
        Event latitude in degrees
    event_lon : float
        Event longitude in degrees
    event_depth : float
        Event depth in meters (positive downward)
    channel_coords : np.ndarray
        Channel coordinates array of shape (n_channels, 3) with (easting, northing, depth)
    model_path : str
        Path to 1D velocity model file
    sampling_rate : float
        Sampling rate in Hz (default: 1000)
    origin_sample : int
        Sample index corresponding to origin time (default: 1000, i.e., 1s into the trace)
    utm_zone : int
        UTM zone for coordinate conversion (default: 12)

    Returns
    -------
    P_picks : np.ndarray
        P-wave pick sample indices, shape (n_channels, 2) with [channel_idx, sample_idx]
    S_picks : np.ndarray
        S-wave pick sample indices, shape (n_channels, 2) with [channel_idx, sample_idx]

    Examples
    --------
    >>> # Load event data
    >>> event = pd.read_csv('event_data.csv').iloc[0]
    >>>
    >>> # Get channel coordinates (easting, northing, depth)
    >>> coords = np.load('channel_coords.npy')
    >>>
    >>> # Calculate theoretical picks
    >>> P_picks, S_picks = get_theoretical_picks(
    ...     event_lat=event['lat'],
    ...     event_lon=event['lon'],
    ...     event_depth=event['depth[m_local]'],
    ...     channel_coords=coords,
    ...     model_path='FORGE_1d.nd',
    ...     sampling_rate=1000.0,
    ...     origin_sample=1000  # Origin at 1s into trace
    ... )
    """
    # Calculate travel times
    tt_p, tt_s = calculate_travel_times(
        event_lat, event_lon, event_depth,
        channel_coords, model_path, utm_zone
    )

    # Convert to sample indices
    samples_p = travel_times_to_samples(tt_p, sampling_rate, origin_sample)
    samples_s = travel_times_to_samples(tt_s, sampling_rate, origin_sample)

    # Format as pick arrays [channel_idx, sample_idx]
    n_channels = len(channel_coords)
    P_picks = np.column_stack([np.arange(n_channels), samples_p])
    S_picks = np.column_stack([np.arange(n_channels), samples_s])

    return P_picks, S_picks


def load_event_from_csv(csv_path: str) -> dict:
    """
    Load event information from CSV file.

    Parameters
    ----------
    csv_path : str
        Path to event CSV file

    Returns
    -------
    event_info : dict
        Dictionary with keys: 'lat', 'lon', 'depth', 'origin_time'
    """
    df = pd.read_csv(csv_path)
    event = df.iloc[0]

    return {
        'lat': event['lat'],
        'lon': event['lon'],
        'depth': event['depth[m_local]'],
        'origin_time': event['time(UTC)']
    }


# Convenience function for batch processing
def calculate_travel_times_batch(
    event_lat: float,
    event_lon: float,
    event_depth: float,
    channel_coords: np.ndarray,
    model_path: str,
    utm_zone: int = 12,
    n_jobs: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate travel times with optional parallel processing.

    For large numbers of channels, this can use parallel processing
    to speed up calculation.

    Parameters
    ----------
    event_lat, event_lon, event_depth : float
        Event location
    channel_coords : np.ndarray
        Channel coordinates
    model_path : str
        Path to velocity model
    utm_zone : int
        UTM zone
    n_jobs : int
        Number of parallel jobs (default: 1, no parallelism)

    Returns
    -------
    travel_times_p, travel_times_s : np.ndarray
        Travel times for P and S waves
    """
    if n_jobs == 1:
        return calculate_travel_times(
            event_lat, event_lon, event_depth,
            channel_coords, model_path, utm_zone
        )

    # Parallel implementation using joblib
    try:
        from joblib import Parallel, delayed
    except ImportError:
        print("joblib not available, using serial processing")
        return calculate_travel_times(
            event_lat, event_lon, event_depth,
            channel_coords, model_path, utm_zone
        )

    if cake is None:
        raise ImportError("PyRocko is required")

    event_x, event_y = latlon_to_utm(event_lat, event_lon, utm_zone)
    model = cake.load_model(model_path)

    def calc_single(i):
        rec_x, rec_y, rec_depth = channel_coords[i]
        dx = rec_x - event_x
        dy = rec_y - event_y
        distance = np.sqrt(dx**2 + dy**2)

        # P-wave
        phases_p = cake.PhaseDef("p")
        rays_p = model.arrivals(
            phases=[phases_p],
            distances=[distance * cake.m2d],
            zstart=event_depth,
            zstop=rec_depth,
        )
        tt_p = rays_p[0].t if rays_p else np.nan

        # S-wave
        phases_s = cake.PhaseDef("s")
        rays_s = model.arrivals(
            phases=[phases_s],
            distances=[distance * cake.m2d],
            zstart=event_depth,
            zstop=rec_depth,
        )
        tt_s = rays_s[0].t if rays_s else np.nan

        return tt_p, tt_s

    results = Parallel(n_jobs=n_jobs)(
        delayed(calc_single)(i) for i in range(len(channel_coords))
    )

    travel_times_p = np.array([r[0] for r in results])
    travel_times_s = np.array([r[1] for r in results])

    return travel_times_p, travel_times_s
