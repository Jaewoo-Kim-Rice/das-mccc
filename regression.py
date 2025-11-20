"""
Regression utilities for DAS pick smoothing.

This module provides RANSAC-based polynomial regression
for robust fitting of seismic phase picks.
"""

import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


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
        Array of [X, Y_pred], where Y_pred is the model's prediction for each X in `data`.
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
