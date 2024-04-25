"""Utility statistics functions."""

from collections.abc import Callable
from typing import Any

import numpy as np
import scipy as sc
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

from statsplotly.constants import DEFAULT_KDE_BANDWIDTH


def compute_ssquares(y: NDArray[Any], yhat: NDArray[Any]) -> tuple[float, float, float]:
    """Evaluates sum of squares of a least-square regression."""

    ybar = np.sum(y) / len(y)  # Mean of the observed data

    TSS = np.sum((y - ybar) ** 2)  # Total sum of squares
    ESS = np.sum((yhat - ybar) ** 2)  # Explained sum of squares
    RSS = np.sum((y - yhat) ** 2)  # Residual sum of squares

    return TSS, ESS, RSS


def inverse_func(x: NDArray[Any], a: float, b: float) -> NDArray[Any]:
    """The reciprocal function"""
    return b + a / x


def affine_func(x: NDArray[Any], a: float, b: float) -> NDArray[Any]:
    """The affine function"""
    return a * x + b


def logarithmic_func(x: NDArray[Any], a: float, b: float) -> float:
    """The logarithmic function"""
    return b + a * np.log(x)


# Regression functions
def regress(
    x: NDArray[Any],
    y: NDArray[Any],
    func: Callable[[NDArray[Any], float, float], NDArray[Any]],
    p0: float | None = None,
    maxfev: int = 1000,
) -> tuple[NDArray[Any], float, tuple[NDArray[Any], NDArray[Any]]]:
    """Regresses y on x using the curve fit method from Scipy."""
    p, _ = curve_fit(func, x, y, p0, maxfev=maxfev)
    tss, ess, rss = compute_ssquares(y, func(x, *p))
    # Compute R2
    r2 = 1 - (rss / tss)

    # Draw a curve
    x_grid = np.linspace(x.min(), x.max(), 100)
    y_fit = func(x_grid, *p)

    return p, r2, (x_grid, y_fit)


def exponential_regress(
    x: NDArray[Any], y: NDArray[Any]
) -> tuple[NDArray[Any], float, tuple[NDArray[Any], NDArray[Any]]]:
    """Exponential regression via linear regression of the logarithm"""
    # For fitting y = AeBx, take the logarithm of both side gives
    # log y = log A + Bx. So fit (log y) against x
    # We weigh the points by the square root of their magnitude
    p = np.polyfit(x, np.log(y), 1, w=np.sqrt(y))

    # Fit the data and get sum of squares
    # y ≈ exp(p[1]) * exp(p[0] * x)
    tss, ess, rss = compute_ssquares(y, np.exp(p[1]) * np.exp(p[0] * x))
    # Compute R2
    r2 = 1 - (rss / tss)

    # Draw a curve
    x_grid = np.linspace(x.min(), x.max(), 100)
    y_fit = np.exp(p[1]) * np.exp(p[0] * x_grid)

    return p, r2, (x_grid, y_fit)


# KDE utilities functions
def kde_1d(x_data: NDArray[Any], x_grid: NDArray[Any]) -> NDArray[Any]:
    try:
        kde = gaussian_kde(x_data[~np.isnan(x_data)], bw_method=DEFAULT_KDE_BANDWIDTH)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Not enough values to compute KDE") from exc

    return kde.evaluate(x_grid)


def kde_2d(
    x_data: NDArray[Any],
    y_data: NDArray[Any],
    x_grid: NDArray[Any],
    y_grid: NDArray[Any],
) -> NDArray[Any]:
    data = np.vstack([x_data[~np.isnan(x_data)], y_data[~np.isnan(y_data)]])
    kde = gaussian_kde(data, bw_method=DEFAULT_KDE_BANDWIDTH)

    x_grid_mesh, y_grid_mesh = np.meshgrid(x_grid, y_grid)
    z = kde.evaluate(np.vstack([x_grid_mesh.ravel(), y_grid_mesh.ravel()]))

    return z.reshape(x_grid_mesh.shape)


def sem(data: NDArray[Any], confidence_level: float = 0.95) -> float:
    """Returns the margin of error at the given confidence level."""
    confidence_level = confidence_level / 2  # Converts to 2-tail

    # Anonymous function based on the Inverse of the complementary error function erfc.
    # see https://www.mathworks.com/help/matlab/ref/erfcinv.html?searchHighlight=erfcinv&s_tid=doc_srchtitle
    def my_norm_inv(confidence_level: float) -> float:
        return -np.sqrt(2) * sc.special.erfcinv(2 * confidence_level)

    zscore_ci = abs(
        my_norm_inv(confidence_level)
    )  # This is the same as doing: abs(sc.stats.norm.ppf(ci, 0, 1))

    return np.std(data) / np.sqrt(len(data)) * zscore_ci


def get_iqr(x: NDArray[Any]) -> NDArray[Any]:
    """Returns inter-quartile range."""
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return iqr


def range_normalize(data: NDArray[Any], a: float, b: float) -> NDArray[Any]:
    """Normalizes an array between a and b (min and max) values."""
    if min(data) == max(data):
        return np.clip(data, a, b)
    return (b - a) * (data - min(data)) / (max(data) - min(data)) + a


def reject_outliers(data: NDArray[Any], m: float = 2.0) -> NDArray[Any]:
    """Uses distance from the median of a distribution to remove outliers.
    (from https://stackoverflow.com/a/45399188/4696032)
    Returns the masks of non outliers.
    """

    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d / (mdev if mdev else 1.0)

    return s < m
