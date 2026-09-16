"""
Feature extraction: converts raw per-sensor time-series curves into a
fixed-length feature vector per breath sample, following the kind of
hand-engineered kinetic features used in published e-nose studies.

For each sensor channel, six features are computed:
    1. peak response        - max value reached
    2. time_to_peak          - fraction of the window at which peak occurs
    3. rise_slope            - average slope over the first 20% of the window
    4. auc                   - area under the curve (trapezoidal)
    5. decay_half_life       - time (fraction of window) for the signal to
                                fall to half its peak-above-baseline value
                                after the peak
    6. end_recovery          - value at the end of the window (how much the
                                sensor has recovered toward baseline)

With N_SENSORS channels this produces N_SENSORS * 6 features per sample.
"""
from __future__ import annotations

import numpy as np


FEATURE_NAMES = [
    "peak", "time_to_peak", "rise_slope", "auc", "decay_half_life", "end_recovery",
]


def _extract_single_curve(curve: np.ndarray) -> np.ndarray:
    n = len(curve)
    t = np.linspace(0, 1, n)

    baseline = curve[0]
    peak_idx = int(np.argmax(curve))
    peak_val = curve[peak_idx]
    time_to_peak = t[peak_idx]

    # rise slope over first 20% of the window
    cutoff = max(int(0.2 * n), 2)
    rise_slope = (curve[cutoff] - curve[0]) / (t[cutoff] - t[0] + 1e-8)

    auc = np.trapezoid(curve, t) if hasattr(np, "trapezoid") else np.trapz(curve, t)

    # decay half-life: time after peak for signal to drop to
    # baseline + 0.5*(peak - baseline)
    half_level = baseline + 0.5 * (peak_val - baseline)
    post_peak = curve[peak_idx:]
    below_half = np.where(post_peak <= half_level)[0]
    if len(below_half) > 0 and len(post_peak) > 1:
        decay_half_life = t[peak_idx + below_half[0]] - time_to_peak
    else:
        decay_half_life = t[-1] - time_to_peak  # never decayed to half within window

    end_recovery = curve[-1]

    return np.array([peak_val, time_to_peak, rise_slope, auc, decay_half_life, end_recovery])


def extract_features(X: np.ndarray) -> np.ndarray:
    """Extract features for a batch of samples.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_sensors, n_timepoints)

    Returns
    -------
    np.ndarray, shape (n_samples, n_sensors * len(FEATURE_NAMES))
    """
    n_samples, n_sensors, _ = X.shape
    features = np.empty((n_samples, n_sensors * len(FEATURE_NAMES)))

    for i in range(n_samples):
        row = []
        for s in range(n_sensors):
            row.append(_extract_single_curve(X[i, s]))
        features[i] = np.concatenate(row)

    return features


def feature_column_names(n_sensors: int) -> list[str]:
    return [f"sensor{s}_{name}" for s in range(n_sensors) for name in FEATURE_NAMES]
