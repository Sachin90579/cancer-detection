"""
Synthetic e-nose breath sensor data generator.

Simulates a 13-channel metal-oxide / chemiresistive sensor array recording
the response of each sensor over a fixed exposure window when presented
with a breath sample, for two classes: "healthy" and "cancer".

This is a *simulation* built to have realistic, noisy, overlapping class
structure (not trivially separable) so the downstream ML pipeline has
something non-trivial to learn from. It is NOT derived from real patient
data or real cancer biology, and must not be treated as such.

Each sensor's response curve follows a simple rise-then-recover shape:

    r(t) = baseline + amplitude * (1 - exp(-t / tau_rise))          for t <= t_peak
    r(t) = baseline + amplitude * exp(-(t - t_peak) / tau_decay)    for t >  t_peak

Class-conditional shifts are applied to a subset of channels (loosely
inspired by which sensors tend to respond to cancer-associated VOCs such
as certain aldehydes and alkanes in the real literature), with per-sample
biological noise and per-timepoint measurement noise layered on top.

Usage:
    python -m src.data_simulation --n_healthy 28 --n_cancer 18 --augment_to 400
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

N_SENSORS = 13          # 12 MOX sensors + 1 chemiresistive alkane sensor
N_TIMEPOINTS = 60        # samples across a 5-minute exposure window
T_PEAK_FRAC = 0.35        # exposure phase ends ~35% into the window

# Sensors that show a stronger class-conditional shift (indices are
# arbitrary but fixed, standing in for VOC-specific sensitive channels).
SHIFTED_SENSORS = [1, 3, 4, 7, 9, 11]


def _sensor_curve(
    rng: np.random.Generator,
    n_timepoints: int,
    t_peak_frac: float,
    baseline: float,
    amplitude: float,
    tau_rise: float,
    tau_decay: float,
    noise_std: float,
) -> np.ndarray:
    """Generate one rise-then-recover sensor response curve with noise."""
    t = np.linspace(0, 1, n_timepoints)
    t_peak = t_peak_frac
    curve = np.empty(n_timepoints)

    rise_mask = t <= t_peak
    decay_mask = ~rise_mask

    curve[rise_mask] = baseline + amplitude * (1 - np.exp(-t[rise_mask] / tau_rise))
    peak_val = baseline + amplitude * (1 - np.exp(-t_peak / tau_rise))
    curve[decay_mask] = baseline + (peak_val - baseline) * np.exp(
        -(t[decay_mask] - t_peak) / tau_decay
    )

    curve += rng.normal(0, noise_std, size=n_timepoints)
    return curve


def _generate_sample(rng: np.random.Generator, label: int) -> np.ndarray:
    """Generate one full breath sample: (n_sensors, n_timepoints)."""
    sample = np.empty((N_SENSORS, N_TIMEPOINTS))

    for s in range(N_SENSORS):
        baseline = rng.normal(0.05, 0.01)
        amplitude = rng.normal(1.0, 0.15)
        tau_rise = np.clip(rng.normal(0.15, 0.03), 0.03, 0.6)
        tau_decay = np.clip(rng.normal(0.25, 0.05), 0.05, 0.8)
        noise_std = 0.035

        if label == 1 and s in SHIFTED_SENSORS:
            # Cancer-associated VOC channels: higher amplitude, faster rise.
            # Kept modest on purpose so classes overlap and aren't trivially
            # separable, similar to real biological VOC data.
            amplitude *= rng.normal(1.22, 0.18)
            tau_rise *= rng.normal(0.85, 0.15)

        sample[s] = _sensor_curve(
            rng, N_TIMEPOINTS, T_PEAK_FRAC, baseline, amplitude,
            tau_rise, tau_decay, noise_std,
        )

    return sample


def generate_dataset(
    n_healthy: int, n_cancer: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a raw (unaugmented) synthetic dataset.

    Returns
    -------
    X : np.ndarray, shape (n_healthy + n_cancer, N_SENSORS, N_TIMEPOINTS)
    y : np.ndarray, shape (n_healthy + n_cancer,)  -- 0 = healthy, 1 = cancer
    """
    rng = np.random.default_rng(seed)
    X, y = [], []

    for _ in range(n_healthy):
        X.append(_generate_sample(rng, label=0))
        y.append(0)
    for _ in range(n_cancer):
        X.append(_generate_sample(rng, label=1))
        y.append(1)

    return np.array(X), np.array(y)


def augment_with_noise(
    X: np.ndarray, y: np.ndarray, target_n: int, noise_scale: float = 0.03,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand a small dataset by duplicating samples with added Gaussian
    noise, a technique used in published e-nose pilot studies to cope with
    small cohort sizes while preserving each sample's statistical shape.

    IMPORTANT: each augmented copy stays tagged with a `group` id equal to
    its original sample's index. Near-duplicate noisy copies of the same
    underlying sample are highly correlated, so if they end up split across
    train/test in cross-validation, scores are inflated by leakage rather
    than real generalization. Downstream training must use a *group-aware*
    CV splitter (e.g. StratifiedGroupKFold) keyed on `groups` so that all
    copies of a given original sample always land on the same side of a
    fold. See src/train.py.

    Returns
    -------
    X_aug, y_aug, groups : np.ndarray
        `groups[i]` gives the original-sample index that row i was derived
        from (original samples are their own group).
    """
    base_groups = np.arange(len(X))

    if target_n <= len(X):
        return X, y, base_groups

    rng = np.random.default_rng(seed)
    n_extra = target_n - len(X)
    idx = rng.integers(0, len(X), size=n_extra)

    extra_X = X[idx] + rng.normal(0, noise_scale, size=(n_extra, *X.shape[1:]))
    extra_y = y[idx]
    extra_groups = idx  # tag with the original sample's group id

    X_aug = np.concatenate([X, extra_X], axis=0)
    y_aug = np.concatenate([y, extra_y], axis=0)
    groups_aug = np.concatenate([base_groups, extra_groups], axis=0)

    # shuffle (all three arrays together, keeping row correspondence)
    perm = rng.permutation(len(X_aug))
    return X_aug[perm], y_aug[perm], groups_aug[perm]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_healthy", type=int, default=28,
                         help="Number of healthy samples to simulate (base cohort).")
    parser.add_argument("--n_cancer", type=int, default=18,
                         help="Number of cancer samples to simulate (base cohort).")
    parser.add_argument("--augment_to", type=int, default=None,
                         help="If set, augment the dataset up to this many total "
                              "samples using Gaussian-noise duplication.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/breath_sensor_data.npz")
    args = parser.parse_args()

    X, y = generate_dataset(args.n_healthy, args.n_cancer, seed=args.seed)
    print(f"Generated base dataset: X={X.shape}, y={y.shape} "
          f"({(y == 0).sum()} healthy, {(y == 1).sum()} cancer)")

    groups = np.arange(len(X))
    if args.augment_to:
        X, y, groups = augment_with_noise(X, y, args.augment_to, seed=args.seed)
        print(f"Augmented dataset to: X={X.shape}, y={y.shape} "
              f"({(y == 0).sum()} healthy, {(y == 1).sum()} cancer)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y, groups=groups)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
