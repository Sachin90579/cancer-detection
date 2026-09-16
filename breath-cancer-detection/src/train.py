"""
End-to-end training script: loads a breath sensor dataset, extracts
features, trains each model in src.models with stratified k-fold cross
-validation, evaluates with clinical-style metrics, and saves plots +
a metrics table + the best fitted model.

Usage:
    python -m src.train --data data/breath_sensor_data.npz --folds 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
)

from src.evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curve
from src.features import extract_features
from src.models import get_models


def run(data_path: Path, folds: int, results_dir: Path, seed: int = 42):
    npz = np.load(data_path)
    X_raw, y = npz["X"], npz["y"]
    groups = npz["groups"] if "groups" in npz.files else np.arange(len(y))
    n_unique_groups = len(np.unique(groups))
    print(f"Loaded dataset: X={X_raw.shape}, y={y.shape} "
          f"({(y == 0).sum()} healthy, {(y == 1).sum()} cancer), "
          f"{n_unique_groups} unique source samples")

    X = extract_features(X_raw)
    print(f"Extracted features: X={X.shape}")

    if n_unique_groups < len(y):
        # Data was augmented (noise-duplicated). Use group-aware CV so all
        # noisy copies of the same original sample stay on the same side
        # of every fold -- otherwise near-duplicates leak between train
        # and test and inflate scores.
        print("Detected augmented data -> using StratifiedGroupKFold "
              "(grouped by original sample) to avoid leakage.")
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        cv_kwargs = {"groups": groups}
    else:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        cv_kwargs = {}

    models = get_models(random_state=seed)

    rows = []
    best_name, best_auc, best_model = None, -1.0, None

    for name, pipeline in models.items():
        y_pred = cross_val_predict(pipeline, X, y, cv=cv, method="predict", **cv_kwargs)
        y_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", **cv_kwargs)[:, 1]

        metrics = compute_metrics(y, y_pred, y_proba)
        metrics["model"] = name
        rows.append(metrics)

        plot_confusion_matrix(y, y_pred, name, results_dir)
        plot_roc_curve(y, y_proba, name, results_dir)

        print(f"[{name}] accuracy={metrics['accuracy']:.3f} "
              f"sensitivity={metrics['sensitivity']:.3f} "
              f"specificity={metrics['specificity']:.3f} "
              f"roc_auc={metrics['roc_auc']:.3f}")

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_name = name
            best_model = pipeline

    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(rows).set_index("model")
    metrics_df.to_csv(results_dir / "metrics.csv")
    print(f"\nSaved metrics table to {results_dir / 'metrics.csv'}")
    print(metrics_df[["accuracy", "sensitivity", "specificity", "roc_auc"]])

    # Refit the best model on the full dataset and save it
    best_model.fit(X, y)
    joblib.dump(best_model, results_dir / f"best_model_{best_name}.joblib")
    print(f"\nBest model by ROC-AUC: {best_name} (AUC={best_auc:.3f}) "
          f"-> saved to {results_dir / f'best_model_{best_name}.joblib'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="data/breath_sensor_data.npz")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(Path(args.data), args.folds, Path(args.results_dir), seed=args.seed)


if __name__ == "__main__":
    main()
