# Breath-Based Cancer Detection (E-Nose + ML)

A machine learning pipeline that mimics an **electronic nose (e-nose)** system for
non-invasive cancer screening from exhaled breath — inspired by recent research
using metal-oxide (MOX) gas sensor arrays and machine learning to classify
volatile organic compound (VOC) patterns associated with disease.

> **⚠️ Research / educational project — not a medical device.**
> This repo ships with a **synthetic data generator**, not real patient data.
> Public clinical breath-VOC datasets are scarce (most published studies use
> small, private cohorts), so the pipeline is built to run end-to-end on
> simulated sensor data and to be easy to point at real data once you have it
> (e.g. from an IRB-approved study or a public release). Nothing here is
> validated for, or intended for, clinical or diagnostic use.

## Why breath?

Cancer-related metabolic changes alter the mix of volatile organic compounds
(VOCs) exhaled in breath. An array of gas sensors (each with a different
chemical sensitivity) produces a *time-series response* per sensor when
exposed to a breath sample — rising as VOCs bind to the sensor surface and
falling as it recovers. The pattern across all sensors forms a "breath print"
that a classifier can learn to associate with health status.

This project simulates that pipeline:

```
raw sensor time-series (per breath sample)
        │
        ▼
 feature extraction (peak, AUC, slope, decay, ...)
        │
        ▼
 classical ML classifiers (LogReg / RandomForest / MLP)
        │
        ▼
 evaluation (accuracy, sensitivity, specificity, ROC-AUC, confusion matrix)
```

## Project structure

```
breath-cancer-detection/
├── src/
│   ├── data_simulation.py   # synthetic e-nose sensor dataset + noise augmentation
│   ├── features.py          # time-series -> feature vector extraction
│   ├── models.py            # model factory: LogReg, RandomForest, MLP
│   ├── train.py             # end-to-end training + cross-validation script
│   └── evaluate.py          # metrics + plotting utilities
├── data/                    # generated datasets land here (gitignored)
├── results/                 # metrics, plots, saved models land here
├── tests/
│   └── test_pipeline.py     # sanity tests for data + features + leakage-safe grouping
├── requirements.txt
└── README.md
```

## Quickstart

```bash
git clone <your-fork-url>
cd breath-cancer-detection
pip install -r requirements.txt

# Generate a synthetic dataset (simulates a small pilot cohort, e.g. 28 vs 18,
# then optionally augments it with Gaussian-noise samples as in real e-nose studies)
python -m src.data_simulation --n_healthy 28 --n_cancer 18 --augment_to 400

# Train and evaluate all models with 5-fold stratified cross-validation
python -m src.train --data data/breath_sensor_data.npz --folds 5
```

Outputs land in `results/`: a metrics table (`results/metrics.csv`), ROC
curves, and a confusion matrix per model.

Example output (synthetic data, 5-fold CV):

| model               | accuracy | sensitivity | specificity | roc_auc |
|---------------------|---------:|------------:|------------:|--------:|
| logistic_regression |    0.920 |       0.890 |       0.939 |   0.973 |
| random_forest        |    0.870 |       0.740 |       0.951 |   0.970 |
| mlp                  |    0.888 |       0.805 |       0.939 |   0.930 |

(Numbers will vary slightly by seed — these are intentionally in the same
ballpark as published e-nose lung-cancer studies rather than a suspicious
100%, since the synthetic classes are built to overlap and aren't trivially
separable.)

### A note on avoiding data leakage

Small pilot cohorts (like the 28-vs-18 sample cohort simulated here) are
often expanded via Gaussian-noise duplication, a real technique used in
published e-nose studies. But noisy copies of the *same* underlying sample
are highly correlated — if a copy ends up in the training fold and another
copy of the same sample ends up in the test fold, cross-validation scores
get inflated by leakage rather than real generalization (this is a common,
easy-to-miss bug and is worth knowing about if you build on this).
`data_simulation.py` tracks a `groups` array (which original sample each
row was derived from), and `train.py` automatically switches to
`StratifiedGroupKFold` whenever it detects augmented data, so every copy of
a given original sample always stays on the same side of a fold.

## The simulated sensor array

Following the design used in several published e-nose studies, the simulator
models **13 sensor channels** (12 metal-oxide semiconductor sensors + 1
chemiresistive alkane-sensitive sensor), each producing a response curve
over a ~5-minute exposure window. Class-conditional differences (amplitude,
rise/decay kinetics, baseline drift) are injected per channel so the
"cancer" and "healthy" classes are separable but noisy and overlapping —
similar to real biological data — rather than trivially separable.

**This is a simulation of plausible sensor behavior, not a model of real
cancer biology.** Swap `src/data_simulation.py` for a real data loader when
you have real sensor recordings; `src/features.py` and everything downstream
only need the same `(n_samples, n_sensors, n_timepoints)` array shape.

## Feature extraction

For each sensor channel's response curve, six kinetic/statistical features
are extracted: peak response, time-to-peak, initial rise slope, area under
the curve (AUC), decay half-life, and end-of-window baseline recovery. With
13 sensors this yields a 78-dimensional feature vector per breath sample,
which is what the classical ML models train on (mirroring how published
e-nose papers hand-engineer features before classification, rather than
learning directly from raw curves).

## Models

- **Logistic Regression** — linear baseline
- **Random Forest** — handles nonlinear feature interactions, gives feature
  importances (useful for seeing which sensors/features drive predictions)
- **MLP (multi-layer perceptron)** — the approach most published breath-VOC
  studies actually use

All models are evaluated with stratified k-fold cross-validation and report
accuracy, sensitivity (recall on the cancer class), specificity, and
ROC-AUC — the same metrics reported in the clinical breath-analysis
literature, so results are easy to compare against published numbers.

## Extending this project

- Swap in a real dataset (see `src/data_simulation.py` for the expected shape)
- Add more sensor channels or VOC-specific features
- Try a 1D-CNN directly on raw time series instead of hand-engineered features
- Add SHAP/permutation importance for interpretability
- Package the trained model behind a small FastAPI inference endpoint

## Disclaimer

This is a learning/portfolio project demonstrating an ML pipeline for a
research-relevant problem. It is **not** a diagnostic tool, has **not** been
validated on real patients, and should not be used to make any health
decisions. Real deployment of breath-based diagnostics requires clinical
trials, regulatory approval, and rigorous external validation.

## License

MIT — see [LICENSE](LICENSE).
