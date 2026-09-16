import numpy as np
import pytest

from src.data_simulation import N_SENSORS, N_TIMEPOINTS, augment_with_noise, generate_dataset
from src.features import extract_features, FEATURE_NAMES
from src.models import get_models


def test_generate_dataset_shapes():
    X, y = generate_dataset(n_healthy=5, n_cancer=3, seed=0)
    assert X.shape == (8, N_SENSORS, N_TIMEPOINTS)
    assert y.shape == (8,)
    assert set(np.unique(y)) == {0, 1}
    assert (y == 0).sum() == 5
    assert (y == 1).sum() == 3


def test_generate_dataset_no_nans():
    X, _ = generate_dataset(n_healthy=4, n_cancer=4, seed=1)
    assert np.isfinite(X).all()


def test_augment_with_noise_expands_dataset():
    X, y = generate_dataset(n_healthy=5, n_cancer=3, seed=0)
    X_aug, y_aug, groups = augment_with_noise(X, y, target_n=50, seed=0)
    assert X_aug.shape[0] == 50
    assert y_aug.shape[0] == 50
    assert groups.shape[0] == 50
    # every group id must point back to one of the 8 original samples
    assert set(np.unique(groups)) <= set(range(8))
    # class balance roughly preserved (within reason, since duplication is random)
    assert 0 < (y_aug == 1).sum() < 50


def test_augment_with_noise_no_shrink():
    X, y = generate_dataset(n_healthy=5, n_cancer=3, seed=0)
    X_aug, y_aug, groups = augment_with_noise(X, y, target_n=4, seed=0)
    # target smaller than current size -> dataset unchanged
    assert X_aug.shape[0] == 8
    assert groups.shape[0] == 8


def test_augment_groups_prevent_leakage_when_used_correctly():
    """Same original sample's noisy copies must share a group id, so a
    group-aware CV splitter can keep them on the same side of a fold."""
    X, y = generate_dataset(n_healthy=5, n_cancer=3, seed=0)
    _, _, groups = augment_with_noise(X, y, target_n=40, seed=0)
    # at least one original sample should have been duplicated
    _, counts = np.unique(groups, return_counts=True)
    assert counts.max() > 1


def test_extract_features_shape():
    X, _ = generate_dataset(n_healthy=3, n_cancer=3, seed=2)
    feats = extract_features(X)
    assert feats.shape == (6, N_SENSORS * len(FEATURE_NAMES))
    assert np.isfinite(feats).all()


def test_models_fit_predict_on_toy_data():
    X, y = generate_dataset(n_healthy=10, n_cancer=10, seed=3)
    feats = extract_features(X)
    models = get_models(random_state=3)

    for name, pipeline in models.items():
        pipeline.fit(feats, y)
        preds = pipeline.predict(feats)
        proba = pipeline.predict_proba(feats)
        assert preds.shape == (20,)
        assert proba.shape == (20, 2)


@pytest.mark.parametrize("n_healthy,n_cancer", [(1, 1), (10, 2)])
def test_generate_dataset_various_sizes(n_healthy, n_cancer):
    X, y = generate_dataset(n_healthy=n_healthy, n_cancer=n_cancer, seed=0)
    assert X.shape[0] == n_healthy + n_cancer
