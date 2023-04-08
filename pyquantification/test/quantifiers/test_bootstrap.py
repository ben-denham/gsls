import numpy as np
import pandas as pd
from typing import cast
import pytest

from pyquantification.quantifiers.base import PredictionInterval
from pyquantification.quantifiers.bootstrap import EmBootstrapQuantifier


def test_EmBootstrapQuantifier() -> None:
    rng = np.random.RandomState(1)
    assert EmBootstrapQuantifier.quantify(
        classes=np.array(['a', 'b', 'c']),
        target_probs=np.array([]),
    ) == {
        'a': PredictionInterval(0, 0, 0),
        'b': PredictionInterval(0, 0, 0),
        'c': PredictionInterval(0, 0, 0),
    }

    calib_probs = np.clip(np.array([
        # Columns
        rng.normal(size=1000, loc=0.2, scale=0.1),
        rng.normal(size=1000, loc=0.3, scale=0.1),
        rng.normal(size=1000, loc=0.5, scale=0.1),
    ]), 0, 1).T
    calib_y = pd.Series(np.concatenate([
        np.repeat('a', 333),
        np.repeat('b', 333),
        np.repeat('b', 334),
    ]))

    low_var_probs = np.clip(np.array([
        # Columns
        rng.normal(size=1000, loc=0.2, scale=0.1),
        rng.normal(size=1000, loc=0.3, scale=0.1),
        rng.normal(size=1000, loc=0.5, scale=0.1),
    ]), 0, 1).T
    low_var_probs = low_var_probs / low_var_probs.sum(axis=1)[:, np.newaxis]
    low_var_intervals = EmBootstrapQuantifier.quantify(
        classes=np.array(['a', 'b', 'c']),
        target_probs=cast(np.ndarray, low_var_probs),
        calib_probs=cast(np.ndarray, calib_probs),
        prediction_interval_mass=0.8,
        calib_y=calib_y,
        random_state=123,
    )
    assert low_var_intervals['a'].prediction == pytest.approx(70.81, 0.01)
    assert low_var_intervals['a'].lower == 6
    assert low_var_intervals['a'].upper == 158
    assert low_var_intervals['b'].prediction == pytest.approx(376.99, 0.01)
    assert low_var_intervals['b'].lower == 229
    assert low_var_intervals['b'].upper == 515
    assert low_var_intervals['c'].prediction == pytest.approx(552.20, 0.01)
    assert low_var_intervals['c'].lower == 403
    assert low_var_intervals['c'].upper == 702

    high_var_probs = np.clip(np.array([
        # Columns
        rng.normal(size=100, loc=0.2, scale=0.2),
        rng.normal(size=100, loc=0.3, scale=0.2),
        rng.normal(size=100, loc=0.5, scale=0.2),
    ]), 0, 1).T
    high_var_probs = high_var_probs / high_var_probs.sum(axis=1)[:, np.newaxis]
    high_var_intervals = EmBootstrapQuantifier.quantify(
        classes=np.array(['a', 'b', 'c']),
        target_probs=cast(np.ndarray, high_var_probs),
        calib_probs=cast(np.ndarray, calib_probs),
        prediction_interval_mass=0.8,
        calib_y=calib_y,
        random_state=123,
    )
    assert high_var_intervals['a'].prediction == pytest.approx(23.85, 0.01)
    assert high_var_intervals['a'].lower == 10
    assert high_var_intervals['a'].upper == 39
    assert high_var_intervals['b'].prediction == pytest.approx(10.74, 0.01)
    assert high_var_intervals['b'].lower == 0
    assert high_var_intervals['b'].upper == 24
    assert high_var_intervals['c'].prediction == pytest.approx(65.41, 0.01)
    assert high_var_intervals['c'].lower == 47
    assert high_var_intervals['c'].upper == 82
