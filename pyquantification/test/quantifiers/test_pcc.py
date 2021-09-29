import numpy as np
from typing import cast
import pytest

from pyquantification.quantifiers.base import PredictionInterval
from pyquantification.quantifiers.pcc import PccQuantifier


def test_PccQuantifier() -> None:
    rng = np.random.RandomState(1)
    assert PccQuantifier.quantify(
        classes=np.array(['a', 'b', 'c']),
        target_probs=np.array([]),
    ) == {
        'a': PredictionInterval(0, 0, 0),
        'b': PredictionInterval(0, 0, 0),
        'c': PredictionInterval(0, 0, 0),
    }

    low_var_probs = np.clip(np.array([
        # Columns
        rng.normal(size=1000, loc=0.2, scale=0.1),
        rng.normal(size=1000, loc=0.3, scale=0.1),
        rng.normal(size=1000, loc=0.5, scale=0.1),
    ]), 0, 1).T
    low_var_probs = low_var_probs / low_var_probs.sum(axis=1)[:, np.newaxis]
    low_var_intervals = PccQuantifier.quantify(
        classes=np.array(['a', 'b', 'c']),
        target_probs=cast(np.ndarray, low_var_probs),
        prediction_interval_mass=0.8,
    )
    assert low_var_intervals['a'].prediction == pytest.approx(200.46, 0.01)
    assert low_var_intervals['a'].lower == 185
    assert low_var_intervals['a'].upper == 216
    assert low_var_intervals['b'].prediction == pytest.approx(298.85, 0.01)
    assert low_var_intervals['b'].lower == 281
    assert low_var_intervals['b'].upper == 317
    assert low_var_intervals['c'].prediction == pytest.approx(500.68, 0.01)
    assert low_var_intervals['c'].lower == 481
    assert low_var_intervals['c'].upper == 521

    high_var_probs = np.clip(np.array([
        # Columns
        rng.normal(size=100, loc=0.2, scale=0.2),
        rng.normal(size=100, loc=0.3, scale=0.2),
        rng.normal(size=100, loc=0.5, scale=0.2),
    ]), 0, 1).T
    high_var_probs = high_var_probs / high_var_probs.sum(axis=1)[:, np.newaxis]
    high_var_intervals = PccQuantifier.quantify(
        classes=np.array(['a', 'b', 'c']),
        target_probs=cast(np.ndarray, high_var_probs),
        prediction_interval_mass=0.8,
    )
    assert high_var_intervals['a'].prediction == pytest.approx(25.03, 0.01)
    assert high_var_intervals['a'].lower == 20
    assert high_var_intervals['a'].upper == 30
    assert high_var_intervals['b'].prediction == pytest.approx(30.20, 0.01)
    assert high_var_intervals['b'].lower == 25
    assert high_var_intervals['b'].upper == 36
    assert high_var_intervals['c'].prediction == pytest.approx(44.76, 0.01)
    assert high_var_intervals['c'].lower == 39
    assert high_var_intervals['c'].upper == 51
