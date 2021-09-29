import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from typing import cast
import pytest

from pyquantification.quantifiers.base import PredictionInterval
from pyquantification.quantifiers.em import (
    normalise_prob_rows,
    adjust_to_priors,
    get_em_confidence_interval,
    EmQuantifier,
)


def test_normalise_prob_rows() -> None:
    assert_allclose(
        normalise_prob_rows(np.array([
            [1, 2, 3, 4],
            [0.4, 0.3, 0.2, 0.1],
            [1.0, 0.0, 0.0, 0.0],
        ])),
        np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
            [1.0, 0.0, 0.0, 0.0],
        ]),
    )


def test_adjust_to_priors() -> None:
    assert_array_equal(
        adjust_to_priors(
            source_priors=np.array([0.5, 0.5]),
            target_priors=np.array([0.8, 0.2]),
            probs=np.array([
                [0.5, 0.5],
                [1.0, 0.0],
            ]),
        ),
        np.array([
            [0.8, 0.2],
            [1.0, 0.0],
        ]),
    )


def test_get_em_confidence_interval() -> None:
    # Directly test edge cases
    assert get_em_confidence_interval(0.5, 0.8, np.array([]), 0.8) == PredictionInterval(0, 0, 0)
    assert get_em_confidence_interval(0.5, 0.8, np.array([0.5] * 100), 0.5) == PredictionInterval(80, 0, 100)


def test_EmQuantifier() -> None:
    rng = np.random.RandomState(1)
    assert EmQuantifier.quantify(
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

    low_var_probs = np.clip(np.array([
        # Columns
        rng.normal(size=1000, loc=0.2, scale=0.1),
        rng.normal(size=1000, loc=0.3, scale=0.1),
        rng.normal(size=1000, loc=0.5, scale=0.1),
    ]), 0, 1).T
    low_var_probs = low_var_probs / low_var_probs.sum(axis=1)[:, np.newaxis]
    low_var_intervals = EmQuantifier.quantify(
        classes=np.array(['a', 'b', 'c']),
        target_probs=cast(np.ndarray, low_var_probs),
        calib_probs=cast(np.ndarray, calib_probs),
        prediction_interval_mass=0.8,
    )
    assert low_var_intervals['a'].prediction == pytest.approx(56.82, 0.01)
    assert low_var_intervals['a'].lower == 17
    assert low_var_intervals['a'].upper == 139
    assert low_var_intervals['b'].prediction == pytest.approx(376.55, 0.01)
    assert low_var_intervals['b'].lower == 278
    assert low_var_intervals['b'].upper == 475
    assert low_var_intervals['c'].prediction == pytest.approx(566.63, 0.01)
    assert low_var_intervals['c'].lower == 452
    assert low_var_intervals['c'].upper == 681

    high_var_probs = np.clip(np.array([
        # Columns
        rng.normal(size=100, loc=0.2, scale=0.2),
        rng.normal(size=100, loc=0.3, scale=0.2),
        rng.normal(size=100, loc=0.5, scale=0.2),
    ]), 0, 1).T
    high_var_probs = high_var_probs / high_var_probs.sum(axis=1)[:, np.newaxis]
    high_var_intervals = EmQuantifier.quantify(
        classes=np.array(['a', 'b', 'c']),
        target_probs=cast(np.ndarray, high_var_probs),
        calib_probs=cast(np.ndarray, calib_probs),
        prediction_interval_mass=0.8,
    )
    assert high_var_intervals['a'].prediction == pytest.approx(24.26, 0.01)
    assert high_var_intervals['a'].lower == 11
    assert high_var_intervals['a'].upper == 38
    assert high_var_intervals['b'].prediction == pytest.approx(11.09, 0.01)
    assert high_var_intervals['b'].lower == 3
    assert high_var_intervals['b'].upper == 23
    assert high_var_intervals['c'].prediction == pytest.approx(64.64, 0.01)
    assert high_var_intervals['c'].lower == 47
    assert high_var_intervals['c'].upper == 82
