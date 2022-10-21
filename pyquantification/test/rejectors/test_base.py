import numpy as np
from typing import Dict
import pytest

from pyquantification.quantifiers.base import PredictionInterval
from pyquantification.rejectors.base import (
    Cache, Classes, IntervalRejector,
    ProbThresholdRejector, MipRejector,
)


class StubIntervalRejector(IntervalRejector):

    @classmethod
    def get_class_intervals(
            cls, *,
            cache: Cache,
            prediction_interval_mass: float,
            selection_mask: np.ndarray,
            random_state: int,
    ) -> Dict[str, PredictionInterval]:
        # The interval is derived from the class index.
        intervals = {}
        for class_num, target_class in enumerate(cache['classes'], start=2):
            prediction = (1 / class_num)
            interval_width = (1 / (class_num ** 2))
            intervals[target_class] = PredictionInterval(
                prediction=(prediction * cache['target_n']),
                lower=((prediction - (interval_width / 2)) * cache['target_n']),
                upper=((prediction + (interval_width / 2)) * cache['target_n']),
            )
        return intervals


def test_get_class_interval_width_limits() -> None:

    def test_rejection_limit(rejection_limit, expected_width_limits,
                             *, class_count=2):
        classes = np.array([str(i) for i in range(class_count)])
        target_probs = np.array([
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
        ])
        np.testing.assert_array_almost_equal(
            StubIntervalRejector.get_class_interval_width_limits(
                cache={'classes': classes, 'target_n': target_probs.shape[0]},
                classes=classes,
                target_probs=target_probs,
                prediction_interval_mass=0.8,
                random_state=0,
                rejection_limit=rejection_limit,
            ),
            expected_width_limits,
        )

    test_rejection_limit('abs:0.01', [0.01, 0.01])
    test_rejection_limit('abs:0.01', [0.01, 0.01, 0.01], class_count=3)
    test_rejection_limit('rel:0.5', [0.5 * (1 / 2), 0.5 * (1 / 3)])
    test_rejection_limit('frac:0.5', [0.5 * (1 / 2**2), 0.5 * (1 / 3**2)])
    with pytest.raises(ValueError):
        test_rejection_limit('unknown:0.5', None)
    with pytest.raises(ValueError):
        test_rejection_limit('unknown:asdf', None)
    with pytest.raises(ValueError):
        test_rejection_limit('unknown', None)


class StubProbThresholdRejector(ProbThresholdRejector):

    @classmethod
    def get_class_count_interval_widths(
            cls, *,
            cache: Cache,
            classes: Classes,
            prediction_interval_mass: float,
            selection_mask: np.ndarray,
            random_state: int,
    ) -> np.ndarray:
        # The interval width will be equal to the number of selected instances.
        return np.full(classes.shape, fill_value=np.sum(selection_mask))


def test_prob_threshold_get_rejected_mask() -> None:

    def test_interval_width_limits(limit, expected_rejected_mask):
        target_n = 10
        classes = np.array(['a', 'b'])
        # Probs will be more extreme at each end, rejection will start
        # in the middle.
        target_probs = np.array([
            [(i / (target_n - 1)), (1 - (i / (target_n - 1)))]
            for i in range(target_n)
        ])
        np.testing.assert_array_equal(
            StubProbThresholdRejector.get_rejected_mask(
                cache={},
                classes=classes,
                target_probs=target_probs,
                prediction_interval_mass=0.8,
                random_state=0,
                class_interval_width_limits=np.full(classes.shape,
                                                    fill_value=(limit / target_n)),
            ),
            np.array(expected_rejected_mask).astype(bool),
        )

    # Because the interval width will be equal to the number of
    # selected instances, the limit acts as the number of instances to
    # select.
    test_interval_width_limits(0, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    test_interval_width_limits(1, [0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    test_interval_width_limits(4, [0, 0, 1, 1, 1, 1, 1, 1, 0, 0])
    test_interval_width_limits(5, [0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
    test_interval_width_limits(6, [0, 0, 0, 1, 1, 1, 1, 0, 0, 0])
    test_interval_width_limits(9, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    test_interval_width_limits(10, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    with pytest.raises(AssertionError):
        test_interval_width_limits(11, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    with pytest.raises(AssertionError):
        test_interval_width_limits(-1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


class StubMipRejector(MipRejector):

    @classmethod
    def get_interval_width_for_variance(
            cls, *,
            variance: float,
            interval_mass: float,
    ) -> float:
        # The interval width will be equal to the variance
        return variance


def test_mip_find_variance_for_interval_width() -> None:

    def test_variance_finder(interval_width):
        np.testing.assert_almost_equal(
            StubMipRejector.find_variance_for_interval_width(
                interval_mass=0.8,
                interval_width=interval_width,
            ),
            # Variance should be the same as the interval_width with
            # the StubMipRejector
            interval_width,
        )

    test_variance_finder(0.0)
    test_variance_finder(0.5)
    test_variance_finder(1.0)
