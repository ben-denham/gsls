import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from typing import cast

from pyquantification.quantifiers.gsls import (
    sum_ind_var_pmfs,
    build_histogram,
    GslsQuantifier,
    TrueWeightGslsQuantifier,
)


def test_sum_ind_var_pmfs() -> None:
    # 100% prob of zero in one array.
    assert_array_equal(
        sum_ind_var_pmfs(np.array([0.2, 0.3, 0.5]), np.array([1.0])),
        np.array([0.2, 0.3, 0.5]),
    )
    assert_array_equal(
        sum_ind_var_pmfs(np.array([1.0]), np.array([0.2, 0.3, 0.5])),
        np.array([0.2, 0.3, 0.5]),
    )
    # Different array lengths.
    assert_allclose(
        sum_ind_var_pmfs(np.array([0.2, 0.3, 0.5]), np.array([0.1, 0.1, 0.2, 0.2, 0.4])),
        np.array([0.02, 0.05, 0.12, 0.15, 0.24, 0.22, 0.2]),
    )
    assert_allclose(
        sum_ind_var_pmfs(np.array([0.1, 0.1, 0.2, 0.2, 0.4]), np.array([0.2, 0.3, 0.5])),
        np.array([0.02, 0.05, 0.12, 0.15, 0.24, 0.22, 0.2]),
    )


def test_build_histogram() -> None:
    # Empty probs.
    assert_array_equal(
        build_histogram(bin_edges=np.array([0.0, 0.5, 1.0]),
                        class_probs=np.array([])),
        np.array([0.0, 0.0])
    )
    # Typical example
    assert_array_equal(
        build_histogram(bin_edges=np.array([0.0, 0.5, 1.0]),
                        class_probs=np.array([0.0, 0.1, 0.6, 0.5, 0.1, 0.8, 1.0, 1.0])),
        np.array([0.375, 0.625])
    )


def test_decompose_mixture() -> None:
    hist1, weight1 = GslsQuantifier.decompose_mixture(
        mixture_hist=np.array([0.2, 0.3, 0.5]),
        component_hist=np.array([0.2, 0.3, 0.5]),
    )
    assert_array_equal(hist1, np.array([0.0, 0.0, 0.0]))
    assert weight1 == 0
    hist2, weight2 = GslsQuantifier.decompose_mixture(
        mixture_hist=np.array([0.2, 0.3, 0.5]),
        component_hist=np.array([0.0, 0.0, 1.0]),
    )
    assert_array_equal(hist2, np.array([0.4, 0.6, 0.0]))
    assert weight2 == 0.5
    hist3, weight3 = GslsQuantifier.decompose_mixture(
        mixture_hist=np.array([0.6, 0.4, 0.0]),
        component_hist=np.array([0.4, 0.5, 0.1]),
    )
    assert_array_equal(hist3, np.array([0.6, 0.4, 0.0]))
    assert weight3 == 1.0


def test_get_gsls_interval() -> None:
    target_probs = np.array([0.2] * 100)
    interval1 = GslsQuantifier.get_gsls_interval(
        loss_weight=0.0,
        gain_weight=0.0,
        target_class_probs=target_probs,
        class_count=3,
        interval_mass=0.8,
    )
    assert interval1.prediction == 20.0
    assert interval1.lower == 15
    assert interval1.upper == 25
    interval2 = GslsQuantifier.get_gsls_interval(
        loss_weight=0.2,
        gain_weight=0.6,
        target_class_probs=target_probs,
        class_count=3,
        interval_mass=0.8,
    )
    assert interval2.prediction == 13.0
    assert interval2.lower == 9
    assert interval2.upper == 48
    interval3 = GslsQuantifier.get_gsls_interval(
        loss_weight=1.0,
        gain_weight=0.6,
        target_class_probs=target_probs,
        class_count=3,
        interval_mass=0.8,
    )
    assert interval3.prediction == 0.0
    assert interval3.lower == 5
    assert interval3.upper == 68
    interval4 = GslsQuantifier.get_gsls_interval(
        loss_weight=0.6,
        gain_weight=1.0,
        target_class_probs=target_probs,
        class_count=3,
        interval_mass=0.8,
    )
    assert interval4.prediction == 0.0
    assert interval4.lower == 5
    assert interval4.upper == 68


def test_get_auto_hist_bins() -> None:
    assert GslsQuantifier.get_auto_hist_bins(calib_count=100, target_count=10) == 5.0
    assert GslsQuantifier.get_auto_hist_bins(calib_count=10, target_count=100) == 5.0
    assert GslsQuantifier.get_auto_hist_bins(calib_count=100, target_count=100) == 13.0


def test_gsls_quantify() -> None:
    rng = np.random.RandomState(1)
    calib_probs = np.clip(np.array([
        # Columns
        rng.normal(size=1000, loc=0.2, scale=0.1),
        rng.normal(size=1000, loc=0.3, scale=0.1),
        rng.normal(size=1000, loc=0.5, scale=0.1),
    ]), 0, 1).T
    calib_probs = calib_probs / calib_probs.sum(axis=1)[:, np.newaxis]
    target_probs = np.clip(np.array([
        # Columns
        rng.normal(size=100, loc=0.3, scale=0.2),
        rng.normal(size=100, loc=0.5, scale=0.2),
        rng.normal(size=100, loc=0.2, scale=0.2),
    ]), 0, 1).T
    target_probs = target_probs / target_probs.sum(axis=1)[:, np.newaxis]
    intervals = GslsQuantifier.quantify(
        classes=np.array(['a', 'b', 'c']),
        target_probs=cast(np.ndarray, target_probs),
        prediction_interval_mass=0.8,
        calib_probs=calib_probs,
        bin_count='auto',
        true_weights={'loss': 0.5, 'gain': 0.5},
        random_state=2,
    )
    assert intervals['a'].prediction == 11.0
    assert intervals['a'].lower == 12
    assert intervals['a'].upper == 59
    assert intervals['a'].stats['loss_weight'] == 0.0
    assert intervals['a'].stats['gain_weight'] == 0.74
    assert intervals['a'].stats['bins'] == 13
    assert intervals['b'].prediction == 22.0
    assert intervals['b'].lower == 19
    assert intervals['b'].upper == 65
    assert intervals['b'].stats['loss_weight'] == 0.26
    assert intervals['b'].stats['gain_weight'] == 0.71
    assert intervals['b'].stats['bins'] == 13
    assert intervals['c'].prediction == 3.0
    assert intervals['c'].lower == 6
    assert intervals['c'].upper == 65
    assert intervals['c'].stats['loss_weight'] == 0.46
    assert intervals['c'].stats['gain_weight'] == 0.93
    assert intervals['c'].stats['bins'] == 13


def test_true_weight_gsls_quantify() -> None:
    rng = np.random.RandomState(1)
    calib_probs = np.clip(np.array([
        # Columns
        rng.normal(size=1000, loc=0.2, scale=0.1),
        rng.normal(size=1000, loc=0.3, scale=0.1),
        rng.normal(size=1000, loc=0.5, scale=0.1),
    ]), 0, 1).T
    calib_probs = calib_probs / calib_probs.sum(axis=1)[:, np.newaxis]
    target_probs = np.clip(np.array([
        # Columns
        rng.normal(size=100, loc=0.3, scale=0.2),
        rng.normal(size=100, loc=0.5, scale=0.2),
        rng.normal(size=100, loc=0.2, scale=0.2),
    ]), 0, 1).T
    target_probs = target_probs / target_probs.sum(axis=1)[:, np.newaxis]
    intervals = TrueWeightGslsQuantifier.quantify(
        classes=np.array(['a', 'b', 'c']),
        target_probs=cast(np.ndarray, target_probs),
        prediction_interval_mass=0.8,
        calib_probs=calib_probs,
        bin_count='auto',
        true_weights={'loss': 0.5, 'gain': 0.5},
        random_state=2,
    )
    assert intervals['a'].prediction == 34.0
    assert intervals['a'].lower == 16
    assert intervals['a'].upper == 57
    assert intervals['a'].stats['loss_weight'] == 0.5
    assert intervals['a'].stats['gain_weight'] == 0.5
    assert intervals['a'].stats['bins'] == 13
    assert intervals['b'].prediction == 49.0
    assert intervals['b'].lower == 26
    assert intervals['b'].upper == 70
    assert intervals['b'].stats['loss_weight'] == 0.5
    assert intervals['b'].stats['gain_weight'] == 0.5
    assert intervals['b'].stats['bins'] == 13
    assert intervals['c'].prediction == 19.0
    assert intervals['c'].lower == 10
    assert intervals['c'].upper == 45
    assert intervals['c'].stats['loss_weight'] == 0.5
    assert intervals['c'].stats['gain_weight'] == 0.5
    assert intervals['c'].stats['bins'] == 13
