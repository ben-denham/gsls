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
    np.testing.assert_almost_equal(interval1.prediction, 20.0000030)
    assert interval1.lower == 15
    assert interval1.upper == 25
    interval2 = GslsQuantifier.get_gsls_interval(
        loss_weight=0.2,
        gain_weight=0.6,
        target_class_probs=target_probs,
        class_count=3,
        interval_mass=0.8,
    )
    np.testing.assert_almost_equal(interval2.prediction, 35.3905242)
    assert interval2.lower == 11
    assert interval2.upper == 60
    interval3 = GslsQuantifier.get_gsls_interval(
        loss_weight=1.0,
        gain_weight=0.6,
        target_class_probs=target_probs,
        class_count=3,
        interval_mass=0.8,
    )
    np.testing.assert_almost_equal(interval3.prediction, 50.0)
    assert interval3.lower == 10
    assert interval3.upper == 90
    interval4 = GslsQuantifier.get_gsls_interval(
        loss_weight=0.6,
        gain_weight=1.0,
        target_class_probs=target_probs,
        class_count=3,
        interval_mass=0.8,
    )
    np.testing.assert_almost_equal(interval4.prediction, 50.0)
    assert interval4.lower == 10
    assert interval4.upper == 90


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
    np.testing.assert_almost_equal(intervals['a'].prediction, 45.7154284)
    assert intervals['a'].lower == 16
    assert intervals['a'].upper == 76
    assert intervals['a'].stats['loss_weight'] == 0.0
    assert intervals['a'].stats['gain_weight'] == 0.74
    assert intervals['a'].stats['bins'] == 13
    np.testing.assert_array_almost_equal(
        intervals['a'].stats['bin_edges'],
        [0.0, 0.070382, 0.117044, 0.142233, 0.163727, 0.17854,
         0.198875, 0.211378, 0.228507, 0.244196, 0.261838, 0.284858,
         0.317592, 1.0],
    )
    np.testing.assert_array_almost_equal(
        intervals['a'].stats['target_hist'],
        [0.05, 0.02, 0.04, 0.07, 0.02, 0.02, 0.02, 0.03, 0.05, 0.02, 0.04, 0.13, 0.49],
    )
    np.testing.assert_almost_equal(np.sum(intervals['a'].stats['target_hist']), 1.0)
    np.testing.assert_array_almost_equal(
        intervals['a'].stats['gain_hist'],
        [4.049978e-02, 3.492320e-06, 2.699161e-02, 6.751311e-02,
         1.062428e-04, 1.923804e-04, 3.594886e-04, 1.348139e-02,
         4.049557e-02, 4.686214e-18, 2.699109e-02, 1.485553e-01,
         6.348105e-01],
    )
    np.testing.assert_almost_equal(np.sum(intervals['a'].stats['gain_hist']), 1.0)

    np.testing.assert_almost_equal(intervals['b'].prediction, 49.6123272)
    assert intervals['b'].lower == 21
    assert intervals['b'].upper == 78
    assert intervals['b'].stats['loss_weight'] == 0.26
    assert intervals['b'].stats['gain_weight'] == 0.71
    assert intervals['b'].stats['bins'] == 13
    np.testing.assert_array_almost_equal(
        intervals['b'].stats['bin_edges'],
        [0.0, 0.173559, 0.212064, 0.24176, 0.258613, 0.276012,
         0.290162, 0.304967, 0.321542, 0.344741, 0.363878, 0.386814,
         0.421558, 1.0],
    )
    np.testing.assert_array_almost_equal(
        intervals['b'].stats['target_hist'],
        [0.02, 0.0, 0.03, 0.0, 0.01, 0.03, 0.04, 0.03, 0.02, 0.06, 0.05, 0.09, 0.62],
    )
    np.testing.assert_almost_equal(np.sum(intervals['b'].stats['target_hist']), 1.0)
    np.testing.assert_array_almost_equal(
        intervals['b'].stats['gain_hist'],
        [0.0, 0.0, 4.146989e-07, 0.0,
         1.979034e-06, 1.840066e-08, 1.462502e-02, 4.618260e-07,
         2.768588e-07, 4.223080e-02, 2.815500e-02, 8.446052e-02,
         8.305255e-01],
    )
    np.testing.assert_almost_equal(np.sum(intervals['b'].stats['gain_hist']), 1.0)

    np.testing.assert_almost_equal(intervals['c'].prediction, 47.6131842)
    assert intervals['c'].lower == 10
    assert intervals['c'].upper == 85
    assert intervals['c'].stats['loss_weight'] == 0.46
    assert intervals['c'].stats['gain_weight'] == 0.93
    assert intervals['c'].stats['bins'] == 13
    np.testing.assert_array_almost_equal(
        intervals['c'].stats['bin_edges'],
        [0.0, 0.37487, 0.409594, 0.437714, 0.455062, 0.473455,
         0.48891, 0.506762, 0.521039, 0.538711, 0.560365, 0.587568,
         0.635713, 1.0],
    )
    np.testing.assert_array_almost_equal(
        intervals['c'].stats['target_hist'],
        [0.89, 0.04, 0.02, 0.01, 0.0, 0.01, 0.0, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_almost_equal(np.sum(intervals['c'].stats['target_hist']), 1.0)
    np.testing.assert_array_almost_equal(
        intervals['c'].stats['gain_hist'],
        [9.462338e-01, 3.225803e-02, 1.075315e-02, 6.413827e-08,
         0.000000e+00, 2.068965e-06, 0.000000e+00, 1.075287e-02,
         0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
         0.000000e+00],
    )
    np.testing.assert_almost_equal(np.sum(intervals['c'].stats['gain_hist']), 1.0)


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
    np.testing.assert_almost_equal(intervals['a'].prediction, 41.7604394)
    assert intervals['a'].lower == 18
    assert intervals['a'].upper == 66
    assert intervals['a'].stats['loss_weight'] == 0.5
    assert intervals['a'].stats['gain_weight'] == 0.5
    assert intervals['a'].stats['bins'] == 13
    np.testing.assert_array_almost_equal(
        intervals['a'].stats['bin_edges'],
        [0.0, 0.070382, 0.117044, 0.142233, 0.163727, 0.17854,
         0.198875, 0.211378, 0.228507, 0.244196, 0.261838, 0.284858,
         0.317592, 1.0],
    )
    np.testing.assert_array_almost_equal(
        intervals['a'].stats['target_hist'],
        [0.05, 0.02, 0.04, 0.07, 0.02, 0.02, 0.02, 0.03, 0.05, 0.02, 0.04, 0.13, 0.49],
    )
    np.testing.assert_almost_equal(np.sum(intervals['a'].stats['target_hist']), 1.0)
    np.testing.assert_array_almost_equal(
        intervals['a'].stats['gain_hist'],
        [4.049978e-02, 3.492320e-06, 2.699161e-02, 6.751311e-02,
         1.062428e-04, 1.923804e-04, 3.594886e-04, 1.348139e-02,
         4.049557e-02, 4.686214e-18, 2.699109e-02, 1.485553e-01,
         6.348105e-01],
    )
    np.testing.assert_almost_equal(np.sum(intervals['a'].stats['gain_hist']), 1.0)

    np.testing.assert_almost_equal(intervals['b'].prediction, 49.5029836)
    assert intervals['b'].lower == 22
    assert intervals['b'].upper == 77
    assert intervals['b'].stats['loss_weight'] == 0.5
    assert intervals['b'].stats['gain_weight'] == 0.5
    assert intervals['b'].stats['bins'] == 13
    np.testing.assert_array_almost_equal(
        intervals['b'].stats['bin_edges'],
        [0.0, 0.173559, 0.212064, 0.24176, 0.258613, 0.276012,
         0.290162, 0.304967, 0.321542, 0.344741, 0.363878, 0.386814,
         0.421558, 1.0],
    )
    np.testing.assert_array_almost_equal(
        intervals['b'].stats['target_hist'],
        [0.02, 0.0, 0.03, 0.0, 0.01, 0.03, 0.04, 0.03, 0.02, 0.06, 0.05, 0.09, 0.62],
    )
    np.testing.assert_almost_equal(np.sum(intervals['b'].stats['target_hist']), 1.0)
    np.testing.assert_array_almost_equal(
        intervals['b'].stats['gain_hist'],
        [0.0, 0.0, 4.146989e-07, 0.0,
         1.979034e-06, 1.840066e-08, 1.462502e-02, 4.618260e-07,
         2.768588e-07, 4.223080e-02, 2.815500e-02, 8.446052e-02,
         8.305255e-01],
    )
    np.testing.assert_almost_equal(np.sum(intervals['b'].stats['gain_hist']), 1.0)

    np.testing.assert_almost_equal(intervals['c'].prediction, 33.736578)
    assert intervals['c'].lower == 12
    assert intervals['c'].upper == 55
    assert intervals['c'].stats['loss_weight'] == 0.5
    assert intervals['c'].stats['gain_weight'] == 0.5
    assert intervals['c'].stats['bins'] == 13
    np.testing.assert_array_almost_equal(
        intervals['c'].stats['bin_edges'],
        [0.0, 0.37487, 0.409594, 0.437714, 0.455062, 0.473455,
         0.48891, 0.506762, 0.521039, 0.538711, 0.560365, 0.587568,
         0.635713, 1.0],
    )
    np.testing.assert_array_almost_equal(
        intervals['c'].stats['target_hist'],
        [0.89, 0.04, 0.02, 0.01, 0.0, 0.01, 0.0, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_almost_equal(np.sum(intervals['c'].stats['target_hist']), 1.0)
    np.testing.assert_array_almost_equal(
        intervals['c'].stats['gain_hist'],
        [9.462338e-01, 3.225803e-02, 1.075315e-02, 6.413827e-08,
         0.000000e+00, 2.068965e-06, 0.000000e+00, 1.075287e-02,
         0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
         0.000000e+00],
    )
    np.testing.assert_almost_equal(np.sum(intervals['c'].stats['gain_hist']), 1.0)
