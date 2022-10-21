import numpy as np

from pyquantification.quantifiers.gsls import GslsQuantifier, build_histogram
from pyquantification.rejectors.gsls import LossDist, ApproxGslsRejectorMixin, StaticGainGslsMixin


class StaticGainApproxGslsRejectorMixin(StaticGainGslsMixin, ApproxGslsRejectorMixin):
    pass


def test_loss_dist():
    target_n = 100
    target_class_probs = np.linspace(0, 1.0, target_n)

    loss_dist = LossDist(target_class_probs=target_class_probs)

    # Key conditioned_loss_mean() points

    assert loss_dist.conditioned_loss_mean(q_s=0, loss_weight=0.01) == 0.0
    assert loss_dist.conditioned_loss_mean(q_s=0, loss_weight=0.5) == 0.0
    assert loss_dist.conditioned_loss_mean(q_s=0, loss_weight=1.0) == 0.0

    assert loss_dist.conditioned_loss_mean(q_s=0.5, loss_weight=0.01) == 0.5
    assert loss_dist.conditioned_loss_mean(q_s=0.5, loss_weight=0.5) == 0.5
    assert loss_dist.conditioned_loss_mean(q_s=0.5, loss_weight=1.0) == 0.5

    assert loss_dist.conditioned_loss_mean(q_s=1.0, loss_weight=0.01) == 1.0
    assert loss_dist.conditioned_loss_mean(q_s=1.0, loss_weight=0.5) == 1.0
    assert loss_dist.conditioned_loss_mean(q_s=1.0, loss_weight=1.0) == 1.0

    # Key conditioned_loss_variance() points

    assert loss_dist.conditioned_loss_variance(q_s=0, loss_weight=0.01) == 0.0
    assert loss_dist.conditioned_loss_variance(q_s=0, loss_weight=0.5) == 0.0
    assert loss_dist.conditioned_loss_variance(q_s=0, loss_weight=1.0) == 0.0

    assert loss_dist.conditioned_loss_variance(q_s=0.5, loss_weight=0.01) == 1 / 12
    assert loss_dist.conditioned_loss_variance(q_s=0.5, loss_weight=0.5) == 1 / 12
    assert loss_dist.conditioned_loss_variance(q_s=0.5, loss_weight=1.0) == 0.0

    assert loss_dist.conditioned_loss_variance(q_s=1.0, loss_weight=0.01) == 0.0
    assert loss_dist.conditioned_loss_variance(q_s=1.0, loss_weight=0.5) == 0.0
    assert loss_dist.conditioned_loss_variance(q_s=1.0, loss_weight=1.0) == 0.0

    # Key get_loss_variance_covariance() points

    assert loss_dist.get_loss_variance_covariance(loss_weight=0.0) == (0.0, 0.0)
    np.testing.assert_almost_equal(
        loss_dist.get_loss_variance_covariance(loss_weight=0.5),
        (0.074730252, 0.0016498),
    )
    np.testing.assert_almost_equal(
        loss_dist.get_loss_variance_covariance(loss_weight=1.0),
        (0.0016498316, 0.0016498316),
    )

    # Key get_class_loss_coefficient() points

    assert loss_dist.get_class_loss_coefficient(loss_weight=0.0) == 0.0
    np.testing.assert_almost_equal(
        loss_dist.get_class_loss_coefficient(loss_weight=0.5),
        0.0681309253951668,
    )
    np.testing.assert_almost_equal(
        loss_dist.get_class_loss_coefficient(loss_weight=1.0),
        -0.0016498316498316234,
    )


def test_loss_dist_zero_variance():
    target_n = 100

    loss_dist = LossDist(target_class_probs=np.zeros(target_n))
    # Key get_loss_variance_covariance() points
    assert loss_dist.get_loss_variance_covariance(loss_weight=0.0) == (0.0, 0.0)
    assert loss_dist.get_loss_variance_covariance(loss_weight=0.5) == (0.0, 0.0)
    assert loss_dist.get_loss_variance_covariance(loss_weight=1.0) == (0.0, 0.0)
    # Key get_class_loss_coefficient() points
    assert loss_dist.get_class_loss_coefficient(loss_weight=0.0) == 0.0
    assert loss_dist.get_class_loss_coefficient(loss_weight=0.5) == 0.0
    assert loss_dist.get_class_loss_coefficient(loss_weight=1.0) == 0.0

    loss_dist = LossDist(target_class_probs=np.ones(target_n))
    # Key get_loss_variance_covariance() points
    assert loss_dist.get_loss_variance_covariance(loss_weight=0.0) == (0.0, 0.0)
    assert loss_dist.get_loss_variance_covariance(loss_weight=0.5) == (0.0, 0.0)
    assert loss_dist.get_loss_variance_covariance(loss_weight=1.0) == (0.0, 0.0)
    # Key get_class_loss_coefficient() points
    assert loss_dist.get_class_loss_coefficient(loss_weight=0.0) == 0.0
    assert loss_dist.get_class_loss_coefficient(loss_weight=0.5) == 0.0
    assert loss_dist.get_class_loss_coefficient(loss_weight=1.0) == 0.0


def test_get_gain_bin_weights():
    target_n = 100
    target_class_probs = np.random.RandomState(0).uniform(size=target_n)
    bin_edges = np.linspace(0, 1, 6)
    target_hist = build_histogram(bin_edges=bin_edges, class_probs=target_class_probs)
    assert np.sum(target_hist) == 1.0
    gain_hist = np.array([0, 0.1, 0.2, 0.3, 0.4])
    assert np.sum(gain_hist) == 1.0
    # Select a gain_weight that allows the gain_hist to still fit under the target_hist
    gain_weight = np.min(
        np.divide(target_hist, gain_hist,
                  out=np.full(gain_hist.shape, fill_value=np.inf),
                  where=(gain_hist > 0.0))
    ) - 0.01
    gain_bin_weights = ApproxGslsRejectorMixin.get_gain_bin_weights(
        target_hist=target_hist,
        gain_weight=gain_weight,
        gain_hist=gain_hist,
    )
    assert gain_bin_weights.shape == gain_hist.shape
    # Ensure the target-weighted mean of the gain_bin_weights equals the gain_weight
    np.mean(gain_bin_weights * target_hist) == gain_weight


def test_gsls_get_class_intervals():
    """Ensure rejector intervals are equivalent to quantifier intervals
    for a full selection."""
    prediction_interval_mass = 0.8
    random_state = 0
    target_n = 100
    classes = np.array(['a', 'b', 'c'])
    calib_probs = np.random.RandomState(42).dirichlet(np.ones(classes.shape), target_n)
    assert calib_probs.shape == (target_n, classes.shape[0])
    target_probs = np.random.RandomState(43).dirichlet(np.linspace(1, 1.5, classes.shape[0]), target_n)
    assert target_probs.shape == (target_n, classes.shape[0])

    quant_intervals = GslsQuantifier.quantify(
        classes=classes,
        calib_probs=calib_probs,
        target_probs=target_probs,
        prediction_interval_mass=prediction_interval_mass,
        random_state=random_state,
        bin_count='auto',
        true_weights={'loss': 0, 'gain': 0},
    )

    quantification_method_results = {
        'gsls': {
            'class_intervals': {
                target_class: {
                    'bin_edges': quant_intervals[target_class].stats['bin_edges'],
                    'loss_weight': quant_intervals[target_class].stats['loss_weight'],
                    'gain_weight': quant_intervals[target_class].stats['gain_weight'],
                    'gain_hist': quant_intervals[target_class].stats['gain_hist'],
                    'target_hist': quant_intervals[target_class].stats['target_hist'],
                }
                for target_class in classes
            },
        },
    }

    cache = ApproxGslsRejectorMixin.build_cache(
        classes=classes,
        target_probs=target_probs,
        calib_probs=calib_probs,
        quantification_method_results=quantification_method_results,
    )
    rej_intervals = ApproxGslsRejectorMixin.get_class_intervals(
        cache=cache,
        prediction_interval_mass=prediction_interval_mass,
        selection_mask=np.full((target_n, ), fill_value=True),
        random_state=random_state,
    )

    for target_class in classes:
        q_interval = quant_intervals[target_class]
        r_interval = rej_intervals[target_class]
        np.testing.assert_almost_equal(q_interval.prediction, r_interval.prediction, decimal=2)
        assert q_interval.upper == r_interval.upper
        assert q_interval.lower == r_interval.lower


def test_static_gain_gsls_get_class_intervals():
    """Ensure rejector intervals are equivalent to quantifier intervals
    for a full selection."""
    prediction_interval_mass = 0.8
    random_state = 0
    target_n = 100
    classes = np.array(['a', 'b', 'c'])
    calib_probs = np.random.RandomState(42).dirichlet(np.ones(classes.shape), target_n)
    assert calib_probs.shape == (target_n, classes.shape[0])
    target_probs = np.random.RandomState(43).dirichlet(np.linspace(1, 1.5, classes.shape[0]), target_n)
    assert target_probs.shape == (target_n, classes.shape[0])

    quant_intervals = GslsQuantifier.quantify(
        classes=classes,
        calib_probs=calib_probs,
        target_probs=target_probs,
        prediction_interval_mass=prediction_interval_mass,
        random_state=random_state,
        bin_count='auto',
        true_weights={'loss': 0, 'gain': 0},
    )

    quantification_method_results = {
        'gsls': {
            'class_intervals': {
                target_class: {
                    'bin_edges': quant_intervals[target_class].stats['bin_edges'],
                    'loss_weight': quant_intervals[target_class].stats['loss_weight'],
                    'gain_weight': quant_intervals[target_class].stats['gain_weight'],
                    'gain_hist': quant_intervals[target_class].stats['gain_hist'],
                    'target_hist': quant_intervals[target_class].stats['target_hist'],
                }
                for target_class in classes
            },
        },
    }

    cache = StaticGainApproxGslsRejectorMixin.build_cache(
        classes=classes,
        target_probs=target_probs,
        calib_probs=calib_probs,
        quantification_method_results=quantification_method_results,
    )
    rej_intervals = StaticGainApproxGslsRejectorMixin.get_class_intervals(
        cache=cache,
        prediction_interval_mass=prediction_interval_mass,
        selection_mask=np.full((target_n, ), fill_value=True),
        random_state=random_state,
    )

    for target_class in classes:
        q_interval = quant_intervals[target_class]
        r_interval = rej_intervals[target_class]
        np.testing.assert_almost_equal(q_interval.prediction, r_interval.prediction, decimal=2)
        assert q_interval.upper == r_interval.upper
        assert q_interval.lower == r_interval.lower
