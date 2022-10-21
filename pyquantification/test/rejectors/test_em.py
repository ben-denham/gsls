import numpy as np

from pyquantification.quantifiers.em import EmQuantifier
from pyquantification.rejectors.em import EmRejectorMixin


def test_get_class_intervals():
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

    quant_intervals = EmQuantifier.quantify(
        classes=classes,
        calib_probs=calib_probs,
        target_probs=target_probs,
        prediction_interval_mass=prediction_interval_mass,
    )

    quantification_method_results = {
        'em': {
            'class_intervals': {
                target_class: {
                    'source_prior': quant_intervals[target_class].stats['source_prior'],
                    'adjusted_prior': quant_intervals[target_class].stats['adjusted_prior'],
                }
                for target_class in classes
            },
        },
    }

    cache = EmRejectorMixin.build_cache(
        classes=classes,
        target_probs=target_probs,
        calib_probs=calib_probs,
        quantification_method_results=quantification_method_results,
    )
    rej_intervals = EmRejectorMixin.get_class_intervals(
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
