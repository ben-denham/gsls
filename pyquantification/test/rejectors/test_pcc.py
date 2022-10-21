import numpy as np

from pyquantification.quantifiers.pcc import PccQuantifier
from pyquantification.rejectors.pcc import PccRejectorMixin


def test_get_class_intervals():
    """Ensure rejector intervals are equivalent to quantifier intervals
    for a full selection."""
    prediction_interval_mass = 0.8
    random_state = 0
    target_n = 100
    classes = np.array(['a', 'b', 'c'])
    target_probs = np.random.RandomState(42).dirichlet(np.ones(classes.shape), target_n)
    assert target_probs.shape == (target_n, classes.shape[0])
    # These arguments are not used for PCC
    calib_probs = None
    quantification_method_results = None

    quant_intervals = PccQuantifier.quantify(
        classes=classes,
        target_probs=target_probs,
        prediction_interval_mass=prediction_interval_mass,
    )
    cache = PccRejectorMixin.build_cache(
        classes=classes,
        target_probs=target_probs,
        calib_probs=calib_probs,
        quantification_method_results=quantification_method_results,
    )
    rej_intervals = PccRejectorMixin.get_class_intervals(
        cache=cache,
        prediction_interval_mass=prediction_interval_mass,
        selection_mask=np.full((target_n, ), fill_value=True),
        random_state=random_state,
    )

    for target_class in classes:
        q_interval = quant_intervals[target_class]
        r_interval = rej_intervals[target_class]
        assert q_interval.prediction == r_interval.prediction
        assert q_interval.upper == r_interval.upper
        assert q_interval.lower == r_interval.lower
