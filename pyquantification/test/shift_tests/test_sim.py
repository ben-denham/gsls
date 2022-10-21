import numpy as np

from pyquantification.shift_tests.sim import (
    WpaShiftTester,
    CdtShiftTester,
)


def test_get_sim_priors():
    observed_adjusted_priors = np.array([0.5, 0.3, 0.2])
    for shift_tester in [WpaShiftTester, CdtShiftTester]:
        sim_priors = shift_tester.get_sim_priors(
            observed_adjusted_priors=observed_adjusted_priors,
            rng=np.random.RandomState(0),
        )
        assert np.all([
            (priors.shape == observed_adjusted_priors.shape) and abs(np.sum(priors) - 1) < 0.00001
            for priors in sim_priors
        ])
