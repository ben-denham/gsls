import numpy as np
from numpy.testing import assert_array_equal
from pyquantification.experiments import normalise_probs


def test_normalise_probs() -> None:
    assert_array_equal(
        normalise_probs(np.array([
            [1, 2, 3, 4],
            [1, 0, 0, 0],
            [1, 1e-16, 1e-16, 1e-16],
        ])),
        np.array([
            [1/10, 2/10, 3/10, 4/10],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ]),
    )
