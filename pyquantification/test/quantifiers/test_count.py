import numpy as np

from pyquantification.quantifiers.base import PredictionInterval
from pyquantification.quantifiers.count import CountQuantifier


def test_CountQuantifier() -> None:
    assert CountQuantifier.quantify(
        classes=np.array(['a', 'b', 'c']),
        target_probs=np.array([]),
    ) == {
        'a': PredictionInterval(0, 0, 0),
        'b': PredictionInterval(0, 0, 0),
        'c': PredictionInterval(0, 0, 0),
    }
    assert CountQuantifier.quantify(
        classes=np.array(['a', 'b', 'c']),
        target_probs=np.array([
            [0.1, 0.4, 0.5],
            [0.1, 0.5, 0.4],
            [0.0, 0.5, 0.5],
        ]),
    ) == {
        'a': PredictionInterval(0, 0, 0),
        'b': PredictionInterval(2, 2, 2),
        'c': PredictionInterval(1, 1, 1),
    }
