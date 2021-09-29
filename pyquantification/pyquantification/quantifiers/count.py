import numpy as np
from typing import Dict, Any

from pyquantification.quantifiers.base import (
    Quantifier, Classes, Probs, PredictionInterval)


class CountQuantifier(Quantifier):
    """The classify-and-count quantification method. Works under the
    assumption of a perfect classifier."""

    @classmethod
    def _quantify(cls, *,
                  classes: Classes,
                  target_probs: Probs,
                  **_: Any) -> Dict[str, PredictionInterval]:
        quantifications = {}
        target_preds = classes[np.argmax(target_probs, axis=1)]
        for class_index, target_class in enumerate(classes):
            prediction = (target_preds == target_class).sum()
            quantifications[target_class] = PredictionInterval(
                prediction=prediction,
                upper=prediction,
                lower=prediction,
            )
        return quantifications
