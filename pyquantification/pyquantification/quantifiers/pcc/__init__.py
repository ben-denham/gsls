import numpy as np
import scipy.stats
from typing import Dict

from pyquantification.quantifiers.pcc.poibin import PoiBin  # type: ignore
from pyquantification.quantifiers.base import (
    Quantifier, Classes, Probs, PredictionInterval, interval_for_cdf)


def get_poibin_interval(probs: Probs, mass: float) -> PredictionInterval:
    """Given the probabilities of events for a Poisson binomial
    distribution, return a PredictionInterval for the desired
    probability mass (in terms of the proportion of events).

    NOTE: It is possible for the mean (sum(probs)/len(probs)) to be
    outside this equal-tailed interval - only the median is guaranteed
    to be inside. However, we use the mean as the prediction, because
    it minimises absolute error."""
    mean = np.sum(probs).item()  # Use .item() to get scalar.
    xs = range(len(probs) + 1)
    try:
        cdf = PoiBin(probs).cdf(xs)
    except TypeError as ex:
        if 'pmf / xi values have to be real.' in str(ex):
            # In the rare case that PoiBin fails (usually due to
            # extremely high or low probabilities), fall back to a
            # Gaussian approximation.
            variance = np.sum((1 - probs) * probs)
            pdf = scipy.stats.norm.pdf(xs, loc=mean, scale=np.sqrt(variance))
            # Gaussian cdf extends beyond 0-1 range, so we compute the
            # cdf from a normalised pdf.
            cdf = np.cumsum(pdf / pdf.sum())
        else:
            raise ex
    interval = interval_for_cdf(cdf, mass)
    return PredictionInterval(prediction=mean,
                              lower=interval.lower,
                              upper=interval.upper)


class PccQuantifier(Quantifier):
    """Probabilistic classify-and-count (PCC), with Poisson binomial
    distribution (PB) prediction interval. Works under the assumption
    that class priors do not shift."""

    @classmethod
    def _quantify(cls, *,  # type: ignore[override]
                  classes: Classes,
                  target_probs: Probs,
                  prediction_interval_mass: float) -> Dict[str, PredictionInterval]:
        return {
            target_class: get_poibin_interval(
                probs=target_probs[:, class_index],
                mass=prediction_interval_mass,
            )
            for class_index, target_class in enumerate(classes)
        }
