from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import inspect
import numpy as np
from typing import Any, Dict, Tuple

Classes = np.ndarray
Probs = np.ndarray
Priors = np.ndarray


@dataclass
class Interval():
    lower: float
    upper: float


@dataclass
class PredictionInterval():
    """A prediction interval should return predictions and bounds in the
    scale of instance counts (not a proportion)."""
    prediction: float
    lower: float
    upper: float
    stats: dict = field(default_factory=dict)


class QuantificationError(ValueError):
    pass


class Quantifier(ABC):
    """Contains all computations related to a particular quantification
    method."""

    @classmethod
    def quantify(cls, *,
                 classes: Classes,
                 target_probs: Probs,
                 **kwargs: Any) -> Dict[str, PredictionInterval]:
        """Given the list of classes, and at least a target_probs array (row
        per instance, col per class), return a dict mapping classes
        to PredictionIntervals
        """
        if target_probs.shape[0] == 0:
            # Return per-class quantifications of zero when there are
            # no instances to quantify.
            return {
                target_class: PredictionInterval(prediction=0, upper=0, lower=0)
                for target_class in classes
            }
        # Limit kwargs to those supported by the particular quantifier.
        class_quantify_params = set(inspect.signature(cls._quantify).parameters.keys())
        return cls._quantify(classes=classes, target_probs=target_probs, **{
            key: value for key, value in kwargs.items()
            if key in class_quantify_params
        })

    @classmethod
    @abstractmethod
    def _quantify(cls, *,
                  classes: Classes,
                  target_probs: Probs,
                  **kwargs: Any) -> Dict[str, PredictionInterval]:
        """Inner class-specific quantify()."""
        pass


def find_index_in_sorted(xs: np.ndarray, v: float) -> Tuple[int, int]:
    """Given an array of sorted values xs, return a pair of indexes for
    the maximum value <= v, and the minimum value >= v (NOTE: if
    max_lte = -1, then v is less than all xs, and if min_gte = len(xs)
    then v is greater than all xs).
    """
    # Cast to int for type checking.
    max_lte = int(np.searchsorted(xs, v, side='right')) - 1
    min_gte = int(np.searchsorted(xs, v, side='left'))
    return max_lte, min_gte


def interval_for_cdf(cdf: np.ndarray, mass: float) -> Interval:
    """Given a cdf array, return an equal-tailed credible interval for the
    given probability mass. Given the discrete nature of the cdf
    array, the interval will be at least as wide as the given
    probability mass, and may be wider."""
    lower_alpha = (1 - mass) / 2
    upper_alpha = (1 + mass) / 2
    wide_lower_idx, _ = find_index_in_sorted(cdf, lower_alpha)
    _, wide_upper_idx = find_index_in_sorted(cdf, upper_alpha)
    return Interval(
        # wide_lower_idx is the maximum point with probability sum <=
        # lower_alpha. Therefore it and every point below do not need
        # to be in the interval, so add one to get the first point in
        # the interval. If cdf[0] > lower_alpha, then wide_lower_idx =
        # -1, and we start at index 0.
        lower=(wide_lower_idx + 1),
        # wide_upper_idx is the minimum point with probability sum >=
        # upper_alpha, and therefore represents the last point of the
        # interval. In the highly unlikely scenario cdf[-1] <
        # upper_alpha, wide_upper_idx will be len(cdf), so we cap it
        # to the max index.
        upper=np.minimum(wide_upper_idx, (len(cdf) - 1)),
    )
