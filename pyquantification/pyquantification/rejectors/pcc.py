from typing import Dict, Any, List

import cvxpy
import numpy as np
import scipy.stats

from pyquantification.quantifiers.base import PredictionInterval
from pyquantification.quantifiers.pcc import PccQuantifier
from .base import (Classes, Probs, Cache, ProbThresholdRejector,
                   ApproxProbThresholdRejector, MipRejector)


class PccRejectorMixin:
    """Mixin defining functions for rejection based on PCC quantification."""

    @classmethod
    def build_cache(
            cls, *,
            classes: Classes,
            target_probs: Probs,
            calib_probs: Probs,
            quantification_method_results: Dict[str, Dict[str, Any]],
    ) -> Cache:
        return {
            'classes': classes,
            'target_probs': target_probs,
        }

    @classmethod
    def get_class_intervals(
            cls, *,
            cache: Cache,
            prediction_interval_mass: float,
            selection_mask: np.ndarray,
            random_state: int,
    ) -> Dict[str, PredictionInterval]:
        # As the PCC interval only depends on the selected probs, we
        # can produce an interval by applying the PccQuantifier to the
        # selected probs.
        selected_target_probs = cache['target_probs'][selection_mask]
        return PccQuantifier.quantify(
            classes=cache['classes'],
            target_probs=selected_target_probs,
            prediction_interval_mass=prediction_interval_mass,
        )


# First parent class methods take precedence
class PccProbThresholdRejector(PccRejectorMixin, ProbThresholdRejector):
    pass


class ApproxPccRejectorMixin(PccRejectorMixin):
    """The more expensive build_cache() steps in this class are not needed
    for the base ProbThresholdRejector."""

    @classmethod
    def build_cache(
            cls, *,
            classes: Classes,
            target_probs: Probs,
            calib_probs: Probs,
            quantification_method_results: Dict[str, Dict[str, Any]],
    ) -> Cache:
        cache = super().build_cache(
            classes=classes,
            target_probs=target_probs,
            calib_probs=calib_probs,
            quantification_method_results=quantification_method_results,
        )
        return {
            **cache,
            # Z[i, j] = g_j(x_i) * (1 - g_j(x_i)), for instance i and class j
            'Z': target_probs * (1 - target_probs),
        }

    @classmethod
    def get_interval_width_for_variance(
            cls, *,
            variance: float,
            interval_mass: float,
    ) -> float:
        """The interval for PCC is determined by the Poisson Binomial
        Distribution (PBD), but that cannot be produced based on
        variance alone (it needs all prediction probabilities). We
        therefore use a Gaussian approximation, as is commonly done
        (see: Neammanee, K. (2005). A refinement of normal
        approximation to Poisson binomial. International Journal of
        Mathematics and Mathematical Sciences, 2005(5), 717-728.)

        We do not truncate the resulting interval to [0, 1] so that
        the interval shape/width does not depend on the mean
        probability or the number of instances used to produce the
        variance - resulting in equivalent width calculation for
        MipRejectors and ApproxProbThresholdRejectors.

        """
        # With no variance, interval is width zero.
        if variance == 0:
            return 0
        # With infinite variance, interval is infinite.
        if variance == np.inf:
            return np.inf
        low, high = scipy.stats.norm.interval(
            loc=0,
            scale=np.sqrt(variance),
            alpha=interval_mass,
        )
        return high - low

    @classmethod
    def get_class_count_variances(
            cls, *,
            cache: Cache,
            selection_mask: np.ndarray,
    ) -> np.ndarray:
        # The count_variance for each class j is the sum of that class's
        # column in Z for selected instances.
        # class_count_variances[j] = sum(selection_mask[i] * Z[i,j] for each instance i)
        class_count_variances = selection_mask @ cache['Z']
        assert class_count_variances.shape == cache['classes'].shape
        return class_count_variances

    @classmethod
    def get_cvxpy_constraints(
            cls, *,
            cache: Cache,
            selection_mask: cvxpy.Variable,
            class_count_variance_limits: np.ndarray,
    ) -> List[cvxpy.constraints.constraint.Constraint]:
        class_count_variances = selection_mask @ cache['Z']
        return [(class_count_variances <= class_count_variance_limits)]


# First parent class methods take precedence
class PccApproxProbThresholdRejector(ApproxPccRejectorMixin, ApproxProbThresholdRejector):
    pass


# First parent class methods take precedence
class PccMipRejector(ApproxPccRejectorMixin, MipRejector):
    pass
