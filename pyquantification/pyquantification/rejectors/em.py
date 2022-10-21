# coding: utf-8

from typing import Dict, Any, List

import cvxpy as cp
import numpy as np
import scipy.stats

from pyquantification.quantifiers.base import PredictionInterval
from pyquantification.quantifiers.em import adjust_to_priors
from .base import (Classes, Probs, Cache, ProbThresholdRejector,
                   ApproxProbThresholdRejector, MipRejector)


class EmRejectorMixin:
    """Mixin defining functions for rejection based on EM quantification."""

    @classmethod
    def build_cache(
            cls, *,
            classes: Classes,
            target_probs: Probs,
            calib_probs: Probs,
            quantification_method_results: Dict[str, Dict[str, Any]],
    ) -> Cache:
        source_priors = np.array([
            quantification_method_results['em']['class_intervals'][target_class]['source_prior']
            for target_class in classes
        ])
        adjusted_priors = np.array([
            quantification_method_results['em']['class_intervals'][target_class]['adjusted_prior']
            for target_class in classes
        ])
        adjusted_probs = adjust_to_priors(
            source_priors=source_priors,
            target_priors=adjusted_priors,
            probs=target_probs,
        )
        # Assertions for valid division.
        assert np.all(source_priors > 0)
        assert np.all(source_priors < 1)
        # pos_row_likelihoods[i, j] = target_probs[i, j] / source_priors[j]
        # L^+_{i,j} = p(x|y^j) ∝ p(y^j|x)/p(y^j)
        pos_row_likelihoods = target_probs / source_priors
        # L^-_{i,j} = p(x|\neg y^j) ∝ p(\neg y^j|x)/p(\neg y^j)
        neg_row_likelihoods = (1 - target_probs) / (1 - source_priors)
        # Z[i, j] = \frac{L^+_{i,j} - L^-_{i,j}}{\theta_j*L^+_i + (1 - \theta_j)*L^-_i}^2
        # where: \theta_j = adjusted_priors[j]
        Z = np.square(
            (pos_row_likelihoods - neg_row_likelihoods) /
            ((pos_row_likelihoods * adjusted_priors) +
             (neg_row_likelihoods * (1 - adjusted_priors)))
        )
        return {
            'classes': classes,
            'target_probs': target_probs,
            'adjusted_probs': adjusted_probs,
            'calib_probs': calib_probs,
            'Z': Z,
        }

    @classmethod
    def get_interval_width_for_variance(
            cls, *,
            variance: float,
            interval_mass: float,
    ) -> float:
        """The interval is based on the normal distribution because
        the EM curvature provides a normal distribution over the
        prediction.

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
        # variance[j] = 1 / sum(selection_mask[i] * Z[i,j] for each instance i)
        class_selected_Z_sums = selection_mask @ cache['Z']
        class_variances = np.divide(1, class_selected_Z_sums,
                                    # Variance is infinite if Z_sum == 0
                                    out=np.full(class_selected_Z_sums.shape[0],
                                                fill_value=np.inf),
                                    where=(class_selected_Z_sums != 0))
        # Re-scale variance from 0-1 scale to count scale.
        # count_variance[j] = sum(selection_mask[i])^2 * variance[j]
        selected_n = np.sum(selection_mask)
        class_count_variances = (selected_n ** 2) * class_variances
        return class_count_variances

    @classmethod
    def get_cvxpy_constraints(
            cls, *,
            cache: Cache,
            selection_mask: cp.Variable,
            class_count_variance_limits: np.ndarray,
    ) -> List[cp.constraints.constraint.Constraint]:
        """Re-arrangement of get_class_count_variances() into a
        valid DCP constraint for cvxpy (by avoiding multiplication
        of two variable-based expressions)."""
        return [(cp.square(cp.sum(selection_mask)) <=
                 (selection_mask @ (class_count_variance_limits * cache['Z'])))]

    @classmethod
    def get_class_intervals(
            cls, *,
            cache: Cache,
            prediction_interval_mass: float,
            selection_mask: np.ndarray,
            random_state: int,
    ) -> Dict[str, PredictionInterval]:
        """This mirrors the interval calculation of EmQuantifier, with the
        following difference to produce an interval for just the
        selected instances: We use the previously adjusted
        probabilities - selection may break the prior shift assumption
        that is assumed to exist between the full source and target
        sets.
        """
        selected_n = np.sum(selection_mask)
        # Avoid division by zero by exiting early with no selected instances.
        if selected_n == 0:
            return {
                target_class: PredictionInterval(
                    prediction=0,
                    lower=0,
                    upper=0,
                )
                for target_class in cache['classes']
            }

        class_means = np.mean(cache['adjusted_probs'][selection_mask], axis=0)

        class_selected_Z_sums = (selection_mask @ cache['Z'])
        class_variances = np.divide(1, class_selected_Z_sums,
                                    # Variance is infinite if Z_sum == 0
                                    out=np.full(class_selected_Z_sums.shape[0],
                                                fill_value=np.inf),
                                    where=(class_selected_Z_sums != 0))

        intervals = {}
        for class_idx, target_class in enumerate(cache['classes']):
            class_mean = class_means[class_idx]
            class_variance = class_variances[class_idx]

            if class_variance == np.inf:
                lower, upper = (0.0, 1.0)
            else:
                class_std = np.sqrt(class_variance)
                lower, upper = scipy.stats.truncnorm.interval(
                    a=((0 - class_mean) / class_std),
                    b=((1 - class_mean) / class_std),
                    loc=class_mean, scale=class_std,
                    alpha=prediction_interval_mass,
                )
            intervals[target_class] = PredictionInterval(
                prediction=(class_mean * selected_n),
                lower=int(np.floor(lower * selected_n)),
                upper=int(np.ceil(upper * selected_n)),
            )
        return intervals


# First parent class methods take precedence
class EmProbThresholdRejector(EmRejectorMixin, ProbThresholdRejector):
    pass


# First parent class methods take precedence
class EmApproxProbThresholdRejector(EmRejectorMixin, ApproxProbThresholdRejector):
    pass


# First parent class methods take precedence
class EmMipRejector(EmRejectorMixin, MipRejector):
    pass
