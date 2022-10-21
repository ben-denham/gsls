# coding: utf-8

from typing import Callable, Dict, Any, List, Tuple, Union, cast

import cvxpy as cp
import numpy as np
import scipy.stats
import warnings

from pyquantification.quantifiers.base import PredictionInterval
from pyquantification.quantifiers.gsls import GslsQuantifier
from .base import (Classes, Probs, Cache, ProbThresholdRejector,
                   ApproxProbThresholdRejector, MipRejector)


class LossDist:
    """Helper class for calculating the Variance(q^-) and
    Covariance(q^S, q^-) given target set prediction probabilities
    p(y|x_i) and a loss_weight w^-.

    Assumes q^S has a PBD distribution (approximated with a normal
    distribution) and q^- has a uniform distribution between points
    determined by q^S."""

    def __init__(self, *, target_class_probs: Probs):
        # E[q^s] - mean of source PBD
        self.source_mean = np.mean(target_class_probs)
        # Var(n^T * q_s) - variance of source PBD
        source_count_variance = np.sum(target_class_probs * (1 - target_class_probs))
        # Var(q^S) - scale variance from count scale to 0-1 scale
        self.source_variance = source_count_variance / (target_class_probs.shape[0] ** 2)
        # Approximate PBD of q^S with a normal distribution. Using a
        # truncnorm distribution here would have little impact (as
        # variance will be low when mean is close to 0 or 1).
        self.source_dist = scipy.stats.norm(loc=self.source_mean, scale=np.sqrt(self.source_variance))

    def source_dist_expect(self, func: Callable[[np.number], Union[float, np.number]]) -> float:
        """Computes the expected value of the given func over values of q_s from source_dist."""
        if self.source_variance == 0:
            # If the source_dist has no variance, the mean q_s has
            # 100% probability.
            return float(func(self.source_mean))
        else:
            return float(self.source_dist.expect(func))

    def conditioned_loss_mean(self, *, q_s: float, loss_weight: float) -> float:
        """Compute the mean/expected value of the uniform q^- distribution for
        a given value of q^S: E[q^- | q^S]"""
        lower_bound = max(0, (((q_s - 1) / loss_weight) + 1))
        upper_bound = min(1, q_s / loss_weight)
        # Mean of a normal distribution between the bounds.
        return 0.5 * (lower_bound + upper_bound)

    def conditioned_loss_variance(self, *, q_s: float, loss_weight: float) -> float:
        """Compute the variance of the uniform q^- distribution for
        a given value of q^S: Var(q^- | q^S)"""
        lower_bound = max(0, (((q_s - 1) / loss_weight) + 1))
        upper_bound = min(1, q_s / loss_weight)
        # Variance of a normal distribution between the bounds.
        return (1 / 12) * ((upper_bound - lower_bound) ** 2)

    def get_loss_variance_covariance(self, loss_weight: float) -> Tuple[float, float]:
        """Compute Variance(q^-) and Covariance(q^S, q^-).

        Note that expectations over the normally approximated
        source_dist may produce probabilities that are slightly
        outside the 0-1 range, so we clip them.

        """
        # Avoid division by zero by handling the case where q^- is
        # always zero: there is no variance.
        if loss_weight == 0:
            return 0, 0

        # Expectation of mean/expected q^- over q^S: E[E[q^- | q^S]]
        expected_loss = self.source_dist_expect(
            lambda q_s: self.conditioned_loss_mean(q_s=np.clip(q_s, 0, 1),
                                                   loss_weight=loss_weight)
        )
        # Expectation of (variance of q^- + square mean/expected q^-)
        # over q^S: E[Var(q^- | q^S) + E[q^- | q^S]^2]
        expected_loss_variance_plus_squared_mean = self.source_dist_expect(
            lambda q_s: (self.conditioned_loss_variance(q_s=np.clip(q_s, 0, 1),
                                                        loss_weight=loss_weight)
                         + (self.conditioned_loss_mean(q_s=np.clip(q_s, 0, 1),
                                                       loss_weight=loss_weight) ** 2))
        )
        # E[Var(q^- | q^S) + E[q^- | q^S]^2] - E[E[q^- | q^S]]^2
        variance = expected_loss_variance_plus_squared_mean - (expected_loss ** 2)
        # Expectation of (q^S multiplied by expected/mean q^-) over q^S:
        # E[q^S * E[q^- | q^S]]
        with warnings.catch_warnings():
            # In rare cases small values may result in an integration
            # warning, but the returned small values still appear
            # reasonable.
            warnings.filterwarnings('ignore', message='The integral is probably divergent, or slowly convergent')
            expected_source_loss_product = self.source_dist_expect(
                lambda q_s: q_s * self.conditioned_loss_mean(q_s=np.clip(q_s, 0, 1),
                                                             loss_weight=loss_weight)
            )
        # E[qs * E[q- | qs]] - (E[qs] * E[E[q- | qs]])
        covariance = expected_source_loss_product - (self.source_mean * expected_loss)
        return variance, float(covariance)

    def get_class_loss_coefficient(self, loss_weight: float) -> float:
        """Calculate the coefficient of the GSLS variance model's z^- terms
        that indicates the impact of the loss distribution."""
        if loss_weight == 0:
            return 0.0

        variance, covariance = self.get_loss_variance_covariance(loss_weight)
        # Var(q^-) - ((2 / w^-) * Cov(q^S, q^-))
        return variance - (covariance * (2 / loss_weight))


class GslsRejectorMixin:
    """Mixin defining functions for rejection based on GSLS quantification."""

    @classmethod
    def get_gain_bin_weights(
            cls, *,
            target_hist: np.ndarray,
            gain_weight: float,
            gain_hist: np.ndarray,
    ) -> np.ndarray:
        """Returns the gain_weight for instances in each bin of the
        target_hist. The mean of these weights across all instances
        (i.e. weighted by target_hist) should produce the gain_weight."""
        # The gain_bin_weight is the proportion of the target
        # bin that is made up of gain,
        # which is given by: (gain_weight * gain_hist) / target_hist
        # because: target_hist = (gain_weight * gain_hist) + ((1 - gain_weight) * remain_hist)
        gain_bin_weights = np.divide((gain_weight * gain_hist),
                                     target_hist,
                                     # If a target_hist bin is empty,
                                     # the weight will not be used,
                                     # so we safely set it to zero.
                                     out=np.zeros(target_hist.shape),
                                     where=(target_hist > 0.0))
        # As with the overall gain_weight, each gain_bin_weight must
        # be between 0 and 1
        assert np.all(gain_bin_weights >= 0)
        # Precision issues may result in weights that exceed 1, so
        # clip:
        gain_bin_weights = np.minimum(1, gain_bin_weights)
        return gain_bin_weights

    @classmethod
    def build_cache(
            cls, *,
            classes: Classes,
            target_probs: Probs,
            calib_probs: Probs,
            quantification_method_results: Dict[str, Dict[str, Any]],
    ) -> Cache:
        class_loss_weights = {}
        class_target_gain_weights = {}

        # Compute interval-related values for each target_class.
        for class_idx, target_class in enumerate(classes):
            interval = quantification_method_results['gsls']['class_intervals'][target_class]
            target_class_probs = target_probs[:, class_idx]

            loss_weight = interval['loss_weight']
            class_loss_weights[target_class] = loss_weight

            if loss_weight <= 0:
                # Handle 100% loss as 100% gain
                target_gain_weights = np.full(target_class_probs.shape, fill_value=1.0)
            else:
                _gain_bin_weights = cls.get_gain_bin_weights(
                    target_hist=interval['target_hist'],
                    gain_weight=interval['gain_weight'],
                    gain_hist=interval['gain_hist'],
                )
                assert _gain_bin_weights.shape == interval['gain_hist'].shape
                # Digitize assigns each digit d so: bins[d - 1] <= x < bins[d],
                # so we clamp to len(bins)-1 so: x <= bins[d] for the final bin
                # (same as np.histogram used in build_histogram). We then subtract
                # 1 so that the indexing ranges from 0 to len(bins)-2 (= len(hist)-1)
                bin_edges = interval['bin_edges']
                instance_digits = np.digitize(target_class_probs, bins=bin_edges, right=False)
                instance_bins = np.minimum(instance_digits, len(bin_edges) - 1) - 1
                # w^+_{c,i} values
                target_gain_weights = _gain_bin_weights[instance_bins]
            assert target_gain_weights.shape == target_class_probs.shape
            class_target_gain_weights[target_class] = target_gain_weights

        return {
            'classes': classes,
            'target_probs': target_probs,
            'class_loss_weights': class_loss_weights,
            'class_target_gain_weights': class_target_gain_weights,
        }

    @classmethod
    def get_class_intervals(
            cls, *,
            cache: Cache,
            prediction_interval_mass: float,
            selection_mask: np.ndarray,
            random_state: int,
    ) -> Dict[str, PredictionInterval]:
        """Compute intervals for the selected instances using GslsQuantifier.
        The class loss_weights estimated for the full target set are
        used (we cannot determine how loss changes based on a
        selection), while the class gain_weights can be estimated by
        taking the mean of the target instance gain weights for the
        selected instances."""
        selected_n = np.sum(selection_mask)
        class_count = len(cache['classes'])

        intervals = {}
        for class_idx, target_class in enumerate(cache['classes']):
            if selected_n == 0:
                intervals[target_class] = PredictionInterval(
                    prediction=0,
                    lower=0,
                    upper=0,
                )
            else:
                intervals[target_class] = GslsQuantifier.get_gsls_interval(
                    loss_weight=cache['class_loss_weights'][target_class],
                    gain_weight=cast(float, np.mean(cache['class_target_gain_weights'][target_class][selection_mask])),
                    target_class_probs=cache['target_probs'][selection_mask][:, class_idx],
                    class_count=class_count,
                    interval_mass=prediction_interval_mass,
                )
        return intervals


# First parent class methods take precedence
class GslsProbThresholdRejector(GslsRejectorMixin, ProbThresholdRejector):
    pass


class ApproxGslsRejectorMixin(GslsRejectorMixin):
    """The more expensive build_cache() steps in this class are not needed
    for the base ProbThresholdRejector."""

    @classmethod
    def get_target_class_mean_weights(cls, target_weights: np.ndarray) -> np.ndarray:
        """Given the bin-local gain_weights of each target instance,
        return the gain_weight that should be used for coefficients of
        instances in cases where DCP will not allow us to compute the
        true overall gain_weight based on the selected instances.

        This implementation heuristically sets this weight equal to
        the bin-local gain_weight to make the impact of the remain
        distribution relative to each instance's local gain_weight.
        """
        return target_weights

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

        # The standard deviation of the gain distribution, assuming a
        # uniform/beta(1, 1) distribution.
        gain_dist_std = np.sqrt(scipy.stats.beta.var(1, 1))

        Z_gain = np.full(target_probs.shape, fill_value=np.nan)
        Z_source = np.full(target_probs.shape, fill_value=np.nan)
        Z_loss = np.full(target_probs.shape, fill_value=np.nan)
        # Compute Z values for each target_class.
        for class_idx, target_class in enumerate(classes):
            target_class_probs = target_probs[:, class_idx]
            loss_weight = cache['class_loss_weights'][target_class]
            target_gain_weights = cache['class_target_gain_weights'][target_class]

            if loss_weight <= 0:
                # Handle 100% loss as 100% gain
                inverse_weight_ratios = np.full(target_class_probs.shape, fill_value=0.0)
            else:
                # inverse_weight_ratio = (1 - w^+_c) / (1 - w^-_c)
                inverse_weight_ratios = (
                    # Get w^+_c from the w^+_{c,i} values
                    (1 - cls.get_target_class_mean_weights(target_gain_weights))
                    / (1 - loss_weight)
                )
            assert inverse_weight_ratios.shape == target_class_probs.shape

            # z^+_c[i] = w^+_{c,i} * gain_dist_std
            Z_gain[:, class_idx] = target_gain_weights * gain_dist_std

            # z^S_c[i] = ((1 - w^+_c) / (1 - w^-_c))^2 * (g_c(x_i) * (1 - g_c(x_i)))
            Z_source[:, class_idx] = (
                (inverse_weight_ratios ** 2) * (target_class_probs * (1 - target_class_probs))
            )

            class_loss_coefficient = (LossDist(target_class_probs=target_class_probs)
                                      .get_class_loss_coefficient(loss_weight))
            # z^-_c = ((1 - w^+_c) / (1 - w^-_c)) * w^-_c
            #         * sqrt{max(0 , Var(q^-_c) - (2 / w^-_c)Cov(q^S_c, q^-_c))}
            Z_loss[:, class_idx] = (
                inverse_weight_ratios * loss_weight * np.sqrt(max(0, class_loss_coefficient))
            )

        return {
            **cache,
            'Z_gain': Z_gain,
            'Z_source': Z_source,
            'Z_loss': Z_loss,
        }

    @classmethod
    def get_interval_width_for_variance(
            cls, *,
            variance: float,
            interval_mass: float,
    ) -> float:
        """The interval is based on the normal distribution because the PBD
        distribution of the source distribution can be approximated as
        a normal distribution prediction (although gain and loss may
        re-shape the distribution to a more uniform shape, see: UniformDistMixin).
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
            selection_mask: Union[np.ndarray, cp.Variable],
    ) -> np.ndarray:
        gain = (selection_mask @ cache['Z_gain']) ** 2
        source = selection_mask @ cache['Z_source']
        loss = (selection_mask @ cache['Z_loss']) ** 2
        return gain + source + loss

    @classmethod
    def get_cvxpy_constraints(
            cls, *,
            cache: Cache,
            selection_mask: cp.Variable,
            class_count_variance_limits: np.ndarray,
    ) -> List[cp.constraints.constraint.Constraint]:
        class_count_variances = cls.get_class_count_variances(
            cache=cache,
            selection_mask=selection_mask,
        )
        return [(class_count_variances <= class_count_variance_limits)]


# First parent class methods take precedence
class GslsApproxProbThresholdRejector(ApproxGslsRejectorMixin, ApproxProbThresholdRejector):
    pass


# First parent class methods take precedence
class GslsMipRejector(ApproxGslsRejectorMixin, MipRejector):
    pass


class StaticGainGslsMixin:
    """Variant of the GSLS rejector that uses the same gain_weight for all
    instances - as if the gain_hist is not known. Doing so removes the
    heuristic nature of target_class_mean_weights(), as all
    gain_weights will be constant."""

    @classmethod
    def get_gain_bin_weights(
            cls, *,
            target_hist: np.ndarray,
            gain_weight: float,
            gain_hist: np.ndarray,
    ) -> np.ndarray:
        return np.full(gain_hist.shape, fill_value=gain_weight)


# First parent class methods take precedence
class StaticGainGslsProbThresholdRejector(StaticGainGslsMixin, GslsRejectorMixin, ProbThresholdRejector):
    pass


# First parent class methods take precedence
class StaticGainGslsApproxProbThresholdRejector(StaticGainGslsMixin, ApproxGslsRejectorMixin, ApproxProbThresholdRejector):
    pass


# First parent class methods take precedence
class StaticGainGslsMipRejector(StaticGainGslsMixin, ApproxGslsRejectorMixin, MipRejector):
    pass


class UniformDistMixin:

    @classmethod
    def get_interval_width_for_variance(
            cls, *,
            variance: float,
            interval_mass: float,
    ) -> float:
        """Because GSLS intervals are a mixture of PB and uniform
        distributions, we conservatively get interval widths for a
        uniform distribution."""
        # With no variance, interval is width zero.
        if variance == 0:
            return 0
        # With infinite variance, interval is infinite.
        if variance == np.inf:
            return np.inf

        width = np.sqrt(12 * variance)
        low, high = scipy.stats.uniform.interval(
            loc=0,
            scale=width,
            alpha=interval_mass,
        )
        return high - low


# First parent class methods take precedence
class UniformGslsProbThresholdRejector(UniformDistMixin, GslsRejectorMixin, ProbThresholdRejector):
    pass


# First parent class methods take precedence
class UniformGslsApproxProbThresholdRejector(UniformDistMixin, ApproxGslsRejectorMixin, ApproxProbThresholdRejector):
    pass


# First parent class methods take precedence
class UniformGslsMipRejector(UniformDistMixin, ApproxGslsRejectorMixin, MipRejector):
    pass


# First parent class methods take precedence
class UniformStaticGainGslsProbThresholdRejector(UniformDistMixin, StaticGainGslsMixin, GslsRejectorMixin, ProbThresholdRejector):
    pass


# First parent class methods take precedence
class UniformStaticGainGslsApproxProbThresholdRejector(UniformDistMixin, StaticGainGslsMixin, ApproxGslsRejectorMixin, ApproxProbThresholdRejector):
    pass


# First parent class methods take precedence
class UniformStaticGainGslsMipRejector(UniformDistMixin, StaticGainGslsMixin, ApproxGslsRejectorMixin, MipRejector):
    pass
