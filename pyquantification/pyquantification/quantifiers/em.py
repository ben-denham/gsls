# coding: utf-8

import numpy as np
import scipy.stats
from typing import Tuple, Dict

from pyquantification.quantifiers.base import (
    Quantifier, Classes, Probs, Priors, PredictionInterval, QuantificationError)


def normalise_prob_rows(probs: Probs) -> Probs:
    """Normalise the class probabilities within each row of probs
    (i.e. divide each row by its sum)."""
    row_sums = probs.sum(axis=1).reshape((probs.shape[0], 1))
    return probs / row_sums


def adjust_to_priors(*,
                     source_priors: Priors,
                     target_priors: Priors,
                     probs: Probs) -> Probs:
    """Given class source_priors and target_priors for each class, use the
    Saerens et al. 2002 method to adjust the predicted posterior
    probabilities for each class.
    """
    prior_adjustments = target_priors / source_priors
    adjusted_probs = prior_adjustments * probs
    return normalise_prob_rows(adjusted_probs)


def adjust_priors_with_em(*,
                          source_priors: Priors,
                          probs: Probs,
                          max_iterations: int = 10_000,
                          convergence_tolerance: float = 1e-4) -> Tuple[Priors, Probs]:
    """Given class source_priors and unknown (but assumed different)
    target_priors, use the Saerens et al. 2002
    Expectation-Maximisation (EM) MLE method to adjust the predicted
    posterior probabilities for each class.

    Return the target_priors and adjusted probs.

    Convergence tolerance set to 1e-4, as higher (e.g. 1e-8) results
    in underflow when normalising probabilities. Reducing our error
    tolerance thus prevents probabilities from being zeroed out by the
    adjustment process (zeroing out probabilities may have adverse
    effects on statistical tests and prediction intervals).

    """
    source_priors = np.array(source_priors)
    class_count = source_priors.shape[0]

    assert len(source_priors.shape) == 1
    assert len(probs.shape) == 2
    assert probs.shape[1] == class_count

    target_priors = np.copy(source_priors)
    for i in range(max_iterations):
        last_target_priors = np.copy(target_priors)
        adjusted_probs = adjust_to_priors(
            source_priors=source_priors,
            target_priors=target_priors,
            probs=probs,
        )
        target_priors = np.mean(adjusted_probs, axis=0)

        if np.min(np.abs(target_priors - last_target_priors) < convergence_tolerance):
            break
    else:
        raise Exception('Maximum iterations reached without convergence.')
    return target_priors, adjusted_probs


def get_em_interval_instance_weights(source_pos_prior: float,
                                     adjusted_pos_prior: float,
                                     unadjusted_pos_probs: Probs) -> np.ndarray:
    """Each weight is:
    $\frac{L^{+}_i - L^{-}_i}{\theta*L^{+}_i + (1 - \theta)*L^{-}_i}^2$
    (See: https://www.aclweb.org/anthology/D18-1487.pdf)"""
    # We can get the row likelihoods by normalising out the
    # priors. While we could perform a similar normalisation of the
    # adjusted_probs with the adjusted_pos_prior, the
    # adjusted_pos_prior can sometimes reduce to zero, resulting in
    # likelihoods of zero and infinity. source_pos_prior should alway
    # be > 0, as it should be based on a stratified calibration set.
    assert source_pos_prior > 0
    pos_row_likelihoods = unadjusted_pos_probs / source_pos_prior  # p(x|y) ∝ p(y|x)/p(y)
    neg_row_likelihoods = (1 - unadjusted_pos_probs) / (1 - source_pos_prior)  # p(x|y) ∝ p(y|x)/p(y)
    return np.square(
        (pos_row_likelihoods - neg_row_likelihoods) /
        ((pos_row_likelihoods * adjusted_pos_prior) +
         (neg_row_likelihoods * (1 - adjusted_pos_prior)))
    )


def get_em_confidence_interval(source_pos_prior: float,
                               adjusted_pos_prior: float,
                               unadjusted_pos_probs: Probs,
                               interval_mass: float) -> PredictionInterval:
    """The confidence interval for adjusted quantification is determined
    by a Gaussian distribution (as per the central limit theory for
    the Fisher Information for Maximum Likelihood Estimates like
    Expectation Maximisation), but we can use a truncated normal
    distribution as we know that the quantification must be within
    those bounds.

    We use this confidence interval as an approximation of a true
    prediction interval (scaled to the range of target instances)."""
    assert len(unadjusted_pos_probs.shape) == 1
    target_count = unadjusted_pos_probs.shape[0]

    instance_weights = get_em_interval_instance_weights(
        source_pos_prior=source_pos_prior,
        adjusted_pos_prior=adjusted_pos_prior,
        unadjusted_pos_probs=unadjusted_pos_probs,
    )
    weight_sum = np.sum(instance_weights)

    if target_count == 0:
        # No variance when there are no target instances, interval is
        # width zero.
        lower, upper = (adjusted_pos_prior, adjusted_pos_prior)
    elif weight_sum == 0:
        # Variance is infinite when pos_row_likelihoods all equal
        # neg_row_likelihoods, so interval has maximum width.
        lower, upper = (0.0, 1.0)
    else:
        # Weights sum to 1 / variance of the confidence distribution.
        std_dev = np.sqrt(1 / weight_sum)
        mean = adjusted_pos_prior
        # Truncnorm bounds are specified as a, b (relative to mean and std_dev).
        a = (0 - mean) / std_dev
        b = (1 - mean) / std_dev
        # Generate an equal-tailed interval for a truncated Gaussian.
        lower, upper = scipy.stats.truncnorm.interval(
            alpha=interval_mass, a=a, b=b, loc=mean, scale=std_dev)

    # Convert [0, 1]-scale prior, lower, and upper to target_count
    # scale.
    return PredictionInterval(
        prediction=(adjusted_pos_prior * target_count),
        # Use ceil/floor to round to slightly wider bounds that
        # represent discrete counts.
        lower=int(np.floor(lower * target_count)),
        upper=int(np.ceil(upper * target_count)),
    )


class EmQuantifier(Quantifier):
    """Expectation-Maximization (EM) quantification method. Prediction
    interval is actually a confidence interval based on an
    central-limit-theorem approximation of EM accuracy derived from
    Fisher Information. Works under the prior shift assumption: p^S(y)
    != p^T(y), but p^S(x|y) = p^T(x|y)."""

    @classmethod
    def _quantify(cls, *,  # type: ignore[override]
                  classes: Classes,
                  target_probs: Probs,
                  prediction_interval_mass: float,
                  calib_probs: Probs) -> Dict[str, PredictionInterval]:
        source_priors = calib_probs.mean(axis=0).reshape(calib_probs.shape[1],)

        # Check that all source_priors are non-zero, otherwise
        # adjustment cannot be performed.
        zero_priors_mask = (source_priors == 0)
        if zero_priors_mask.any():
            formatted_zero_prior_classes = ', '.join([
                f'"{target_class}"'
                for target_class in np.array(classes)[zero_priors_mask]
            ])
            raise QuantificationError(
                'Adjusted quantification cannot be performed because no instances '
                'in the calibration dataset were assigned a probability greater '
                f'than zero for class(es): {formatted_zero_prior_classes}.')

        adjusted_priors, adjusted_probs = adjust_priors_with_em(
            source_priors=source_priors,
            probs=target_probs,
        )

        return {
            target_class: get_em_confidence_interval(
                source_pos_prior=source_priors[class_index],
                adjusted_pos_prior=adjusted_priors[class_index],
                unadjusted_pos_probs=target_probs[:, class_index],
                interval_mass=prediction_interval_mass,
            )
            for class_index, target_class in enumerate(classes)
        }
