import numpy as np
import pandas as pd
import scipy.stats
from typing import cast, Dict, Tuple, Union

from pyquantification.quantifiers.base import (Quantifier, Classes, Probs,
                                               PredictionInterval,
                                               interval_for_cdf)
from pyquantification.quantifiers.gsls.fitters import WeightSumFitter


def sum_ind_var_pmfs(pmf_a: np.ndarray, pmf_b: np.ndarray) -> np.ndarray:
    """Given pmfs representing p(A), p(B) return a histogram representing p(A + B)"""
    assert len(pmf_a) > 0
    assert len(pmf_b) > 0
    # Sum maximum is the sum of all pmf maximum values (the first
    # item in a pmf represents a value of zero, so we subtract 1
    # from each length).
    sum_max = (len(pmf_a) - 1) + (len(pmf_b) - 1)
    sum_pmf = np.zeros(sum_max + 1)
    for point_a, prob_a in enumerate(pmf_a):
        # Add pmf_b scaled by prob_a over the range of possible
        # values given a = point_a.
        pmf_b_for_point_a = prob_a * pmf_b
        assert pmf_b_for_point_a.shape == pmf_b.shape
        sum_pmf[point_a:(point_a + len(pmf_b))] += pmf_b_for_point_a
    return sum_pmf


def build_histogram(*, bin_edges: np.ndarray, class_probs: Probs) -> np.ndarray:
    """Return a histogram array of len(bin_edges) - 1 bins produced by
    binning class_probs according to bin_edges."""
    assert len(class_probs.shape) == 1
    if class_probs.shape[0] == 0:
        # Return a blank histogram if there are no probs.
        return np.zeros(len(bin_edges) - 1)
    bin_counts, _ = np.histogram(class_probs, bins=bin_edges)
    hist = bin_counts / class_probs.shape[0]
    assert len(hist.shape) == 1
    assert hist.shape[0] == len(bin_edges) - 1
    np.testing.assert_almost_equal(np.sum(hist), 1.0)
    return hist


class GslsQuantifier(Quantifier):
    """
    Quantification method that accounts for gain-some-lose-some shift.
    """

    gsls_fitter = WeightSumFitter

    @classmethod
    def decompose_mixture(cls, *,
                          mixture_hist: np.ndarray,
                          component_hist: np.ndarray) -> Tuple[np.ndarray, float]:
        """Given histograms representing a mixture and one of two components
        in the mixture, return the complement component histogram, and
        the weight of the complement in the mixture (range 0-1). The
        choice of complement histogram minimizes the complement's
        weight."""
        a_hist = component_hist
        nonzero_a_mask = (a_hist != 0)
        a_weight = (
            # Find the maximum a_weight that allows a_hist to fit
            # inside mixture_hist.
            np.min(mixture_hist[nonzero_a_mask] / a_hist[nonzero_a_mask])
            # If a_hist is all zeroes, then the weight of that
            # distribution is 0 (mixture_hist == b_hist).
            if np.any(nonzero_a_mask) else 0
        )
        b_weight = 1 - a_weight
        if b_weight == 0:
            # b_hist is irrelevant, as it has zero weight in the
            # mixture.
            b_hist = np.zeros(mixture_hist.shape)
        else:
            # b_scaled_hist is found by subtracting the
            # a_weight-scaled a_hist from the mixture_hist.
            a_scaled_hist = a_hist * a_weight
            b_scaled_hist = mixture_hist - a_scaled_hist
            # Scale b_hist to have sum 1.
            b_hist = b_scaled_hist / b_weight
        return b_hist, b_weight

    @classmethod
    def get_gsls_interval(cls, *,
                          loss_weight: float,
                          gain_weight: float,
                          target_class_probs: Probs,
                          class_count: int,
                          interval_mass: float) -> PredictionInterval:
        """Return an interval based on a Poisson Binomial distribution over
        target_class_probs adjusted to account for loss_weight and
        gain_weight in the GSLS model."""
        assert 0 <= loss_weight <= 1
        assert 0 <= gain_weight <= 1
        assert 0 <= interval_mass <= 1
        assert len(target_class_probs.shape) == 1

        # Losing 100% of the training distribution is equivalent to a
        # 100% gain (avoids division by zero to determine source_n).
        if loss_weight == 1:
            gain_weight = 1
            loss_weight = 0
        full_n = target_class_probs.shape[0]
        gain_n = int(round(full_n * gain_weight))
        remain_n = full_n - gain_n
        source_n = int(round(remain_n / (1 - loss_weight)))
        loss_n = source_n - remain_n

        # Source distribution
        if source_n == 0:
            # No instances from source - source pmf has 100% prob at zero.
            source_pmf = np.array([1.0])
        else:
            # The mean and variance are calculated based on a Poisson
            # binomial distribution on target_class_probs scaled from
            # the full_n of target_class_probs to the intended
            # source_n.
            source_scale = source_n / full_n
            source_mean = np.sum(target_class_probs) * source_scale
            source_var = np.sum(target_class_probs * (1 - target_class_probs)) * source_scale**2

            # Because the scaling means we cannot use a discrete
            # Poisson Binomial PMF, we approximate with a Gaussian
            # distribution.
            source_pmf = scipy.stats.norm.pdf(
                x=np.arange(source_n + 1),
                loc=source_mean,
                scale=np.sqrt(source_var),
            )
            # As the Gaussian distribution may extend past the range
            # of source_pmf, we normalise it (giving a truncated
            # Gaussian distribution).
            source_pmf = source_pmf / source_pmf.sum()

        # Remain distribution (Source - Loss)
        if loss_n == 0:
            remain_pmf = source_pmf
        else:
            remain_pmf = np.zeros(shape=remain_n+1)
            for source_pos, source_prob in enumerate(source_pmf):
                min_loss_pos = max(0, source_pos - remain_n)
                max_loss_pos = min(loss_n, source_pos)
                # The probability of a given loss_pos is drawn from a
                # beta distribution over range [0, loss_n], but
                # truncated to the possible values in range
                # [min_loss_pos, max_loss_pos].
                loss_pos_ns = list(range(min_loss_pos, max_loss_pos + 1))
                loss_probs = scipy.stats.beta.pdf(
                    x=loss_pos_ns,
                    a=1,
                    b=(class_count - 1),
                    scale=loss_n
                )
                if loss_probs.sum() == 0:
                    # If beta.pdf returns probs so small they can only
                    # be represented as zero (e.g. when class_count=3
                    # and loss_pos_ns=[loss_n]), we will use a uniform
                    # distribution.
                    loss_probs = np.full(loss_probs.shape[0],
                                         fill_value=(1.0 / loss_probs.shape[0]))
                # Normalise probs to give a discrete truncated beta
                # distribution.
                loss_probs = loss_probs / loss_probs.sum()
                for loss_pos, loss_prob in zip(loss_pos_ns, loss_probs):
                    remain_pmf[source_pos - loss_pos] += source_prob * loss_prob

        # Target distribution (Remain + Gain)
        if gain_n == 0:
            target_pmf = remain_pmf
        else:
            gain_pmf = scipy.stats.beta.pdf(
                x=np.arange(gain_n + 1),
                a=1,
                b=(class_count - 1),
                scale=gain_n,
            )
            # Normalise PDF to a PMF.
            gain_pmf = gain_pmf / gain_pmf.sum()
            target_pmf = sum_ind_var_pmfs(remain_pmf, gain_pmf)

        target_cdf = np.cumsum(target_pmf)
        assert len(target_cdf) == full_n + 1
        interval = interval_for_cdf(target_cdf, interval_mass)
        return PredictionInterval(
            # Prediction is the mean of the maximum points in the PMF.
            prediction=cast(float, np.mean(
                # Treat any point very close to the maximum as a
                # maximum point (to account for very small differences
                # in probabilities).
                np.argwhere(target_pmf >= np.amax(target_pmf))
            )),
            lower=interval.lower,
            upper=interval.upper,
        )

    @classmethod
    def get_auto_hist_bins(cls, *, calib_count: int, target_count: int) -> int:
        """Set hist_bins according to rule-of-thumb for equiprobable bins
        (based on minimum count of instances for either distribution
        sample):
        https://itl.nist.gov/div898/handbook/prc/section2/prc211.htm
        """
        return round(2 * (min(calib_count, target_count) ** (2/5)))

    @classmethod
    def update_weights(cls, *,
                       loss_weight: float,
                       gain_weight: float,
                       true_loss_weight: float,
                       true_gain_weight: float,
                       hist_bins: int,
                       random_state: int) -> Tuple[float, float]:
        """Hook to allow sub-classes to update the weights."""
        return loss_weight, gain_weight

    @classmethod
    def _quantify(cls, *,  # type: ignore[override]
                  classes: Classes,
                  target_probs: Probs,
                  prediction_interval_mass: float,
                  calib_probs: Probs,
                  bin_count: Union[int, str],
                  true_weights: Dict[str, float],
                  random_state: int,
                  hist_equal_counts: bool = True) -> Dict[str, PredictionInterval]:
        # Assert all probs are between 0 and 1 (otherwise histograms
        # will be incorrect).
        assert np.all(calib_probs >= 0) and np.all(calib_probs <= 1)
        assert np.all(target_probs >= 0) and np.all(target_probs <= 1)
        target_count = target_probs.shape[0]

        if bin_count == 'auto':
            hist_bins = cls.get_auto_hist_bins(calib_count=calib_probs.shape[0],
                                               target_count=target_probs.shape[0])
        elif isinstance(bin_count, int):
            hist_bins = bin_count
        else:
            raise ValueError(f'Unrecognised bin_count: {bin_count}')

        quantifications = {}
        for class_index, target_class in enumerate(classes):
            calib_class_probs = calib_probs[:, class_index]
            target_class_probs = target_probs[:, class_index]

            if hist_equal_counts:
                # Create equiprobable bins based on the calib probs
                # (may give fewer than hist_bins).
                _, bin_edges = pd.qcut(calib_class_probs, hist_bins, retbins=True, duplicates='drop')
                # Ensure all bin_edges are in the range [0, 1], and
                # that the first and last bin edges are 0 and 1
                # respectively. For np.histogram, this final edge will
                # include records where prob = 1.
                bin_edges = np.clip(bin_edges, 0, 1)
                bin_edges[0] = 0
                bin_edges[-1] = 1
            else:
                bin_edges = np.linspace(0, 1, num=(hist_bins + 1), endpoint=True)

            calib_hist = build_histogram(
                bin_edges=bin_edges, class_probs=calib_class_probs)
            target_hist = build_histogram(
                bin_edges=bin_edges, class_probs=target_class_probs)

            remain_hist = cls.gsls_fitter.find_remain_hist(
                calib_hist=calib_hist, target_hist=target_hist, random_state=random_state)
            loss_hist, loss_weight = cls.decompose_mixture(
                mixture_hist=calib_hist, component_hist=remain_hist)
            gain_hist, gain_weight = cls.decompose_mixture(
                mixture_hist=target_hist, component_hist=remain_hist)

            # Discretise loss_weight and gain_weight to the number of
            # target probs (prevents very large source_n from division
            # by 1-loss_weight in get_gsls_interval).
            loss_weight = round(loss_weight * target_count) / target_count
            gain_weight = round(gain_weight * target_count) / target_count

            loss_weight, gain_weight = cls.update_weights(
                loss_weight=loss_weight,
                gain_weight=gain_weight,
                true_loss_weight=true_weights['loss'],
                true_gain_weight=true_weights['gain'],
                hist_bins=hist_bins,
                random_state=random_state)

            quantifications[target_class] = cls.get_gsls_interval(
                loss_weight=loss_weight,
                gain_weight=gain_weight,
                target_class_probs=target_class_probs,
                class_count=len(classes),
                interval_mass=prediction_interval_mass,
            )
            quantifications[target_class].stats = {
                **quantifications[target_class].stats,
                'loss_weight': loss_weight,
                'gain_weight': gain_weight,
                'bins': hist_bins,
            }
        return quantifications


class TrueWeightGslsQuantifier(GslsQuantifier):
    """
    Uses true gain/loss weights instead of estimated weights.
    """

    @classmethod
    def update_weights(cls, *,
                       loss_weight: float,
                       gain_weight: float,
                       true_loss_weight: float,
                       true_gain_weight: float,
                       hist_bins: int,
                       random_state: int) -> Tuple[float, float]:
        """Replace estimated weights with true weights"""
        return true_loss_weight, true_gain_weight
