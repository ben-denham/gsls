from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.stats
from typing import Dict, List, Any, Type

from pyquantification.shift_tests.base import (
    ShiftTester, TestResult, Classes, Priors, Probs, ProbAdjustment
)
from pyquantification.shift_tests.dists import (
    Hist,
    HistGrid,
    NullHistGrid,
    EqualSpaceHistGrid,
    EqualProbHistGrid,
    ClassConditionalMixture,
    SampledClassConditionalMixture,
    get_hists,
    mix_per_class_hists,
    DistanceMeasure,
    null_distance,
    ks_distance,
    hellinger_distance,
    DistanceAggregator,
    mean_distance_aggregator,
    max_distance_aggregator,
)
from pyquantification.quantifiers.em import adjust_priors_with_em


class SimShiftTester(ShiftTester, metaclass=ABCMeta):
    """Simulation-based prior-shift tests generalising the tests proposed
    in:

    * Vaz, A. F., Izbicki, R., & Stern, R. B. (2019). Quantification
      Under Prior Probability Shift: the Ratio Estimator and its
      Extensions. J. Mach. Learn. Res., 20, 79-1.
    * Maletzke, A., Reis, D., Cherman, E., & Batista, G. (2018,
      November). On the need of class ratio insensitive drift tests
      for data streams. In Second international workshop on learning
      with imbalanced domains: theory and applications
      (pp. 110-124). PMLR.

    The test determines a p-value by counting how often a simulated
    prior shift of the observed/estimated magnitude has greater
    distance to a best-fit mixture of class-conditional calibration
    distributions than the distance between the target distribution
    and a best-fit mixture of class-conditional calibration
    distributions.

    The p-value is low when the observed shift distance appears
    greater than expected for prior shift, indicating that non-prior
    shift has likely occurred.

    This implementation is based on that of the original paper
    (https://github.com/afonsofvaz/ratio_estimator) but extended for
    the multi-class case. Some key differences:

    * We don't grid search for the best class priors, we use
      the EM prior adjustment method (a more practical assessment, as
      that is how we will be selecting the proportion when the method is
      later applied to the target set). Because of this, we also measure
      target distribution distances to a mixture of the original
      calibration class distributions (as the classifier outputs used
      for EM adjustment are learned from the training distribution the
      calibration set is drawn from).
    * We don't use a Kernel Smoother to model class-conditional
      classifier output distributions, as the resulting distribution
      may not fit well if sample sizes are small. Instead, we resample
      from the calibration set probabilities (using
      SampledClassConditionalMixture).

    """
    SIMS: int = 1000
    MIXTURE_CLASS: Type[ClassConditionalMixture] = SampledClassConditionalMixture
    HIST_GRID: HistGrid = NullHistGrid()
    DISTANCE_MEASURE: DistanceMeasure = null_distance
    # We default to a mean over class-specific distances, as each
    # class may have a different number of Hist bins.
    DISTANCE_AGGREGATOR: DistanceAggregator = mean_distance_aggregator

    @classmethod
    def estimate_adjusted_priors(cls, *, source_priors: Priors, target_probs: Probs) -> Priors:
        """Return estimated adjusted_priors given the priors of the source
        distribution and the prob outputs for the target sample."""
        adjusted_priors, _ = adjust_priors_with_em(
            source_priors=source_priors,
            probs=target_probs,
        )
        return adjusted_priors

    @classmethod
    def compute_distance(cls, hists_a: List[Hist], hists_b: List[Hist]) -> float:
        """Compute a distance measure between the given Hists, where each Hist
        row represents the Hist of a particular class probability
        output."""
        assert len(hists_a) == len(hists_b)
        class_distances = [cls.DISTANCE_MEASURE(hist_a, hist_b)
                           for hist_a, hist_b in zip(hists_a, hists_b)]
        return cls.DISTANCE_AGGREGATOR(class_distances)

    @classmethod
    def get_test_result(cls, *,
                        observed_distance: float,
                        sim_distances: np.ndarray,
                        test_alpha: float) -> TestResult:
        """Produce a TestResult for a given test_alpha based on the observed
        and simulated distances."""
        # The p value of the test is given by the proportion of sims where
        # the simulated test stat was greater than the observed test stat.
        p_value = np.sum(sim_distances >= observed_distance) / sim_distances.shape[0]
        return TestResult(
            shift_detected=(p_value <= test_alpha),
            stats=dict(
                p_value=p_value,
            ),
        )

    @classmethod
    @abstractmethod
    def get_sim_priors(cls, *,
                       observed_adjusted_priors: Priors,
                       rng: np.random.RandomState) -> List[Priors]:
        """Get the prior probabilities to use in each of the cls.SIMS
        simulations."""

    @classmethod
    def _run(cls, *,  # type: ignore[override]
             classes: Classes,
             calib_y: np.ndarray,
             calib_probs: Probs,
             target_probs: Probs,
             quantification_method_results: Dict[str, Dict[str, Any]],
             test_alpha: float,
             random_state: int) -> Dict[str, TestResult]:
        adjustment = ProbAdjustment.from_class_intervals(
            classes=classes,
            class_intervals=quantification_method_results['em']['class_intervals'],
            calib_probs=calib_probs,
            target_probs=target_probs,
        )
        source_priors = adjustment.source_priors
        adjusted_priors = adjustment.adjusted_priors

        bin_edges = cls.HIST_GRID.get_bin_edges(calib_probs=calib_probs, target_probs=target_probs)

        # Produce class-conditional Hists of classifier output probs
        # from the calibration set.
        calib_per_class_probs = [
            calib_probs[(calib_y == target_class)] for target_class in classes
        ]
        calib_per_class_hists = [get_hists(class_probs, bin_edges=bin_edges) for class_probs in calib_per_class_probs]
        calib_class_mixture = cls.MIXTURE_CLASS(per_class_probs=calib_per_class_probs)

        # Measure the distance between the real target Hists and the
        # mixture Hists with the estimated class priors.
        observed_distance = cls.compute_distance(
            mix_per_class_hists(class_priors=adjusted_priors,
                                per_class_hists=calib_per_class_hists),
            get_hists(target_probs, bin_edges=bin_edges),
        )

        rng = np.random.RandomState(random_state)
        sim_distances = []
        sim_priors = cls.get_sim_priors(
            observed_adjusted_priors=adjusted_priors,
            rng=rng,
        )
        for sim_target_priors in sim_priors:
            # Generate a simulated target set with the estimated class mix.
            sim_target_probs = calib_class_mixture.sample_probs(
                class_priors=sim_target_priors, target_size=target_probs.shape[0], rng=rng)
            sim_adjusted_priors = cls.estimate_adjusted_priors(
                source_priors=source_priors, target_probs=sim_target_probs)

            # Measure the distance between the simulated target Hists
            # and the mixture Hists with the simulation-estimated class
            # priors. We use simulation-estimated priors instead of
            # the true priors of the simulated target set to account
            # for the fact that the observed distance is between the
            # target distribution and a calibration mixture based on
            # estimated priors.
            sim_distances.append(cls.compute_distance(
                mix_per_class_hists(class_priors=sim_adjusted_priors,
                                    per_class_hists=calib_per_class_hists),
                get_hists(sim_target_probs, bin_edges=bin_edges),
            ))

        test_result = cls.get_test_result(
            observed_distance=observed_distance,
            sim_distances=np.array(sim_distances),
            test_alpha=test_alpha,
        )
        test_result.stats['bin_counts'] = map(len, bin_edges)
        return {
            y_class: test_result
            for y_class in classes
        }


class WpaShiftTester(SimShiftTester):
    """Weak-Prior-Shift Assumption test (Vaz et al. 2019) that uses fixed
    prior probabilities in each simulation."""

    @classmethod
    def get_sim_priors(cls, *,
                       observed_adjusted_priors: Priors,
                       rng: np.random.RandomState) -> List[Priors]:
        return [observed_adjusted_priors for _ in range(cls.SIMS)]


class CdtShiftTester(SimShiftTester):
    """Concept-Distance-Threshold test (Maletzke et al. 2018) that uses a
    range. While the original paper uses a fixed grid of priors, we
    improve scalability to multi-class settings by randomly sampling
    priors."""

    @classmethod
    def get_sim_priors(cls, *,
                       observed_adjusted_priors: Priors,
                       rng: np.random.RandomState) -> List[Priors]:
        sim_priors = scipy.stats.dirichlet.rvs(
            alpha=np.ones(observed_adjusted_priors.shape[0]),
            size=cls.SIMS,
            random_state=rng,
        )
        return [sim_priors[i, :] for i in range(sim_priors.shape[0])]


class KsWpaShiftTester(WpaShiftTester):
    HIST_GRID = EqualSpaceHistGrid(1000)
    DISTANCE_MEASURE = ks_distance


class HdWpaShiftTester(WpaShiftTester):
    HIST_GRID = EqualSpaceHistGrid(11)
    DISTANCE_MEASURE = hellinger_distance


class DynHdWpaShiftTester(WpaShiftTester):
    HIST_GRID = EqualProbHistGrid()
    DISTANCE_MEASURE = hellinger_distance


class KsCdtShiftTester(CdtShiftTester):
    HIST_GRID = EqualSpaceHistGrid(1000)
    DISTANCE_MEASURE = ks_distance


class HdCdtShiftTester(CdtShiftTester):
    HIST_GRID = EqualSpaceHistGrid(11)
    DISTANCE_MEASURE = hellinger_distance


class DynHdCdtShiftTester(CdtShiftTester):
    HIST_GRID = EqualProbHistGrid()
    DISTANCE_MEASURE = hellinger_distance


class MaxKsWpaShiftTester(KsWpaShiftTester):
    DISTANCE_AGGREGATOR = max_distance_aggregator


class MaxDynHdWpaShiftTester(DynHdWpaShiftTester):
    DISTANCE_AGGREGATOR = max_distance_aggregator


class MaxKsCdtShiftTester(KsCdtShiftTester):
    DISTANCE_AGGREGATOR = max_distance_aggregator


class MaxDynHdCdtShiftTester(DynHdCdtShiftTester):
    DISTANCE_AGGREGATOR = max_distance_aggregator
