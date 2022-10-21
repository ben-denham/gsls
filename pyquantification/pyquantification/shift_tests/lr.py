import numpy as np
import scipy.stats
from typing import Any, Dict, Tuple

from pyquantification.shift_tests.base import (
    ShiftTester, TestResult, Classes, Probs, ProbAdjustment
)


def log_likelihood_shift_test(*,
                              orig_class_probs: Probs,
                              source_class_prior: float,
                              adjusted_class_probs: Probs,
                              adjusted_class_prior: float,
                              dof: int = 1) -> Tuple[float, float]:
    """When the LLR is high (and therefore p value is close to zero), we
    reject the null-hypothesis that the distribution hasn't shifted -
    therefore the distribution is likely to have shifted.
    """
    n = orig_class_probs.shape[0]
    # Because some probs may be zero, we add a small eps term to each
    # prob.
    eps = 0.001 * (1 / n)
    log_likelihood_ratio = (
        np.log(orig_class_probs + eps).sum()
        - np.log(adjusted_class_probs + eps).sum()
        + (n * np.log(adjusted_class_prior))
        - (n * np.log(source_class_prior))
    )
    # 2 * Log-Likelihood Ratio is distributed as Chi2 with n_classes-1
    # degrees of freedom.
    likelihood_ratio_p = scipy.stats.chi2.sf((2 * log_likelihood_ratio), df=dof)
    return log_likelihood_ratio, likelihood_ratio_p


class LrShiftTester(ShiftTester):
    """Likelihood Ratio Test for prior shift, as proposed by:

    * Saerens, M., Latinne, P., & Decaestecker, C. (2002). Adjusting
      the outputs of a classifier to new a priori probabilities: a
      simple procedure. Neural computation, 14(1), 21-41.

    Performs multiple tests with Bonferroni correction for multi-class
    datasets.

    """

    @classmethod
    def _run(cls, *,  # type: ignore[override]
             classes: Classes,
             calib_probs: Probs,
             target_probs: Probs,
             quantification_method_results: Dict[str, Dict[str, Any]],
             test_alpha: float) -> Dict[str, TestResult]:
        adjustment = ProbAdjustment.from_class_intervals(
            classes=classes,
            class_intervals=quantification_method_results['em']['class_intervals'],
            calib_probs=calib_probs,
            target_probs=target_probs,
        )

        if len(classes) == 2:
            _, p_value = log_likelihood_shift_test(
                orig_class_probs=adjustment.orig_probs[:, 0],
                source_class_prior=adjustment.source_priors[0],
                adjusted_class_probs=adjustment.adjusted_probs[:, 0],
                adjusted_class_prior=adjustment.adjusted_priors[0],
                # Degrees of freedom = n_classes - 1 (all classes were optimised at once).
                dof=(len(classes) - 1),
            )
            class_p_values = {y_class: p_value for y_class in classes}
            n_tests = 1
            shift_detected = p_value <= test_alpha
        else:
            class_p_values = {}
            for y_idx, y_class in enumerate(classes):
                _, class_p_value = log_likelihood_shift_test(
                    orig_class_probs=adjustment.orig_probs[:, y_idx],
                    source_class_prior=adjustment.source_priors[y_idx],
                    adjusted_class_probs=adjustment.adjusted_probs[:, y_idx],
                    adjusted_class_prior=adjustment.adjusted_priors[y_idx],
                    # Degrees of freedom = n_classes - 1 (all classes were optimised at once).
                    dof=(len(classes) - 1),
                )
                class_p_values[y_class] = class_p_value
            # Perform a single test for shift across all classes by
            # checking if any class prior has shifted (using Bonferroni
            # correction on the test alpha that conservatively assumes all
            # tests are independent).
            n_tests = len(classes)
            bonferroni_test_alpha = test_alpha / n_tests
            p_value = min(class_p_values.values())
            shift_detected = p_value <= bonferroni_test_alpha

        return {
            y_class: TestResult(
                shift_detected=shift_detected,
                stats=dict(
                    p_value=p_value,
                    n_tests=n_tests,
                    class_p_value=class_p_values[y_class],
                ),
            )
            for y_class in classes
        }
