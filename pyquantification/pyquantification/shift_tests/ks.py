# coding: utf-8

import scipy.stats
from typing import Dict

from pyquantification.shift_tests.base import (
    ShiftTester, TestResult, Classes, Probs
)


class KsShiftTester(ShiftTester):
    """Performs a Kolmogorov-Smirnoff test for dataset shift between the
    calib_probs and target_probs, as used by:

    * Lipton, Z., Wang, Y. X., & Smola, A. (2018). Detecting and
      correcting for label shift with black box predictors. In
      International conference on machine learning
      (pp. 3122-3130). PMLR.
    * Rabanser, S., Günnemann, S., & Lipton, Z. C. (2018). Failing
      loudly: An empirical study of methods for detecting dataset
      shift. arXiv preprint arXiv:1810.11953.

    Performs multiple tests with Bonferroni correction for multi-class
    datasets (as used by Rabanser et al.).

    """

    @classmethod
    def _run(cls, *,  # type: ignore[override]
             classes: Classes,
             calib_probs: Probs,
             target_probs: Probs,
             test_alpha: float) -> Dict[str, TestResult]:
        if len(classes) == 2:
            _, p_value = scipy.stats.ks_2samp(
                calib_probs[:, 0],
                target_probs[:, 0],
                # Issues have been reported with large sample sizes
                # and 'exact' mode
                mode='asymp',
            )
            class_p_values = {y_class: p_value for y_class in classes}
            n_tests = 1
            shift_detected = p_value <= test_alpha
        else:
            class_p_values = {}
            for y_idx, y_class in enumerate(classes):
                _, class_p_value = scipy.stats.ks_2samp(
                    calib_probs[:, y_idx],
                    target_probs[:, y_idx],
                    # Issues have been reported with large sample
                    # sizes and 'exact' mode
                    mode='asymp',
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
