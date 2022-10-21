import numpy as np
import scipy.stats
from typing import Any, Dict

from pyquantification.shift_tests.base import (
    ShiftTester, TestResult, Classes, Probs, ProbAdjustment,
)
from pyquantification.shift_tests.dists import (
    HistGrid,
    EqualSpaceHistGrid,
    get_hists,
    mix_per_class_hists,
    ks_distance,
)


class AksShiftTester(ShiftTester):
    """Shift test that performs a 2-sample KS-test between the target
    sample and the best-fit mixture of class-conditional calibration
    distributions (with class proportions determined by EM).

    Performs multiple tests with Bonferroni correction for multi-class
    datasets.

    """

    HIST_GRID: HistGrid = EqualSpaceHistGrid(1000)

    @classmethod
    def _run(cls, *,  # type: ignore[override]
             classes: Classes,
             calib_y: np.ndarray,
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
        bin_edges = cls.HIST_GRID.get_bin_edges(calib_probs=calib_probs, target_probs=target_probs)

        # Produce class-conditional Hists of classifier output probs
        # from the calibration set.
        calib_per_class_probs = [
            calib_probs[(calib_y == target_class)] for target_class in classes
        ]
        calib_per_class_hists = [get_hists(class_probs, bin_edges=bin_edges)
                                 for class_probs in calib_per_class_probs]

        class_distances = [
            ks_distance(hist_a, hist_b) for hist_a, hist_b in zip(
                mix_per_class_hists(class_priors=adjustment.adjusted_priors,
                                    per_class_hists=calib_per_class_hists),
                get_hists(target_probs, bin_edges=bin_edges),
            )
        ]
        distance = np.max(class_distances)
        # Based on implementation in scipy.stats.ks_2samp
        p_value = scipy.stats.distributions.kstwo.sf(distance, np.round(
            (calib_probs.shape[0] * target_probs.shape[0]) /
            (calib_probs.shape[0] + target_probs.shape[0])
        ))

        if len(classes) > 2:
            bonferroni_test_alpha = test_alpha / len(classes)
            shift_detected = p_value <= bonferroni_test_alpha
        else:
            shift_detected = p_value <= test_alpha

        return {
            y_class: TestResult(
                shift_detected=shift_detected,
                stats=dict(
                    p_value=p_value,
                    class_distance=class_distances[y_idx],
                ),
            )
            for y_idx, y_class in enumerate(classes)
        }
