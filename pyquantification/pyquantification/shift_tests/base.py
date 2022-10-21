from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import inspect
import numpy as np
from typing import Any, Dict, cast

from pyquantification.quantifiers.em import adjust_to_priors

Classes = np.ndarray
Probs = np.ndarray
Priors = np.ndarray


@dataclass
class TestResult():
    shift_detected: bool
    stats: Dict[str, Any] = field(default_factory=dict)


class ShiftTester(ABC):
    """Contains all computations related to a particular shift test."""

    @classmethod
    def run(cls, *,
            classes: Classes,
            target_probs: Probs,
            **kwargs: Any) -> Dict[str, TestResult]:
        """Given the list of classes, and at least a target_probs array (row
        per instance, col per class), return a dict mapping classes
        to TestResults
        """
        if target_probs.shape[0] == 0:
            # Return empty results when there are no instances to quantify.
            return {
                target_class: TestResult(shift_detected=False)
                for target_class in classes
            }
        # Limit kwargs to those supported by the particular test.
        class_run_params = set(inspect.signature(cls._run).parameters.keys())
        return cls._run(classes=classes, target_probs=target_probs, **{
            key: value for key, value in kwargs.items()
            if key in class_run_params
        })

    @classmethod
    @abstractmethod
    def _run(cls, *,
             classes: Classes,
             target_probs: Probs,
             **kwargs: Any) -> Dict[str, TestResult]:
        """Inner class-specific run()."""
        pass


@dataclass
class ProbAdjustment:
    """Stores priors and probs related to an EM prior-probability adjustment."""

    source_priors: Priors
    orig_probs: Probs
    adjusted_priors: Priors
    target_probs: Probs

    @classmethod
    def from_class_intervals(cls, *,
                             classes: Classes,
                             class_intervals: Dict[str, Dict[str, Any]],
                             calib_probs: Probs,
                             target_probs: Probs) -> 'ProbAdjustment':
        """Construct a ProbAdjustment from EM class intervals."""
        source_priors = cast(Priors, calib_probs.mean(axis=0))
        adjusted_priors = np.array([
            (class_intervals[y_class]['count'] / target_probs.shape[0])
            for y_class in classes
        ])
        return cls(
            source_priors=source_priors,
            orig_probs=target_probs,
            adjusted_priors=adjusted_priors,
            target_probs=target_probs,
        )

    @property
    def adjusted_probs(self) -> Probs:
        if not hasattr(self, '_adjusted_probs'):
            self._adjusted_probs = adjust_to_priors(
                source_priors=self.source_priors,
                target_priors=self.adjusted_priors,
                probs=self.target_probs,
            )
        return self._adjusted_probs
