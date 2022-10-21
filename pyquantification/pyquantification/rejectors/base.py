from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
from math import floor
from time import process_time_ns
import warnings
from typing import Any, Dict, List, cast

import cvxpy as cp
import numpy as np
import pandas as pd
from xpress import SolverError

from pyquantification.utils import mask_to_indexes
from pyquantification.quantifiers.base import PredictionInterval

Classes = np.ndarray
Probs = np.ndarray
Cache = Dict[str, Any]


@dataclass
class RejectResult:
    rejected_indexes: np.ndarray
    post_class_intervals: Dict[str, PredictionInterval]


class Rejector(ABC):

    @classmethod
    def run(cls, *,
            classes: Classes,
            target_y: pd.Series,
            target_probs: Probs,
            **kwargs: Any) -> RejectResult:
        """Given the list of classes, and at least a target_probs array (row
        per instance, col per class), return a dict mapping classes
        to RejectResults
        """
        if target_probs.shape[0] == 0:
            # Return empty results when there are no instances to reject.
            return RejectResult(
                rejected_indexes=np.array([]),
                post_class_intervals={
                    target_class: PredictionInterval(0, 0, 0)
                    for target_class in classes
                },
            )
        return cls._run(
            classes=classes,
            target_y=target_y,
            target_probs=target_probs,
            # Limit kwargs to those supported by the particular rejector.
            **{
                key: value for key, value in kwargs.items()
                if key in set(inspect.signature(cls._run).parameters.keys())
            },
        )

    @classmethod
    @abstractmethod
    def _run(cls, *,
             classes: Classes,
             target_y: pd.Series,
             target_probs: Probs,
             **kwargs: Any) -> RejectResult:
        """Inner class-specific run()."""


class IntervalRejector(Rejector):
    """Abstract base class for rejectors that select the instances to
    reject that will achieve desired interval widths."""

    @classmethod
    def _run(cls, *,  # type: ignore[override]
             classes: Classes,
             target_y: pd.Series,
             target_probs: Probs,
             calib_probs: Probs,
             quantification_method_results: Dict[str, Dict[str, Any]],
             prediction_interval_mass: float,
             rejection_limit: str,
             random_state: int) -> RejectResult:
        assert 0 <= prediction_interval_mass <= 1

        cache_start_ns = process_time_ns()
        # Pre-compute and store values that are repeatedly used
        # repeatedly by the rejector.
        cache = cls.build_cache(
            classes=classes,
            target_probs=target_probs,
            calib_probs=calib_probs,
            quantification_method_results=quantification_method_results,
        )
        cache_end_ns = process_time_ns()
        # Get the maximum allowed interval width for each class, as
        # determined by the rejection_limit.
        class_interval_width_limits = cls.get_class_interval_width_limits(
            cache=cache,
            classes=classes,
            target_probs=target_probs,
            prediction_interval_mass=prediction_interval_mass,
            rejection_limit=rejection_limit,
            random_state=random_state,
        )
        assert class_interval_width_limits.shape == classes.shape
        rejection_start_ns = process_time_ns()
        # Get a mask indicating which instances should be rejected to
        # reduce class intervals to their target widths.
        rejected_mask = cls.get_rejected_mask(
            cache=cache,
            classes=classes,
            target_probs=target_probs,
            prediction_interval_mass=prediction_interval_mass,
            class_interval_width_limits=class_interval_width_limits,
            random_state=random_state,
        )
        rejection_end_ns = process_time_ns()
        assert rejected_mask.shape == (target_probs.shape[0],)
        # Get the final class intervals for the selected/non-rejected instances.
        selected_class_intervals = cls.get_class_intervals(
            cache=cache,
            prediction_interval_mass=prediction_interval_mass,
            selection_mask=~rejected_mask,
            random_state=random_state,
        )
        rejected_class_counts = cast(pd.Series, target_y[rejected_mask]).value_counts().to_dict()

        # Update the final class intervals with the true labels of the
        # rejected instances.
        post_class_intervals = {}
        for target_class, interval_width_limit in zip(classes, class_interval_width_limits):
            selected_interval = selected_class_intervals[target_class]
            rejected_class_count = rejected_class_counts.get(target_class, 0)
            post_class_intervals[target_class] = PredictionInterval(
                prediction=selected_interval.prediction + rejected_class_count,
                lower=selected_interval.lower + rejected_class_count,
                upper=selected_interval.upper + rejected_class_count,
                stats={
                    'target_width_limit': interval_width_limit * target_probs.shape[0],
                    'all_class_time_ns': (
                        (cache_end_ns - cache_start_ns) +
                        (rejection_end_ns - rejection_start_ns)
                    )
                },
            )

        return RejectResult(
            rejected_indexes=mask_to_indexes(rejected_mask),
            post_class_intervals=post_class_intervals,
        )

    @classmethod
    def get_class_interval_width_limits(
            cls, *,
            cache: Cache,
            classes: Classes,
            target_probs: Probs,
            prediction_interval_mass: float,
            rejection_limit: str,
            random_state: int,
    ) -> np.ndarray:
        """Parse the given rejection_limit and return the 0-1 scale absolute
        interval width limit for each class."""
        limit_parts = str(rejection_limit).split(':')
        if len(limit_parts) != 2:
            raise ValueError(f'Invalid rejection_limit: {rejection_limit}')
        limit_type, limit_value_str = limit_parts

        try:
            limit_value = float(limit_value_str)
        except ValueError:
            raise ValueError(f'Invalid numeric limit value: {limit_value_str}')

        initial_class_intervals = cls.get_class_intervals(
            cache=cache,
            prediction_interval_mass=prediction_interval_mass,
            selection_mask=np.full(target_probs.shape[0], fill_value=True),
            random_state=random_state,
        )
        assert len(initial_class_intervals) == classes.shape[0]

        if limit_type == 'abs':
            # Absolute limit: expressed in absolute proportion points.
            return np.full(shape=classes.shape, fill_value=limit_value)
        elif limit_type == 'rel':
            # Relative limit: expressed as a proportion of the target
            # point prediction's value.
            initial_class_predictions = np.array([
                initial_class_intervals[target_class].prediction / target_probs.shape[0]
                for target_class in classes
            ])
            return initial_class_predictions * limit_value
        elif limit_type == 'frac':
            # Fractional limit: expressed as a proportion of the
            # original interval width.
            initial_class_interval_widths = np.array([
                ((initial_class_intervals[target_class].upper
                  - initial_class_intervals[target_class].lower)) / target_probs.shape[0]
                for target_class in classes
            ])
            return initial_class_interval_widths * limit_value
        elif limit_type == 'fracmax':
            # Fractional of max limit: expressed as a proportion of
            # the maximum original interval width across classes.
            max_initial_class_interval_width = np.max([
                ((initial_class_intervals[target_class].upper
                  - initial_class_intervals[target_class].lower)) / target_probs.shape[0]
                for target_class in classes
            ])
            return np.full(shape=classes.shape, fill_value=(max_initial_class_interval_width * limit_value))
        else:
            raise ValueError(f'Invalid limit type: {limit_type}')

    @classmethod
    @abstractmethod
    def build_cache(
            cls, *,
            classes: Classes,
            target_probs: Probs,
            calib_probs: Probs,
            quantification_method_results: Dict[str, Dict[str, Any]],
    ) -> Cache:
        """Pre-compute and store values that are repeatedly used repeatedly by
        the rejector."""

    @classmethod
    @abstractmethod
    def get_rejected_mask(
            cls, *,
            cache: Cache,
            classes: Classes,
            target_probs: Probs,
            prediction_interval_mass: float,
            class_interval_width_limits: np.ndarray,
            random_state: int,
    ) -> np.ndarray:
        """Get a mask indicating which instances should be rejected to reduce
        class intervals to their target widths."""

    @classmethod
    @abstractmethod
    def get_class_intervals(
            cls, *,
            cache: Cache,
            prediction_interval_mass: float,
            selection_mask: np.ndarray,
            random_state: int,
    ) -> Dict[str, PredictionInterval]:
        """Get the class intervals for only the selected (non-rejected) instances."""

    @classmethod
    @abstractmethod
    def get_interval_width_for_variance(
            cls, *,
            variance: float,
            interval_mass: float,
    ) -> float:
        """Given the mean, variance for a prediction distribution, and the
        desired prediction interval mass, return the prediction
        interval width.

        This should be an equal-tailed interval, so that we are not biased
        to over or under estimate."""


class ProbThresholdRejector(IntervalRejector):
    """Abstract base class for rejectors that use a threshold on probabilities to
    select the instances to reject that will achieve the desired interval."""

    @classmethod
    def get_rejected_mask(
            cls, *,
            cache: Cache,
            classes: Classes,
            target_probs: Probs,
            prediction_interval_mass: float,
            class_interval_width_limits: np.ndarray,
            random_state: int,
    ) -> np.ndarray:
        assert np.all(0 <= class_interval_width_limits)
        assert np.all(class_interval_width_limits <= 1)
        # Scale the 0-1 interval-width limits to the count of instances.
        class_count_interval_width_limits = class_interval_width_limits * target_probs.shape[0]

        # We perform a binary search to reject the minimal number of
        # instances with the lowest maximum class-probability
        # (i.e. lowest confidence) in order to achieve the interval
        # width limits.
        best_to_worst_target_indexes = np.argsort(-target_probs.max(axis=1))

        lower_selected_n = 0
        # Upper starts at (max + 1) so that the floor() used to update
        # selected_n can take us to selecting all instances
        # (target_probs.shape[0]).
        upper_selected_n = target_probs.shape[0] + 1
        while True:
            selected_n = int(floor((lower_selected_n + upper_selected_n) / 2))
            selected_indexes = best_to_worst_target_indexes[:selected_n]
            selection_mask = np.full(target_probs.shape[0], fill_value=False)
            selection_mask[selected_indexes] = True
            # Get the intervals for the selected instances.
            class_count_interval_widths = cls.get_class_count_interval_widths(
                cache=cache,
                classes=classes,
                prediction_interval_mass=prediction_interval_mass,
                selection_mask=selection_mask,
                random_state=random_state,
            )
            valid_selection = np.all(class_count_interval_widths <= class_count_interval_width_limits)
            if valid_selection:
                # Because we set selected_n with floor, and only
                # update lower_selected_n when valid, we are
                # guaranteed to converge to a valid selection (even if
                # the only valid_selection is no instances).
                if (upper_selected_n - lower_selected_n) <= 1:
                    rejected_mask = ~selection_mask
                    return rejected_mask
                # Select more (reject fewer) instances
                lower_selected_n = selected_n
            else:
                # Select fewer (reject more) instances
                upper_selected_n = selected_n

    @classmethod
    def get_class_count_interval_widths(
            cls, *,
            cache: Cache,
            classes: Classes,
            prediction_interval_mass: float,
            selection_mask: np.ndarray,
            random_state: int,
    ) -> np.ndarray:
        """Get the count-scale interval widths for a given set of selected
        instances. Used during the search for a prob-threshold."""
        # Use the same interval computation as used for the final
        # intervals.
        class_intervals = cls.get_class_intervals(
            cache=cache,
            prediction_interval_mass=prediction_interval_mass,
            selection_mask=selection_mask,
            random_state=random_state,
        )
        return np.array([
            class_intervals[target_class].upper - class_intervals[target_class].lower
            for target_class in classes
        ])


class ApproxProbThresholdRejector(ProbThresholdRejector):
    """Abstract base class for approximate probability-threshold
    rejectors.

    These differ from the base probability-threshold rejectors by
    computing the interval widths during the probability-threshold
    search using the variance-based approximation of intervals that
    assumes a particular distribution shape.

    Because this strategy uses the same distribution assumption as
    MipRejector, it provides insight into the differences of the
    prob-threshold and MIP strategies for the same interval width
    computation. This strategy will also typically be faster than
    using get_class_intervals().
    """

    @classmethod
    def get_class_count_interval_widths(
            cls, *,
            cache: Cache,
            classes: Classes,
            prediction_interval_mass: float,
            selection_mask: np.ndarray,
            random_state: int,
    ) -> np.ndarray:
        selected_n = selection_mask.sum()
        if selected_n == 0:
            return np.full(len(classes), fill_value=0.0)

        # Get count-scale variances.
        class_count_variances = cls.get_class_count_variances(
            cache=cache,
            selection_mask=selection_mask,
        )
        # Convert count-scale variances to 0-1 scale.
        class_variances = class_count_variances / (selected_n ** 2)
        interval_widths = np.array([
            cls.get_interval_width_for_variance(
                variance=variance,
                interval_mass=prediction_interval_mass,
            )
            for variance in class_variances
        ])
        # Convert 0-1 scale interval widths to count scale.
        count_interval_widths = np.array(interval_widths) * selected_n
        return count_interval_widths

    @classmethod
    @abstractmethod
    def get_class_count_variances(
            cls, *,
            cache: Cache,
            selection_mask: np.ndarray,
    ) -> np.ndarray:
        """Return the variances of each class prediction distribution for a
        given set of selected instances."""


class MipRejector(IntervalRejector):
    """Abstract base class for rejectors that use Mixed-Integer-Programming to
    select the instances to reject that will achieve the desired interval."""

    @classmethod
    def get_rejected_mask(
            cls, *,
            cache: Cache,
            classes: Classes,
            target_probs: Probs,
            prediction_interval_mass: float,
            class_interval_width_limits: np.ndarray,
            random_state: int,
    ) -> np.ndarray:
        target_n = target_probs.shape[0]
        # Convert each interval width limit to a limit on variance.
        class_variance_limits = np.array([
            cls.find_variance_for_interval_width(
                interval_mass=prediction_interval_mass,
                interval_width=interval_width_limit,
            )
            for interval_width_limit in class_interval_width_limits
        ])
        # Scale the variance limits from 0-1 scale to count scale.
        class_count_variance_limits = class_variance_limits * (target_n ** 2)

        # Boolean vector indicating whether an instance is
        # selected for machine classification (the inverse of a
        # rejection mask)
        selection_mask = cp.Variable(target_n, boolean=True)
        assert len(classes) == target_probs.shape[1] == class_count_variance_limits.shape[0]
        # Set up variance constraints for the MIP problem.
        constraints = cls.get_cvxpy_constraints(
            cache=cache,
            selection_mask=selection_mask,
            class_count_variance_limits=class_count_variance_limits,
        )
        # We want to maximize the number of instances selected (i.e. minimize rejection).
        objective = cp.Maximize(cp.sum(selection_mask))
        problem = cp.Problem(objective, constraints)
        # 15 minute timeout
        timeout_seconds = int(15 * 60)
        with warnings.catch_warnings():
            # We already handle the optimal_inaccurate condition
            # below, so ignore this warning.
            warnings.filterwarnings(
                'ignore',
                message=('Solution may be inaccurate. Try another solver, '
                         'adjusting the solver settings, or solve with '
                         'verbose=True for more information.'))
            solver_error = None
            try:
                problem.solve(
                    solver=cp.XPRESS,
                    maxtime=timeout_seconds,
                    # We have observed instances where the xpress solver
                    # fails to terminate (even after several hours). The
                    # problem appears similar to that described by:
                    # https://community.fico.com/s/question/0D52E00005z4VilSAE/xpress-solver-occasionally-gets-stuck-when-newton-barrier-algorithm-is-selected-to-solve-lp
                    # (as ours also occurs when the barrier method is
                    # used). We implement the suggested solution of
                    # manually setting the cvxpy solver to use the dual
                    # simplex method (see:
                    # https://www.fico.com/fico-xpress-optimization/docs/latest/solver/optimizer/HTML/LPFLAGS.html
                    # and the setControl() method in the PDF documentation
                    # that accompanies the xpress Python library). Some
                    # problems fail to terminate with dual simplex but not
                    # with the default barrier method, but we try dual
                    # simplex first because maxtime appears to be
                    # ineffective with the barrier method in at least some
                    # circumstances.
                    lpflags=0b0001,
                )
            except SolverError as ex:
                # This error is observed for a small number of
                # plankton dataset samples.
                if '722 Error: IIS number 0 is not yet identified' in str(ex):
                    solver_error = str(ex)
                else:
                    raise ex

        # The status will be optimal_inaccurate in the case of a timeout
        if problem.status == 'optimal_inaccurate':
            warnings.warn('MIP problem likely timed out; trying again with different solver options.')
            problem = cp.Problem(objective, constraints)
            problem.solve(
                solver=cp.XPRESS,
                maxtime=timeout_seconds,
            )

        if problem.status == 'infeasible':
            # The only known case of infeasible problems are where
            # full rejection is required, so check if there is any
            # solution selecting at least one instance.
            for single_instance_idx in range(target_n):
                single_instance_mask = np.zeros(target_n)
                single_instance_mask[single_instance_idx] = 1.0
                single_instance_class_count_variances = cls.get_class_count_variances(
                    cache=cache,
                    selection_mask=single_instance_mask,
                )
                if np.all(single_instance_class_count_variances <= class_count_variance_limits):
                    raise ValueError('Infesible cvxpy problem where selecting at least one instance is a valid solution')
            # If we couldn't find a solution that selects a single
            # instance, then full rejection is the correct solution:
            selected_mask = np.full(target_n, fill_value=False)
        elif solver_error is not None:
            # In the case of solver errors with unknown resolutions,
            # perform full rejection as a fallback.
            warnings.warn(f'Falling back to full rejection for MIP problem that failed with solver error: {solver_error}')
            selected_mask = np.full(target_n, fill_value=False)
        else:
            assert problem.status == 'optimal'
            # Check that boolean variable values are "almost" 0 or
            # 1 before rounding them to 0 or 1. (cvxpy cannot guarantee
            # better precision than about 1e-12:
            # https://github.com/cvxgrp/cvxpy/issues/286).
            assert np.all((np.abs(selection_mask.value) < 1e-2) |
                          (np.abs(1 - selection_mask.value) < 1e-2))
            selected_mask = np.round(np.array(selection_mask.value)).astype(bool)
        rejected_mask = ~selected_mask
        return rejected_mask

    @classmethod
    def find_variance_for_interval_width(
            cls, *,
            interval_mass: float,
            interval_width: float,
            precision: float = 1e-7,
    ) -> float:
        """For a known prediction interval mass and width, find the
        variance. Found by binary search, which stops when desired
        precision is reached."""
        lower_var = 0.0
        upper_var = None
        current_var = 0.01
        while True:
            current_interval_width = cls.get_interval_width_for_variance(
                variance=current_var,
                interval_mass=interval_mass,
            )
            # Check for breaking condition.
            if np.abs(current_interval_width - interval_width) <= precision:
                return current_var
            if current_interval_width > interval_width:
                # Search lower variance values.
                upper_var = current_var
                current_var = (current_var + lower_var) / 2
            elif current_interval_width < interval_width:
                # Search higher variance values.
                lower_var = current_var
                if upper_var is None:
                    current_var = current_var * 10
                else:
                    current_var = (current_var + upper_var) / 2

    @classmethod
    @abstractmethod
    def get_cvxpy_constraints(
            cls, *,
            cache: Cache,
            selection_mask: cp.Variable,
            class_count_variance_limits: np.ndarray,
    ) -> List[cp.constraints.constraint.Constraint]:
        """Get the constraints for the MIP problem that ensure the variances
        of the class prediction distributions for selected instances
        are within the given limits."""

    @classmethod
    @abstractmethod
    def get_class_count_variances(
            cls, *,
            cache: Cache,
            selection_mask: np.ndarray,
    ) -> np.ndarray:
        """Return the variances of each class prediction distribution for a
        given set of selected instances."""
