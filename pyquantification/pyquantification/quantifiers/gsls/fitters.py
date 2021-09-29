from abc import ABC, abstractmethod
from functools import partial
import numpy as np
import scipy.optimize
from typing import cast, Any, Dict, Sequence


class GslsFitter(ABC):

    @classmethod
    @abstractmethod
    def find_remain_hist(cls, *,
                         calib_hist: np.ndarray,
                         target_hist: np.ndarray,
                         random_state: int) -> np.ndarray:
        """Fits the GSLS model by finding the 'remaining' histogram for a
        given calibration and target histogram."""
        pass


class WeightSumFitter(GslsFitter):
    """
    GSLS fitting method that minimizes: `gain_weight + loss_weight`.
    """

    @classmethod
    def get_seed_hists(cls, *,
                       calib_hist: np.ndarray,
                       target_hist: np.ndarray,
                       random_state: int) -> Sequence[np.ndarray]:
        """Provide a list of random seed states for the optimisation problem."""
        rng = np.random.RandomState(random_state)
        seed_hists = [rng.dirichlet(np.ones(calib_hist.shape)) for _ in range(10)]
        return seed_hists

    @classmethod
    def target_func(cls, remain_hist: np.ndarray, *,
                    calib_hist: np.ndarray,
                    target_hist: np.ndarray) -> float:
        """We want to find the "remain" distribution histogram that maximizes
        the sum of minimum calib/remain and target/remain bin ratios
        (which are equivalent to (1 - loss_weight) and (1 -
        gain_weight) respectively, so this is equivalent to minimizing
        (loss_weight + gain_weight)). Excludes remain_hist bins with
        probability of zero to avoid division by zero - these bins
        will always fit inside calib/target.
        """
        remain_nonzero_mask = (remain_hist != 0)
        masked_remain_hist = remain_hist[remain_nonzero_mask]
        masked_calib_hist = calib_hist[remain_nonzero_mask]
        masked_target_hist = target_hist[remain_nonzero_mask]

        max_target = (
            np.min(masked_calib_hist / masked_remain_hist) +
            np.min(masked_target_hist / masked_remain_hist)
        )
        return -max_target

    @classmethod
    def target_jacobian(cls, remain_hist: np.ndarray, *,
                        calib_hist: np.ndarray,
                        target_hist: np.ndarray) -> np.ndarray:
        """
        Jacobian (i.e. gradient) function for target_func.
        """
        remain_nonzero_mask = (remain_hist != 0)
        masked_remain_hist = remain_hist[remain_nonzero_mask]
        masked_calib_hist = calib_hist[remain_nonzero_mask]
        masked_target_hist = target_hist[remain_nonzero_mask]

        masked_jac = np.zeros(masked_remain_hist.shape)
        for masked_match_hist in [masked_calib_hist, masked_target_hist]:
            ratios = masked_match_hist / masked_remain_hist
            # nonzero() returns a tuple of arrays (one for each
            # dimension), so we just take the first (and only)
            # dimension by using [0].
            min_ratio_indexes = np.asarray(ratios == np.min(ratios)).nonzero()[0]
            # By the definition of the gradient for min (see use of
            # Heaviside in Min gradient:
            # https://github.com/sympy/sympy/blob/46e00feeef5204d896a2fbec65390bd4145c3902/sympy/functions/elementary/miscellaneous.py#L837
            # and gradient for Heaviside:
            # https://docs.sympy.org/latest/modules/functions/special.html#sympy.functions.special.delta_functions.Heaviside),
            # only the bins responsible for the results of min in the
            # target function have non-zero gradients. In very rare
            # circumstances, min_ratio_indexes may have length > 1
            # (multiple ratios are the same minimum). Technically, the
            # derivative is undefined for multiple minima, but we
            # assign the derivative to all minimal indexes, which is
            # commonly the behaviour of the automatic finite
            # difference gradient computation.
            masked_jac[min_ratio_indexes] += (masked_match_hist[min_ratio_indexes] /
                                              masked_remain_hist[min_ratio_indexes]**2)
        # For bins where calib_hist and target_hist are zero, the
        # gradient is technically undefined. To to prevent an error we
        # treat it as a gradient of zero, which is the same as the
        # automatic finite difference gradient computation.
        jac = np.full(remain_hist.shape, fill_value=0.0)
        jac[remain_nonzero_mask] = masked_jac
        return jac

    @classmethod
    def equality_constraint_func(cls, remain_hist: np.ndarray) -> np.number:
        """Equality constraint that passes when result is zero."""
        # Ensure remain_hist sums to 1.
        return 1 - remain_hist.sum()

    @classmethod
    def equality_constraint_jacobian(cls, remain_hist: np.ndarray) -> np.ndarray:
        """Jacobian of constraint func is always -1 for all elements in remain_hist."""
        return np.full(remain_hist.shape, fill_value=-1.0)

    @classmethod
    def optimize_remain_hist(cls, *,
                             calib_hist: np.ndarray,
                             target_hist: np.ndarray,
                             seed_hist: np.ndarray,
                             max_iterations: int = 1500) -> Dict[str, Any]:
        """Find the remain_hist that minimizes difference from calib_hist and
        target_hist. Initialise minimization of remain_hist from the
        given seed_hist."""
        result = scipy.optimize.minimize(
            partial(cls.target_func, calib_hist=calib_hist, target_hist=target_hist),
            jac=partial(cls.target_jacobian, calib_hist=calib_hist, target_hist=target_hist),
            x0=seed_hist,
            # Ensure there are no negative values in the histogram.
            bounds=scipy.optimize.Bounds(lb=0, ub=np.inf),
            constraints={
                'type': 'eq',
                'fun': cls.equality_constraint_func,
                'jac': cls.equality_constraint_jacobian,
            },
            options={'maxiter': max_iterations},
            # Supports constraints and bounds.
            method='SLSQP',
        )

        converged = True
        if not result.success:
            # Handle known possible convergence failures, but still
            # allow the result minimised to this point to be a
            # candidate for selection if it achieves the smallest
            # value of the target function.
            if result.message in {
                    'Positive directional derivative for linesearch',
                    'Inequality constraints incompatible',
                    'More than 3*n iterations in LSQ subproblem',
            }:
                converged = False
            else:
                raise ValueError(result.message)

        return {
            'min_target_value': result.fun,
            # Final normalisation of remain_hist.
            'remain_hist': result.x / np.sum(result.x),
            'converged': converged,
            'message': result.message,
        }

    @classmethod
    def find_remain_hist(cls, *,
                         calib_hist: np.ndarray,
                         target_hist: np.ndarray,
                         random_state: int) -> np.ndarray:
        """Find the remain_hist that minimizes difference from calib_hist and
        target_hist. Does not optimize histogram bins that are zero in
        either calib_hist or target_hist (as the optimal remain_hist
        must also be zero for such bins)."""
        both_nonzero_mask = ((calib_hist != 0) & (target_hist != 0))

        remain_hist = np.zeros(calib_hist.shape)
        if not np.any(both_nonzero_mask):
            return remain_hist

        masked_calib_hist = calib_hist[both_nonzero_mask]
        masked_target_hist = target_hist[both_nonzero_mask]
        seed_hists = cls.get_seed_hists(calib_hist=masked_calib_hist,
                                        target_hist=masked_target_hist,
                                        random_state=random_state)
        results = [cls.optimize_remain_hist(calib_hist=masked_calib_hist,
                                            target_hist=masked_target_hist,
                                            seed_hist=seed_hist)
                   for seed_hist in seed_hists]
        best_result_index = np.argmin([result['min_target_value'] for result in results])
        best_result = results[cast(int, best_result_index)]
        # Fill in optimized nonzero bins in remain_hist, leaving
        # remaining bins at value zero.
        remain_hist[both_nonzero_mask] = best_result['remain_hist']
        return remain_hist
