from abc import abstractmethod
from collections import Counter
from typing import cast, Dict

import numpy as np
import pandas as pd

from pyquantification.utils import mask_to_indexes
from pyquantification.quantifiers.base import (
    Quantifier, Classes, Probs, Priors, PredictionInterval, QuantificationError)
from pyquantification.quantifiers.em import adjust_priors_with_em


class BootstrapQuantifier(Quantifier):
    """Base class for quantification methods that produce prediction
    intervals through the bootstrapping method proposed in: Tasche, D.
    (2019). Confidence intervals for class prevalences under prior
    probability shift. Machine Learning and Knowledge Extraction,
    1(3), 805-831."""
    R = 999
    PREDICTION_INTERVAL = True

    @classmethod
    def _quantify(cls, *,  # type: ignore[override]
                  classes: Classes,
                  target_probs: Probs,
                  prediction_interval_mass: float,
                  calib_probs: Probs,
                  calib_y: pd.Series,
                  random_state: int) -> Dict[str, PredictionInterval]:
        rng = np.random.RandomState(random_state)
        # Separate rng for prediction interval sampling so that
        # PREDICTION_INTERVAL does not alter the bootstrapping.
        rng_b = np.random.RandomState(rng.randint(2**32))

        target_count = target_probs.shape[0]
        target_indexes = np.arange(target_count)
        calib_y_array = calib_y.to_numpy()
        class_calib_indexes = {
            target_class: mask_to_indexes(calib_y_array == target_class)
            for target_class in classes
        }

        sample_quantification_list = []
        for _ in range(cls.R):
            # Re-sample the calibration and target sets
            sample_calib_indexes = np.concatenate([
                rng.choice(class_calib_indexes[target_class], size=class_calib_indexes[target_class].shape[0], replace=True)
                for target_class in classes
            ], axis=0)
            sample_calib_probs = calib_probs[sample_calib_indexes]
            sample_target_indexes = rng.choice(target_indexes, size=target_count, replace=True)
            sample_target_probs = target_probs[sample_target_indexes]
            # Quantify the bootstrapped sample
            adjusted_prior = cls.get_point_quantification(
                classes=classes,
                target_probs=sample_target_probs,
                calib_probs=sample_calib_probs,
            )
            if cls.PREDICTION_INTERVAL:
                # We will compute a prediction interval on sample
                # counts simulated from adjusted priors
                adjusted_sample_class_counts: Counter = Counter(rng_b.choice(
                    classes, size=target_count, replace=True, p=adjusted_prior))
                adjusted_sample_prior = np.array([
                    adjusted_sample_class_counts[target_class] / target_count
                    for target_class in classes
                ])
                sample_quantification_list.append(adjusted_sample_prior)
            else:
                # We will compute a confidence interval on the adjusted priors
                sample_quantification_list.append(adjusted_prior)
        sample_qs = np.array(sample_quantification_list)

        lower_alpha = (1 - prediction_interval_mass) / 2
        upper_alpha = (1 + prediction_interval_mass) / 2
        intervals = {}
        for class_index, target_class in enumerate(classes):
            class_sample_qs = sample_qs[:, class_index]
            intervals[target_class] = PredictionInterval(
                prediction=cast(float, np.mean(class_sample_qs)) * target_count,
                # Use ceil/floor to round to slightly wider bounds that
                # represent discrete counts.
                lower=int(np.floor(np.quantile(class_sample_qs, lower_alpha) * target_count)),
                upper=int(np.ceil(np.quantile(class_sample_qs, upper_alpha) * target_count)),
                stats={
                    'R': cls.R,
                    'PREDICTION_INTERVAL': cls.PREDICTION_INTERVAL,
                },
            )
        return intervals

    @classmethod
    @abstractmethod
    def get_point_quantification(cls, *,
                                 classes: Classes,
                                 target_probs: Probs,
                                 calib_probs: Probs) -> Priors:
        """Performs quantification of a single target sample."""


class EmBootstrapQuantifier(BootstrapQuantifier):
    """EM quantification with bootstrapped prediction intervals."""

    @classmethod
    def get_point_quantification(cls, *,
                                 classes: Classes,
                                 target_probs: Probs,
                                 calib_probs: Probs) -> Priors:
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

        adjusted_priors, _ = adjust_priors_with_em(
            source_priors=source_priors,
            probs=target_probs,
        )
        return adjusted_priors
