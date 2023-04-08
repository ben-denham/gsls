from concurrent.futures import Future
from datetime import datetime
import os
from joblib import dump, load
from math import ceil
import numpy as np
import pandas as pd
from time import process_time_ns
from sklearn.base import ClassifierMixin
from typing import cast, Any, Union, Optional, Dict, Sequence, Tuple, List

from pyquantification.utils import prefix_keys, dict_first
from pyquantification.datasets import DATASETS, Dataset, ConceptsDataset, SamplesDataset
from pyquantification.classifiers import CLASSIFIERS, SourceProbClassifier
from pyquantification.quantifiers import QUANTIFIERS
from pyquantification.shift_tests import SHIFT_TESTERS
from pyquantification.rejectors import REJECTORS
from pyquantification.experiments.executors import get_executor, as_completed, Executor
from pyquantification.experiments.splitting import (split_train_calib, split_test,
                                                    get_class_priors_for_components)
from pyquantification.experiments.metrics import prepare_results
from pyquantification.experiments.calibration import PrefitCalibratedClassifier

# Suppress message about community license.
import xpress
xpress.init('/home/jovyan/work/.pip-packages/lib/python3.8/site-packages/xpress/license/community-xpauth.xpr')

VERBOSE = False


def log(message: str) -> None:
    print(f'{message} - {datetime.now()}')


# CACHING

CACHE_DIR = 'results'


def cache_filepath(key: str) -> str:
    return os.path.join(CACHE_DIR, f'{key}.joblib')


def is_cached(key: str) -> bool:
    return os.path.isfile(cache_filepath(key))


def save_to_cache(key: str, obj: Any) -> None:
    filepath = cache_filepath(key)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    dump({key: obj, 'timestamp': datetime.now()}, filepath)


def load_from_cache(key: str) -> Any:
    return load(cache_filepath(key))[key]


# CLASSIFICATION

def classification_cache_key(*,
                             dataset_name: str, classifier_name: str,
                             loss_weight: float, random_priors: bool,
                             random_state: int) -> str:
    """Cache key/path for the given experiment classification configuration."""
    return os.path.join(
        dataset_name,
        classifier_name,
        f'lw{loss_weight}-r{random_state}-rp{random_priors}',
        f'classification-{dataset_name}-{classifier_name}-lw{loss_weight}-rp{random_priors}-r{random_state}',
    )


def normalise_probs(probs: np.ndarray) -> np.ndarray:
    """
    Normalise probs so each row sums to 1.

    Some probs can be so extreme that normalisation suffers from
    precision issues, and we still have rows with one prob of
    1.0 and the rest ~1e-20. We correct this to prob 1.0 and the
    rest 0.0.
    """
    probs = probs / probs.sum(axis=1)[:, np.newaxis]
    full_prob_indexes = np.nonzero(probs == 1.0)
    # Ensure there is only one full_prob per row
    assert len(full_prob_indexes[0]) == len(np.unique(full_prob_indexes[0]))
    for full_row, full_col in zip(*full_prob_indexes):
        # Rest of cols in the row excluding full_col
        rest_cols = list(set(range(probs.shape[1])) - {full_col})
        # Set rest_cols in this row to 0.0
        probs[full_row, rest_cols] = 0.0
    return probs


def apply_classifier(classifier: ClassifierMixin,
                     X_df: pd.DataFrame,
                     batch_size: int = 100_000) -> np.ndarray:
    """Apply classifier to produce prediction probabilities in batches, as
    all-at-once application to large test sets may result in memory
    exhaustion."""
    batch_count = ceil(X_df.shape[0] / batch_size)
    batch_probs = []
    for i in range(batch_count):
        batch_start = batch_size * i
        batch_end = batch_start + batch_size
        batch_X_df = X_df.iloc[batch_start:batch_end]
        batch_probs.append(normalise_probs(classifier.predict_proba(X=batch_X_df)))
    probs = np.vstack(batch_probs)
    assert probs.shape == (X_df.shape[0], classifier.classes_.shape[0])
    return probs


def execute_classification(dataset: Dataset, *,
                           classifier_name: str,
                           loss_weight: float,
                           random_priors: bool,
                           random_state: int,
                           cache_key: str) -> Dict[str, Any]:
    """Generate a train/calib/rest split for the given configuration,
    train and apply a classifier, and return the classification
    result."""
    train_split = split_train_calib(dataset,
                                    random_state=random_state,
                                    loss_weight=loss_weight,
                                    loss_random_prior=random_priors,
                                    remain_random_prior=random_priors)
    train_dataset = train_split['datasets']['train']
    calib_dataset = train_split['datasets']['calib']
    rest_dataset = train_split['datasets']['rest']
    # Assert datasets have non-overlapping indexes.
    all_indexes = np.concatenate([train_dataset.indexes,
                                  calib_dataset.indexes,
                                  rest_dataset.indexes])
    assert len(np.unique(all_indexes)) == len(all_indexes)
    # Train Classifier
    classifier = CLASSIFIERS[classifier_name](train_dataset)
    classifier.fit(X=train_dataset.X, y=train_dataset.y)
    # Post-processing feature names can be retrieved with:
    # classifier.named_steps['all_features'].get_feature_names()

    if isinstance(classifier, SourceProbClassifier):
        # SourceProbClassifier needs access to all columns
        calib_X = calib_dataset.df
        rest_X = rest_dataset.df
    else:
        calib_X = calib_dataset.X
        rest_X = rest_dataset.X

    clfs = {}
    clfs['uncalibrated'] = classifier
    uncalibrated_calib_probs = clfs['uncalibrated'].predict_proba(calib_X)

    clfs['sigmoid'] = PrefitCalibratedClassifier(base_estimator=clfs['uncalibrated'], method='sigmoid')
    clfs['sigmoid'].fit(calib_X, calib_dataset.y, uncalibrated_calib_probs)

    clfs['isotonic'] = PrefitCalibratedClassifier(base_estimator=clfs['uncalibrated'], method='isotonic')
    clfs['isotonic'].fit(calib_X, calib_dataset.y, uncalibrated_calib_probs)

    classification_result = {
        'cache_key': cache_key,
        'classes': classifier.classes_,
        'train_components': train_split['components'],
        'indexes': {
            'calib': calib_dataset.indexes,
            'rest': rest_dataset.indexes,
        },
        'probs': {
            calibration_method: {
                'calib': apply_classifier(clf, calib_X),
                'rest': apply_classifier(clf, rest_X),
            }
            for calibration_method, clf in clfs.items()
        },
    }
    return classification_result


def run_classification(dataset: Dataset, *,
                       classifier_name: str,
                       loss_weight: float,
                       random_priors: bool,
                       random_state: int) -> Dict:
    """Wrapper around execute_classification that performs caching."""
    cache_key = classification_cache_key(
        dataset_name=dataset.name,
        classifier_name=classifier_name,
        loss_weight=loss_weight,
        random_priors=random_priors,
        random_state=random_state,
    )
    if not is_cached(cache_key):
        classification_result = execute_classification(
            dataset,
            classifier_name=classifier_name,
            loss_weight=loss_weight,
            random_priors=random_priors,
            random_state=random_state,
            cache_key=cache_key,
        )
        save_to_cache(cache_key, classification_result)
    # Load from cache even after execution, in case the save/load
    # introduces subtle changes.
    return cast(Dict, load_from_cache(cache_key))


# QUANTIFICATION

def quantification_cache_key(*,
                             classification_result: Dict[str, Any],
                             calibration_method: str,
                             shift_type: str,
                             gain_weight: float,
                             bin_count: Union[int, str],
                             random_priors: bool,
                             random_state: int,
                             sample_idx: Optional[int],
                             quantification_method: str) -> str:
    """Cache key/path for the given experiment quantification configuration."""
    sample_idx_part = '' if sample_idx is None else f'-si{sample_idx}'
    return os.path.join(
        os.path.dirname(classification_result['cache_key']),
        calibration_method,
        shift_type,
        quantification_method,
        f'quantification-{calibration_method}-{shift_type}-{quantification_method}-gw{gain_weight}-b{bin_count}-rp{random_priors}-r{random_state}{sample_idx_part}',  # noqa: E501
    )


def shift_test_cache_key(*,
                         classification_result: Dict[str, Any],
                         calibration_method: str,
                         shift_type: str,
                         gain_weight: float,
                         bin_count: Union[int, str],
                         random_priors: bool,
                         random_state: int,
                         sample_idx: Optional[int],
                         shift_test: str) -> str:
    """Cache key/path for the given experiment shift test configuration."""
    sample_idx_part = '' if sample_idx is None else f'-si{sample_idx}'
    return os.path.join(
        os.path.dirname(classification_result['cache_key']),
        calibration_method,
        shift_type,
        f'st{shift_test}',
        f'shifttest-{calibration_method}-{shift_type}-st{shift_test}-gw{gain_weight}-b{bin_count}-rp{random_priors}-r{random_state}{sample_idx_part}',
    )


def rejector_cache_key(*,
                       classification_result: Dict[str, Any],
                       calibration_method: str,
                       shift_type: str,
                       gain_weight: float,
                       bin_count: Union[int, str],
                       random_priors: bool,
                       random_state: int,
                       sample_idx: Optional[int],
                       rejector: str,
                       rejection_limit: str) -> str:
    """Cache key/path for the given experiment rejector configuration."""
    sample_idx_part = '' if sample_idx is None else f'-si{sample_idx}'
    return os.path.join(
        os.path.dirname(classification_result['cache_key']),
        calibration_method,
        shift_type,
        f'rej{rejector}',
        f'l{rejection_limit.replace(":", "_")}',
        f'rejector-{calibration_method}-{shift_type}-rej{rejector}-l{rejection_limit.replace(":", "_")}-gw{gain_weight}-b{bin_count}-rp{random_priors}-r{random_state}{sample_idx_part}',  # noqa: E501
    )


def get_base_calibration_method(calibration_method: str) -> str:
    """Return the calibration_method in a classification_result that
    should be used as the base for a given calibration_method."""
    base_calibration_method_map = {
        'uncalibrated': 'uncalibrated',
        'isotonic': 'isotonic',
        'sigmoid': 'sigmoid',
        'perfect': 'uncalibrated',
        'clipped_isotonic': 'isotonic',
        'clipped_sigmoid': 'sigmoid',
    }
    return base_calibration_method_map[calibration_method]


def quantification_shared_precomputation(dataset: Dataset, *,
                                         classification_result: Dict[str, Any],
                                         calibration_method: str,
                                         shift_type: str,
                                         bin_count: Union[int, str],
                                         gain_weight: float,
                                         random_priors: bool,
                                         random_state: int,
                                         sample_idx: Optional[int]) -> Dict[str, Any]:
    """Prepare required inputs for quantification that can be precomputed
    and shared between all quantification methods."""
    base_calibration_method = get_base_calibration_method(calibration_method)
    calib_probs = classification_result['probs'][base_calibration_method]['calib']
    rest_probs = classification_result['probs'][base_calibration_method]['rest']

    calib_indexes = classification_result['indexes']['calib']
    rest_indexes = classification_result['indexes']['rest']
    calib_dataset = dataset.subset(dataset.df.loc[calib_indexes])
    rest_dataset = dataset.subset(dataset.df.loc[rest_indexes])

    test_split = split_test(
        rest_dataset,
        train_components=classification_result['train_components'],
        shift_type=shift_type,
        gain_weight=gain_weight,
        random_state=random_state,
        sample_idx=sample_idx,
        gain_random_prior=random_priors,
    )
    test_dataset = test_split['datasets']['test']
    test_indexes = test_dataset.indexes
    test_in_rest_mask = np.isin(rest_indexes, test_indexes)
    test_probs = rest_probs[test_in_rest_mask]
    # Assert selected probs have the same length as the test indexes.
    assert test_probs.shape[0] == test_indexes.shape[0]

    classes = classification_result['classes']
    dataset_components = {
        'train': classification_result['train_components'],
        'test': test_split['components'],
    }
    sample_class_counts = {
        'calib': calib_dataset.y.value_counts().to_dict(),
        'test': test_dataset.y.value_counts().to_dict()
    }

    if calibration_method == 'perfect':
        # "Perfect" calibration bins the probs, and assigns all probs
        # in a bin the same calibrated prob based on the true contents
        # of the target_y for that bin.
        for class_idx, target_class in enumerate(classes):
            # Use the same bin_edges for both calib and test probs.
            bin_edges = np.histogram_bin_edges(test_probs[:, class_idx], bins=10)
            # Perfectly calibrate test probs
            test_prob_bins = np.digitize(test_probs[:, class_idx], bins=bin_edges)
            for test_bin_val in np.unique(test_prob_bins):
                test_bin_mask = test_prob_bins == test_bin_val
                test_probs[test_bin_mask, class_idx] = np.mean(
                    test_dataset.y[test_bin_mask] == target_class)
            # Perfectly calibrate calib probs
            calib_prob_bins = np.digitize(calib_probs[:, class_idx], bins=bin_edges)
            for calib_bin_val in np.unique(calib_prob_bins):
                calib_bin_mask = calib_prob_bins == calib_bin_val
                calib_probs[calib_bin_mask, class_idx] = np.mean(
                    calib_dataset.y[calib_bin_mask] == target_class)
        test_probs /= np.sum(test_probs, axis=1)[:, np.newaxis]
        calib_probs /= np.sum(calib_probs, axis=1)[:, np.newaxis]
    elif calibration_method in ['clipped_isotonic', 'clipped_sigmoid']:
        # Clipped variants prevent probabilities equal to 0 or 1 by
        # limiting the min/max probs to those of the uncalibrated
        # probs.
        uncalibrated_calib_probs = classification_result['probs']['uncalibrated']['calib']
        min_prob, max_prob = np.min(uncalibrated_calib_probs), np.max(uncalibrated_calib_probs)
        np.clip(test_probs, min_prob, max_prob, test_probs)
        np.clip(calib_probs, min_prob, max_prob, calib_probs)
        test_probs /= np.sum(test_probs, axis=1)[:, np.newaxis]
        calib_probs /= np.sum(calib_probs, axis=1)[:, np.newaxis]

    if classification_result['train_components']:
        # For ConceptsDatasets with components, specify the true
        # weights used to construct the dataset.
        true_weights = {
            'gain': gain_weight,
            'loss': classification_result['train_components']['loss'].weight,
        }
    else:
        # For SamplesDatasets without components, we do not know the
        # true shift weights.
        true_weights = {
            'gain': 0.0,
            'loss': 0.0,
        }

    return {
        'random_state': random_state,
        'calibration_method': calibration_method,
        'classes': classes,
        'calib_y': calib_dataset.y,
        'calib_probs': calib_probs,
        'test_y': test_dataset.y,
        'test_probs': test_probs,
        'true_weights': true_weights,
        'bin_count': bin_count,
        'concepts': {
            dataset_key: {component_name: component.concepts
                          for component_name, component in components.items()}
            for dataset_key, components in dataset_components.items()
        },
        'class_priors': {
            dataset_key: get_class_priors_for_components(components)
            for dataset_key, components in dataset_components.items()
        },
        'class_counts': {
            sample_key: {y_class: class_counts.get(y_class, 0)
                         for y_class in classes}
            for sample_key, class_counts in sample_class_counts.items()
        },
    }


def execute_quantification(*,
                           shared_precomputation: Dict[str, Any],
                           quantification_method: str,
                           prediction_interval_mass: float) -> Dict[str, Any]:
    """Apply a quantification method to the given shared_precomputation
    values, and return the quantification result for all classes."""
    quantifier = QUANTIFIERS[quantification_method]
    quantify_params = dict(
        classes=shared_precomputation['classes'],
        calib_y=shared_precomputation['calib_y'],
        calib_probs=shared_precomputation['calib_probs'],
        target_probs=shared_precomputation['test_probs'],
        prediction_interval_mass=prediction_interval_mass,
        true_weights=shared_precomputation['true_weights'],
        bin_count=shared_precomputation['bin_count'],
        random_state=shared_precomputation['random_state']
    )
    start_time_ns = process_time_ns()
    class_intervals = quantifier.quantify(**quantify_params)
    stop_time_ns = process_time_ns()
    return {
        'classes': shared_precomputation['classes'],
        'test_n': shared_precomputation['test_y'].shape[0],
        'calib_n': shared_precomputation['calib_y'].shape[0],
        'concepts': shared_precomputation['concepts'],
        'class_priors': shared_precomputation['class_priors'],
        'class_counts': shared_precomputation['class_counts'],
        'time_ns': stop_time_ns - start_time_ns,
        'class_intervals': {
            y_class: {
                'count': interval.prediction,
                'count_lower': interval.lower,
                'count_upper': interval.upper,
                **interval.stats,
            }
            for y_class, interval in class_intervals.items()
        },
    }


def execute_shift_test(*,
                       shared_precomputation: Dict[str, Any],
                       quantification_method_results: Dict[str, Dict[str, Any]],
                       shift_test: str,
                       test_alpha: float = 0.05) -> Dict[str, Any]:
    """Apply a shift test to the given shared_precomputation
    values, and return the test result for all classes."""
    shift_tester = SHIFT_TESTERS[shift_test]
    start_time_ns = process_time_ns()
    class_results = shift_tester.run(
        classes=shared_precomputation['classes'],
        calib_y=shared_precomputation['calib_y'],
        calib_probs=shared_precomputation['calib_probs'],
        target_probs=shared_precomputation['test_probs'],
        quantification_method_results=quantification_method_results,
        test_alpha=test_alpha,
        random_state=shared_precomputation['random_state']
    )
    stop_time_ns = process_time_ns()
    return {
        'classes': shared_precomputation['classes'],
        'test_n': shared_precomputation['test_y'].shape[0],
        'calib_n': shared_precomputation['calib_y'].shape[0],
        'concepts': shared_precomputation['concepts'],
        'class_priors': shared_precomputation['class_priors'],
        'class_counts': shared_precomputation['class_counts'],
        'time_ns': stop_time_ns - start_time_ns,
        'class_results': {
            y_class: {
                'shift_detected': result.shift_detected,
                **result.stats,
            }
            for y_class, result in class_results.items()
        },
    }


def execute_rejector(*,
                     shared_precomputation: Dict[str, Any],
                     prediction_interval_mass: float,
                     quantification_method_results: Dict[str, Dict[str, Any]],
                     rejection_limit: str,
                     rejector: str):
    """Apply a rejector to the given shared_precomputation
    values, and return results for all classes."""
    result = REJECTORS[rejector].run(
        classes=shared_precomputation['classes'],
        target_y=shared_precomputation['test_y'],
        target_probs=shared_precomputation['test_probs'],
        calib_probs=shared_precomputation['calib_probs'],
        quantification_method_results=quantification_method_results,
        prediction_interval_mass=prediction_interval_mass,
        rejection_limit=rejection_limit,
        random_state=shared_precomputation['random_state']
    )
    return {
        'rejection_limit': rejection_limit,
        'classes': shared_precomputation['classes'],
        'test_n': shared_precomputation['test_y'].shape[0],
        'calib_n': shared_precomputation['calib_y'].shape[0],
        'concepts': shared_precomputation['concepts'],
        'class_priors': shared_precomputation['class_priors'],
        'class_counts': shared_precomputation['class_counts'],
        'rejected_indexes': result.rejected_indexes,
        'post_class_intervals': {
            y_class: {
                'count': interval.prediction,
                'count_lower': interval.lower,
                'count_upper': interval.upper,
                **interval.stats,
            }
            for y_class, interval in result.post_class_intervals.items()
        },
    }


def run_quantifications(*, classes: np.ndarray,
                        base_row: Dict[str, Any],
                        quantification_cache_keys: Dict[str, str],
                        shift_test_cache_keys: Dict[str, str],
                        rejector_cache_keys: Dict[Tuple[str, str], str],
                        shared_precomputation: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Wrapper around execute_quantification, execute_shift_test, and
    execute_rejector that handles caching and preparing per-class
    result rows."""
    PREDICTION_INTERVAL_MASS = 0.8

    # Get the all-class quantification result for each method.
    test_results = {}
    method_results = {}
    rejector_results = {}

    for quantification_method, cache_key in quantification_cache_keys.items():
        if not is_cached(cache_key):
            assert shared_precomputation is not None
            quantification_result = execute_quantification(
                shared_precomputation=cast(Dict[str, Any], shared_precomputation),
                quantification_method=quantification_method,
                prediction_interval_mass=PREDICTION_INTERVAL_MASS,
            )
            save_to_cache(cache_key, quantification_result)
        # Load from cache even after execution, in case the save/load
        # introduces subtle changes.
        method_results[quantification_method] = load_from_cache(cache_key)

    for shift_test, cache_key in shift_test_cache_keys.items():
        if not is_cached(cache_key):
            assert shared_precomputation is not None
            test_result = execute_shift_test(
                shared_precomputation=cast(Dict[str, Any], shared_precomputation),
                quantification_method_results=method_results,
                shift_test=shift_test,
            )
            save_to_cache(cache_key, test_result)
        # Load from cache even after execution, in case the save/load
        # introduces subtle changes.
        test_results[shift_test] = load_from_cache(cache_key)

    for (rejector, rejection_limit), cache_key in rejector_cache_keys.items():
        if not is_cached(cache_key):
            assert shared_precomputation is not None
            rejector_result = execute_rejector(
                shared_precomputation=cast(Dict[str, Any], shared_precomputation),
                quantification_method_results=method_results,
                prediction_interval_mass=PREDICTION_INTERVAL_MASS,
                rejector=rejector,
                rejection_limit=rejection_limit,
            )
            save_to_cache(cache_key, rejector_result)
        # Load from cache even after execution, in case the save/load
        # introduces subtle changes.
        rejector_results[(rejector, rejection_limit)] = load_from_cache(cache_key)

    # Prepare per-class result rows.
    class_rows = []
    for y_class in classes:
        if len({**method_results, **test_results}) == 0:
            continue
        # Some fields will be identical between results, so we only
        # need to include them from the first result.
        first_result = dict_first({**method_results, **test_results})
        class_row = {
            **base_row,
            'target_class': y_class,
            'concepts': first_result['concepts'],
            'calib_n': first_result['calib_n'],
            'test_n': first_result['test_n'],
            'calib_true_count': first_result['class_counts']['calib'][y_class],
            'test_true_count': first_result['class_counts']['test'][y_class],
            'train_prior': first_result['class_priors']['train'].get(y_class, None),
            'test_prior': first_result['class_priors']['test'].get(y_class, None),
        }
        # Add method-prefixed shift test, quantification interval, and rejection results.
        for shift_test, test_result in test_results.items():
            class_row.update(prefix_keys(test_result['class_results'][y_class], f'{shift_test}_'))
            class_row[f'{shift_test}_all_class_time_ns'] = test_result['time_ns']
        for q_method, q_result in method_results.items():
            class_row.update(prefix_keys(q_result['class_intervals'][y_class], f'{q_method}_'))
            class_row[f'{q_method}_all_class_time_ns'] = q_result['time_ns']
        for (rejector, rejection_limit), rejector_result in rejector_results.items():
            rejection_prefix = f'{rejector}_{rejection_limit}'
            rejector_class_interval = rejector_result['post_class_intervals'][y_class]
            class_row.update(prefix_keys(rejector_class_interval, f'{rejection_prefix}_'))
            class_row[f'{rejection_prefix}_rejected_count'] = len(rejector_result['rejected_indexes'])
            class_row[f'{rejection_prefix}_interval_width'] = rejector_class_interval['count_upper'] - rejector_class_interval['count_lower']
            class_row[f'{rejection_prefix}_target_width_limit'] = rejector_class_interval['target_width_limit']
        class_rows.append(class_row)
    return pd.DataFrame(class_rows)


def spawn_quantification_task(*,
                              q_executor: Executor,
                              dataset: Dataset,
                              classification_result: Dict[str, Any],
                              calibration_method: str,
                              gain_weight: float,
                              shift_type: str,
                              bin_count: Union[int, str],
                              random_priors: bool,
                              quantification_methods: Sequence[str],
                              shift_tests: Sequence[str],
                              rejectors: Sequence[str],
                              rejection_limits: Sequence[str],
                              random_state: int,
                              sample_idx: Optional[int] = None) -> Future:
    """Responsible for performing memory-expensive precomputation before
    spawning a sub-process for quantification."""
    classes = classification_result['classes']
    base_row = {
        'calibration_method': calibration_method,
        'gain_weight': gain_weight,
        'shift_type': shift_type,
        'bin_count': bin_count,
        'random_state': random_state,
        'sample_idx': sample_idx,
    }

    quantification_cache_keys = {
        quantification_method: quantification_cache_key(
            classification_result=classification_result,
            calibration_method=calibration_method,
            shift_type=shift_type,
            gain_weight=gain_weight,
            bin_count=bin_count,
            random_priors=random_priors,
            random_state=random_state,
            sample_idx=sample_idx,
            quantification_method=quantification_method,
        )
        for quantification_method in quantification_methods
    }
    shift_test_cache_keys = {
        shift_test: shift_test_cache_key(
            classification_result=classification_result,
            calibration_method=calibration_method,
            shift_type=shift_type,
            gain_weight=gain_weight,
            bin_count=bin_count,
            random_priors=random_priors,
            random_state=random_state,
            sample_idx=sample_idx,
            shift_test=shift_test,
        )
        for shift_test in shift_tests
    }
    rejector_cache_keys = {
        (rejector, rejection_limit): rejector_cache_key(
            classification_result=classification_result,
            calibration_method=calibration_method,
            shift_type=shift_type,
            gain_weight=gain_weight,
            bin_count=bin_count,
            random_priors=random_priors,
            random_state=random_state,
            sample_idx=sample_idx,
            rejector=rejector,
            rejection_limit=rejection_limit,
        )
        for rejector in rejectors
        for rejection_limit in rejection_limits
    }

    shared_precomputation = None
    if any([not is_cached(cache_key) for cache_key
            in [*quantification_cache_keys.values(),
                *shift_test_cache_keys.values(),
                *rejector_cache_keys.values()]]):
        # Prepare shared_precomputation so it can be shared among
        # quantification executions, but only prepare if at least one
        # shift test or quantification requires execution. We prepare
        # the shared_precomputation before spawning the quantification
        # task process because its result requires much less memory
        # than its inputs (saving time/space copying memory).
        shared_precomputation = quantification_shared_precomputation(
            dataset,
            classification_result=classification_result,
            calibration_method=calibration_method,
            shift_type=shift_type,
            bin_count=bin_count,
            gain_weight=gain_weight,
            random_priors=random_priors,
            random_state=random_state,
            sample_idx=sample_idx,
        )

    return q_executor.submit(
        run_quantifications,
        classes=classes,
        base_row=base_row,
        quantification_cache_keys=quantification_cache_keys,
        shift_test_cache_keys=shift_test_cache_keys,
        rejector_cache_keys=rejector_cache_keys,
        shared_precomputation=shared_precomputation,
    )


# EXPERIMENTS

def classification_task(*,
                        dataset: Dataset,
                        classifier_name: str,
                        loss_weight: float,
                        random_state: int,
                        calibration_methods: Sequence[str],
                        gain_weights: Sequence[float],
                        shift_types: Sequence[str],
                        bin_counts: Sequence[Union[int, str]],
                        random_priors: bool,
                        quantification_methods: Sequence[str],
                        sample_idxs:  Optional[Sequence[int]],
                        shift_tests: Sequence[str],
                        rejectors: Sequence[str],
                        rejection_limits: Sequence[str],
                        quantification_workers: int,
                        continue_on_failure: bool) -> Tuple[pd.DataFrame, int]:
    """Sub-task of run_experiments that allows all quantification relating
    to a single classifier to be spun off into a sub-process."""
    classification_result = run_classification(
        dataset,
        classifier_name=classifier_name,
        loss_weight=loss_weight,
        random_priors=random_priors,
        random_state=random_state,
    )
    # After classification has been performed, we can save memory by
    # using a copy of the dataset without features in quantification
    # sub-processes.
    dataset = dataset.without_X()
    # Execute quantification for each test-set configuration as a
    # concurrent sub-task.
    q_futures = []
    q_executor = get_executor(max_workers=quantification_workers)
    for calibration_method in calibration_methods:
        for gain_weight in gain_weights:
            for shift_type in shift_types:
                for bin_count in bin_counts:
                    if isinstance(dataset, ConceptsDataset):
                        future = spawn_quantification_task(
                            q_executor=q_executor,
                            dataset=dataset,
                            classification_result=classification_result,
                            calibration_method=calibration_method,
                            gain_weight=gain_weight,
                            shift_type=shift_type,
                            bin_count=bin_count,
                            random_priors=random_priors,
                            quantification_methods=quantification_methods,
                            shift_tests=shift_tests,
                            rejectors=rejectors,
                            rejection_limits=rejection_limits,
                            random_state=random_state,
                        )
                        q_futures.append(future)
                    elif isinstance(dataset, SamplesDataset):
                        dataset = cast(SamplesDataset, dataset)
                        sample_idxs = (range(len(dataset.test_samples))
                                       if sample_idxs is None else sample_idxs)
                        for sample_idx in sample_idxs:
                            future = spawn_quantification_task(
                                q_executor=q_executor,
                                dataset=dataset,
                                classification_result=classification_result,
                                calibration_method=calibration_method,
                                gain_weight=gain_weight,
                                shift_type=shift_type,
                                bin_count=bin_count,
                                random_priors=random_priors,
                                quantification_methods=quantification_methods,
                                shift_tests=shift_tests,
                                rejectors=rejectors,
                                rejection_limits=rejection_limits,
                                random_state=random_state,
                                sample_idx=sample_idx,
                            )
                            q_futures.append(future)
    # Aggregate all class-quantification rows returned by
    # run_quantifications into a list of all rows for this
    # classification task.
    classification_dfs = []
    q_failures = 0
    for i, future in enumerate(as_completed(q_executor, q_futures), start=1):
        try:
            result_df = future.result()
        except Exception as ex:
            q_failures += 1
            if continue_on_failure:
                log(f'Failed q_task {i}/{len(q_futures)} for '
                    f'c_task: {os.path.dirname(classification_result["cache_key"])}: '
                    f'{type(ex).__name__}({ex})')
            else:
                for f in q_futures:
                    f.cancel()
                raise ex
        else:
            # Add classification configuration values to each row.
            classification_dfs.append(result_df.assign(**{
                'dataset_name': dataset.name,
                'classifier_name': classifier_name,
                'loss_weight': loss_weight,
                'full_train_n': dataset.train_n,
            }))
            if VERBOSE:
                log(f'- Completed q_task {i}/{len(q_futures)} for '
                    f'c_task: {os.path.dirname(classification_result["cache_key"])}')
    q_executor.shutdown()
    return pd.concat(classification_dfs), q_failures


def run_experiments(*,
                    dataset_names: Sequence[str],
                    classifier_names: Sequence[str],
                    calibration_methods: Sequence[str],
                    loss_weights: Sequence[float],
                    gain_weights: Sequence[float],
                    shift_types: Sequence[str],
                    quantification_methods: Sequence[str],
                    bin_counts: Sequence[Union[int, str]],
                    random_priors_options: Sequence[bool],
                    random_states: Sequence[int],
                    sample_idxs: Optional[Sequence[int]] = None,
                    shift_tests: Optional[Sequence[str]] = None,
                    rejectors: Optional[Sequence[str]] = None,
                    rejection_limits: Optional[Sequence[str]] = None,
                    classification_workers: int = 12,
                    quantification_workers: int = 1,
                    continue_on_failure: bool = False) -> pd.DataFrame:
    """Top-level function for executing experiments for the cross-product
    of provided configuration options. Returns a DataFrame of
    quantification results. *_workers parameters control concurrency -
    their product should not be greater than the number of available
    cores on your machine.
    """
    if shift_tests is None:
        shift_tests = []
    if rejectors is None:
        rejectors = []
    if rejection_limits is None:
        rejection_limits = []

    log('Beginning experiments')
    # Execute computation for each classification configuration as a
    # concurrent sub-task.
    c_futures = []
    c_executor = get_executor(max_workers=classification_workers)
    for dataset_name in dataset_names:
        dataset = DATASETS[dataset_name]()

        if isinstance(dataset, ConceptsDataset):
            dataset_shift_types = shift_types
            dataset_loss_weights = loss_weights
            dataset_gain_weights = gain_weights
            dataset_random_priors_options = random_priors_options
            dataset_random_states = random_states
        elif isinstance(dataset, SamplesDataset):
            log(f'Eliminating any redundant shift config permutations for samples-based dataset {dataset_name}')
            dataset_shift_types = ['no_shift']
            dataset_loss_weights = [0]
            dataset_gain_weights = [0]
            dataset_random_priors_options = [False]
            dataset_random_states = [0]
        else:
            raise ValueError(f'Unrecognised dataset type: {type(dataset)}')

        for classifier_name in classifier_names:
            for loss_weight in dataset_loss_weights:
                for random_priors in dataset_random_priors_options:
                    for random_state in dataset_random_states:
                        future = c_executor.submit(
                            classification_task,
                            dataset=dataset,
                            classifier_name=classifier_name,
                            random_state=random_state,
                            loss_weight=loss_weight,
                            gain_weights=dataset_gain_weights,
                            calibration_methods=calibration_methods,
                            shift_types=dataset_shift_types,
                            bin_counts=bin_counts,
                            random_priors=random_priors,
                            quantification_methods=quantification_methods,
                            sample_idxs=sample_idxs,
                            shift_tests=shift_tests,
                            rejectors=rejectors,
                            rejection_limits=rejection_limits,
                            quantification_workers=quantification_workers,
                            continue_on_failure=continue_on_failure,
                        )
                        c_futures.append(future)
    # Aggregate all result rows returned by classification sub-tasks.
    all_dfs: List[pd.DataFrame] = []
    c_failures = 0
    for i, future in enumerate(as_completed(c_executor, c_futures), start=1):
        try:
            result_df, q_failures = future.result()
        except Exception as ex:
            c_failures += 1
            if continue_on_failure:
                log(f'Failed c_task {i}/{len(c_futures)}: {type(ex).__name__}({ex})')
            else:
                for f in c_futures:
                    f.cancel()
                raise ex
        else:
            all_dfs.append(result_df)
            c_failures += q_failures
            log(f'Completed c_task {i}/{len(c_futures)}')
    c_executor.shutdown()

    if c_failures > 0:
        print(f"""
========================================
WARNING! there were {c_failures} failures!
========================================
""")

    return pd.concat(all_dfs)


def cached_experiments(*, cache_key: str, **kwargs: Any) -> pd.DataFrame:
    """Wrapper around run_experiments() that provides caching of the final
    results_df, and final preparation of metrics in results_df."""
    if not is_cached(cache_key):
        results_df = run_experiments(**kwargs)
        log('Saving to cache...')
        save_to_cache(cache_key, results_df)
        del results_df
    log('Loading from cache...')
    results_df = load_from_cache(cache_key)
    results_df = results_df.reset_index()
    return prepare_results(results_df)


def cached_test_sample_indexes(*, cache_key: str,
                               dataset_name: str,
                               max_test_n: int) -> Sequence[int]:
    """Return a list of sample indexes for the given dataset that meet
    specified criteria."""
    cache_key = f'test_sample_indexes_{cache_key}_{dataset_name}_{max_test_n}'
    if not is_cached(cache_key):
        dataset = DATASETS[dataset_name]()
        assert isinstance(dataset, SamplesDataset)
        sample_sizes = dataset.df['sample'].value_counts().to_dict()
        sample_indexes = [
            sample_index
            for sample_index, sample in enumerate(dataset.test_samples)
            if sample_sizes[sample] <= max_test_n
        ]
        save_to_cache(cache_key, sample_indexes)
        del dataset
        del sample_indexes
    return load_from_cache(cache_key)
