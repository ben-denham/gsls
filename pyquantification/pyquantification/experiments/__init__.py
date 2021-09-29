from datetime import datetime
import os
from joblib import dump, load
import numpy as np
import pandas as pd
from time import process_time_ns
from typing import cast, Any, Union, Optional, Dict, Sequence, Tuple, List

from pyquantification.utils import prefix_keys, dict_first
from pyquantification.datasets import DATASETS, Dataset
from pyquantification.classifiers import CLASSIFIERS
from pyquantification.quantifiers import QUANTIFIERS
from pyquantification.experiments.executors import get_executor, as_completed
from pyquantification.experiments.splitting import (split_train_calib, split_test,
                                                    get_class_priors_for_components)
from pyquantification.experiments.metrics import prepare_results

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
    classification_result = {
        'cache_key': cache_key,
        'classes': classifier.classes_,
        'train_components': train_split['components'],
        'indexes': {
            'calib': calib_dataset.indexes,
            'rest': rest_dataset.indexes,
        },
        'probs': {
            'uncalibrated': {
                'calib': normalise_probs(classifier.predict_proba(X=calib_dataset.X)),
                'rest': normalise_probs(classifier.predict_proba(X=rest_dataset.X)),
            },
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
                             shift_type: str,
                             gain_weight: float,
                             bin_count: Union[int, str],
                             random_priors: bool,
                             random_state: int,
                             quantification_method: str) -> str:
    """Cache key/path for the given experiment quantification configuration."""
    return os.path.join(
        os.path.dirname(classification_result['cache_key']),
        shift_type,
        quantification_method,
        f'quantification-{shift_type}-{quantification_method}-gw{gain_weight}-b{bin_count}-rp{random_priors}-r{random_state}',
    )


def quantification_shared_precomputation(dataset: Dataset, *,
                                         classification_result: Dict[str, Any],
                                         shift_type: str,
                                         gain_weight: float,
                                         random_priors: bool,
                                         random_state: int) -> Dict[str, Any]:
    """Prepare required inputs for quantification that can be precomputed
    and shared between all quantification methods."""
    calib_probs = classification_result['probs']['uncalibrated']['calib']
    rest_probs = classification_result['probs']['uncalibrated']['rest']
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

    return {
        'random_state': random_state,
        'classes': classes,
        'calib_y': calib_dataset.y,
        'calib_probs': calib_probs,
        'test_y': test_dataset.y,
        'test_probs': test_probs,
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
        'component_indexes': {
            'calib': calib_dataset.components_index(classification_result['train_components']),
            'test': test_dataset.components_index(test_split['components']),
        },
    }


def execute_quantification(*,
                           shared_precomputation: Dict[str, Any],
                           quantification_method: str,
                           bin_count: Union[int, str],
                           prediction_interval_mass: float = 0.8,
                           true_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Apply a quantification method to the given shared_precomputation
    values, and return the quantification result for all classes."""
    quantifier = QUANTIFIERS[quantification_method]
    quantify_params = dict(
        classes=shared_precomputation['classes'],
        calib_y=shared_precomputation['calib_y'],
        calib_probs=shared_precomputation['calib_probs'],
        target_probs=shared_precomputation['test_probs'],
        prediction_interval_mass=prediction_interval_mass,
        true_weights=true_weights,
        bin_count=bin_count,
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


def run_quantifications(dataset: Dataset, *,
                        classification_result: Dict[str, Any],
                        gain_weight: float,
                        shift_type: str,
                        bin_count: Union[int, str],
                        random_priors: bool,
                        quantification_methods: Sequence[str],
                        random_state: int) -> Sequence[Dict[str, Any]]:
    """Wrapper around execute_quantification that handles caching, sharing
    precomputation between quantification methods, and preparing
    per-class result rows."""
    # Get the all-class quantification result for each method.
    shared_precomputation = None
    method_results = {}
    for quantification_method in quantification_methods:
        cache_key = quantification_cache_key(
            classification_result=classification_result,
            shift_type=shift_type,
            gain_weight=gain_weight,
            bin_count=bin_count,
            random_priors=random_priors,
            random_state=random_state,
            quantification_method=quantification_method
        )
        if not is_cached(cache_key):
            # Prepare shared_precomputation so it can be shared among
            # quantification executions, but only prepare if at least
            # one quantification requires execution.
            if shared_precomputation is None:
                shared_precomputation = quantification_shared_precomputation(
                    dataset,
                    classification_result=classification_result,
                    shift_type=shift_type,
                    gain_weight=gain_weight,
                    random_priors=random_priors,
                    random_state=random_state,
                )
            quantification_result = execute_quantification(
                shared_precomputation=shared_precomputation,
                quantification_method=quantification_method,
                bin_count=bin_count,
                true_weights={
                    'gain': gain_weight,
                    'loss': classification_result['train_components']['loss'].weight,
                },
            )
            save_to_cache(cache_key, quantification_result)
        # Load from cache even after execution, in case the save/load
        # introduces subtle changes.
        method_results[quantification_method] = load_from_cache(cache_key)
    # Prepare per-class result rows.
    class_rows = []
    for y_class in classification_result['classes']:
        # Some fields will be identical between results, so we only
        # need to include them from the first result.
        first_result = dict_first(method_results)
        class_row = {
            'gain_weight': gain_weight,
            'shift_type': shift_type,
            'bin_count': bin_count,
            'random_state': random_state,
            'target_class': y_class,
            'concepts': first_result['concepts'],
            'calib_n': first_result['calib_n'],
            'test_n': first_result['test_n'],
            'calib_true_count': first_result['class_counts']['calib'][y_class],
            'test_true_count': first_result['class_counts']['test'][y_class],
            'train_prior': first_result['class_priors']['train'][y_class],
            'test_prior': first_result['class_priors']['test'][y_class],
        }
        # Add method-prefixed quantification interval results.
        for q_method, q_result in method_results.items():
            class_row.update(prefix_keys(q_result['class_intervals'][y_class], f'{q_method}_'))
            class_row[f'{q_method}_all_class_time_ns'] = q_result['time_ns']
        class_rows.append(class_row)
    return class_rows


# EXPERIMENTS

def classification_task(*,
                        dataset: Dataset,
                        classifier_name: str,
                        loss_weight: float,
                        random_state: int,
                        gain_weights: Sequence[float],
                        shift_types: Sequence[str],
                        bin_counts: Sequence[Union[int, str]],
                        random_priors: bool,
                        quantification_methods: Sequence[str],
                        quantification_workers: int,
                        continue_on_failure: bool) -> Tuple[Sequence[Dict[str, Any]], int]:
    """Sub-task of run_experiments that allows all quantification relating
    to a single classifier to be spun off into a sub-process."""
    classification_result = run_classification(
        dataset,
        classifier_name=classifier_name,
        loss_weight=loss_weight,
        random_priors=random_priors,
        random_state=random_state,
    )
    # Execute quantification for each test-set configuration as a
    # concurrent sub-task.
    q_futures = []
    q_executor = get_executor(max_workers=quantification_workers)
    for gain_weight in gain_weights:
        for shift_type in shift_types:
            for bin_count in bin_counts:
                future = q_executor.submit(
                    run_quantifications,
                    dataset=dataset,
                    classification_result=classification_result,
                    gain_weight=gain_weight,
                    shift_type=shift_type,
                    bin_count=bin_count,
                    random_priors=random_priors,
                    quantification_methods=quantification_methods,
                    random_state=random_state,
                )
                q_futures.append(future)
    # Aggregate all class-quantification rows returned by
    # run_quantifications into a list of all rows for this
    # classification task.
    classification_rows = []
    q_failures = 0
    for i, future in enumerate(as_completed(q_executor, q_futures), start=1):
        try:
            result = future.result()
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
            classification_rows += [{
                'dataset_name': dataset.name,
                'classifier_name': classifier_name,
                'loss_weight': loss_weight,
                **row,
            } for row in result]
            if VERBOSE:
                log(f'- Completed q_task {i}/{len(q_futures)} for '
                    f'c_task: {os.path.dirname(classification_result["cache_key"])}')
    q_executor.shutdown()
    return classification_rows, q_failures


def run_experiments(*,
                    dataset_names: Sequence[str],
                    classifier_names: Sequence[str],
                    loss_weights: Sequence[float],
                    gain_weights: Sequence[float],
                    shift_types: Sequence[str],
                    quantification_methods: Sequence[str],
                    bin_counts: Sequence[Union[int, str]],
                    random_priors_options: Sequence[bool],
                    random_states: Sequence[int],
                    classification_workers: int = 12,
                    quantification_workers: int = 1,
                    continue_on_failure: bool = False) -> pd.DataFrame:
    """Top-level function for executing experiments for the cross-product
    of provided configuration options. Returns a DataFrame of
    quantification results. *_workers parameters control concurrency -
    their product should not be greater than the number of available
    cores on your machine.
    """
    log('Beginning experiments')
    # Execute computation for each classification configuration as a
    # concurrent sub-task.
    c_futures = []
    c_executor = get_executor(max_workers=classification_workers)
    for dataset_name in dataset_names:
        dataset = DATASETS[dataset_name]()
        for classifier_name in classifier_names:
            for loss_weight in loss_weights:
                for random_priors in random_priors_options:
                    for random_state in random_states:
                        future = c_executor.submit(
                            classification_task,
                            dataset=dataset,
                            classifier_name=classifier_name,
                            random_state=random_state,
                            loss_weight=loss_weight,
                            gain_weights=gain_weights,
                            shift_types=shift_types,
                            bin_counts=bin_counts,
                            random_priors=random_priors,
                            quantification_methods=quantification_methods,
                            quantification_workers=quantification_workers,
                            continue_on_failure=continue_on_failure,
                        )
                        c_futures.append(future)
    # Aggregate all result rows returned by classification sub-tasks.
    all_rows: List = []
    c_failures = 0
    for i, future in enumerate(as_completed(c_executor, c_futures), start=1):
        try:
            result, q_failures = future.result()
        except Exception as ex:
            c_failures += 1
            if continue_on_failure:
                log(f'Failed c_task {i}/{len(c_futures)}: {type(ex).__name__}({ex})')
            else:
                for f in c_futures:
                    f.cancel()
                raise ex
        else:
            all_rows += result
            c_failures += q_failures
            log(f'Completed c_task {i}/{len(c_futures)}')
    c_executor.shutdown()

    if c_failures > 0:
        print(f"""
========================================
WARNING! there were {c_failures} failures!
========================================
""")

    return pd.DataFrame(all_rows)


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
    return prepare_results(results_df)
