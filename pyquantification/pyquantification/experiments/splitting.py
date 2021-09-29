from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Optional, Dict, Any, Sequence

from pyquantification.utils import normalise_dict, check_dict_almost_equal
from pyquantification.datasets import (Dataset, Component, Components,
                                       Concepts, ClassPriors)

# Use a different seed for each stage of an experiment to prevent
# overlaps and unintended correlation. Different orders of magnitude
# so that they can be repeated several times if needed.
TRAIN_SPLIT_RANDOM_SEED = 1
TEST_SPLIT_RANDOM_SEED = 1_000


# CLASS PRIORS

def check_class_priors(class_priors: ClassPriors, raise_error: bool = False) -> bool:
    """Returns True if class_priors are valid."""
    priors = np.array(list(class_priors.values()))
    valid = bool(
        # All priors should be between (and not equal to) 0 and 1.
        np.all(priors < 1) and
        np.all(priors > 0) and
        # Check the priors sum to 1.
        # Same test as np.testing.assert_almost_equal(np.sum(priors), 1)
        (np.abs(1 - np.sum(priors)) < (1.5 * 10**(-7)))
    )
    if raise_error and not valid:
        raise ValueError(f'Class priors are not valid: {str(class_priors)}')
    return valid


def get_class_priors(dataset: Dataset, *,
                     concepts: Optional[Concepts] = None) -> ClassPriors:
    """Return the empirical class_priors of the given dataset (optionally
    filtered to the given concepts)."""
    df = dataset.df
    if concepts is not None:
        df = df[df['concept'].isin(concepts)]
    priors_series = df['class'].value_counts() / df.shape[0]
    priors = {y_class: 0.0 for y_class in df['class'].unique()}
    priors.update(priors_series.to_dict())
    return priors


def get_class_priors_for_components(components: Components) -> ClassPriors:
    """Return the overall class_priors for the weighted component
    class_priors (i.e. P(y) = sum_concept P(y|concept)P(concept))."""
    # Each column is a y_class, each row in a component, each
    # cell is the component's weighted class prior
    weighted_class_priors_df = pd.DataFrame([
        {y_class: class_prior * component.weight
         for y_class, class_prior in component.class_priors.items()}
        for component in components.values()
    ])
    assert not weighted_class_priors_df.isnull().any(axis=None)
    # Sum each column to get the overall class_priors
    class_priors = weighted_class_priors_df.sum(axis='rows').to_dict()
    check_class_priors(class_priors, raise_error=True)
    return class_priors


def check_prior_shift_assumption(source_components: Components,
                                 target_components: Components,
                                 *, decimal: int = 7) -> bool:
    """Check that P_s(c|y) = P_t(c|y) for all c and y."""
    if set(source_components.keys()) != set(target_components.keys()):
        return False
    source_class_priors = get_class_priors_for_components(source_components)
    target_class_priors = get_class_priors_for_components(target_components)
    for component_name in source_components.keys():
        for y_class in source_components[component_name].class_priors.keys():
            # P_s(c|y) = P_s(y|c) * P_s(c) / P_s(y)
            source_class_conditional = (source_components[component_name].class_priors[y_class] *
                                        source_components[component_name].weight /
                                        source_class_priors[y_class])
            # P_t(c|y) = P_t(y|c) * P_t(c) / P_t(y)
            target_class_conditional = (target_components[component_name].class_priors[y_class] *
                                        target_components[component_name].weight /
                                        target_class_priors[y_class])
            # Same test as np.testing.assert_almost_equal
            if np.abs(source_class_conditional - target_class_conditional) >= (1.5 * 10**(-decimal)):
                return False
    return True


def random_class_priors(dataset: Dataset, *, rng: np.random.RandomState) -> ClassPriors:
    """Draw a set of random class_priors from a balanced Dirichlet distribution."""
    return dict(zip(dataset.classes, rng.dirichlet([1] * len(dataset.classes))))


def simulate_class_counts(*, n: int, class_priors: ClassPriors,
                          rng: np.random.RandomState,
                          min_class_count: int) -> Dict[str, int]:
    """Return the number of instances to draw from each class, based on
    simulation of drawing from a distribution with the given
    class_priors. Guarantees that there will be at least
    min_class_count instances from each class.
    """
    check_class_priors(class_priors, raise_error=True)
    classes = sorted(class_priors.keys())
    priors = [class_priors[y_class] for y_class in classes]
    choices = rng.choice(classes, size=n, replace=True, p=priors)
    class_counts: Dict[str, int] = dict(Counter(choices))
    # If we do not have min_class_count instances of one or more
    # classes, we will randomly swap some of the choices to them. In
    # the unlikely event that we end up reducing another class below
    # min_class_count or swap it's own instances, we loop until the
    # condition is satisfied for all classes. This approach is deemed
    # to minimally impact distributions while guaranteeing we have
    # min_class_count (e.g. for having at least one example of each
    # class for training).
    while np.any([class_counts.get(y_class, 0) < min_class_count
                  for y_class in classes]):
        for y_class in classes:
            count = class_counts.get(y_class, 0)
            if count >= min_class_count:
                continue
            swap_indexes = rng.choice(np.arange(n), size=(min_class_count - count), replace=False)
            choices[swap_indexes] = y_class
        # Update class_counts from choices
        class_counts = dict(Counter(choices))
    assert np.sum(list(class_counts.values())) == n
    return class_counts


# COMPONENT WEIGHTS

def check_component_weights(components: Components, raise_error: bool = False) -> bool:
    """Returns True if the component weights are valid."""
    weights = np.array([component.weight for component in components.values()])
    valid = bool(
        # All weights should be between (or equal to) 0 and 1.
        np.all(weights <= 1) and
        np.all(weights >= 0) and
        # Check the weights sum to 1.
        # Same test as np.testing.assert_almost_equal(np.sum(weights), 1)
        (np.abs(1 - np.sum(weights)) < (1.5 * 10**(-7)))
    )
    if raise_error and not valid:
        raise ValueError(f'Component weights are not valid: {str(weights)}')
    return valid


def get_component_counts(*, n: int, components: Components) -> Dict[str, int]:
    """Return the number of instances to draw from each weighted component
    to sum to n instances. Will always return the same counts given
    the same n and components."""
    check_component_weights(components, raise_error=True)
    allocated_n = 0
    component_counts = {}
    for i, component_name in enumerate(sorted(components.keys())):
        if i == (len(components) - 1):
            # Ensure the total sums to n by computing the final
            # component's count from the difference of the total what
            # has been allocated so far.
            component_counts[component_name] = n - allocated_n
        else:
            component_counts[component_name] = int(round(components[component_name].weight * n))
        allocated_n += component_counts[component_name]
    assert np.sum(list(component_counts.values())) == n
    return component_counts


# ARTIFICIAL PRIOR_SHIFT

def adjust_components_for_prior_shift(*, classes: Sequence[str],
                                      components: Components,
                                      target_class_priors: ClassPriors) -> Components:
    r"""To achieve prior shift in the target dataset, we must update the
    weights and class_priors of the given components, while ensuring
    we do not shift class-conditional distributions (which is required
    by the prior-shift assumption).

    Key for maths:

    * source distribution = s; target distribution = t
    * class = y; inputs = X; component = c
    * Prior-shift assumption: P_s(X|y) = P_t(X|y)
    * The distribution of inputs depends on the component:
      * P(X|c) != P(X) (not necessarily equal)
      * Therefore, for the prior shift assumption, we must
        ensure: P_s(c|y) = P_t(c|y)
    * source_class_priors[y] = P_s(y)
    * target_class_priors[y] = P_t(y)
    * components[c].class_priors[y] = P_s(y|c)
    * components[c].weight = P_s(c)

    Step 1: Find each P_t(y|c) using an approach analogous to Saerens
    et al. 2002 for prior adjustment:

    * P_t(y|c) = P_t(c|y) * P_t(y) / P_t(c)
      * By Bayes rule.
    * P_t(y|c) = P_s(c|y) * P_t(y) / P_t(c)
      * By prior-shift assumption.
    * P_t(y|c) = (P_s(y|c) * P_s(c) / P_s(y)) * P_t(y) / P_t(c)
      * By Bayes rule.
    * P_t(y|c) = P_s(y|c) * (P_t(y)/P_s(y)) / \sum{y in Y} P_s(y|c) * (P_t(y)/P_s(y))
      * This is just a normalisation over posteriors given concept.

    Step 2: Use P_t(y|c) to find P_t(c):

    * P_t(y|c) = P_t(c|y) * P_t(y) / P_t(c)
      * By Bayes rule.
    * P_t(c) = P_t(c|y) * P_t(y) / P_t(y|c)
    * P_t(c) = P_s(c|y) * P_t(y) / P_t(y|c)
      * By prior-shift assumption.
    * P_t(c) = (P_s(y|c) * P_s(c) / P_s(y)) * P_t(y) / P_t(y|c)
      * By Bayes rule.
    * P_t(c) = P_s(c) * (P_s(y|c)/P_t(y|c)) * (P_t(y)/P_s(y))
      * The same P_t(c) will be arrived at for any value of y.

    """
    def clip_component_weight(weight: float) -> float:
        """Our calculations can result in weights that are outside the [0, 1]
        range by a floating point error, so we clip the values in those cases."""
        if (weight > 1) and np.isclose(weight, 1):
            return 1.0
        elif (weight < 0) and np.isclose(weight, 0):
            return 0.0
        else:
            return weight

    assert sorted(components.keys()) == ['loss', 'remain']
    check_component_weights(components, raise_error=True)
    for component in components.values():
        check_class_priors(component.class_priors, raise_error=True)

    source_class_priors = get_class_priors_for_components(components)
    target_components = {}
    for component_name, component in components.items():
        # Step 1
        updated_class_priors = normalise_dict({y_class: (
            component.class_priors[y_class] *
            (target_class_priors[y_class] / source_class_priors[y_class])
        ) for y_class in classes})
        # Step 2
        updated_weights = [
            component.weight *
            (component.class_priors[y_class] / updated_class_priors[y_class]) *
            (target_class_priors[y_class] / source_class_priors[y_class])
            for y_class in classes
        ]
        # The same value should be produced for all values of y_class.
        np.testing.assert_almost_equal(updated_weights, updated_weights[0])
        target_components[component_name] = Component(
            concepts=component.concepts,
            # All updated_weights will be (almost) equal.
            weight=clip_component_weight(float(np.mean(updated_weights))),
            class_priors=updated_class_priors,
        )
    # Check that we have successfully shifted the class priors without
    # affecting the class-conditional distributions.
    assert check_dict_almost_equal(
        get_class_priors_for_components(target_components),
        target_class_priors
    )
    assert check_prior_shift_assumption(components, target_components)
    return target_components


# DATASET SAMPLING

def sample_indexes_for_components(dataset: Dataset, *,
                                  n: int,
                                  components: Components,
                                  rng: np.random.RandomState,
                                  # min_class_count=2 so that train
                                  # and calib get at least one
                                  # instance of each class. Used for
                                  # all samples so that the approach
                                  # is consistent.
                                  min_class_count: int = 2) -> np.ndarray:
    """Sample n instances from dataset with the distribution determined by
    the weights and class_priors of components. Return the indexes of
    those instances, such that dataset.df.loc[indexes] will return the
    actual instances.

    The component weights are treated as exact - given the same
    weights, exactly the same number of instances will be drawn from
    each component.

    The component class_priors are treated as priors - we will
    simulate sampling from a distribution with those class_priors.

    There is guaranteed to be at least min_class_count instances of
    each unique class in the sample.
    """
    index_arrays = []
    component_counts = get_component_counts(n=n, components=components)
    for component_name in sorted(components.keys()):
        component = components[component_name]
        component_df = dataset.df[dataset.df['concept'].isin(component.concepts)]
        class_counts = simulate_class_counts(
            n=component_counts[component_name],
            class_priors=component.class_priors,
            rng=rng,
            # We only need to require min_class_count in each
            # component relative to the size of the component.
            min_class_count=int(round(component.weight * min_class_count)),
        )
        for y_class, class_n in class_counts.items():
            component_class_df = component_df[component_df['class'] == y_class]
            sample_df = component_class_df.sample(
                n=class_n,
                replace=False,
                random_state=rng,
            )
            index_arrays.append(sample_df.index.to_numpy())
    return np.concatenate(index_arrays)


def split_train_calib(dataset: Dataset, *,
                      random_state: int,
                      loss_weight: float = 0.0,
                      loss_random_prior: bool = False,
                      remain_random_prior: bool = False,
                      loss_concept_count: int = 1,
                      remain_concept_count: int = 1) -> Dict[str, Any]:
    """Given a dataset, split it into training, calibration, and 'rest'
    datasets. The training and calibration sets combined will have
    dataset.train_n instances, and will have a distribution based on
    the provided loss/remain parameters.

    Returns the datasets, as well as the components of the
    training+calib distribution."""
    rng = np.random.RandomState(random_state * TRAIN_SPLIT_RANDOM_SEED)
    # Pick random loss and remain concepts.
    concepts = rng.choice(dataset.concepts,
                          size=(loss_concept_count + remain_concept_count),
                          replace=False)
    loss_concepts = concepts[:loss_concept_count]
    remain_concepts = concepts[loss_concept_count:]
    # Configure loss/remain components for the training distribution.
    components = {
        'loss': Component(
            concepts=loss_concepts,
            weight=loss_weight,
            # Optionally randomly shift the loss class_priors.
            class_priors=(random_class_priors(dataset, rng=rng)
                          if loss_random_prior else
                          get_class_priors(dataset, concepts=loss_concepts))
        ),
        'remain': Component(
            concepts=remain_concepts,
            weight=(1 - loss_weight),
            # Optionally randomly shift the remain class_priors.
            class_priors=(random_class_priors(dataset, rng=rng)
                          if remain_random_prior else
                          get_class_priors(dataset, concepts=remain_concepts))
        ),
    }
    # Create dfs of instances in/not-in the full training set.
    full_train_index = sample_indexes_for_components(
        dataset,
        n=dataset.train_n,
        components=components,
        rng=rng,
    )
    full_train_df = dataset.df.loc[full_train_index]
    rest_df = dataset.df.loc[dataset.df.index.difference(full_train_index)]
    # Split the full training set into training/calibration sets.
    train_df, calib_df = train_test_split(
        full_train_df,
        test_size=dataset.calib_size,
        random_state=rng,
        shuffle=True,
        # Stratifying by class maintains per-class distributions.  The
        # per-concept-per-class distributions will be approximately
        # retained, but not exactly - stratifying based on concept
        # would not be possible in a real scenario - also, stratifying
        # by both can often result in only 1 record per group, which
        # is disallowed by train_test_split.
        stratify=full_train_df['class'],
    )
    return {
        'components': components,
        'datasets': {
            'train': dataset.subset(train_df),
            'calib': dataset.subset(calib_df),
            'rest': dataset.subset(rest_df),
        },
    }


def split_test_without_gsls_shift(dataset: Dataset, *,
                                  train_components: Components,
                                  random_state: int,
                                  random_prior: bool = False) -> Dict[str, Any]:
    """Given a dataset (assumed to be the 'rest' dataset from training
    sampling), sample a test set of dataset.test_n instances. The
    distribution will be the same as that of the training set (defined
    by train_components) with optional prior-shift (if random_prior
    is True).

    Returns the test dataset, as well as the components of the test
    distribution."""
    rng = np.random.RandomState(random_state * TEST_SPLIT_RANDOM_SEED)
    components = (
        adjust_components_for_prior_shift(
            classes=dataset.classes,
            components=train_components,
            target_class_priors=random_class_priors(dataset, rng=rng),
        )
        if random_prior else train_components
    )
    test_index = sample_indexes_for_components(
        dataset,
        n=dataset.test_n,
        components=components,
        rng=rng,
    )
    return {
        'components': components,
        'datasets': {
            'test': dataset.subset(dataset.df.loc[test_index]),
        },
    }


def split_test_with_gsls_shift(dataset: Dataset, *,
                               train_components: Components,
                               random_state: int,
                               gain_concept_count: int = 1,
                               gain_weight: float = 0.0,
                               gain_random_prior: bool = False) -> Dict[str, Any]:
    """Given a dataset (assumed to be the 'rest' dataset from training
    sampling), sample a test set of dataset.test_n instances. The
    distribution will have "loss shift" (of train_components,
    remain is kept, but loss is removed) and "gain shift" (a new
    component is added based on the gain parameters).

    Returns the test dataset, as well as the components of the test
    distribution."""
    rng = np.random.RandomState(random_state * TEST_SPLIT_RANDOM_SEED)
    # Pick random concepts for the gain component.
    possible_gain_concepts = set(dataset.concepts).difference(set(np.concatenate([
        train_components['loss'].concepts,
        train_components['remain'].concepts,
    ])))
    gain_concepts = rng.choice(sorted(possible_gain_concepts), size=gain_concept_count, replace=False)
    # Configure gain/remain components for the test distribution.
    components = {
        'gain': Component(
            concepts=gain_concepts,
            weight=gain_weight,
            # Optionally randomly shift the gain class_priors.
            class_priors=(random_class_priors(dataset, rng=rng)
                          if gain_random_prior else
                          get_class_priors(dataset, concepts=gain_concepts)),
        ),
        'remain': Component(
            concepts=train_components['remain'].concepts,
            weight=(1 - gain_weight),
            class_priors=train_components['remain'].class_priors,
        ),
    }
    test_index = sample_indexes_for_components(
        dataset,
        n=dataset.test_n,
        components=components,
        rng=rng,
    )
    return {
        'components': components,
        'datasets': {
            'test': dataset.subset(dataset.df.loc[test_index]),
        },
    }


def split_test(dataset: Dataset, *,
               train_components: Components,
               shift_type: str,
               gain_weight: float,
               random_state: int,
               gain_random_prior: bool) -> Dict[str, Any]:
    """Given a dataset (assumed to be the 'rest' dataset from training
    sampling), sample a test set of dataset.test_n instances. The
    nature of the shift in the generated dataset will be determined by
    the shift_type parameter.

    Returns the test dataset, as well as the components of the test
    distribution.
    """
    if shift_type == 'gsls_shift':
        return split_test_with_gsls_shift(dataset,
                                          train_components=train_components,
                                          gain_weight=gain_weight,
                                          gain_random_prior=gain_random_prior,
                                          random_state=random_state)
    elif shift_type in ['no_shift', 'prior_shift']:
        random_prior = (shift_type == 'prior_shift')
        return split_test_without_gsls_shift(dataset,
                                             train_components=train_components,
                                             random_prior=random_prior,
                                             random_state=random_state)
    else:
        raise ValueError(f'Unknown shift_type: {shift_type}')
