import numpy as np
import pandas as pd
import pytest

from pyquantification.utils import check_dict_almost_equal
from pyquantification.datasets import Dataset, Component
from pyquantification.experiments.splitting import (
    check_class_priors,
    get_class_priors,
    get_class_priors_for_components,
    check_prior_shift_assumption,
    random_class_priors,
    simulate_class_counts,
    get_component_counts,
    adjust_components_for_prior_shift,
    sample_indexes_for_components,
    split_train_calib,
    split_test,
)


def test_get_class_priors() -> None:
    dataset = Dataset(pd.DataFrame({
        'concept': ['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
        'class':   ['t', 't', 't', 't', 'f', 'f', 't', 't', 'f', 'f'],
    }), train_n=1, test_n=1)
    assert get_class_priors(dataset) == {'t': 6/10, 'f': 4/10}
    assert get_class_priors(dataset, concepts=['a']) == {'t': 4/6, 'f': 2/6}


def test_get_class_priors_for_components() -> None:
    check_dict_almost_equal(
        get_class_priors_for_components({
            'a': Component([], 0.2, {'t': 0.5, 'f': 0.5}),
            'b': Component([], 0.8, {'t': 0.25, 'f': 0.75}),
        }),
        {'t': 0.3, 'f': 0.7}
    )
    check_dict_almost_equal(
        get_class_priors_for_components({
            'a': Component([], 1.0, {'t': 0.5, 'f': 0.5}),
            'b': Component([], 0.0, {'t': 0.25, 'f': 0.75}),
        }),
        {'t': 0.5, 'f': 0.5}
    )
    # Weights don't sum to 1.
    with pytest.raises(ValueError, match='Class priors are not valid'):
        get_class_priors_for_components({
            'a': Component([], 0.5, {'t': 0.5, 'f': 0.5}),
            'b': Component([], 0.0, {'t': 0.25, 'f': 0.75}),
        })
    # Component prior doesn't sum to 1.
    with pytest.raises(ValueError, match='Class priors are not valid'):
        get_class_priors_for_components({
            'a': Component([], 0.5, {'t': 0.25, 'f': 0.25}),
            'b': Component([], 0.5, {'t': 0.25, 'f': 0.75}),
        })


def test_check_prior_shift_assumption() -> None:
    # No shift
    assert check_prior_shift_assumption(
        {
            'a': Component([], 0.5, {'t': 0.5, 'f': 0.5}),
            'b': Component([], 0.5, {'t': 0.25, 'f': 0.75}),
        },
        {
            'a': Component([], 0.5, {'t': 0.5, 'f': 0.5}),
            'b': Component([], 0.5, {'t': 0.25, 'f': 0.75}),
        },
    ) is True
    # Complex prior shift
    assert check_prior_shift_assumption(
        {
            'a': Component([], 0.5, {'t': 1.0, 'f': 0.0}),
            'b': Component([], 0.5, {'t': 0.0, 'f': 1.0}),
        },
        {
            'a': Component([], 0.3, {'t': 1.0, 'f': 0.0}),
            'b': Component([], 0.7, {'t': 0.0, 'f': 1.0}),
        },
    ) is True
    # Not prior shift
    assert check_prior_shift_assumption(
        {
            'a': Component([], 0.2, {'t': 0.5, 'f': 0.5}),
            'b': Component([], 0.8, {'t': 0.25, 'f': 0.75}),
        },
        {
            'a': Component([], 0.4, {'t': 0.5, 'f': 0.5}),
            'b': Component([], 0.6, {'t': 0.25, 'f': 0.75}),
        },
    ) is False


def test_random_class_priors() -> None:
    dataset = Dataset(pd.DataFrame({
        'concept': ['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b'],
        'class':   ['t', 't', 'u', 'u', 'f', 'f', 't', 't', 'f', 'f', 'u', 'u'],
    }), train_n=1, test_n=1)
    priors = random_class_priors(dataset, rng=np.random.RandomState(1))
    assert sorted(priors.keys()) == ['f', 't', 'u']
    assert (1 - np.sum(list(priors.values()))) < 1e-7


def test_simulate_class_counts() -> None:
    min_class_count = 2
    n = 200
    priors = {'t': 0.95, 'f': 0.04, 'u': 0.01}
    sims = [simulate_class_counts(
        n=n,
        class_priors=priors,
        min_class_count=min_class_count,
        rng=np.random.RandomState(i),
    ) for i in range(1000)]
    sim_df = pd.DataFrame(sims)
    # Check total n in each row
    assert np.all(sim_df.sum(axis='columns') == n)
    # Check min_class_count in each column
    assert np.all(sim_df.min(axis='rows') >= 2)
    # Sampling approaches priors
    assert check_dict_almost_equal(
        (sim_df.mean(axis='rows') / n).to_dict(),
        priors,
        decimal=2
    )


def test_get_component_counts() -> None:
    assert get_component_counts(n=100, components={
        'a': Component([], 0.3, {}),
        'b': Component([], 0.7, {}),
    }) == {'a': 30, 'b': 70}
    assert get_component_counts(n=10, components={
        'a': Component([], 0.25, {}),
        'b': Component([], 0.75, {}),
    }) == {'a': 2, 'b': 8}
    assert get_component_counts(n=10, components={
        'a': Component([], 0.0, {}),
        'b': Component([], 1.0, {}),
    }) == {'a': 0, 'b': 10}
    with pytest.raises(ValueError, match='Component weights are not valid'):
        get_component_counts(n=10, components={
            'a': Component([], 0.25, {}),
            'b': Component([], 0.25, {}),
        })


def test_adjust_components_for_prior_shift() -> None:
    adjusted_components = adjust_components_for_prior_shift(
        classes=['a', 'b', 'c'],
        components={
            'loss': Component([], 0.5, {'a': 0.2, 'b': 0.3, 'c': 0.5}),
            'remain': Component([], 0.5, {'a': 0.5, 'b': 0.3, 'c': 0.2}),
        },
        target_class_priors={'a': 0.2, 'b': 0.3, 'c': 0.5})
    # Many assertions of behaviour in function, so test a known point
    # for regressions.
    assert abs(0.5642857142857143 - adjusted_components['loss'].weight) < 1e-7
    assert abs(0.4357142857142857 - adjusted_components['remain'].weight) < 1e-7
    assert check_dict_almost_equal(adjusted_components['loss'].class_priors,
                                   {'a': 0.10126582278481015,
                                    'b': 0.26582278481012656,
                                    'c': 0.6329113924050633})
    assert check_dict_almost_equal(adjusted_components['remain'].class_priors,
                                   {'a': 0.3278688524590164,
                                    'b': 0.3442622950819672,
                                    'c': 0.3278688524590164})


def test_sample_indexes_for_components() -> None:
    df = pd.DataFrame({
        'concept': (['a'] * 500) + (['b'] * 500),
        'class':   ['t', 'f'] * 500,
    }, index=list(reversed(range(1000))))
    dataset = Dataset(df, train_n=100, test_n=100)
    components = {
        'a': Component(['a'], 0.4, {'t': 0.25, 'f': 0.75}),
        'b': Component(['b'], 0.6, {'t': 0.5, 'f': 0.5}),
    }
    indexes = sample_indexes_for_components(dataset, n=100,
                                            components=components,
                                            min_class_count=2,
                                            rng=np.random.RandomState(1))
    sample_df = dataset.df.loc[indexes]
    class_concept_counts = sample_df.groupby(['concept', 'class']).count().to_dict()
    assert class_concept_counts['index'] == {
        ('a', 't'): 9,
        ('a', 'f'): 31,
        ('b', 't'): 30,
        ('b', 'f'): 30,
    }


def test_split_train_calib() -> None:
    df = pd.DataFrame({
        'concept': (['a'] * 500) + (['b'] * 500),
        'class':   ['t', 'f'] * 500,
    }, index=list(reversed(range(1000))))
    dataset = Dataset(df, train_n=100, test_n=100, calib_size=0.4)
    split = split_train_calib(
        dataset,
        random_state=1,
        loss_weight=0.3,
        loss_random_prior=True,
        remain_random_prior=True,
        loss_concept_count=1,
        remain_concept_count=1,
    )
    assert split['components']['loss'].concepts == ['a']
    assert split['components']['loss'].weight == 0.3
    assert split['components']['loss'].class_priors['t'] != 0.5
    assert split['components']['loss'].class_priors['f'] != 0.5
    assert check_class_priors(split['components']['loss'].class_priors)
    assert split['components']['remain'].concepts == ['b']
    assert split['components']['remain'].weight == 0.7
    assert split['components']['remain'].class_priors['t'] != 0.5
    assert split['components']['remain'].class_priors['f'] != 0.5
    assert check_class_priors(split['components']['remain'].class_priors)
    assert split['datasets']['train'].df.shape[0] == 60
    assert split['datasets']['calib'].df.shape[0] == 40
    assert split['datasets']['rest'].df.shape[0] == 900


def test_split_test_gsls_shift() -> None:
    df = pd.DataFrame({
        'concept': (['a'] * 200) + (['b'] * 300) + (['c'] * 500),
        'class':   ['t', 'f'] * 500,
    }, index=list(reversed(range(1000))))
    dataset = Dataset(df, train_n=50, test_n=50, calib_size=0.4)
    train_components = {
        'loss': Component(['a'], 0.4, {'t': 0.5, 'f': 0.5}),
        'remain': Component(['b'], 0.6, {'t': 0.25, 'f': 0.75}),
    }
    split = split_test(
        dataset,
        train_components=train_components,
        shift_type='gsls_shift',
        gain_weight=0.6,
        random_state=1,
        gain_random_prior=True,
    )
    assert split['components']['gain'].concepts == ['c']
    assert split['components']['gain'].weight == 0.6
    assert split['components']['gain'].class_priors['t'] != 0.5
    assert split['components']['gain'].class_priors['f'] != 0.5
    assert check_class_priors(split['components']['gain'].class_priors)
    assert split['components']['remain'].concepts == ['b']
    assert split['components']['remain'].weight == 0.4
    assert split['components']['remain'].class_priors['t'] == 0.25
    assert split['components']['remain'].class_priors['f'] == 0.75
    assert check_class_priors(split['components']['remain'].class_priors)
    assert split['datasets']['test'].df.shape[0] == 50


def test_split_test_prior_shift() -> None:
    df = pd.DataFrame({
        'concept': (['a'] * 200) + (['b'] * 300) + (['c'] * 500),
        'class':   ['t', 'f'] * 500,
    }, index=list(reversed(range(1000))))
    dataset = Dataset(df, train_n=50, test_n=50, calib_size=0.4)
    train_components = {
        'loss': Component(['a'], 0.4, {'t': 0.5, 'f': 0.5}),
        'remain': Component(['b'], 0.6, {'t': 0.25, 'f': 0.75}),
    }
    split = split_test(
        dataset,
        train_components=train_components,
        shift_type='prior_shift',
        gain_weight=0.6,
        random_state=1,
        gain_random_prior=True,
    )
    assert split['components']['loss'].concepts == ['a']
    assert split['components']['loss'].weight != 0.6
    assert split['components']['loss'].class_priors['t'] != 0.5
    assert split['components']['loss'].class_priors['f'] != 0.5
    assert check_class_priors(split['components']['loss'].class_priors)
    assert split['components']['remain'].concepts == ['b']
    assert split['components']['remain'].weight != 0.4
    assert split['components']['remain'].class_priors['t'] != 0.25
    assert split['components']['remain'].class_priors['f'] != 0.75
    assert check_class_priors(split['components']['remain'].class_priors)
    assert split['datasets']['test'].df.shape[0] == 50
    assert check_prior_shift_assumption(train_components, split['components'])
