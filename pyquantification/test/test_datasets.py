import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal
from string import ascii_lowercase
import pytest

from pyquantification.datasets import Component, ConceptsDataset, SamplesDataset


def build_df(data: dict) -> pd.DataFrame:
    """Construct a DataFrame with a non-default index to test index preservation."""
    df = pd.DataFrame(data)
    df.index = list(ascii_lowercase)[:df.shape[0]]
    return df


def test_ConceptsDataset() -> None:
    orig_df = build_df({
        'class': ['t', 't', 't', 't', 'f', 'f', 'f', 'f'],
        'concept': ['p', 'p', 'q', 'q', 'p', 'p', 'q', 'q'],
        'a': [1, 2, 3, 4, 5, 6, 7, 8],
        'b': [8, 7, 6, 5, 4, 3, 2, 1],
    })

    with pytest.raises(ValueError,
                       match=('Only 2 instances available for class "f" in '
                              'concept "p", while the maximum number that may '
                              'be sampled is the full train and test size: 3')):
        ConceptsDataset(orig_df, train_n=2, test_n=1)

    dataset = ConceptsDataset(
        orig_df,
        train_n=1,
        test_n=1,
        numeric_features={'a', 'b'},
    )
    # Dataset resets index.
    df = orig_df.reset_index(drop=True)

    assert dataset.name == 'UNNAMED'
    dataset.set_name('test')
    assert dataset.name == 'test'

    assert dataset.all_features == {'a', 'b'}
    assert_frame_equal(dataset.X, df[['a', 'b']], check_like=True)
    assert_series_equal(dataset.y, df['class'])
    assert_array_equal(dataset.indexes, np.array(range(df.shape[0])))
    assert_frame_equal(dataset.subset(df.iloc[4:]).df, df.iloc[4:])

    assert dataset.concepts == ['p', 'q']
    assert dataset.calib_n == 1
    assert_array_equal(dataset.components_index({
        'a': Component(['p'], 0.5, {'t': 0.5, 'f': 0.5}),
        'b': Component(['q'], 0.5, {'t': 0.5, 'f': 0.5}),
    }), np.array(['a', 'a', 'b', 'b', 'a', 'a', 'b', 'b']))


def test_SamplesDataset() -> None:
    orig_df = build_df({
        'class': ['t', 't', 't', 't', 'f', 'f', 'f', 'f'],
        'sample': ['p', 'p', 'q', 'q', 'p', 'p', 'q', 'q'],
        'a': [1, 2, 3, 4, 5, 6, 7, 8],
        'b': [8, 7, 6, 5, 4, 3, 2, 1],
    })

    dataset = SamplesDataset(
        orig_df,
        numeric_features={'a', 'b'},
        train_samples={'p'},
    )
    # Dataset resets index.
    df = orig_df.reset_index(drop=True)

    assert dataset.name == 'UNNAMED'
    dataset.set_name('test')
    assert dataset.name == 'test'

    assert dataset.all_features == {'a', 'b'}
    assert_frame_equal(dataset.X, df[['a', 'b']], check_like=True)
    assert_series_equal(dataset.y, df['class'])
    assert_array_equal(dataset.indexes, np.array(range(df.shape[0])))
    assert_frame_equal(dataset.subset(df.iloc[4:]).df, df.iloc[4:])

    assert dataset.test_samples == ['q']
    assert dataset.subset(dataset.df.iloc[:1]).test_samples == []
    np.testing.assert_array_equal(dataset.get_train_index(), np.array([0, 1, 4, 5]))
    np.testing.assert_array_equal(dataset.get_test_index(0), np.array([2, 3, 6, 7]))
