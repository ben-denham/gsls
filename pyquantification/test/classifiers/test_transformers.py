import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from string import ascii_lowercase

from pyquantification.classifiers.transformers import (
    DataFrameTransformer,
    SelectFeatureSubsetDataFrameTransformer,
    TransformFeatureSubsetDataFrameTransformer,
    NumpyTransformer,
    DataFrameStandardNormaliser,
)


def build_df(data: dict) -> pd.DataFrame:
    """Construct a DataFrame with a non-default index to test index preservation."""
    df = pd.DataFrame(data)
    df.index = list(ascii_lowercase)[:df.shape[0]]
    return df


class TestTransformer(DataFrameTransformer):

    def transform(self, df_X: pd.DataFrame) -> pd.DataFrame:
        df_X = df_X.copy()
        df_X.loc[:, :] = 42
        return df_X


def test_SelectFeatureSubsetDataFrameTransformer() -> None:
    orig_df = build_df({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df = orig_df.copy()

    tf = SelectFeatureSubsetDataFrameTransformer(['a'])
    tf.fit(df)
    new_df = tf.transform(df)

    assert_frame_equal(df, orig_df)
    assert_frame_equal(new_df,
                       build_df({'a': [1, 2, 3]}))


def test_TransformFeatureSubsetDataFrameTransformer() -> None:
    orig_df = build_df({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df = orig_df.copy()

    tf = TransformFeatureSubsetDataFrameTransformer(['a'], TestTransformer())
    tf.fit(df)
    new_df = tf.transform(df)

    assert_frame_equal(new_df,
                       build_df({'a': [42, 42, 42], 'b': [4, 5, 6]}),
                       # Don't check column order.
                       check_like=True)
    assert_frame_equal(df, orig_df)


def test_TransformFeatureSubsetDataFrameTransformer_fit_transform() -> None:
    orig_df = build_df({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df = orig_df.copy()

    tf = TransformFeatureSubsetDataFrameTransformer(['a'], TestTransformer())
    new_df = tf.fit_transform(df)

    assert_frame_equal(new_df,
                       build_df({'a': [42, 42, 42], 'b': [4, 5, 6]}),
                       # Don't check column order.
                       check_like=True)
    assert_frame_equal(df, orig_df)


def test_NumpyTransformer() -> None:
    orig_df = build_df({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df = orig_df.copy()

    tf = NumpyTransformer(np.float64)
    tf.fit(df)
    new_array = tf.transform(df)

    assert_array_equal(new_array,
                       np.array([[1, 4], [2, 5], [3, 6]]).astype(np.float64))
    assert_frame_equal(df, orig_df)


def test_DataFrameStandardNormaliser() -> None:
    orig_df = build_df({'a': [1, 2, 3, 4, 5],
                        'b': [100, 200, 300, 400, 500],
                        'c': [4, 4, 4, 4, 4]})
    df = orig_df.copy()

    tf = DataFrameStandardNormaliser()
    tf.fit(df)
    new_df = tf.transform(df)

    assert_frame_equal(new_df,
                       build_df({'a': [-1.264911, -0.632456, 0, 0.632456, 1.264911],
                                 'b': [-1.264911, -0.632456, 0, 0.632456, 1.264911],
                                 'c': [0, 0, 0, 0, 0]}))
    assert_frame_equal(df, orig_df)
