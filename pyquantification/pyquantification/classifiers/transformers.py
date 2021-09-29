from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Type


class DataFrameTransformer(TransformerMixin, BaseEstimator, ABC):
    """Abstract base class for transformers that consume and return pandas
    DataFrames."""

    def fit(self,
            df_X: pd.DataFrame,
            y: pd.Series = None) -> 'DataFrameTransformer':
        return self

    @abstractmethod
    def transform(self, df_X: pd.DataFrame) -> pd.DataFrame:
        pass


class SelectFeatureSubsetDataFrameTransformer(DataFrameTransformer):
    """Transforms a DataFrame by selecting only the given feature_names."""

    def __init__(self, feature_names: List[str]) -> None:
        self.feature_names = feature_names
        super().__init__()

    def transform(self, df_X: pd.DataFrame) -> pd.DataFrame:
        return df_X[self.feature_names]


class TransformFeatureSubsetDataFrameTransformer(DataFrameTransformer):
    """Applies the given transformer to only the named features in the
    DataFrame (column order may change)."""

    def __init__(self,
                 feature_names: List[str],
                 transformer: DataFrameTransformer) -> None:
        self.feature_names = feature_names
        self.transformer = transformer
        super().__init__()

    def fit(self,
            df_X: pd.DataFrame,
            y: pd.Series = None) -> 'TransformFeatureSubsetDataFrameTransformer':
        super().fit(df_X, y)
        if len(self.feature_names) > 0:
            self.transformer.fit(df_X[self.feature_names], y)
        return self

    def reconcat(self,
                 df_A: pd.DataFrame,
                 df_B: pd.DataFrame,
                 index: pd.Index) -> pd.DataFrame:
        """As the transformation may have altered the index of one subset of
        the DataFrame, we reset the index on both, join them, and then
        set the index back."""
        dfs = [df.reset_index(drop=True) for df in [df_A, df_B]]
        df_concat = pd.concat(dfs, axis=1)
        df_concat.index = index
        return df_concat

    def transform(self, df_X: pd.DataFrame) -> pd.DataFrame:
        # Short-circuit if the subset of features is empty.
        if len(self.feature_names) < 1:
            return df_X

        transformed_df = self.transformer.transform(df_X[self.feature_names])
        rest_df = df_X.drop(self.feature_names, axis=1)
        return self.reconcat(rest_df, transformed_df, index=df_X.index)

    def fit_transform(self,
                      df_X: pd.DataFrame,
                      y: pd.Series = None) -> pd.DataFrame:
        # Short-circuit if the subset of features is empty.
        if len(self.feature_names) < 1:
            return df_X

        transformed_df = self.transformer.fit_transform(df_X[self.feature_names], y)
        rest_df = df_X.drop(self.feature_names, axis=1)
        return self.reconcat(rest_df, transformed_df, index=df_X.index)


class NumpyTransformer(DataFrameTransformer):
    """Convert a DataFrame to a numpy array with given dtype."""

    def __init__(self, dtype: Type) -> None:
        self.dtype = dtype

    def transform(self, df_X: pd.DataFrame) -> pd.DataFrame:
        return df_X.to_numpy(self.dtype)


class DataFrameStandardNormaliser(DataFrameTransformer):
    """Apply standard/z-score normalisation to the specified columns in
    the DataFrame."""

    def __init__(self) -> None:
        super().__init__()

    def fit(self,
            df_X: pd.DataFrame,
            y: pd.Series = None) -> 'DataFrameStandardNormaliser':
        # Note: Pandas uses an unbiased estimator of std.
        self.col_means_stds = {col: (df_X[col].mean(), df_X[col].std())
                               for col in df_X.columns}
        return self

    def transform(self, df_X: pd.DataFrame) -> pd.DataFrame:
        updated_columns = {}
        for col, (mean, std) in self.col_means_stds.items():
            if std == 0:
                # Avoid division by zero, there is no deviation from
                # the mean, so all values should be zero.
                updated_columns[col] = 0
            else:
                updated_columns[col] = (df_X[col] - mean) / std
        return df_X.assign(**updated_columns)
