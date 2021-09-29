import re
import numpy as np
import pandas as pd
from typing import cast


def coverage_series(df: pd.DataFrame, method: str) -> pd.Series:
    """Return a boolean series indicating whether the true value falls
    within the prediction interval of the given quantification_method"""
    return (
        (df['test_true_count'] >= df[f'{method}_count_lower']) &
        (df['test_true_count'] <= df[f'{method}_count_upper'])
    )


def relative_rate_series(df: pd.DataFrame, count_column: str) -> pd.Series:
    """Convert any instance 'count' column to 0-1 scale by dividing by
    the number of test instances."""
    relative_rate = df[count_column] / df['test_n']
    # Replace infinite rates with None (where replaces when condition is False).
    return relative_rate.where((df['test_n'] > 0), np.nan)


def absolute_error_series(df: pd.DataFrame, method: str) -> pd.Series:
    """Absolute error is the absolute difference between the true instance
    proportion and that estimated by the given quantification method"""
    return cast(
        pd.Series,
        (relative_rate_series(df, f'{method}_count') -
         relative_rate_series(df, 'test_true_count')).abs().astype(float)
    )


def interval_width_series(df: pd.DataFrame, method: str) -> pd.Series:
    """Return the (0-1 scale) width of the prediction interval produced by
    the given quantification method."""
    width_series = (df[f'{method}_count_upper'] - df[f'{method}_count_lower']) / df['test_n']
    # Replace infinite widths with None (where replaces when condition is False).
    return width_series.where((df['test_n'] > 0), np.nan)


def prepare_results(df: pd.DataFrame) -> pd.DataFrame:
    """Metric pre-computation for a results DataFrame."""
    new_cols = {
        'single_grouping': 'single_grouping',
        'remain_weight': ((1 - df['loss_weight']) * (1 - df['gain_weight']))  # type: ignore
    }

    quantification_methods = []
    for column in df.columns:
        matches = re.search('^(.+)_count_lower$', column)
        if matches is not None:
            quantification_methods.append(matches.group(1))

    for method in quantification_methods:
        new_cols[f'{method}_coverage'] = coverage_series(df, method)
        new_cols[f'{method}_absolute_error'] = absolute_error_series(df, method)
        new_cols[f'{method}_width'] = interval_width_series(df, method)
        if f'{method}_loss_weight' in df.columns:
            new_cols[f'{method}_remain_weight'] = (
                (1 - df[f'{method}_loss_weight']) *  # type: ignore
                (1 - df[f'{method}_gain_weight'])  # type: ignore
            )
    return df.assign(**new_cols)
