import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from pyquantification.experiments.metrics import prepare_results


def test_prepare_results() -> None:
    df = pd.DataFrame([
        {
            'test_n': 0,
            'test_true_count': 0,
            'gain_weight': 0,
            'loss_weight': 0,
            'gsls_gain_weight': 0,
            'gsls_loss_weight': 0,
            'gsls_count': 0,
            'gsls_count_lower': 0,
            'gsls_count_upper': 0,
        },
        {
            'test_n': 10,
            'test_true_count': 4,
            'gain_weight': 0.7,
            'loss_weight': 0.7,
            'gsls_gain_weight': 0.3,
            'gsls_loss_weight': 0.3,
            'gsls_count': 4,
            'gsls_count_lower': 4,
            'gsls_count_upper': 6,
        },
        {
            'test_n': 10,
            'test_true_count': 4,
            'gain_weight': 0.7,
            'loss_weight': 0.7,
            'gsls_gain_weight': 0.3,
            'gsls_loss_weight': 0.3,
            'gsls_count': 9,
            'gsls_count_lower': 7,
            'gsls_count_upper': 10,
        },
    ])
    df = prepare_results(df)
    assert_frame_equal(
        df[['single_grouping', 'remain_weight', 'gsls_coverage', 'gsls_absolute_error',
            'gsls_width', 'gsls_remain_weight']],
        pd.DataFrame([
            {
                'single_grouping': 'single_grouping',
                'remain_weight': 1.0,
                'gsls_coverage': True,
                'gsls_absolute_error': np.nan,
                'gsls_width': np.nan,
                'gsls_remain_weight': 1.0,
            },
            {
                'single_grouping': 'single_grouping',
                'remain_weight': 0.09,
                'gsls_coverage': True,
                'gsls_absolute_error': 0.0,
                'gsls_width': 2/10,
                'gsls_remain_weight': 0.49,
            },
            {
                'single_grouping': 'single_grouping',
                'remain_weight': 0.09,
                'gsls_coverage': False,
                'gsls_absolute_error': 5/10,
                'gsls_width': 3/10,
                'gsls_remain_weight': 0.49,
            },
        ])
    )
