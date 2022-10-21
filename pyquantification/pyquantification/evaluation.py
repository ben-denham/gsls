# coding: utf-8

import itertools
import matplotlib.pyplot as plt
import Orange
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, Markdown
from colorsys import hsv_to_rgb
import scipy.stats
from typing import cast, Union, Any, Optional, Callable, Dict, Sequence, List, Tuple, Iterator

from pyquantification.utils import select_keys
from pyquantification.datasets import DATASETS, ConceptsDataset
from pyquantification.quantifiers.gsls import GslsQuantifier

Colormap = Dict[str, str]

BASE_LAYOUT = dict(
    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(0, 0, 0, 0)',
    margin=dict(b=10, t=10, l=10, r=10, pad=0),
    font=dict(
        size=19,
        color='black',
    ),
    legend=dict(orientation='h'),
    yaxis_gridcolor='#AAAAAA',
    xaxis_gridcolor='#AAAAAA',
    yaxis_zerolinecolor='#333333',
    xaxis_zerolinecolor='#333333',
)


def get_colormap(values: Sequence[str]) -> Colormap:
    """Return a colormap for the given value keys."""
    return dict(zip(values, [px.colors.qualitative.Safe[idx] for idx in [0, 8, 6, 4, 1]]))


def print_markdown(*text: str) -> None:
    """Print the given text as markdown."""
    display(Markdown(*text))


def display_dataset_summary(dataset_labels: Dict[str, str]) -> None:
    """Display a summary of the given datasets."""
    dataset_rows = []
    for dataset_name, dataset_label in dataset_labels.items():
        dataset = DATASETS[dataset_name]()
        dataset_row = {
            'name': dataset.name,
            'label': dataset_label,
            'Features': len(dataset.all_features),
            'Classes': len(dataset.classes),
        }
        if isinstance(dataset, ConceptsDataset):
            dataset = cast(ConceptsDataset, dataset)
            dataset_row = {
                **dataset_row,
                'Concepts': len(dataset.concepts),
                'Train+Calibration instances per experiment': dataset.train_n,
                'Test instances per experiment': dataset.test_n,
                'GSLS Histogram Bins': GslsQuantifier.get_auto_hist_bins(
                    calib_count=dataset.calib_n,
                    target_count=dataset.test_n,
                ),
            }
        dataset_rows.append(dataset_row)
    display(pd.DataFrame(dataset_rows).set_index('name'))


def display_stat_table(df: pd.DataFrame,
                       dataset_names: Optional[List[str]] = None,
                       *,
                       stat: str,
                       row_grouping: Union[str, Sequence[str]],
                       methods: Dict[str, str],
                       color_func: Optional[Callable[[float], str]] = None,
                       include_std: bool = False,
                       format_string: str = '{:.2%}',
                       transpose: bool = False) -> None:
    """Display a table of stat values, with a row for each value for the
    row_grouping column(s) and a column for each method (columns also
    grouped by dataset if dataset_name or dataset_label is not part of
    the row_grouping)."""
    row_grouping = [row_grouping] if isinstance(row_grouping, str) else row_grouping

    ALL_DATASETS = 'ALL_DATASETS'
    if ('dataset_name' in row_grouping) or ('dataset_label' in row_grouping):
        table_dataset_names = [ALL_DATASETS]
    else:
        table_dataset_names = dataset_names or list(sorted(df['dataset_name'].unique()))

    def get_dataset_dfs() -> Iterator[Tuple[str, pd.DataFrame]]:
        dataset_dfs = df.groupby('dataset_name')
        for dataset_name in table_dataset_names:
            if dataset_name == ALL_DATASETS:
                yield '', df
            if dataset_name in dataset_dfs.groups.keys():
                yield dataset_name, dataset_dfs.get_group(dataset_name)

    column_to_series: Dict[Tuple, Any] = {}
    for dataset_name, dataset_df in get_dataset_dfs():
        gain_loss_groups = dataset_df.groupby(row_grouping)
        for method_name, method in methods.items():
            mean = gain_loss_groups[f'{method}_{stat}'].mean().astype(np.float_)
            if include_std:
                std = gain_loss_groups[f'{method}_{stat}'].std().astype(np.float_)
                column_to_series[(dataset_name, method_name, 'mean')] = mean
                column_to_series[(dataset_name, method_name, 'std')] = std.map(('(' + format_string + ')').format)
            else:
                column_to_series[(dataset_name, method_name)] = mean

        if len(table_dataset_names) > 1:
            column_to_series[(dataset_name, '_')] = pd.Series('', index=gain_loss_groups.sum().index)

    df = pd.DataFrame(column_to_series)
    if transpose:
        df = df.T
    df_style = df.style

    if color_func is not None:
        def style_func(val):
            return (f'color: {color_func(val)}'
                    if (isinstance(val, np.float_) or isinstance(val, float))
                    else '')
        df_style = df_style.applymap(style_func)

    print_markdown(f'#### Comparison of `{stat}` across methods')
    display(
        df_style
        .format(lambda val: format_string.format(val)
                if (isinstance(val, np.float_) or isinstance(val, float))
                else val)
        .set_table_styles([{
            'selector': 'th',
            'props': [
                ('vertical-align', 'top'),
                ('text-align', 'left'),
            ]
        }])
    )


def color_scale(*,
                threshold: float = 0,
                min_value: float = 0,
                max_value: float = 1,
                inverted: bool = False) -> Callable[[float], str]:
    """Return color_func to generate readable graduated colours,
    where values above threshold are green and those below are red (colours
    reversed if inverted=True)."""

    def color_func(value):
        high_hue = 120 / 360  # green
        low_hue = 0 / 360  # red
        if inverted:
            high_hue, low_hue = low_hue, high_hue

        if value >= threshold:
            proportion = (value - threshold) / (max_value - threshold)
            h = high_hue
            v = (proportion ** 0.5) * 0.7
        else:
            proportion = (threshold - value) / (threshold - min_value)
            h = low_hue
            v = proportion ** 0.5
        r, g, b = hsv_to_rgb(h, 0.9, v)
        return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

    return color_func


def plot_remain_weight(df: pd.DataFrame,
                       gsls_method: str,
                       *,
                       colormap: Colormap = {}) -> go.Figure:
    """Plot the remain_weight estimated for gsls_method against the true
    remain_weight."""
    fig = px.box(
        df,
        x='remain_weight',
        y=f'{gsls_method}_remain_weight',
        color='dataset_label',
        color_discrete_map=colormap,
        category_orders={'dataset_label': list(colormap.keys())},
        labels={
            'remain_weight': 'True remaining weight: <i>w<sup>R</sup></i>',
            f'{gsls_method}_remain_weight': 'Estimated remaining weight: <i>ŵ<sup>R</sup></i>',
        },
    )
    fig.add_trace(go.Scatter(
        x=df['remain_weight'].unique(),
        y=df['remain_weight'].unique(),
        name='True w<sup>R</sup>    ',
        mode='markers',
        marker_symbol='line-ew',
        marker_line_width=3,
        marker_size=45,
        marker_line_color='black',
    ))
    fig.update_layout(**BASE_LAYOUT)
    fig.update_layout(
        legend=dict(
            x=0.02,
            y=-0.2,
            title=None,
        ),
        xaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        yaxis_range=[-0.02, 1.02],
        xaxis_tickvals=df['remain_weight'].unique(),
        yaxis_tickvals=df['remain_weight'].unique(),
    )
    return fig


def plot_bin_sensitivity(df: pd.DataFrame, *,
                         dataset_label_colors: Dict[str, str],
                         bin_count_symbols: Dict[Union[int, str], Tuple[str, int]]) -> go.Figure:
    """Plot mean estimated gsls_remain_weight vs true remain_weight for
    each bin_count and dataset."""
    # Get a row with the mean gsls_remain_weight of each group.
    group_keys = ['dataset_label', 'bin_count', 'remain_weight']
    groups_df = pd.DataFrame([
        {
            **dict(zip(group_keys, group_values)),
            'gsls_remain_weight': group_df['gsls_remain_weight'].mean(),
        }
        for group_values, group_df in df.groupby(group_keys)
    ])

    # Sort data by provided orderings of datasets and bin_counts.
    groups_df = groups_df.assign(
        dataset_order=groups_df['dataset_label'].map(dict(zip(dataset_label_colors.keys(),
                                                              range(len(dataset_label_colors))))),
        bin_count_order=groups_df['bin_count'].map(dict(zip(bin_count_symbols.keys(),
                                                            range(len(bin_count_symbols))))),
    ).sort_values(['dataset_order', 'bin_count_order'])

    fig = go.Figure()
    for _, row in groups_df.iterrows():
        color = dataset_label_colors[cast(str, row['dataset_label'])]
        symbol, marker_size = bin_count_symbols[cast(Union[int, str], row['bin_count'])]
        fig.add_trace(go.Scatter(
            x=[row['remain_weight']],
            y=[row['gsls_remain_weight']],
            marker=dict(
                color=color,
                symbol=symbol,
                size=marker_size,
                opacity=0.9,
                line=dict(
                    width=2,
                    color='black',
                ),
            ),
            showlegend=False,
        ))
    # Datasets for legend
    for dataset_label, color in dataset_label_colors.items():
        fig.add_trace(go.Bar(
            x=[None],
            y=[None],
            name=dataset_label,
            marker=dict(
                color=color,
                line=dict(
                    width=2,
                    color='black',
                ),
            ),
        ))
    fig.add_trace(go.Scatter(
        x=df['remain_weight'].unique(),
        y=df['remain_weight'].unique(),
        name='True w<sup>R</sup>    ',
        mode='markers',
        marker_symbol='line-ew',
        marker_line_width=3,
        marker_size=40,
        marker_line_color='black',
    ))
    # Bin counts for legend
    for bin_count, (symbol, symbol_size) in bin_count_symbols.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            name=f'{bin_count} bins',
            mode='markers',
            marker=dict(
                color='black',
                symbol=symbol,
                size=symbol_size,
                line=dict(
                    width=2,
                    color='black',
                ),
            ),
        ))
    fig.update_layout(**BASE_LAYOUT)
    fig.update_layout(
        xaxis_title='True remaining weight: <i>w<sup>R</sup></i>',
        yaxis_title='Estimated remaining weight: <i>ŵ<sup>R</sup></i>',
        yaxis_range=[-0.02, 1.02],
        xaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        xaxis_tickvals=df['remain_weight'].unique(),
        yaxis_tickvals=df['remain_weight'].unique(),
        legend=dict(
            y=-0.18,
        ),
    )
    return fig


def plot_error_bars(df: pd.DataFrame,
                    gsls_method: str,
                    *,
                    methods: Sequence[str] = ['pcc', 'gsls'],
                    include_fit_weights: bool = True) -> go.Figure:
    """Plot quantification estimates (with error bars) for the given
    methods against the true quantification. Also plot bars of
    gain/loss weights."""
    markersize = 12
    linewidth = 4
    df = df.sort_values(by=['target_class', 'random_state', 'loss_weight', 'gain_weight', 'bin_count'])

    def x_axis(subcategory):
        return [np.arange(df.shape[0]), np.full(df.shape[0], subcategory)]

    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Bar(
        x=x_axis('True weights'),
        # Scale effect of loss_weight based on the weight of remain
        # distribution in the target distribution.
        y=(df['loss_weight'] * (1 - df['gain_weight'])),  # type: ignore
        name='Loss (relative to gain)    ',
        marker=dict(color='#FFDDDD', line=dict(color='black', width=2)),
    ))
    fig.add_trace(go.Bar(
        x=x_axis('True weights'),
        y=df['gain_weight'],
        name='Gain',
        marker=dict(color='#F5FFF5', line=dict(color='black', width=2)),
    ))

    all_methods = {
        'gsls': ('GSLS estimate', 'gsls', '#544200', 16, 10),
        'em': ('EM estimate', 'em', '#F72DFF', 13, 9),
        'pcc': ('PCC estimate', 'pcc', '#3BD0EB', 10, 8),
    }
    for name, method, color, markersize, barwidth in select_keys(all_methods, methods).values():
        fig.add_trace(go.Scatter(
            x=x_axis('True weights'),
            y=df[f'{method}_count'] / df['test_n'],
            name=name,
            marker=dict(color=color, size=markersize),
            mode='markers',
            error_y=dict(
                type='data',
                symmetric=False,
                array=(df[f'{method}_count_upper'] - df[f'{method}_count']) / df['test_n'],
                arrayminus=(df[f'{method}_count'] - df[f'{method}_count_lower']) / df['test_n'],
                thickness=linewidth * 0.8,
                width=barwidth,
            ),
        ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=x_axis('True weights'),
        y=df['test_true_count'] / df['test_n'],
        name='True class proportion  ',
        marker=dict(color='#000000', size=12, symbol='x'),
        mode='markers',
    ), secondary_y=True)

    if include_fit_weights:
        fig.add_trace(go.Bar(
            x=x_axis('Fit weights'),
            # Scale effect of loss_weight based on the weight of
            # remain distribution in the target distribution.
            y=(df[f'{gsls_method}_loss_weight'] * (1 - df[f'{gsls_method}_gain_weight'])),  # type: ignore
            name='Fit relative loss',
            marker=dict(color='#FF1111', opacity=0.5),
        ))
        fig.add_trace(go.Bar(
            x=x_axis('Fit weights'),
            y=df[f'{gsls_method}_gain_weight'],
            name='Fit gain',
            marker=dict(color='#11FF11', opacity=0.5),
        ))
        fig.add_trace(go.Bar(
            x=x_axis(' '),
            y=np.zeros(df.shape[0]),
            name='divider',
            marker=dict(color='black'),
        ))
    fig.update_yaxes(secondary_y=False, side='right', title='Degree of dataset shift',
                     tickformat='%', showgrid=False)
    fig.update_yaxes(secondary_y=True, side='left', title='Class proportion',
                     tickformat='%', showgrid=False)
    fig.update_layout(
        **BASE_LAYOUT,
        barmode='stack',
        xaxis=dict(visible=False),
    )
    return fig


def plot_error_bars_sample(df: pd.DataFrame,
                           gsls_method: str,
                           *,
                           dataset_name: str,
                           seed: int,
                           **kwargs: Any) -> go.Figure:
    """Error bars plot for a single dataset, class, and random_state to
    see the variation across loss/gain weights."""
    classes = df.loc[df['dataset_name'] == dataset_name, 'target_class'].unique().tolist()
    return plot_error_bars(
        df[(df['dataset_name'] == dataset_name) &
           (df['target_class'] == classes[0]) &
           (df['random_state'] == seed)],
        gsls_method, **kwargs)


def t_test(xs: np.ndarray,
           ys: np.ndarray,
           *, alternative: str = 'two-sided') -> float:
    """
    Based on the t-test as documented in
    Chapter 5, Credibility: Evaluating What’s Been
    Learned, Data Mining: Practical Machine Learning Tools and
    Techniques, Third Edition, 2011.

    Assumes each paired sample is an independent test, for re-sampled test
    sets use the corrected_resampled_t_test.
    """
    assert xs.shape == ys.shape
    assert len(xs.shape) == 1
    ds = xs - ys
    d_mean = np.mean(ds)
    # Using same ddof as: https://github.com/Waikato/weka-3.8/blob/49865490cef763855ede07cd11331a7aeaecd110/weka/src/main/java/weka/experiment/Stats.java#L316
    d_var = np.var(ds, ddof=1)
    # If two series are identical, we cannot perform a t-test, so
    # return an infinite p-value.
    if d_var == 0.0:
        return np.inf
    k = ds.shape[0]
    t = d_mean / np.sqrt(d_var / k)
    # 2-sided t-test (so multiply by 2) with k-1 degrees of freedom
    if alternative == 'two-sided':
        p = (1.0 - scipy.stats.t.cdf(abs(t), k-1)) * 2.0
    elif alternative == 'greater':
        p = (1.0 - scipy.stats.t.cdf(t, k-1))
    elif alternative == 'less':
        p = (1.0 - scipy.stats.t.cdf(-t, k-1))
    else:
        raise ValueError('Unsupported alternative value: {}'.format(alternative))
    return p


def corrected_resampled_t_test(xs: np.ndarray,
                               ys: np.ndarray,
                               *, test_size: float,
                               alternative: str = 'two-sided') -> float:
    """
    Based on corrected resampled t-test as documented in
    Chapter 5, Credibility: Evaluating What’s Been
    Learned, Data Mining: Practical Machine Learning Tools and
    Techniques, Third Edition, 2011.

    Original proposition of correction: https://doi.org/10.1023/A:1024068626366
    Useful discussion of alternatives: https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/
    """
    train_size = 1 - test_size
    assert xs.shape == ys.shape
    assert len(xs.shape) == 1
    ds = xs - ys
    d_mean = np.mean(ds)
    # Using same ddof as: https://github.com/Waikato/weka-3.8/blob/49865490cef763855ede07cd11331a7aeaecd110/weka/src/main/java/weka/experiment/Stats.java#L316
    d_var = np.var(ds, ddof=1)
    # If two series are identical, we cannot perform a t-test, so
    # return an infinite p-value.
    if d_var == 0.0:
        return np.inf
    k = ds.shape[0]
    t = d_mean / np.sqrt(((1 / k) + (test_size / train_size)) * d_var)
    # 2-sided t-test (so multiply by 2) with k-1 degrees of freedom
    if alternative == 'two-sided':
        p = (1.0 - scipy.stats.t.cdf(abs(t), k-1)) * 2.0
    elif alternative == 'greater':
        p = (1.0 - scipy.stats.t.cdf(t, k-1))
    elif alternative == 'less':
        p = (1.0 - scipy.stats.t.cdf(-t, k-1))
    else:
        raise ValueError('Unsupported alternative value: {}'.format(alternative))
    return p


def build_shift_classification_df(df: pd.DataFrame, *,
                                  any_shift_tests: List[str],
                                  non_prior_shift_tests: List[str]) -> Tuple[List[str], pd.DataFrame]:
    """Return a list of all shift_classifiers that can be produced by each
    combination of the given any_shift and non_prior_shift tests, along
    with a DataFrame of quantification stats for each shift_classifier."""
    NO_SHIFT = 'no_shift'
    PRIOR_SHIFT = 'prior_shift'
    GENERAL_SHIFT = 'general_shift'

    def get_shift_type_series(df: pd.DataFrame, *, any_shift_test: str, non_prior_shift_test: str) -> pd.Series:
        """Construct a series of shift types produced by the combination of
        the given shift tests."""
        any_shift_cond = df[f'{any_shift_test}_shift_detected']
        non_prior_shift_cond = df[f'{non_prior_shift_test}_shift_detected']

        first_test, second_test = any_shift_test, non_prior_shift_test
        shift_type_series = pd.Series(NO_SHIFT, index=df.index)
        shift_type_series[any_shift_cond & ~non_prior_shift_cond] = PRIOR_SHIFT
        shift_type_series[any_shift_cond & non_prior_shift_cond] = GENERAL_SHIFT

        shift_type_series.name = f'{first_test}+{second_test}'
        return shift_type_series

    # Get a series of shift types for shift classifiers representing
    # each combination of any_shift_test and non_prior_shift_test.
    shift_type_series_list = []
    for any_shift_test, non_prior_shift_test in itertools.product(any_shift_tests, non_prior_shift_tests):
        shift_type_series_list.append(get_shift_type_series(
            df,
            any_shift_test=any_shift_test,
            non_prior_shift_test=non_prior_shift_test,
        ))
    shift_classifiers = [str(shift_type_series.name) for shift_type_series in shift_type_series_list]

    def get_shift_classifier_df(df: pd.DataFrame, shift_type_series: pd.Series) -> pd.DataFrame:
        """Produce a DataFrame of quantification performance stats by
        switching quantifier based on the detected shift type."""
        return pd.DataFrame({
            'shift_type': shift_type_series,
            'coverage': (df['pcc_coverage']
                         .mask((shift_type_series == GENERAL_SHIFT), df['gsls_coverage'])
                         .mask((shift_type_series == PRIOR_SHIFT), df['em_coverage'])),
            'absolute_error': (df['pcc_absolute_error']
                               # We do not need a rule for general shift, as we use pcc point estimates.
                               .mask((shift_type_series == PRIOR_SHIFT), df['em_absolute_error'])),
        })

    # Produce a combined DataFrame with quantification stats for each
    # shift_classifier.
    shift_classification_df = df
    for shift_type_series in shift_type_series_list:
        shift_classifier_df = (get_shift_classifier_df(df, shift_type_series)
                               .add_prefix(f'{shift_type_series.name}_'))
        shift_classification_df = shift_classification_df.join(shift_classifier_df)

    # Compute additional DataFrame fields
    shift_classification_df['ideal_coverage'] = (
        shift_classification_df[['pcc_coverage', 'em_coverage', 'gsls_coverage']]
        .max(axis=1)
    )
    shift_classification_df['ideal_absolute_error'] = (
        shift_classification_df[['pcc_absolute_error', 'em_absolute_error']]
        .min(axis=1)
    )

    return shift_classifiers, shift_classification_df


def display_friedman_test(metric_df: pd.DataFrame, alpha='0.05',
                          nemenyi_file_prefix: Optional[str] = None) -> None:
    """Display Friedman test p-values and critical distance diagrams for
    each interaction count. Each row is expected to be a different
    sample, and each column is expected to be a different method being
    compared."""
    _, friedman_p = scipy.stats.friedmanchisquare(*[
        row.to_list() for _, row in metric_df.iterrows()
    ])
    print(f'Friedman test p-value: {friedman_p}')
    ranks = metric_df.T.rank(ascending=True)
    avg_ranks_series = ranks.mean(axis=1)
    avg_ranks = avg_ranks_series.tolist()
    names = avg_ranks_series.index.tolist()
    dataset_count = ranks.shape[1]
    cd = Orange.evaluation.compute_CD(avg_ranks, dataset_count, alpha=alpha)
    print('Critical value:', cd)
    Orange.evaluation.graph_ranks(avg_ranks, names, cd=cd,
                                  width=6, textspace=1.5, reverse=False)
    if nemenyi_file_prefix is not None:
        plt.savefig(f'{nemenyi_file_prefix}nemenyi.svg',
                    bbox_inches='tight', pad_inches=0)
    plt.show()


def shift_classifier_plot(shift_plot_df: pd.DataFrame, *,
                          shift_classifiers: List[str],
                          shift_classifier_labels: Dict[str, str],
                          nemenyi_file_prefix: Optional[str] = None) -> go.Figure:
    """Given a shift_classification_df, return a scatter plot of mean
    square absolute_error and mean square (1 - coverage) for
    quantifications selected by each shift_classifier."""
    metrics = ['absolute_error', 'missed_coverage']

    for shift_classifier in shift_classifiers:
        shift_plot_df[f'{shift_classifier}_missed_coverage'] = 1 - shift_plot_df[f'{shift_classifier}_coverage']  # type: ignore

    summary_cols = {}
    by_dataset_and_shift_groups = (
        shift_plot_df
        # Compute squared metrics to prefer strategies that achieve
        # "good" performance most of the time over those that achieve
        # a mix of very good and poor performance.
        .assign(**{
            f'{shift_classifier}_squared_{metric}': shift_plot_df[f'{shift_classifier}_{metric}'] ** 2  # type: ignore
            for metric in metrics
            for shift_classifier in shift_classifiers
        })
        .groupby(['dataset_name', 'shift_condition'])
    )
    for metric in metrics:
        print(metric)
        # Compute aggregate metrics across all experiments in each
        # dataset+shift group first (to give equal weight to each in
        # the final aggregation).
        by_dataset_and_shift_df = pd.DataFrame({
            shift_classifier_labels[shift_classifier]: by_dataset_and_shift_groups[f'{shift_classifier}_squared_{metric}'].mean()
            for shift_classifier in shift_classifiers
        })
        display_friedman_test(by_dataset_and_shift_df, nemenyi_file_prefix=nemenyi_file_prefix)
        # Compute final metric mean for each shift_classifier
        summary_cols[f'squared_{metric}'] = by_dataset_and_shift_df.mean()
    summary_df = pd.DataFrame(summary_cols)

    any_shift_test_series = summary_df.index.str.extract(r'(.*)\+.*', expand=False).fillna('N/A')
    any_shift_test_label_symbols = {
        'KS': 'triangle-up',
        'LR': 'triangle-down',
        'N/A': 'circle',
    }
    rest_index_series = summary_df.index.str.replace(r'(.*)\+', '', regex=True)
    rest_index_label_colors = {
        'PCC': '#999999',
        'EM': '#f782c1',
        'GSLS': '#e7ac00',
        'WPA-KS': '#112498',
        'WPA-HD': '#13d4f4',
        'CDT-KS': '#0fa779',
        'CDT-HD': '#81bb5a',
        'AKS': '#fe0605',
    }

    fig = px.scatter(
        summary_df,
        x='squared_absolute_error',
        y='squared_missed_coverage',
        symbol=any_shift_test_series,
        symbol_map=any_shift_test_label_symbols,
        color=rest_index_series,
        color_discrete_map=rest_index_label_colors,
        log_x=True,
        log_y=True,
        labels={
            'squared_absolute_error': 'Mean Squared Absolute Error',
            'squared_missed_coverage': 'Mean Squared (1 - Coverage)',
        },
        hover_name=summary_df.index,
    )

    # Hide default legend
    for trace in fig.data:
        trace['showlegend'] = False

    # Color Legend
    for rest_index_label, color in rest_index_label_colors.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(symbol='circle', color=color),
            name=rest_index_label,
        ))

    # Symbol Legend
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        opacity=0,
        name='"Any Shift" Test',
    ))
    for any_shift_test_label, symbol in any_shift_test_label_symbols.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(symbol=symbol, color='black'),
            name=any_shift_test_label,
        ))

    fig.update_layout(**BASE_LAYOUT)
    fig.update_layout(
        legend=dict(
            title_text='',
            orientation='v',
        ),
        xaxis_zeroline=False,
        yaxis_zeroline=False,
    )
    fig.update_traces(
        marker=dict(
            size=12,
        ),
    )
    return fig


def baseline_comparison_table(shift_plot_df: pd.DataFrame, *,
                              methods: Dict[str, str],
                              baseline_method: str,
                              dataset_labels: Dict[str, str],
                              metric: str,
                              row_grouping: Optional[List[str]] = None,
                              relative_values: bool = False,
                              include_std: bool = False,
                              t_test_alpha: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns a pair of DataFrames: One that compares the given metric for
    each method against a baseline method, and one that is a boolean mask
    that indicates where the difference between each method and the
    baseline is significant according to the given t_test_alpha."""
    experiment_grouping = ['dataset_name', 'shift_type', 'gain_weight', 'loss_weight', 'random_state', 'sample_idx']
    if row_grouping is None:
        row_grouping = ['shift_type', 'gain_weight', 'loss_weight']

    ALL_DATASETS = 'ALL_DATASETS'
    if ('dataset_name' in row_grouping) or ('dataset_label' in row_grouping):
        table_dataset_names = [ALL_DATASETS]
    else:
        table_dataset_names = list(dataset_labels.keys())

    def get_dataset_dfs() -> Iterator[Tuple[str, pd.DataFrame]]:
        dataset_dfs = shift_plot_df.groupby('dataset_name')
        for dataset_name in table_dataset_names:
            if dataset_name == ALL_DATASETS:
                yield '', shift_plot_df
            if dataset_name in dataset_labels.keys():
                yield dataset_name, dataset_dfs.get_group(dataset_name)

    def significance_test(group_df: pd.DataFrame, *,
                          method: str, baseline_method: str, metric: str) -> bool:
        if t_test_alpha is None:
            return False

        if method == baseline_method:
            return False

        # Group by experiment to group different target_classes together.
        baseline_series = group_df.groupby(experiment_grouping, dropna=False)[f'{baseline_method}_{metric}'].mean()
        target_series = group_df.groupby(experiment_grouping, dropna=False)[f'{method}_{metric}'].mean()

        test_sizes = group_df['test_n'] / (group_df['test_n'] + group_df['full_train_n'])
        test_size = test_sizes.iloc[0]
        assert (test_sizes == test_size).all()

        return corrected_resampled_t_test(baseline_series.to_numpy(),
                                          target_series.to_numpy(),
                                          test_size=test_size) < t_test_alpha

    def standard_deviation(group_df, *, method, metric):
        # Group by experiment to group different target_classes together.
        experiment_means = group_df.groupby(experiment_grouping, dropna=False)[f'{method}_{metric}'].mean()
        return experiment_means.std()

    significant_column_to_series = {}
    column_to_series = {}
    for dataset_name, dataset_df in get_dataset_dfs():
        dataset_label = dataset_labels.get(dataset_name, dataset_name)
        row_groups = dataset_df.groupby(row_grouping)
        baseline_mean = row_groups[f'{baseline_method}_{metric}'].mean().astype(np.float_)
        for method_name, method in methods.items():
            mean = row_groups[f'{method}_{metric}'].mean().astype(np.float_)
            if method != baseline_method and relative_values:
                mean = mean - baseline_mean

            if include_std:
                std = row_groups.apply(standard_deviation, method=method, metric=metric)
                column_to_series[(dataset_label, method_name)] = pd.Series(zip(mean, std), index=mean.index)
            else:
                column_to_series[(dataset_label, method_name)] = mean
            significant_column_to_series[(dataset_label, method_name)] = row_groups.apply(
                significance_test,
                method=method,
                baseline_method=baseline_method,
                metric=metric,
            )

    df = pd.DataFrame(column_to_series).T
    significance_mask = pd.DataFrame(significant_column_to_series).T
    return df, significance_mask


def shift_test_runtime_table(shift_test_runtime_df: pd.DataFrame, *,
                             shift_test_labels: Dict[str, str],
                             dataset_labels: Dict[str, str]) -> pd.DataFrame:
    """Returns a DataFrame that compares the runtimes of each shift_test
    in milliseconds."""
    column_to_series = {}
    for dataset_name, dataset_df in shift_test_runtime_df.groupby('dataset_name'):
        runtime_df = dataset_df[[f'{shift_test}_all_class_time_ns' for shift_test in shift_test_labels.keys()]]
        # Convert nanoseconds to milliseconds
        runtime_df = runtime_df / 1_000_000
        dataset_label = dataset_labels[dataset_name]
        column_to_series[(dataset_label, 'mean')] = runtime_df.mean()
        column_to_series[(dataset_label, 'std')] = runtime_df.std()
    table_df = pd.DataFrame(column_to_series).T
    table_df.columns = [
        shift_test_labels[column.replace('_all_class_time_ns', '')]
        for column in table_df.columns
    ]
    return table_df


def shift_test_table(shift_plot_df: pd.DataFrame, *,
                     dataset_labels: Dict[str, str],
                     shift_conditions: List[str],
                     shift_test_labels: Dict[str, str]) -> pd.DataFrame:
    """Returns a DataFrame that reports how frequently each shift_test
    detects shift under different conditions of shift."""
    rows = []
    for dataset_name, dataset_label in dataset_labels.items():
        for shift_condition in shift_conditions:
            row = {
                'dataset': dataset_label,
                'shift_condition': shift_condition,
            }
            cell_df = shift_plot_df[
                (shift_plot_df['dataset_name'] == dataset_name) &
                (shift_plot_df['shift_condition'] == shift_condition)
            ]
            for method in shift_test_labels.keys():
                row[shift_test_labels[method]] = cast(pd.Series, cell_df[f'{method}_shift_detected']).mean()
            rows.append(row)
    return pd.DataFrame(rows).set_index(['dataset', 'shift_condition'])


def plot_discrepancy(df: pd.DataFrame,
                     gsls_method: str,
                     *,
                     colormap: Optional[Colormap] = None,
                     dataset_names: Optional[List[str]] = None) -> go.Figure:
    colormap = colormap or {}

    if dataset_names is not None:
        df = df[df['dataset_name'].isin(dataset_names)]
        dataset_order = df['dataset_name'].map({dataset: i for i, dataset in enumerate(dataset_names)})
        df = df.assign(dataset_order=dataset_order).sort_values('dataset_order')
    else:
        dataset_names = list(df['dataset_name'].unique())

    # Add padding to last legend item
    last_dataset_label = df[df['dataset_name'] == dataset_names[-1]].iloc[0]['dataset_label']
    new_last_label = last_dataset_label + '   '
    dataset_label_series = df['dataset_label'].replace(last_dataset_label, new_last_label)
    colormap = {**colormap, new_last_label: colormap[last_dataset_label]}

    fig = px.scatter(
        x=(df['remain_weight'] - df[f'{gsls_method}_remain_weight']),
        y=np.sqrt(df[f'{gsls_method}_discrepancy']),
        # Add padding to last legend item
        color=dataset_label_series,
        marginal_y='box',
        hover_data={
            'index': df['index'],
        },
        labels={
            'x': r'True remaining weight − Estimated remaining weight',
            'y': r'√Discrepancy',
        },
        color_discrete_map=colormap,
    )
    fig.update_layout(**BASE_LAYOUT)
    fig.update_layout(
        legend=dict(
            x=0.73,
            y=0.04,
            title=None,
        ),
    )
    fig.update_traces(
        opacity=0.75,
    )
    return fig
