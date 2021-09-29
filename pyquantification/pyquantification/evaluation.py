# coding: utf-8

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
from pyquantification.datasets import DATASETS
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
        dataset_rows.append({
            'name': dataset.name,
            'label': dataset_label,
            'Features': len(dataset.all_features),
            'Classes': len(dataset.classes),
            'Concepts': len(dataset.concepts),
            'Train+Calibration instances per experiment': dataset.train_n,
            'Test instances per experiment': dataset.test_n,
            'GSLS Histogram Bins': GslsQuantifier.get_auto_hist_bins(
                calib_count=dataset.calib_n,
                target_count=dataset.test_n,
            ),
        })
    display(pd.DataFrame(dataset_rows).set_index('name'))


def display_stat_table(df: pd.DataFrame,
                       dataset_names: Optional[List[str]] = None,
                       *,
                       stat: str,
                       row_grouping: Union[str, Sequence[str]],
                       methods: Dict[str, str],
                       color_func: Optional[Callable[[float], str]] = None,
                       include_std: bool = False,
                       format_string: str = '{:.2%}') -> None:
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

    df_style = pd.DataFrame(column_to_series).style

    if color_func is not None:
        def style_func(val):
            return f'color: {color_func(val)}' if isinstance(val, np.float) else ''
        df_style = df_style.applymap(style_func)

    print_markdown(f'#### Comparison of `{stat}` across quantification methods')
    display(
        df_style
        .format(lambda val: format_string.format(val) if isinstance(val, np.float_) else val)
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
    markersize = 16
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
        marker=dict(color='#FFBBBB', line=dict(color='black', width=2)),
    ))
    fig.add_trace(go.Bar(
        x=x_axis('True weights'),
        y=df['gain_weight'],
        name='Gain',
        marker=dict(color='#DFFFDF', line=dict(color='black', width=2)),
    ))

    all_methods = {
        'pcc': ('PCC estimate', 'pcc', px.colors.qualitative.Alphabet[13]),
        'em': ('EM estimate', 'em', px.colors.qualitative.Plotly[4]),
        'gsls': ('GSLS estimate', gsls_method, px.colors.qualitative.Plotly[1]),
    }
    for name, method, color in select_keys(all_methods, methods).values():
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
                width=10,
            ),
        ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=x_axis('True weights'),
        y=df['test_true_count'] / df['test_n'],
        name='True class proportion  ',
        marker=dict(color='#000000', size=markersize),
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
    fig.update_yaxes(secondary_y=False, side='right', title='Dataset shift',
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
