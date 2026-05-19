'''
 # @ Author: Wanyh
 # @ Create Time: 2026/5/13 10:51
 # @ Modified by: Wanyh
 # @ Modified time: 2026/5/13 10:51
 # @ Description:
 '''



import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from IPython.display import display
except ImportError:
    display = print

plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',
    'SimHei',
    'Arial Unicode MS',
    'DejaVu Sans',
]
plt.rcParams['axes.unicode_minus'] = False


def _safe_div(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b

def _format_pct(x):
    return '' if pd.isna(x) else f'{x:.2%}'

def _format_num4(x):
    return '' if pd.isna(x) else f'{x:.4f}'

def _is_missing_value(x):
    if pd.isna(x):
        return True
    if isinstance(x, str) and x.strip().lower() in ['', 'nan', 'none', 'null', 'na']:
        return True
    return False

def _is_missing_bin(bin_label):
    if pd.isna(bin_label):
        return False
    return str(bin_label).strip().lower() in [
        'missing', 'nan', 'null', 'none', '缺失', '缺失值', '空值', 'na'
    ]

def _is_interval_bin(bin_label):
    if pd.isna(bin_label):
        return False
    bin_label = str(bin_label).strip()
    pattern = r'^([\[\(])\s*([^,]+)\s*,\s*([^\]\)]+)\s*([\]\)])$'
    return re.match(pattern, bin_label) is not None

def _parse_interval_bin(bin_label):
    """
    支持：
    (-inf, 3]
    (3, 10]
    [1, 3)
    (10, inf)
    """
    if _is_missing_bin(bin_label):
        return {'type': 'missing', 'raw': str(bin_label)}

    bin_label = str(bin_label).strip()
    pattern = r'^([\[\(])\s*([^,]+)\s*,\s*([^\]\)]+)\s*([\]\)])$'
    match = re.match(pattern, bin_label)

    if not match:
        raise ValueError(f'非法区间分箱格式: {bin_label}')

    left_symbol, left_value, right_value, right_symbol = match.groups()

    def convert_bound(v):
        v = str(v).strip().lower()
        if v in ['-inf', '-infinity', '负无穷']:
            return -np.inf
        if v in ['inf', '+inf', 'infinity', '+infinity', '正无穷']:
            return np.inf
        return float(v)

    return {
        'type': 'interval',
        'left': convert_bound(left_value),
        'right': convert_bound(right_value),
        'left_closed': left_symbol == '[',
        'right_closed': right_symbol == ']',
        'raw': bin_label
    }

def _value_in_bin(value, bin_label):
    """
    规则：
    1. 缺失值单独进 missing
    2. 支持左闭右开 / 左开右闭
    3. 0 不单独特判，按区间自然落箱
    """
    parsed = _parse_interval_bin(bin_label)

    if _is_missing_value(value):
        return parsed['type'] == 'missing'

    if parsed['type'] == 'missing':
        return False

    value = float(value)

    if parsed['left_closed']:
        left_ok = value >= parsed['left']
    else:
        left_ok = value > parsed['left']

    if parsed['right_closed']:
        right_ok = value <= parsed['right']
    else:
        right_ok = value < parsed['right']

    return left_ok and right_ok

def _get_feature_detail(feature_name, detail_df=None):
    if detail_df is None:
        if 'feature_bin_detail' in globals():
            detail_df = feature_bin_detail
        else:
            detail_df = pd.read_csv('./data/binning_file/feature_bin_detail.csv')

    detail = (
        detail_df.loc[detail_df['feature_name'].eq(feature_name)]
        .sort_values('bin_order')
        .copy()
    )

    if detail.empty:
        available = sorted(detail_df['feature_name'].unique())[:30]
        raise ValueError(
            f'未找到特征 {feature_name} 的分箱结果。可用特征示例: {available}'
        )

    return detail

def build_detail_from_manual_bins(
    feature_name,
    train_df,
    oot_df,
    target_col,
    bins,
):
    """
    根据原始 train / oot 数据 + 手动分箱边界，动态生成分箱明细表。

    参数
    ----
    feature_name : str
        特征名
    train_df : DataFrame
    oot_df : DataFrame
    target_col : str
        坏样本标签列，1=坏，0=好
    bins : list[str]
        例如：
        ['missing', '(-inf, 3]', '(3, 4]', '(4, 6]', '(6, inf)']
    """

    if feature_name not in train_df.columns:
        raise ValueError(f'train_df 中不存在特征列: {feature_name}')
    if feature_name not in oot_df.columns:
        raise ValueError(f'oot_df 中不存在特征列: {feature_name}')
    if target_col not in train_df.columns:
        raise ValueError(f'train_df 中不存在标签列: {target_col}')
    if target_col not in oot_df.columns:
        raise ValueError(f'oot_df 中不存在标签列: {target_col}')

    for b in bins:
        if not (_is_missing_bin(b) or _is_interval_bin(b)):
            raise ValueError(f'不支持的分箱格式: {b}')

    train_total = len(train_df)
    oot_total = len(oot_df)

    train_bad_total = pd.to_numeric(train_df[target_col], errors='coerce').fillna(0).sum()
    train_good_total = train_total - train_bad_total

    oot_bad_total = pd.to_numeric(oot_df[target_col], errors='coerce').fillna(0).sum()
    oot_good_total = oot_total - oot_bad_total

    rows = []

    for i, bin_label in enumerate(bins, start=1):
        train_mask = train_df[feature_name].apply(lambda x: _value_in_bin(x, bin_label))
        oot_mask = oot_df[feature_name].apply(lambda x: _value_in_bin(x, bin_label))

        train_count = int(train_mask.sum())
        oot_count = int(oot_mask.sum())

        train_bad = pd.to_numeric(train_df.loc[train_mask, target_col], errors='coerce').fillna(0).sum()
        oot_bad = pd.to_numeric(oot_df.loc[oot_mask, target_col], errors='coerce').fillna(0).sum()

        train_good = train_count - train_bad
        oot_good = oot_count - oot_bad

        train_pct = _safe_div(train_count, train_total)
        oot_pct = _safe_div(oot_count, oot_total)

        train_bad_rate = _safe_div(train_bad, train_count)
        oot_bad_rate = _safe_div(oot_bad, oot_count)

        # WOE / IV 按 train 算
        dist_good = _safe_div(train_good, train_good_total)
        dist_bad = _safe_div(train_bad, train_bad_total)

        if pd.isna(dist_good) or pd.isna(dist_bad) or dist_good <= 0 or dist_bad <= 0:
            woe = np.nan
            iv_component = np.nan
        else:
            woe = np.log(dist_good / dist_bad)
            iv_component = (dist_good - dist_bad) * woe

        rows.append({
            'feature_name': feature_name,
            'bin_order': i,
            'bin_label': bin_label,
            'train_count': train_count,
            'train_pct': train_pct,
            'train_bad': train_bad,
            'train_bad_rate': train_bad_rate,
            'oot_count': oot_count,
            'oot_pct': oot_pct,
            'oot_bad': oot_bad,
            'oot_bad_rate': oot_bad_rate,
            'woe': woe,
            'iv_component': iv_component,
        })

    detail = pd.DataFrame(rows)
    detail['iv'] = detail['iv_component'].sum(skipna=True)

    return detail

def get_feature_bin_table(
    feature_name,
    detail_df=None,
    train_df=None,
    oot_df=None,
    target_col=None,
    manual_bins=None,
):
    """
    两种模式：
    1. manual_bins=None: 从 detail_df / feature_bin_detail 读取现成分箱
    2. manual_bins 非空: 用 train_df + oot_df + target_col + manual_bins 动态重算
    """
    if manual_bins is not None:
        if train_df is None or oot_df is None or target_col is None:
            raise ValueError('使用 manual_bins 时，必须同时传入 train_df、oot_df、target_col')
        detail = build_detail_from_manual_bins(
            feature_name=feature_name,
            train_df=train_df,
            oot_df=oot_df,
            target_col=target_col,
            bins=manual_bins,
        )
    else:
        detail = _get_feature_detail(feature_name, detail_df)

    table = detail[[
        'feature_name',
        'bin_order',
        'bin_label',
        'train_count',
        'train_pct',
        'train_bad',
        'train_bad_rate',
        'oot_count',
        'oot_pct',
        'oot_bad',
        'oot_bad_rate',
        'woe',
        'iv_component',
        'iv',
    ]].copy()

    table = table.rename(columns={
        'feature_name': '特征名',
        'bin_order': '箱序',
        'bin_label': '分箱边界',
        'train_count': 'train数量',
        'train_pct': 'train占比',
        'train_bad': 'train坏样本',
        'train_bad_rate': 'train坏账率',
        'oot_count': 'oot数量',
        'oot_pct': 'oot占比',
        'oot_bad': 'oot坏样本',
        'oot_bad_rate': 'oot坏账率',
        'woe': 'WOE',
        'iv_component': 'IV分量',
        'iv': 'IV',
    })

    return table

def plot_feature_binning(
    feature_name,
    detail_df=None,
    save_path=None,
    figsize=(17, 6),
    train_df=None,
    oot_df=None,
    target_col=None,
    manual_bins=None,
):
    if manual_bins is not None:
        if train_df is None or oot_df is None or target_col is None:
            raise ValueError('使用 manual_bins 时，必须同时传入 train_df、oot_df、target_col')
        detail = build_detail_from_manual_bins(
            feature_name=feature_name,
            train_df=train_df,
            oot_df=oot_df,
            target_col=target_col,
            bins=manual_bins,
        )
    else:
        detail = _get_feature_detail(feature_name, detail_df)

    iv = detail['iv'].iloc[0] if len(detail) > 0 else np.nan

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'{feature_name} 分箱表现对比 | IV={iv:.4f}', fontsize=15, fontweight='bold')

    for ax, prefix, title in [
        (axes[0], 'train', 'Train'),
        (axes[1], 'oot', 'OOT'),
    ]:
        counts = detail[f'{prefix}_count'].fillna(0).astype(int)
        bad_rates = detail[f'{prefix}_bad_rate'] * 100
        x = range(len(detail))

        bars = ax.bar(x, counts, color='#5B8FF9', alpha=0.78)
        ax.set_title(f'{title}: 箱内数量 + 坏账率')
        ax.set_ylabel('箱内数量')
        ax.grid(axis='y', alpha=0.25)

        ax_rate = ax.twinx()
        ax_rate.plot(x, bad_rates, color='#D4380D', marker='o', linewidth=2)
        ax_rate.set_ylabel('坏账率(%)')

        max_count = max(counts.max(), 1)
        for idx, (bar, rate) in enumerate(zip(bars, bad_rates)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_count * 0.015,
                f'{int(counts.iloc[idx])}',
                ha='center',
                va='bottom',
                fontsize=9,
            )
            if pd.notna(rate):
                ax_rate.text(
                    idx,
                    rate,
                    f'{rate:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    color='#D4380D',
                )

        tick_labels = []
        for _, row in detail.iterrows():
            rate = row[f'{prefix}_bad_rate']
            rate_text = 'NA' if pd.isna(rate) else f'{rate:.1%}'
            tick_labels.append(
                f"{row['bin_order']}. {row['bin_label']}\n"
                f"n={int(row[f'{prefix}_count'])}, bad={rate_text}"
            )

        ax.set_xticks(list(x))
        ax.set_xticklabels(tick_labels, rotation=25, ha='right', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches='tight')

    return fig

def show_feature_binning(
    feature_name,
    detail_df=None,
    save_dir=None,
    train_df=None,
    oot_df=None,
    target_col=None,
    manual_bins=None,
):
    table = get_feature_bin_table(
        feature_name=feature_name,
        detail_df=detail_df,
        train_df=train_df,
        oot_df=oot_df,
        target_col=target_col,
        manual_bins=manual_bins,
    )

    display_table = table.copy()
    for col in ['train占比', 'train坏账率', 'oot占比', 'oot坏账率']:
        display_table[col] = display_table[col].map(_format_pct)
    for col in ['WOE', 'IV分量', 'IV']:
        display_table[col] = display_table[col].map(_format_num4)

    display(display_table)

    save_path = None
    if save_dir is not None:
        safe_name = str(feature_name).replace('/', '_').replace('\\', '_')
        save_path = Path(save_dir) / f'{safe_name}_binning.png'

    fig = plot_feature_binning(
        feature_name=feature_name,
        detail_df=detail_df,
        save_path=save_path,
        train_df=train_df,
        oot_df=oot_df,
        target_col=target_col,
        manual_bins=manual_bins,
    )
    plt.show()

    return table, fig

def build_bins_from_edges(edges, include_nan=True, closed='left'):
    """
    根据数字切点生成分箱标签。

    参数
    ----
    edges : list
        例如 [505, 624]
    include_nan : bool
        是否增加缺失箱 'missing'
    closed : str
        'left'  -> 左闭右开: (-inf, 505), [505, 624), [624, inf)
        'right' -> 左开右闭: (-inf, 505], (505, 624], (624, inf]

    返回
    ----
    list[str]
    """
    if edges is None:
        edges = []

    edges = list(edges)
    edges = sorted(set(edges))

    bins = []

    if include_nan:
        bins.append('missing')

    if len(edges) == 0:
        bins.append('[-inf, inf)' if closed == 'left' else '(-inf, inf]')
        return bins

    if closed == 'left':
        # 第一箱：(-inf, 505)
        bins.append(f'(-inf, {edges[0]})')

        # 中间箱：[505, 624)
        for i in range(len(edges) - 1):
            bins.append(f'[{edges[i]}, {edges[i+1]})')

        # 最后一箱：[624, inf)
        bins.append(f'[{edges[-1]}, inf)')

    elif closed == 'right':
        # 第一箱：(-inf, 505]
        bins.append(f'(-inf, {edges[0]}]')

        # 中间箱：(505, 624]
        for i in range(len(edges) - 1):
            bins.append(f'({edges[i]}, {edges[i+1]}]')

        # 最后一箱：(624, inf]
        bins.append(f'({edges[-1]}, inf]')

    else:
        raise ValueError("closed 只能是 'left' 或 'right'")

    return bins

def show_feature_binning_by_edges(
    feature_name,
    train_df,
    oot_df,
    target_col,
    edges,
    include_nan=True,
    closed='left',
    save_dir=None,
):
    """
    直接传数字切点画分箱图。

    示例：
    show_feature_binning_by_edges(
        feature_name='scorertstdd1_V1_2',
        train_df=train_df,
        oot_df=oot_df,
        target_col='target',
        edges=[505, 624],
        include_nan=True,
        closed='left',
        save_dir='./data/binning_plots'
    )
    """
    manual_bins = build_bins_from_edges(
        edges=edges,
        include_nan=include_nan,
        closed=closed
    )

    return show_feature_binning(
        feature_name=feature_name,
        train_df=train_df,
        oot_df=oot_df,
        target_col=target_col,
        manual_bins=manual_bins,
        save_dir=save_dir,
    )