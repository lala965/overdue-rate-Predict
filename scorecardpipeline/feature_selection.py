# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/8 14:06
@Author  : itlubber
@Site    : itlubber.art
"""

import operator
import sys
import types
from copy import deepcopy
from functools import reduce
from itertools import chain, combinations
from functools import partial
from abc import ABCMeta, abstractmethod

import math
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scipy.stats import sem
from scipy.stats._continuous_distns import t
from sklearn.metrics import check_scoring, get_scorer
from sklearn.model_selection._validation import cross_val_score, _score
from sklearn.utils._encode import _unique
from sklearn.utils._mask import _get_mask
from sklearn.model_selection import check_cv
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.utils import _safe_indexing, check_X_y
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.utils.sparsefuncs import mean_variance_axis, min_max_axis
from sklearn.utils.validation import check_is_fitted, check_array, indexable, column_or_1d
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_classifier, MetaEstimatorMixin
from sklearn.feature_selection import RFECV, RFE, SelectFromModel, SelectKBest, GenericUnivariateSelect
from sklearn.feature_selection._from_model import _calculate_threshold, _get_feature_importances
# from statsmodels.stats.outliers_influence import variance_inflation_factor

from .processing import Combiner


class SelectorMixin(BaseEstimator, TransformerMixin):

    def __init__(self):
        """特征筛选器基类，继承自 sklearn 的 SelectorMixin

        提供通用的 fit/transform 模式，``fit`` 后通过 ``transform`` 筛选特征列。
        子类需实现 ``fit`` 方法，在其中设置 ``select_columns``（保留的特征列名列表）、
        ``scores_``（各特征评分，pd.Series）和 ``dropped``（剔除原因，pd.DataFrame）属性。

        **属性字段**

        :param select_columns: list，``fit`` 后保留的特征列名列表
        :param scores_: pd.Series，``fit`` 后各特征的评分（越高越可能被保留）
        :param dropped: pd.DataFrame，包含 ``variable`` 和 ``rm_reason`` 两列，记录剔除特征及其原因
        :param fitted_: bool，是否已完成拟合
        """
        self.select_columns = None
        self.scores_ = None
        self.dropped = None
        self.n_features_in_ = None
        self.fitted_ = False

    def __sklearn_is_fitted__(self):
        """sklearn 检查是否已拟合"""
        return self.fitted_

    def transform(self, x):
        """根据 ``select_columns`` 筛选保留特征列

        :param x: 原始数据集
        :return: pd.DataFrame，仅包含保留特征列的数据集
        """
        check_is_fitted(self, "select_columns")
        return x[[col for col in self.select_columns if col in x.columns]]

    def plot(self, save=None, figsize=(12, 6), top_k=20, fontsize=12):
        """将特征筛选的分数和剔除原因以表格图片形式保存

        :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
        :param figsize: 图片大小，默认 (12, 6)
        :param top_k: 仅展示分数最高的 top_k 个特征，默认 20
        :param fontsize: 字体大小，默认 12

        :return: matplotlib Figure
        """
        from .utils import dataframe_plot
        check_is_fitted(self, "scores_")

        import matplotlib.pyplot as plt

        scores = self.scores_.sort_values(ascending=True).tail(top_k)

        fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]})
        fig.suptitle(f"{self.__class__.__name__} Feature Selection Scores", fontsize=fontsize + 2)

        ax1 = axes[0]
        ax1.barh(scores.index, scores.values, color='#4472C4', alpha=0.8)
        ax1.set_xlabel("Score", fontsize=fontsize)
        ax1.set_title("Feature Scores", fontsize=fontsize)
        ax1.tick_params(axis='y', labelsize=fontsize - 2)

        ax2 = axes[1]
        ax2.axis('off')
        if self.dropped is not None and len(self.dropped) > 0:
            dropped_subset = self.dropped.head(top_k).rename(columns={"variable": "Variable", "rm_reason": "Removed Reason"})
            tbl = ax2.table(
                cellText=dropped_subset.values,
                colLabels=dropped_subset.columns,
                loc='center',
                cellLoc='center',
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(fontsize - 2)
            tbl.scale(1.2, 1.5)
            for (r, c), cell in tbl.get_celld().items():
                if r == 0:
                    cell.set_facecolor('#F76E6C')
                    cell.set_text_props(color='white', fontsize=fontsize - 2)
                else:
                    cell.set_facecolor('#FFF2CC')
            ax2.set_title("Removed Features", fontsize=fontsize, pad=10)

        plt.tight_layout()

        if save:
            import os
            if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
                os.makedirs(os.path.dirname(save), exist_ok=True)
            fig.savefig(save, dpi=150, format="png", bbox_inches="tight")

        return fig

    def __call__(self, *args, **kwargs):
        """支持以函数方式调用：直接 fit 并返回保留的特征列名

        >>> selector = LiftSelector(threshold=3.0)
        >>> selected_cols = selector(X, y)  # fit 并返回 select_columns
        """
        self.fit(*args, **kwargs)
        return self.select_columns

    def fit(self, x, y=None):
        self.fitted_ = True


class TypeSelector(SelectorMixin):
    """基于数据类型筛选特征的 Selector

    根据 dtype_include / dtype_exclude 条件筛选 DataFrame 中的列。

    **参考样例**

    >>> import pandas as pd
    >>> from scorecardpipeline.feature_selection import TypeSelector
    >>> df = pd.DataFrame({"a": [1,2], "b": ["x","y"], "c": [1.0, 2.0]})
    >>> selector = TypeSelector(dtype_include="number")
    >>> selector.fit_transform(df)
    """

    def __init__(self, dtype_include=None, dtype_exclude=None, exclude=None):
        """按数据类型筛选特征

        :param dtype_include: 包含的数据类型（如 "number", "object", "datetime" 等），默认 None
        :param dtype_exclude: 排除的数据类型，默认 None
        :param exclude: 强制保留的列名（list 或 str），默认 None
        """
        super().__init__()
        self.dtype_include = dtype_include
        self.dtype_exclude = dtype_exclude
        self.exclude = exclude

    def fit(self, x: pd.DataFrame, y=None, **fit_params):
        if not hasattr(x, 'iloc'):
            raise ValueError("make_column_selector can only be applied to pandas dataframes")

        self.n_features_in_ = x.shape[1]

        if self.exclude:
            if not isinstance(self.exclude, (list, tuple, np.ndarray)):
                self.exclude = [self.exclude]

            x = x.drop(columns=[c for c in self.exclude if c in x.columns])

        if self.dtype_include is not None or self.dtype_exclude is not None:
            cols = x.select_dtypes(include=self.dtype_include, exclude=self.dtype_exclude).columns
        else:
            cols = x.columns

        self.scores_ = x.dtypes
        self.select_columns = list(set(cols.tolist()))
        if self.exclude:
            self.select_columns = list(set(self.select_columns + self.exclude))

        self.dropped = pd.DataFrame([(col, f"data type or name not match") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        self.fitted_ = True
        return self


class RegexSelector(SelectorMixin):
    """基于正则表达式筛选特征列名的 Selector

    根据列名是否匹配正则表达式来筛选特征。

    **参考样例**

    >>> import pandas as pd
    >>> from scorecardpipeline.feature_selection import RegexSelector
    >>> df = pd.DataFrame({"feature_a": [1,2], "feature_b": [3,4], "target": [0,1]})
    >>> selector = RegexSelector(pattern=r"^feature_")
    >>> selector.fit_transform(df)
    """

    def __init__(self, pattern=None, exclude=None):
        """按列名正则匹配筛选特征

        :param pattern: 正则表达式字符串，列名匹配该表达式则被保留
        :param exclude: 强制保留的列名（list 或 str），默认 None
        """
        super().__init__()
        self.pattern = pattern
        self.exclude = exclude

        if self.pattern is None:
            raise ValueError("pattern must be a regular expression.")

    def fit(self, x: pd.DataFrame, y=None, **fit_params):
        if not hasattr(x, 'iloc'):
            raise ValueError("make_column_selector can only be applied to pandas dataframes")

        self.n_features_in_ = x.shape[1]

        if self.exclude:
            if not isinstance(self.exclude, (list, tuple, np.ndarray)):
                self.exclude = [self.exclude]

            x = x.drop(columns=[c for c in self.exclude if c in x.columns])

        self.scores_ = x.columns.str.contains(self.pattern, regex=True).astype(int)
        self.select_columns = list(set(x.columns[self.scores_ == 1].tolist()))
        if self.exclude:
            self.select_columns = list(set(self.select_columns + self.exclude))

        self.dropped = pd.DataFrame([(col, f"feature name not match {self.pattern}") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        self.fitted_ = True
        return self


def value_ratio(x, value):
    """计算数组或 DataFrame 中指定值的占比

    :param x: pd.DataFrame 或一维数组
    :param value: 需要统计占比的值（默认为 np.nan）
    :return: float（DataFrame 时返回每列的占比 Series）
    """
    if isinstance(x, pd.DataFrame):
        return np.mean(_get_mask(x.values, value), axis=0)

    return np.mean(_get_mask(x, value), axis=0)


def mode_ratio(x, dropna=True):
    """计算数组的众数及其占比

    :param x: 一维数组或 list
    :param dropna: 是否排除 NaN 值，默认 True
    :return: tuple，(众数值, 众数占比)
    """
    if isinstance(x, (list, np.ndarray)):
        x = pd.Series(x)

    summary = x.value_counts(dropna=dropna)
    return (summary.index[0], summary.iloc[0] / sum(summary)) if len(summary) > 0 else (np.nan, 1.0)


class NullSelector(SelectorMixin):
    """基于缺失率筛选特征的 Selector

    剔除缺失值（由 missing_values 指定）占比超过 threshold 的特征。

    **参考样例**

    >>> import pandas as pd
    >>> import numpy as np
    >>> from scorecardpipeline.feature_selection import NullSelector
    >>> df = pd.DataFrame({"a": [1, np.nan, 3], "b": [1, 2, 3], "c": [np.nan, np.nan, np.nan]})
    >>> selector = NullSelector(threshold=0.5)
    >>> selector.fit_transform(df)
    """

    def __init__(self, threshold=0.95, missing_values=np.nan, exclude=None, **kwargs):
        """按缺失率筛选特征

        :param threshold: 缺失率阈值，缺失率 >= threshold 的特征将被剔除，默认 0.95
        :param missing_values: 视为缺失的值，默认 np.nan
        :param exclude: 强制保留的列名（list 或 str），默认 None
        :param kwargs: 其他参数（保留给 sklearn 兼容性）
        """
        super().__init__()
        self.exclude = exclude
        self.threshold = threshold
        self.missing_values = missing_values
        self.dropped = None
        self.select_columns = None
        self.scores_ = None
        self.n_features_in_ = None
        self.kwargs = kwargs

    def fit(self, x: pd.DataFrame, y=None):
        self.n_features_in_ = x.shape[1]

        if self.exclude:
            if not isinstance(self.exclude, (list, tuple, np.ndarray)):
                self.exclude = [self.exclude]

            x = x.drop(columns=[c for c in self.exclude if c in x.columns])

        self.scores_ = pd.Series(value_ratio(x, self.missing_values), index=x.columns)
        self.threshold = _calculate_threshold(self, self.scores_, self.threshold)
        self.select_columns = list(set((self.scores_[self.scores_ < self.threshold]).index.tolist()))
        if self.exclude:
            self.select_columns = list(set(self.select_columns + self.exclude))

        self.dropped = pd.DataFrame([(col, f"nan ratio >= {self.threshold}") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        self.fitted_ = True
        return self


class ModeSelector(SelectorMixin):
    """基于众数占比筛选特征的 Selector

    剔除单一值出现占比超过 threshold 的特征（常用于剔除方差极低的常量列）。

    **参考样例**

    >>> import pandas as pd
    >>> from scorecardpipeline.feature_selection import ModeSelector
    >>> df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3], "c": [0, 0, 1]})
    >>> selector = ModeSelector(threshold=0.8)
    >>> selector.fit_transform(df)
    """

    def __init__(self, threshold=0.95, exclude=None, dropna=True, n_jobs=None, **kwargs):
        """按众数占比筛选特征

        :param threshold: 众数占比阈值，众数占比 >= threshold 的特征将被剔除，默认 0.95
        :param exclude: 强制保留的列名（list 或 str），默认 None
        :param dropna: 计算众数时是否排除 NaN，默认 True
        :param n_jobs: 并行计算的 worker 数，默认 None（单进程）
        :param kwargs: 其他参数
        """
        super().__init__()
        self.dropna = dropna
        self.exclude = exclude
        self.threshold = threshold
        self.dropped = None
        self.select_columns = None
        self.scores_ = None
        self.n_features_in_ = None
        self.kwargs = kwargs
        self.n_jobs = n_jobs

    def fit(self, x: pd.DataFrame, y=None):
        self.n_features_in_ = x.shape[1]

        if self.exclude:
            if not isinstance(self.exclude, (list, tuple, np.ndarray)):
                self.exclude = [self.exclude]

            x = x.drop(columns=[c for c in self.exclude if c in x.columns])

        self.scores_ = pd.DataFrame(Parallel(n_jobs=self.n_jobs)(delayed(mode_ratio)(x[c], self.dropna) for c in x.columns), columns=["Mode", "Ratio"], index=x.columns)
        self.threshold = _calculate_threshold(self, self.scores_, self.threshold)
        self.select_columns = list(set((self.scores_[self.scores_["Ratio"] < self.threshold]).index.tolist()))
        if self.exclude:
            self.select_columns = list(set(self.select_columns + self.exclude))

        self.dropped = pd.DataFrame([(col, f"mode ratio >= {self.threshold}") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        self.fitted_ = True
        return self


class CardinalitySelector(SelectorMixin):
    """基于类别唯一值数量筛选特征的 Selector

    剔除唯一值数量（cardinality）超过 threshold 的特征。

    **参考样例**

    >>> import pandas as pd
    >>> from scorecardpipeline.feature_selection import CardinalitySelector
    >>> df = pd.DataFrame({"f1": ["A", "B", "A"], "f2": ["X", "Y", "Z"], "f3": [1, 2, 3]})
    >>> selector = CardinalitySelector(threshold=2)
    >>> selector.fit_transform(df)
    """

    def __init__(self, threshold=10, exclude=None, dropna=True):
        """按唯一值数量筛选特征

        :param threshold: 唯一值数量阈值，唯一值数量 > threshold 的特征将被剔除，默认 10
        :param exclude: 强制保留的列名（list 或 str），默认 None
        :param dropna: 计算唯一值时是否排除 NaN，默认 True
        """
        super().__init__()
        self.exclude = exclude
        self.threshold = threshold
        self.dropna = dropna

    def fit(self, x, y=None, **fit_params):
        self.n_features_in_ = x.shape[1]

        if self.exclude:
            if not isinstance(self.exclude, (list, tuple, np.ndarray)):
                self.exclude = [self.exclude]

        self.scores_ = pd.Series(x.nunique(axis=0, dropna=self.dropna).values, index=x.columns)
        self.threshold = _calculate_threshold(self, self.scores_, self.threshold)
        self.select_columns = list(set((self.scores_[self.scores_ < self.threshold]).index.tolist()))

        if self.exclude:
            self.select_columns = list(set(self.select_columns + self.exclude))

        self.dropped = pd.DataFrame([(col, f"cardinality >= {self.threshold}") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        self.fitted_ = True
        return self


def IV(x, y, regularization=1.0):
    """计算特征的信息价值（Information Value, IV）

    IV 是评分卡建模中衡量特征区分能力的常用指标，计算方式为：

    **公式**::

        IV = sum((Distr_Good - Distr_Bad) * ln(Distr_Good / Distr_Bad + 1e-10))

    其中 Distr_Good 和 Distr_Bad 分别是好/坏样本在各分箱中的分布占比。
    IV 越大，特征对目标变量的区分能力越强。经验阈值：IV < 0.02 无用，0.02-0.1 弱，0.1-0.3 中，0.3-0.5 强，> 0.5 极强（可能过拟合）。

    :param x: 特征数组（一维）
    :param y: 目标变量（0/1 二分类标签）
    :param regularization: 平滑正则化参数，防止除零，默认 1.0
    :return: float，IV 值
    """
    uniques = np.unique(x)
    n_cats = len(uniques)

    if n_cats <= 1:
        return 0.0

    event_mask = y == 1
    nonevent_mask = y != 1
    event_tot = np.count_nonzero(event_mask) + 2 * regularization
    nonevent_tot = np.count_nonzero(nonevent_mask) + 2 * regularization

    event_rates = np.zeros(n_cats, dtype=np.float64)
    nonevent_rates = np.zeros(n_cats, dtype=np.float64)
    for i, cat in enumerate(uniques):
        mask = x == cat
        event_rates[i] = np.count_nonzero(mask & event_mask) + regularization
        nonevent_rates[i] = np.count_nonzero(mask & nonevent_mask) + regularization

    # Ignore unique values. This helps to prevent overfitting on id-like columns.
    bad_pos = (event_rates + nonevent_rates) == (2 * regularization + 1)
    event_rates /= event_tot
    nonevent_rates /= nonevent_tot
    ivs = (event_rates - nonevent_rates) * np.log(event_rates / nonevent_rates)
    ivs[bad_pos] = 0.
    return np.sum(ivs).item()


def _IV(x, y, regularization=1.0, n_jobs=None):
    """批量计算 DataFrame 中每个特征的 IV 值（内部函数）

    :param x: pd.DataFrame 或二维数组
    :param y: 目标变量数组
    :param regularization: 平滑正则化参数，默认 1.0
    :param n_jobs: 并行计算的 worker 数，默认 None
    :return: np.ndarray，各特征 IV 值
    """
    x = check_array(x, dtype=None, force_all_finite=True, ensure_2d=True)
    le = LabelEncoder()
    y = le.fit_transform(y)
    if len(le.classes_) != 2:
        raise ValueError("Only support binary label for computing information value!")
    _, n_features = x.shape
    iv_values = Parallel(n_jobs=n_jobs)(delayed(IV)(x[:, i], y, regularization=regularization) for i in range(n_features))
    return np.asarray(iv_values, dtype=np.float64)


class InformationValueSelector(SelectorMixin):
    """基于信息价值（IV）筛选特征的 Selector

    计算每个特征的 IV 值，保留 IV >= threshold 的特征。特征会先经过
    Combiner 分箱（可选）或直接使用原始值计算 IV。

    **参考样例**

    >>> import pandas as pd
    >>> from scorecardpipeline.feature_selection import InformationValueSelector
    >>> df = pd.DataFrame({"a": [1,2,3,4], "b": [1,1,0,0], "target": [0,1,1,0]})
    >>> selector = InformationValueSelector(threshold=0.02)
    >>> selector.fit_transform(df)
    """

    def __init__(self, threshold=0.02, target="target", regularization=1.0, methods=None, n_jobs=None, combiner=None, **kwargs):
        """按 IV 值筛选特征

        :param threshold: IV 阈值，IV < threshold 的特征将被剔除，默认 0.02
        :param target: 数据集中目标变量的列名，默认 "target"
        :param regularization: IV 计算时的平滑正则化参数，防止除零，默认 1.0
        :param methods: 分箱方法（传入则先分箱再算 IV），可选 "chi", "dt", "quantile", "step", "kmeans", "cart", "mdlp", "uniform"
        :param n_jobs: 并行计算的 worker 数，默认 None
        :param combiner: 提前训练好的 Combiner，默认 None
        :param kwargs: Combiner 的其他参数
        """
        super().__init__()
        self.dropped = None
        self.select_columns = None
        self.scores_ = None
        self.n_features_in_ = None
        self.combiner = combiner
        self.threshold = threshold
        self.target = target
        self.regularization = regularization
        self.n_jobs = n_jobs
        self.methods = methods
        self.kwargs = kwargs

    def fit(self, x: pd.DataFrame, y=None):
        if y is None:
            if self.target not in x.columns:
                raise ValueError(f"需要传入 y 或者 x 中包含 {self.target}.")
            y = x[self.target]
            x = x.drop(columns=self.target)

        self.n_features_in_ = x.shape[1]

        if self.combiner:
            xt = self.combiner.transform(x)
        elif self.methods:
            temp = x.copy()
            temp[self.target] = y
            self.combiner = Combiner(target=self.target, method=self.methods, n_jobs=self.n_jobs, **self.kwargs)
            self.combiner.fit(temp)
            xt = self.combiner.transform(x)
        else:
            xt = x.copy()

        self.scores_ = pd.Series(_IV(xt, y, regularization=self.regularization, n_jobs=self.n_jobs), index=xt.columns)
        self.threshold = _calculate_threshold(self, self.scores_, self.threshold)
        self.select_columns = list(set((self.scores_[self.scores_ >= self.threshold]).index.tolist() + [self.target]))
        self.dropped = pd.DataFrame([(col, f"IV <= {self.threshold}") for col in xt.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        self.fitted_ = True
        return self


def LIFT(y_pred, y_true):
    """计算 LIFT 值

    LIFT 衡量预测结果中某一类（如坏样本）的占比相对于整体基线的提升程度。
    计算方式：取 y_pred 中某个唯一值对应的坏样本率，除以整体的坏样本率。

    **公式**::

        LIFT = (命中型坏样本率) / (整体坏样本率)
             = [count(y_true==1 & y_pred==v) / count(y_pred==v)] / mean(y_true)

    LIFT > 1 表示该预测分组相比随机有一定区分能力，越大区分能力越强。

    :param y_pred: 预测结果数组（如分箱标签或规则命中标记）
    :param y_true: 真实标签数组（0/1 二分类）
    :return: float，最大 LIFT 值
    """
    if len(np.unique(y_pred)) <= 1:
        return 1.0

    _y_true = column_or_1d(y_true)
    base_bad_rate = np.average(y_true)

    score = []
    for v in np.unique(y_pred):
        if pd.isnull(v):
            _y_pred = column_or_1d(y_pred.isnull())
        else:
            _y_pred = column_or_1d(y_pred == v)
        hit_bad_rate = np.count_nonzero((_y_true == 1) & (_y_pred == 1)) / np.count_nonzero(_y_pred)
        score.append(hit_bad_rate / base_bad_rate)

    return np.nanmax(score)


class LiftSelector(SelectorMixin):
    """基于 LIFT 分数筛选特征的 Selector

    对特征进行分箱后计算每个特征的 LIFT 值，保留 LIFT >= threshold 的特征。
    特征会先经过 Combiner 分箱（可选）或直接使用原始值计算 LIFT。

    **参考样例**

    >>> import pandas as pd
    >>> from scorecardpipeline.feature_selection import LiftSelector
    >>> df = pd.DataFrame({"a": [1,2,3,4,5], "b": [1,1,0,0,1], "target": [0,1,1,0,1]})
    >>> selector = LiftSelector(threshold=1.5)
    >>> selector.fit_transform(df)
    """

    def __init__(self, target="target", threshold=3.0, n_jobs=None, methods=None, combiner=None, **kwargs):
        """按 LIFT 值筛选特征

        :param target: 数据集中目标变量的列名，默认 "target"
        :param threshold: LIFT 阈值，LIFT < threshold 的特征将被剔除，默认 3.0
        :param n_jobs: 并行计算的 worker 数，默认 None
        :param methods: 分箱方法，可选 "chi", "dt", "quantile", "step", "kmeans", "cart", "mdlp", "uniform"
        :param combiner: 提前训练好的 Combiner，默认 None
        :param kwargs: Combiner 的其他参数
        """
        super().__init__()
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.target = target
        self.methods = methods
        self.combiner = combiner
        self.kwargs = kwargs

    def fit(self, x: pd.DataFrame, y=None, **fit_params):
        if y is None:
            if self.target not in x.columns:
                raise ValueError(f"需要传入 y 或者 x 中包含 {self.target}.")
            y = x[self.target]
            x = x.drop(columns=self.target)

        self.n_features_in_ = x.shape[1]

        if self.combiner:
            xt = self.combiner.transform(x)
        elif self.methods:
            temp = x.copy()
            temp[self.target] = y
            self.combiner = Combiner(target=self.target, method=self.methods, n_jobs=self.n_jobs, **self.kwargs)
            self.combiner.fit(temp)
            xt = self.combiner.transform(x)
        else:
            xt = x.copy()

        self.scores_ = pd.Series(Parallel(n_jobs=self.n_jobs)(delayed(LIFT)(xt[c], y) for c in xt.columns), index=xt.columns)
        self.threshold = _calculate_threshold(self, self.scores_, self.threshold)
        self.select_columns = list(set((self.scores_[self.scores_ >= self.threshold]).index.tolist() + [self.target]))
        self.dropped = pd.DataFrame([(col, f"LIFT < {self.threshold}") for col in xt.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        self.fitted_ = True
        return self


class VarianceSelector(SelectorMixin):
    """基于方差筛选特征的 Selector

    剔除方差低于 threshold 的特征。threshold=0 时会额外使用峰值（max-min）比较，
    避免常量特征因数值精度问题产生的误差。

    **参考样例**

    >>> import pandas as pd
    >>> from scorecardpipeline.feature_selection import VarianceSelector
    >>> df = pd.DataFrame({"a": [1,1,1], "b": [1,2,3], "c": [0,1,2]})
    >>> selector = VarianceSelector(threshold=0.1)
    >>> selector.fit_transform(df)
    """

    def __init__(self, threshold=0.0, exclude=None):
        """按方差筛选特征

        :param threshold: 方差阈值，方差 <= threshold 的特征将被剔除，默认 0.0
        :param exclude: 强制保留的列名（list 或 str），默认 None
        """
        super().__init__()
        self.threshold = threshold
        if exclude is not None:
            self.exclude = exclude if isinstance(exclude, (list, np.ndarray)) else [exclude]
        else:
            self.exclude = []

    def fit(self, x, y=None):
        self.n_features_in_ = x.shape[1]

        if hasattr(x, "toarray"):  # sparse matrix
            _, scores = mean_variance_axis(x, axis=0)
            if self.threshold == 0:
                mins, maxes = min_max_axis(x, axis=0)
                peak_to_peaks = maxes - mins
        else:
            scores = np.nanvar(x, axis=0)
            if self.threshold == 0:
                peak_to_peaks = np.ptp(x, axis=0)

        if self.threshold == 0:
            # Use peak-to-peak to avoid numeric precision issues for constant features
            compare_arr = np.array([scores, peak_to_peaks])
            scores = np.nanmin(compare_arr, axis=0)

        if np.all(~np.isfinite(scores) | (scores <= self.threshold)):
            msg = "No feature in x meets the variance threshold {0:.5f}"
            if x.shape[0] == 1:
                msg += " (x contains only one sample)"
            raise ValueError(msg.format(self.threshold))

        self.scores_ = pd.Series(scores, index=x.columns)
        self.threshold = _calculate_threshold(self, self.scores_, self.threshold)
        self.select_columns = list(set((self.scores_[self.scores_ > self.threshold]).index.tolist() + self.exclude))
        self.dropped = pd.DataFrame([(col, f"Variance <= {self.threshold}") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        self.fitted_ = True
        return self


def VIF(x, n_jobs=None, missing=-1):
    """计算方差膨胀因子（Variance Inflation Factor, VIF）

    VIF 衡量线性回归中每个特征的多重共线性程度。计算方式为：
    用其他特征作为自变量回归当前特征，VIF = 1 / (1 - R²)。

    VIF 越大，多重共线性越严重。经验阈值：VIF > 4 存在共线性问题，> 10 严重共线性。

    :param x: pd.DataFrame，特征数据集
    :param n_jobs: 并行计算的 worker 数，默认 None
    :param missing: 缺失值填充值，默认 -1
    :return: pd.Series，各特征的 VIF 值
    """
    columns = x.columns
    x = x.fillna(missing).values
    lr = partial(lambda x, y: LinearRegression(fit_intercept=False).fit(x, y).predict(x))
    y_pred = Parallel(n_jobs=n_jobs)(delayed(lr)(x[:, np.arange(x.shape[1]) != i], x[:, i]) for i in range(x.shape[1]))
    vif = [np.sum(x[:, i] ** 2) / np.sum((y_pred[i] - x[:, i]) ** 2) for i in range(x.shape[1])]

    return pd.Series(vif, index=columns)


class VIFSelector(SelectorMixin):
    """基于方差膨胀因子（VIF）筛选特征的 Selector

    VIF 越高，多重共线性的影响越严重。在金融风控中通常使用经验法则：
    若 VIF > 4，则认为存在多重共线性问题。计算较消耗资源，数据维度大时慎用。

    **参考样例**

    >>> import pandas as pd
    >>> from scorecardpipeline.feature_selection import VIFSelector
    >>> df = pd.DataFrame({"a": [1,2,3], "b": [2,4,6], "c": [1,3,5]})  # b 与 a 高度相关
    >>> selector = VIFSelector(threshold=4.0)
    >>> selector.fit_transform(df)
    """

    def __init__(self, threshold=4.0, exclude=None, missing=-1, n_jobs=None):
        """按 VIF 值筛选特征

        :param threshold: VIF 阈值，VIF >= threshold 的特征将被剔除，默认 4.0
        :param exclude: 强制保留的列名（list 或 str），默认 None
        :param missing: 缺失值默认填充值，默认 -1
        :param n_jobs: 并行计算的 worker 数，默认 None
        """
        super().__init__()
        self.threshold = threshold
        self.missing = missing
        self.n_jobs = n_jobs
        if exclude is not None:
            self.exclude = exclude if isinstance(exclude, (list, np.ndarray)) else [exclude]
        else:
            self.exclude = []

    def fit(self, x: pd.DataFrame, y=None):
        if self.exclude:
            x = x.drop(columns=self.exclude)

        self.n_features_in_ = x.shape[1]

        self.scores_ = VIF(x, missing=self.missing, n_jobs=self.n_jobs)

        self.threshold = _calculate_threshold(self, self.scores_, self.threshold)
        self.select_columns = list(set((self.scores_[self.scores_ < self.threshold]).index.tolist() + self.exclude))
        self.dropped = pd.DataFrame([(col, f"VIF >= {self.threshold}") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        self.fitted_ = True
        return self


class CorrSelector(SelectorMixin):
    """基于相关性筛选特征的 Selector

    通过相关性矩阵剔除与已有特征相关性过高的特征。
    当两个特征的相关性超过 threshold 时，保留权重（weights）较高的特征。

    **参考样例**

    >>> import pandas as pd
    >>> from scorecardpipeline.feature_selection import CorrSelector
    >>> df = pd.DataFrame({"a": [1,2,3], "b": [2,4,6], "c": [1,3,5]})
    >>> selector = CorrSelector(threshold=0.9)
    >>> selector.fit_transform(df)
    """

    def __init__(self, threshold=0.7, method="pearson", weights=None, exclude=None, **kwargs):
        """按特征相关性筛选特征

        :param threshold: 相关系数阈值，|corr| > threshold 时触发剔除，默认 0.7
        :param method: 相关系数计算方法，默认 "pearson"，还支持 "spearman", "kendall"
        :param weights: 特征重要性权重（pd.Series 或 list），权重高的特征优先保留，默认 None
        :param exclude: 强制保留的列名（list 或 str），默认 None
        :param kwargs: pd.DataFrame.corr() 的其他参数
        """
        super().__init__()
        self.threshold = threshold
        self.method = method
        self.weights = weights
        if exclude is not None:
            self.exclude = exclude if isinstance(exclude, (list, np.ndarray)) else [exclude]
        else:
            self.exclude = []
        self.kwargs = kwargs

    def fit(self, x: pd.DataFrame, y=None):
        if self.exclude:
            x = x.drop(columns=self.exclude)

        self.n_features_in_ = x.shape[1]

        _weight = pd.Series(np.zeros(self.n_features_in_), index=x.columns)

        if self.weights is not None:
            if isinstance(self.weights, pd.Series):
                _weight_columns = list(set(self.weights.index) & set(x.columns))
                _weight.loc[_weight_columns] = self.weights[_weight_columns]
            else:
                _weight = pd.Series(self.weights, index=x.columns)

        self.weights = _weight
        x = x[sorted(x.columns, key=lambda c: self.weights.loc[c], reverse=True)]

        corr = x.corr(method=self.method, **self.kwargs)
        self.scores_ = corr
        self.threshold = _calculate_threshold(self, self.scores_, self.threshold)

        drops = []
        ix, cn = np.where(np.triu(corr.values, 1) > self.threshold)
        weights = self.weights.values

        if len(ix):
            graph = np.hstack([ix.reshape((-1, 1)), cn.reshape((-1, 1))])
            uni, counts = np.unique(graph, return_counts=True)

            while True:
                nodes = uni[np.argwhere(counts == np.amax(counts))].flatten()
                n = nodes[np.argsort(weights[nodes])[0]]

                i, c = np.where(graph == n)
                pairs = graph[(i, 1 - c)]

                if weights[pairs].sum() > weights[n]:
                    dro = [n]
                else:
                    dro = pairs.tolist()

                drops += dro

                di, _ = np.where(np.isin(graph, dro))
                graph = np.delete(graph, di, axis=0)

                if len(graph) <= 0:
                    break

                uni, counts = np.unique(graph, return_counts=True)

        self.dropped = pd.DataFrame([(col, f"corr > {self.threshold}") for col in corr.index[drops].values], columns=["variable", "rm_reason"])
        self.select_columns = list(set([c for c in x.columns if c not in corr.index[drops].values] + self.exclude))
        self.fitted_ = True
        return self


def _psi_score(expected, actual):
    """计算单个特征的 PSI（Population Stability Index，群体稳定性指标）

    PSI 衡量实际分布与期望分布之间的差异，广泛用于评估特征在跨时间或跨数据集上的稳定性。
    PSI < 0.1 表示分布稳定，0.1-0.2 表示略有变化，> 0.2 表示显著变化。

    :param expected: 期望分布（基准数据集），通常为训练集
    :param actual: 实际分布（当前数据集），通常为测试集
    :return: float，PSI 值
    """
    n_expected = len(expected)
    n_actual = len(actual)

    psi = []
    for value in _unique(expected):
        expected_cnt = np.count_nonzero(expected == value)
        actual_cnt = np.count_nonzero(actual == value)
        expected_cnt = expected_cnt if expected_cnt else 1.
        actual_cnt = actual_cnt if actual_cnt else 1.
        expected_rate = expected_cnt / n_expected
        actual_rate = actual_cnt / n_actual
        psi.append((actual_rate - expected_rate) * np.log(actual_rate / expected_rate))

    return sum(psi)


def PSI(train, test, n_jobs=None, verbose=0, pre_dispatch='2*n_jobs'):
    """批量计算 DataFrame 中每个特征的 PSI 值（内部函数）

    使用交叉验证方式，将 train 分为多折，计算每折的 PSI 后取平均。

    :param train: 训练/基准数据（pd.DataFrame 或 np.ndarray）
    :param test: 测试/实际数据（pd.DataFrame 或 np.ndarray）
    :param n_jobs: 并行计算的 worker 数，默认 None
    :param verbose: 是否打印进度信息，默认 0
    :param pre_dispatch: 任务分发策略，默认 '2*n_jobs'
    :return: np.ndarray，各特征的 PSI 均值
    """
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    n_cols = train.shape[1] if hasattr(train, 'shape') else len(train.columns)
    scores = parallel(delayed(_psi_score)(train.iloc[:, i] if hasattr(train, 'iloc') else train[:, i], test.iloc[:, i] if hasattr(test, 'iloc') else test[:, i]) for i in range(n_cols))
    return scores


class PSISelector(SelectorMixin):
    """基于群体稳定性指标（PSI）筛选特征的 Selector

    通过交叉验证方式计算每个特征在训练集内部不同折之间的 PSI 均值，
    保留 PSI < threshold（分布稳定）的特征。

    **参考样例**

    >>> import pandas as pd
    >>> from scorecardpipeline.feature_selection import PSISelector
    >>> df_train = pd.DataFrame({"a": [1,2,3,4], "b": [1,2,3,4]})
    >>> df_test = pd.DataFrame({"a": [1,2,3,5], "b": [1,2,3,4]})
    >>> selector = PSISelector(threshold=0.1)
    >>> # 注意：PSISelector 内部会进行 train/test 分割来计算 PSI
    >>> selector.fit(df_train)
    """

    def __init__(self, threshold=0.1, cv=None, method=None, exclude=None, n_jobs=None, verbose=0, pre_dispatch='2*n_jobs', **kwargs):
        """按 PSI 值筛选特征

        :param threshold: PSI 阈值，PSI >= threshold 的特征将被剔除，默认 0.1
        :param cv: 交叉验证折数，默认 None（使用 StratifiedKFold(3)）
        :param method: 分箱方法（传入则先分箱再算 PSI），可选 "chi", "dt", "quantile", "step", "kmeans", "cart", "mdlp", "uniform"
        :param exclude: 强制保留的列名（list 或 str），默认 None
        :param n_jobs: 并行计算的 worker 数，默认 None
        :param verbose: 是否打印进度信息，默认 0
        :param pre_dispatch: 任务分发策略，默认 '2*n_jobs'
        :param kwargs: Combiner 的其他参数
        """
        super().__init__()
        self.threshold = threshold
        self.cv = cv
        self.method = method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        if exclude is not None:
            self.exclude = exclude if isinstance(exclude, (list, np.ndarray)) else [exclude]
        else:
            self.exclude = []
        self.kwargs = kwargs

    def fit(self, x: pd.DataFrame, y=None, groups=None):
        if self.method is not None:
            temp = x.copy()
            if y is not None:
                if self.kwargs and "target" in self.kwargs and self.kwargs["target"] not in temp.columns:
                    temp[self.kwargs["target"]] = y
                elif "target" not in temp.columns:
                    temp["target"] = y

            self.combiner = Combiner(method=self.method, n_jobs=self.n_jobs, **self.kwargs).fit(temp)
            x = self.combiner.transform(x)

        if self.exclude:
            x = x.drop(columns=self.exclude)

        self.n_features_in_ = x.shape[1]
        x, groups = indexable(x, groups)
        cv = check_cv(self.cv)
        n_jobs = self.n_jobs
        verbose = self.verbose
        pre_dispatch = self.pre_dispatch

        cv_scores = []
        for train, test in cv.split(x, y, groups):
            scores = PSI(_safe_indexing(x, train), _safe_indexing(x, test), n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
            cv_scores.append(scores)

        self.scores_ = pd.Series(np.mean(cv_scores, axis=0), index=x.columns)
        self.threshold = _calculate_threshold(self, self.scores_, self.threshold)
        self.select_columns = list(set((self.scores_[self.scores_ >= self.threshold]).index.tolist() + self.exclude))
        self.dropped = pd.DataFrame([(col, f"PSI >= {self.threshold}") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        self.fitted_ = True
        return self


class NullImportanceSelector(SelectorMixin):
    """基于 Null Importance 筛选特征的 Selector

    通过比较特征重要性（基于 shuffle 前后的差异）来识别真正有预测能力的特征。
    核心思想：如果特征在目标变量被打乱（shuffle）后仍然具有高重要性，则该重要性是虚假的。
    采用多次交叉验证 shuffle 的方式估计 Null Distribution。

    :param threshold: 阈值。> 1.0 时取分数最高的前 threshold 个特征；<= 1.0 时保留分数 > threshold 的特征，默认 1.0

    **参考样例**

    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> from scorecardpipeline.feature_selection import NullImportanceSelector
    >>> # estimator = GradientBoostingClassifier()
    >>> # selector = NullImportanceSelector(estimator=estimator, threshold=0.5)
    >>> # selector.fit(X, y)
    """

    def __init__(self, estimator, target="target", threshold=1.0, norm_order=1, importance_getter='auto', cv=3, n_runs=5, **kwargs):
        """按 Null Importance 分数筛选特征

        :param estimator: sklearn 兼容的估算器（如 GradientBoostingClassifier, RandomForestClassifier 等）
        :param target: 数据集中目标变量的列名，默认 "target"
        :param threshold: 阈值，默认 1.0。> 1.0 时取分数最高的前 threshold 个特征，<= 1.0 时保留分数 > threshold 的特征
        :param norm_order: 特征重要性归一化阶数，默认 1
        :param importance_getter: 重要性获取方式，默认 'auto'
        :param cv: 交叉验证折数，默认 3
        :param n_runs: shuffle 次数，默认 5
        :param kwargs: 其他参数
        """
        super().__init__()
        self.estimator = estimator
        self.threshold = threshold
        self.norm_order = norm_order
        self.importance_getter = importance_getter
        self.cv = cv
        self.n_runs = n_runs
        self.target = target

    @staticmethod
    def _feature_score_v0(actual_importances, null_importances):
        """计算方法 v0：实际重要性均值 / Null 重要性均值"""
        return actual_importances.mean(axis=1) / null_importances.mean(axis=1)

    @staticmethod
    def _feature_score_v1(actual_importances, null_importances):
        """计算方法 v1：实际重要性的 log 比值（相对于 Null 重要性的 75 分位数）"""
        actual_importance = actual_importances.mean()
        return np.log(1e-10 + actual_importance / (1. + np.percentile(null_importances, 75)))

    @staticmethod
    def _feature_score_v2(actual_importances, null_importances):
        """计算方法 v2（默认）：shuffle 后重要性低于真实重要性 25 分位数的比例"""
        return np.count_nonzero(null_importances < np.percentile(actual_importances, 25)) / null_importances.shape[0]

    def fit(self, x: pd.DataFrame, y=None):
        if self.target in x.columns:
            y = x[self.target]
            x = x.drop(columns=self.target)

        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))

        n_splits = cv.get_n_splits()
        n_runs = self.n_runs
        getter = self.importance_getter
        norm_order = self.norm_order

        # 计算 shuffle 之后的特征重要性
        estimator = deepcopy(self.estimator)
        n_samples, n_features = x.shape
        null_importances = np.zeros((n_features, n_splits * n_runs))
        idx = np.arange(n_samples)
        for run in range(n_runs):
            np.random.shuffle(idx)
            y_shuffled = y[idx]

            for fold_, (train_idx, valid_idx) in enumerate(cv.split(y_shuffled, y_shuffled)):
                estimator.fit(x.loc[train_idx], y_shuffled.loc[train_idx])
                null_importance = _get_feature_importances(estimator, getter, transform_func=None, norm_order=norm_order)
                null_importances[:, n_splits * run + fold_] = null_importance

        # 计算未 shuffle 的特征重要性
        estimator = clone(self.estimator)
        actual_importances = np.zeros((n_features, n_splits * n_runs))
        for run in range(n_runs):
            np.random.shuffle(idx)
            y_shuffled = y[idx]
            x_shuffled = x[idx]

            for fold_, (train_idx, valid_idx) in enumerate(cv.split(y_shuffled, y_shuffled)):
                estimator.fit(x_shuffled.loc[train_idx], y_shuffled.loc[train_idx])
                actual_importance = _get_feature_importances(estimator, getter, transform_func=None, norm_order=norm_order)
                actual_importances[:, n_splits * run + fold_] = actual_importance

        self.null_importances = null_importances
        self.actual_importances_ = actual_importances

        scores = np.zeros(n_features)
        for i in range(n_features):
            scores[i] = self._feature_score_v2(actual_importances[i, :], null_importances[i, :])

        self.scores_ = pd.Series(scores, index=x.columns)
        self.threshold = _calculate_threshold(self.estimator, scores, self.threshold)

        if self.threshold > 1.0:
            self.select_columns = list(set(self.scores_.sort_values(ascending=False).iloc[:math.floor(self.threshold)].index.tolist() + [self.target]))
            self.dropped = pd.DataFrame([(col, f"nullimportance not top {self.threshold}") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        else:
            self.select_columns = list(set((self.scores_[self.scores_ > self.threshold]).index.tolist() + [self.target]))
            self.dropped = pd.DataFrame([(col, f"nullimportance <= {self.threshold}") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        self.fitted_ = True
        return self


class TargetPermutationSelector(NullImportanceSelector):
    """基于目标变量排列（Target Permutation）筛选特征的 Selector

    继承自 ``NullImportanceSelector``，使用相同的算法逻辑。
    通过多次打乱目标变量，计算特征在随机标签下的重要性基准线，
    识别哪些特征的真实重要性显著高于随机基准。

    **参考样例**

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from scorecardpipeline.feature_selection import TargetPermutationSelector
    >>> # estimator = RandomForestClassifier()
    >>> # selector = TargetPermutationSelector(estimator=estimator, threshold=0.5)
    >>> # selector.fit(X, y)
    """

    def __init__(self, estimator, target="target", threshold=1.0, norm_order=1, importance_getter='auto', cv=3, n_runs=5, **kwargs):
        """按目标变量排列重要性筛选特征

        :param estimator: sklearn 兼容的估算器
        :param target: 数据集中目标变量的列名，默认 "target"
        :param threshold: 阈值，默认 1.0
        :param norm_order: 特征重要性归一化阶数，默认 1
        :param importance_getter: 重要性获取方式，默认 'auto'
        :param cv: 交叉验证折数，默认 3
        :param n_runs: 排列次数，默认 5
        :param kwargs: 其他参数
        """
        super().__init__(estimator, target=target, threshold=threshold, norm_order=norm_order, importance_getter=importance_getter, cv=cv, n_runs=n_runs, **kwargs)


class ExhaustiveSelector(SelectorMixin, MetaEstimatorMixin):
    """穷举式特征组合选择器

    在给定的 min_features 和 max_features 范围内，穷举所有可能的特征组合，
    使用交叉验证评估每种组合的效果，返回最优特征子集。
    适用于特征数量较少（< 20）的场景。

    **属性字段**

    :param subset_info_: list[dict]，每步选择的子集信息，包含 'support_mask'（特征掩码）和 'cv_scores'（交叉验证分数）
    :param support_mask_: np.ndarray，最终选择的特征掩码
    :param best_idx_: int，最优特征子集的索引
    :param best_score_: float，最优子集的交叉验证平均分数
    :param best_feature_indices_: np.ndarray，最优特征子集对应的特征索引

    **参考样例**

    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.datasets import load_iris
    >>> from scorecardpipeline.feature_selection import ExhaustiveSelector
    >>> X, y = load_iris(return_X_y=True, as_frame=True)
    >>> knn = KNeighborsClassifier(n_neighbors=3)
    >>> efs = ExhaustiveSelector(knn, min_features=1, max_features=4, cv=3)
    >>> efs.fit(X, y)
    >>> efs.best_score_
    >>> efs.best_idx_
    """

    def __init__(self, estimator, min_features=1, max_features=1, scoring="accuracy", cv=3, verbose=0, n_jobs=None, pre_dispatch='2*n_jobs'):
        """穷举特征组合选择

        :param estimator: sklearn 兼容的分类器或回归器
        :param min_features: 最小选择的特征数量，默认 1
        :param max_features: 最大选择的特征数量，默认 1
        :param scoring: 评分指标，默认 "accuracy"。分类器还支持 "f1", "precision", "recall", "roc_auc"；回归器支持 "neg_mean_squared_error", "neg_mean_absolute_error", "r2"
        :param cv: 交叉验证折数，默认 3。若为 None 则不使用交叉验证
        :param verbose: 是否打印进度信息，默认 0
        :param n_jobs: 并行计算的 CPU 核心数，默认 None（1 个核心），-1 表示使用所有核心
        :param pre_dispatch: 任务分发策略，默认 '2*n_jobs'
        """
        super().__init__()
        self.estimator = estimator
        self.min_features = min_features
        self.max_features = max_features
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
    
    def _validate_params(self, x, y):
        check_X_y(x, y, estimator=self.estimator)
        _, n_features = x.shape
        if not isinstance(self.min_features, int) or (self.max_features > n_features or self.max_features < 1):
            raise AttributeError("max_features must be smaller than %d and larger than 0" % (n_features + 1))
        if not isinstance(self.min_features, int) or (self.min_features > n_features or self.min_features < 1):
            raise AttributeError("min_features must be smaller than %d and larger than 0" % (n_features + 1))
        
        if self.max_features < self.min_features:
            raise AttributeError("min_features must be less equal than max_features")
        return x, y
    
    @staticmethod
    def _calc_score(estimator, x, y, indices, groups=None, scoring=None, cv=None, **fit_params):
        _, n_features = x.shape
        mask = np.in1d(np.arange(n_features), indices)
        x = x[:, mask]
        
        if cv is None:
            try:
                estimator.fit(x, y, **fit_params)
            except:
                scores = np.nan
            else:
                scores = _score(estimator, x, y, scoring)
            
            scores = np.asarray([scores], dtype=np.float64)
        else:
            scores = cross_val_score(estimator, x, y, groups=groups, cv=cv, scoring=scoring, n_jobs=None, pre_dispatch='2*n_jobs', error_score=np.nan, fit_params=fit_params)
        
        return mask, scores

    @staticmethod
    def ncr(n, r):
        """Return the number of combinations of length r from n items.

        :param n: int, Total number of items
        :param r: int, Number of items to select from n
        :return: Number of combinations, integer
        """
        r = min(r, n - r)
        if r == 0:
            return 1
        numerator = reduce(operator.mul, range(n, n - r, -1))
        denominator = reduce(operator.mul, range(1, r + 1))
        return numerator // denominator

    @staticmethod
    def _calc_confidence(scores, confidence=0.95):
        std_err = sem(scores)
        bound = std_err * t._ppf((1 + confidence) / 2.0, len(scores))
        return bound, std_err

    def fit(self, X, y, groups=None, **fit_params):
        """Perform feature selection and learn model from training data.

        :param X: array-like of shape (n_samples, n_features)
        :param y: array-like of shape (n_samples, ), Target values.
        :param groups: array-like of shape (n_samples,), Group labels for the samples used while splitting the dataset into train/test set. Passed to the fit method of the cross-validator.
        :param fit_params: dict, Parameters to pass to the fit method of classifier
        :return: ExhaustiveFeatureSelector
        """
        X, y = self._validate_params(X, y)
        _, n_features = X.shape
        min_features, max_features = self.min_features, self.max_features
        candidates = chain.from_iterable(combinations(range(n_features), r=i) for i in range(min_features, max_features + 1))
        # chain has no __len__ method
        n_combinations = sum(self.ncr(n=n_features, r=i) for i in range(min_features, max_features + 1))

        estimator = self.estimator
        scoring = check_scoring(estimator, self.scoring)
        cv = self.cv
        n_jobs = self.n_jobs
        pre_dispatch = self.pre_dispatch
        parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)
        work = enumerate(parallel(delayed(self._calc_score)(clone(estimator), X, y, c, groups=groups, scoring=scoring, cv=cv, **fit_params) for c in candidates))
        
        subset_info = []
        append_subset_info = subset_info.append
        try:
            for iteration, (mask, cv_scores) in work:
                avg_score = np.nanmean(cv_scores).item()
                append_subset_info({"support_mask": mask, "cv_scores": cv_scores, "avg_score": avg_score})
                if self.verbose:
                    print("Feature set: %d/%d, avg score: %.3f" % (iteration + 1, n_combinations, avg_score))
        except KeyboardInterrupt:
            print("Stopping early due to keyboard interrupt...")
        finally:
            max_score = float("-inf")
            best_idx, best_info = -1, {}
            for i, info in enumerate(subset_info):
                if info["avg_score"] > max_score:
                    max_score = info["avg_score"]
                    best_idx, best_info = i, info
            score = max_score
            mask = best_info["support_mask"]
            self.subset_info_ = subset_info
            self.support_mask_ = mask
            self.best_idx_ = best_idx
            self.best_score_ = score
            self.best_feature_indices_ = np.where(mask)[0]
            self.fitted_ = True
            return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_mask_")
        return self.support_mask_


class BorutaSelector(SelectorMixin):

    def __init__(self):
        # 对原始特征进行复制一份，并且将其按行进行随机打乱，称为Shadow Feature。将Shadow Feature与原始特征Real Feature进行横向拼接在一起，使用某种模型（随机森林、GBDT）进行计算特征重要性。将Shadow Feature中重要性最高的值为基准，删除Real Feature中重要性低于其的特征。多重复几个迭代。（一般来说随机生成的特征效果不如原始的，因此可以以Shadow Feature的特征重要性作为基准来判断Real Feature的好坏）
        super().__init__()


class MICSelector(SelectorMixin):
    pass


class FeatureImportanceSelector(SelectorMixin):
    pass


class StabilitySelector(SelectorMixin):
    pass


class REFSelector(SelectorMixin):
    pass


class SequentialFeatureSelector(SelectorMixin):
    pass


# class SelectFromModel(SelectorMixin):
#     pass
