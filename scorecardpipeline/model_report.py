# -*- coding: utf-8 -*-
"""
@Time    : 2024/01/15
@Author  : itlubber
@Site    : itlubber.art

模型评估报告快速输出

参考风控建模标准报告模板，提供多 Sheet 结构的模型报告，包括：
- 目录（带超链接）
- 基本信息（项目目标、样本统计、分月分布）
- 模型性能（KS/AUC/PSI、TOP n% LIFT、评分分箱）
- 入模变量重要性 & 分布
- 入模变量有效性分析（逐特征分箱表 + PSI）
- 模型参数（评分卡详情）
- 模型部署需求
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from toad.metrics import KS, AUC, PSI


# ==================== 日期频率兼容性 ====================
def _period_freq(freq: str) -> str:
    """将旧的 pandas 频率别名转换为 to_period() 支持的格式。

    >>> _period_freq("M")
    'ME'
    >>> _period_freq("3M")
    '3ME'
    >>> _period_freq("H")
    'H'
    >>> _period_freq("ME")
    'ME'
    """
    import re
    _f = str(freq).strip()
    if re.match(r'^\d*M$', _f):
        return _f[:-1] + 'ME'
    if _f == 'M':
        return 'ME'
    return _f


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _ensure_dataframe(X, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """确保输入为 DataFrame"""
    if isinstance(X, pd.DataFrame):
        return X.copy()
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    cols = feature_names or [f"feature_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)


def _ensure_series(y, name: str = "target") -> pd.Series:
    """确保输入为 Series"""
    if isinstance(y, pd.Series):
        out = y.copy()
        if out.name is None:
            out.name = name
        return out
    return pd.Series(np.asarray(y), name=name)


def _proba_pos(model, X) -> np.ndarray:
    """获取正类概率"""
    proba = np.asarray(model.predict_proba(X), dtype=float)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    return proba.reshape(-1)


def _score_from_model(model, X) -> np.ndarray:
    """从模型获取评分向量，兼容 ScoreCard / sklearn"""
    # ScoreCard.predict → 评分
    if hasattr(model, "predict"):
        try:
            result = np.asarray(model.predict(X), dtype=float)
            if np.nanmax(np.abs(result)) > 2.0:
                return result
        except Exception:
            pass
    # 兜底：概率转评分
    proba = _proba_pos(model, X)
    return (1.0 - proba) * 1000.0


def _safe_close_figs():
    """安全关闭 matplotlib 图形以释放内存"""
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


def _ks(y_pred, y_true):
    """计算 KS 值，参数顺序为 (score, target)"""
    return KS(y_pred, y_true)


def _auc(y_pred, y_true):
    """计算 AUC 值，参数顺序为 (score, target)"""
    return AUC(y_pred, y_true)


def _psi(expected, actual):
    """计算 PSI 值"""
    return PSI(expected, actual)


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------

@dataclass
class ReportDataset:
    """数据集容器"""
    name: str
    label: str  # 中文标签: "训练集" / "测试集" / "OOT"
    X: pd.DataFrame
    y: pd.Series
    y_proba: np.ndarray
    score: np.ndarray
    amount: Optional[np.ndarray] = None  # 金额字段数组（用于金额口径指标计算）


# ---------------------------------------------------------------------------
# QuickModelReport
# ---------------------------------------------------------------------------

class QuickModelReport:
    """面向报表输出的快速模型报告封装

    参考风控建模标准报告模板，生成多 Sheet 结构的 Excel 报告

    **参考样例**

    >>> from scorecardpipeline import QuickModelReport, ScoreCard
    >>> # 方式1: datasets dict
    >>> report = QuickModelReport(model, datasets={'train': train_df, 'test': test_df})
    >>> report.to_excel("model_report.xlsx")
    >>>
    >>> # 方式2: 兼容 sklearn API
    >>> report = QuickModelReport(model, X_train=X, y_train=y, X_test=X_val, y_test=y_val)
    >>> report.to_excel("model_report.xlsx")
    """

    _PERCENT_COLS = [
        "样本占比", "好样本占比", "坏样本占比", "坏样本率",
        "LIFT值", "坏账改善", "累积LIFT值", "累积坏账改善", "分档KS值",
    ]
    _CONDITION_COLS = ["坏样本率", "LIFT值", "累积LIFT值"]

    def __init__(
        self,
        model,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        X_oot=None,
        y_oot=None,
        feature_names: Optional[List[str]] = None,
        target: Optional[Union[str, Dict]] = None,
        datasets: Optional[Union[List, Dict]] = None,
        overdue: Optional[Union[str, List[str]]] = None,
        dpds: Optional[Union[int, float, List[Union[int, float]]]] = None,
        amount_col: Optional[str] = None,
    ):
        """初始化模型报告

        支持三种调用方式：

        1. datasets API（推荐）：传入数据集字典/列表
           - dict: {'train': DataFrame, 'test': DataFrame, 'oot': DataFrame}
             DataFrame 需包含目标列，或通过 overdue/dpds 自动构建标签
           - list: [DataFrame, DataFrame, ...] 自动命名为训练集、测试集、OOT集...

        2. 兼容 API：传入 X_train/y_train/X_test/y_test/X_oot/y_oot
           - sklearn 风格：target='target'
           - overdue/dpds 风格：传入单独的 overdue/dpds 参数

        示例::

            # 方式1: datasets dict（DataFrame 直接传入，X 中含目标列）
            report = QuickModelReport(model, datasets={'train': train_df, 'test': test_df})

            # 方式2: 兼容 sklearn API
            report = QuickModelReport(model, X_train=X, y_train=y, X_test=X_val, y_test=y_val)

        :param model: 训练好的模型（ScoreCard / XGBoost / LightGBM / sklearn 等）
        :param datasets: 数据集字典/列表（推荐方式）
        :param X_train: 训练集特征（兼容旧 API）
        :param y_train: 训练集标签（兼容旧 API）
        :param X_test: 测试集特征（兼容旧 API）
        :param y_test: 测试集标签（兼容旧 API）
        :param feature_names: 特征名称列表
        :param target: 目标列配置
            - str: 列名，如 'target'
            - dict: {'overdue': col, 'dpds': threshold} 或 {'overdue': col, 'dpds': [15, 7, 0]}
        :param overdue: 逾期列名（str）或多个列名（List[str]），与 dpds 配合自动构建标签
        :param dpds: 逾期天数阈值（int/float）或多个阈值（List），与 overdue 配合使用
        :param amount_col: 金额字段名（可选），用于金额口径指标计算
        """
        self.model = model
        self._feature_names = feature_names

        # overdue/dpds 优先，构造 target dict
        if overdue is not None and dpds is not None:
            self._target_cfg: Optional[Union[str, Dict]] = {
                "overdue": overdue,
                "dpds": dpds,
            }
        else:
            self._target_cfg = target
        self._amount_col = amount_col  # 金额字段名（用于金额口径指标计算）

        # 构建数据集
        self._datasets: Dict[str, ReportDataset] = {}
        self._datasets_info: Dict[str, str] = {}

        # 确定目标列名
        self._target_name = self._resolve_target_name(target)

        if datasets is not None:
            self._init_from_datasets(datasets, self._amount_col)
        else:
            self._init_from_xy(X_train, y_train, X_test, y_test, X_oot, y_oot, self._amount_col)

        # 从第一个数据集获取特征名
        if not hasattr(self, 'feature_names') or not self.feature_names:
            if self._datasets:
                first_ds = next(iter(self._datasets.values()))
                self.feature_names = list(first_ds.X.columns)
            elif self._feature_names:
                self.feature_names = self._feature_names
            else:
                self.feature_names = []

        # 统一为模型实际入模特征
        model_required: Optional[List[str]] = None
        if hasattr(self.model, 'feature_names_') and self.model.feature_names_ is not None:
            model_required = list(self.model.feature_names_)
        elif hasattr(self.model, 'feature_names_in_') and self.model.feature_names_in_ is not None:
            model_required = list(self.model.feature_names_in_)

        if model_required:
            self.feature_names = [f for f in self.feature_names if f in model_required]

        # 缓存
        self._metrics_cache: Optional[pd.DataFrame] = None
        self._importance_cache: Optional[pd.DataFrame] = None
        self._features_describe_cache: Optional[pd.DataFrame] = None

    def _resolve_target_name(self, target) -> str:
        """解析目标配置，返回标签列名"""
        if isinstance(target, str):
            return target
        if isinstance(target, dict) and "overdue" in target:
            return target.get("label", "target")
        return "target"

    def _build_y(self, X: pd.DataFrame, target_cfg) -> pd.Series:
        """根据 target 配置从 X 构建 y 标签"""
        if target_cfg is None:
            for col in ("target", "label", "y", "flag", "overdue"):
                if col in X.columns:
                    return _ensure_series(X[col], name="target")
            raise ValueError(
                "未找到目标列（target），请通过 target 参数指定标签列名，"
                "或传入 dict={'overdue': col, 'dpds': threshold} 联合构建"
            )

        if isinstance(target_cfg, str):
            if target_cfg in X.columns:
                return _ensure_series(X[target_cfg], name=target_cfg)
            raise ValueError(f"目标列 '{target_cfg}' 不存在于数据中")

        if isinstance(target_cfg, dict) and "overdue" in target_cfg:
            overdue_cols = target_cfg["overdue"]
            dpds_vals = target_cfg.get("dpds")
            threshold = target_cfg.get("threshold")
            label_name = target_cfg.get("label", "target")

            if isinstance(overdue_cols, str):
                overdue_cols = [overdue_cols]

            if threshold is not None:
                dpds_col = dpds_vals if isinstance(dpds_vals, str) else None
                thresholds = [threshold]
            elif dpds_vals is not None:
                if isinstance(dpds_vals, (int, float)):
                    dpds_vals = [dpds_vals]
                thresholds = dpds_vals
                dpds_col = None
            else:
                thresholds = [0]
                dpds_col = None

            for col in overdue_cols:
                if col not in X.columns:
                    raise ValueError(f"逾期列 '{col}' 不存在，请检查列名")

            indicators = pd.DataFrame(index=X.index)
            for col in overdue_cols:
                for t in thresholds:
                    if dpds_col is not None and dpds_col in X.columns:
                        indicators[f"{col}>{t}"] = X[dpds_col] > t
                    else:
                        indicators[f"{col}>{t}"] = X[col] > t

            y = indicators.any(axis=1).astype(int)
            return _ensure_series(y, name=label_name)

        raise ValueError(f"target 参数格式错误：{target_cfg}")

    def _init_from_datasets(self, datasets, amount_col: Optional[str] = None):
        """从 datasets 初始化数据集

        :param datasets: 数据集字典/列表
        :param amount_col: 金额字段名（可选）
        """
        if isinstance(datasets, dict):
            for key, value in datasets.items():
                if isinstance(value, (tuple, list)) and len(value) >= 2:
                    X_raw, y_raw = value[0], value[1]
                    label = key
                    X_df = _ensure_dataframe(X_raw, feature_names=self._feature_names)
                    if y_raw is None:
                        y_s = self._build_y(X_df, self._target_cfg)
                    else:
                        y_s = _ensure_series(y_raw, name=self._target_name)
                else:
                    X_raw = value
                    label = key
                    X_df = _ensure_dataframe(X_raw, feature_names=self._feature_names)
                    y_s = self._build_y(X_df, self._target_cfg)

                self._add_dataset(key, label, X_df, y_s, amount_col)
                self._datasets_info[key] = label

        elif isinstance(datasets, (list, tuple)):
            for i, value in enumerate(datasets):
                key = f"dataset_{i}"
                label = f"数据集{i + 1}"
                if isinstance(value, (tuple, list)) and len(value) >= 2:
                    X_raw, y_raw = value[0], value[1]
                    X_df = _ensure_dataframe(X_raw, feature_names=self._feature_names)
                    if y_raw is None:
                        y_s = self._build_y(X_df, self._target_cfg)
                    else:
                        y_s = _ensure_series(y_raw, name=self._target_name)
                else:
                    X_raw = value
                    X_df = _ensure_dataframe(X_raw, feature_names=self._feature_names)
                    y_s = self._build_y(X_df, self._target_cfg)

                self._add_dataset(key, label, X_df, y_s, amount_col)
                self._datasets_info[key] = label

    def _init_from_xy(self, X_train, y_train, X_test, y_test, X_oot=None, y_oot=None, amount_col: Optional[str] = None):
        """从 X/y 参数初始化

        :param amount_col: 金额字段名（可选）
        """
        X_train_df = _ensure_dataframe(X_train, feature_names=self._feature_names)

        if y_train is None:
            y_train_s = self._build_y(X_train_df, self._target_cfg)
        else:
            y_train_s = _ensure_series(y_train, name=self._target_name)

        self._add_dataset("train", "训练集", X_train_df, y_train_s, amount_col)
        self._datasets_info["train"] = "训练集"

        if X_test is not None:
            X_test_df = _ensure_dataframe(X_test, feature_names=list(X_train_df.columns))
            if y_test is None:
                y_test_s = self._build_y(X_test_df, self._target_cfg)
            else:
                y_test_s = _ensure_series(y_test, name=self._target_name)
            self._add_dataset("test", "测试集", X_test_df, y_test_s, amount_col)
            self._datasets_info["test"] = "测试集"

        if X_oot is not None:
            X_oot_df = _ensure_dataframe(X_oot, feature_names=list(X_train_df.columns))
            if y_oot is None:
                y_oot_s = self._build_y(X_oot_df, self._target_cfg)
            else:
                y_oot_s = _ensure_series(y_oot, name=self._target_name)
            self._add_dataset("oot", "跨时间验证集", X_oot_df, y_oot_s, amount_col)
            self._datasets_info["oot"] = "跨时间验证集"

    def _add_dataset(self, key: str, label: str, X: pd.DataFrame, y: pd.Series, amount_col: Optional[str] = None):
        """添加数据集

        :param key: 数据集标识
        :param label: 数据集标签
        :param X: 特征 DataFrame
        :param y: 标签 Series
        :param amount_col: 金额字段名（可选），用于金额口径指标计算
        """
        required_features: Optional[List[str]] = None
        if hasattr(self.model, 'feature_names_') and self.model.feature_names_ is not None:
            required_features = list(self.model.feature_names_)
        elif hasattr(self.model, 'feature_names_in_') and self.model.feature_names_in_ is not None:
            required_features = list(self.model.feature_names_in_)

        if required_features:
            missing = set(required_features) - set(X.columns)
            if missing:
                raise ValueError(f"数据集缺少以下模型特征: {missing}")
            X_for_pred = X[required_features]
        else:
            X_for_pred = X

        # 金额字段数组（用于金额口径指标计算）
        amount_arr: Optional[np.ndarray] = None
        if amount_col and amount_col in X.columns:
            amount_arr = X[amount_col].to_numpy()

        self._datasets[key] = ReportDataset(
            name=key,
            label=label,
            X=X,
            y=y,
            y_proba=_proba_pos(self.model, X_for_pred),
            score=_score_from_model(self.model, X_for_pred),
            amount=amount_arr,
        )

    def add_dataset(self, key: str, label: str, X, y=None, feature_names: Optional[List[str]] = None):
        """添加额外数据集（如 OOT）用于报告

        :param key: 数据集标识
        :param label: 数据集标签
        :param X: DataFrame（含目标列时 y 可为 None，自动构建标签）
        :param y: 标签列，None 时从 X 中通过 target / overdue+dpds 自动构建
        :param feature_names: 特征名列表
        """
        X = _ensure_dataframe(X, feature_names=feature_names or self.feature_names)
        if y is None:
            y = self._build_y(X, self._target_cfg)
        y = _ensure_series(y, name=self._target_name)
        self._add_dataset(key, label, X, y)

    # ---------- 模型性能指标 ----------

    def get_metrics(self) -> pd.DataFrame:
        """KS / AUC / PSI 等核心指标"""
        ds_keys = [k for k in ["train", "test"] + [k for k in self._datasets if k not in ("train", "test")] if k in self._datasets]
        labels_map = {k: self._datasets[k].label for k in ds_keys}

        rows = []
        rows.append({"统计项": "KS", **{labels_map[k]: _ks(self._datasets[k].y_proba, self._datasets[k].y) for k in ds_keys}})
        rows.append({"统计项": "AUC", **{labels_map[k]: _auc(self._datasets[k].y_proba, self._datasets[k].y) for k in ds_keys}})
        rows.append({"统计项": "样本数", **{labels_map[k]: len(self._datasets[k].y) for k in ds_keys}})
        rows.append({"统计项": "坏样本率", **{labels_map[k]: float(self._datasets[k].y.mean()) for k in ds_keys}})

        if len(ds_keys) >= 2:
            psi_row: Dict[str, Any] = {"统计项": "PSI", labels_map[ds_keys[0]]: "\\"}
            for k in ds_keys[1:]:
                try:
                    psi_row[labels_map[k]] = _psi(self._datasets[ds_keys[0]].score, self._datasets[k].score)
                except Exception:
                    psi_row[labels_map[k]] = np.nan
            rows.append(psi_row)

        return pd.DataFrame(rows)

    # ---------- 评分分箱效果表 ----------

    def get_bin_table(
        self,
        dataset: str = "train",
        method: str = "quantile",
        max_n_bins: int = 10,
        amount_col: Optional[str] = None,
        margins: bool = True,
        rules: Optional[List] = None,
    ) -> pd.DataFrame:
        """生成评分分箱效果表

        :param dataset: 数据集标识，默认 'train'
        :param method: 分箱方法，默认 'quantile'
        :param max_n_bins: 最大分箱数，默认 10
        :param amount_col: 金额字段名
        :param margins: 是否包含合计行
        :param rules: 分箱规则列表，用于将相同的分箱规则应用到其他数据集进行PSI计算
        """
        from scorecardpipeline.processing import feature_bin_stats

        ds = self._datasets[dataset]
        target_col = "__target__"
        score_col = "__score__"
        df = ds.X.copy()
        df[target_col] = ds.y.values
        df[score_col] = ds.score

        kw: Dict[str, Any] = dict(
            data=df,
            feature=score_col,
            target=target_col,
            method=method,
            desc="模型评分",
            max_n_bins=max_n_bins,
            margins=margins,
            return_cols=['分箱', '样本总数', '好样本数', '坏样本数', '样本占比', '好样本占比', '坏样本占比', '坏样本率', 'LIFT值', '累积LIFT值', '坏账改善', '累积坏账改善', '分档KS值'],
        )
        if amount_col and amount_col in df.columns:
            kw["amount"] = amount_col

        # 如果传入rules，则使用rules进行分箱（用于PSI计算时保持分箱一致性）
        if rules is not None:
            kw["rules"] = rules

        table = feature_bin_stats(**kw)
        if isinstance(table, tuple):
            table = table[0]
        return table

    def get_bin_table_rules(
        self,
        dataset: str = "train",
        method: str = "quantile",
        max_n_bins: int = 10,
    ) -> Tuple[pd.DataFrame, List]:
        """获取评分分箱表及其分箱规则

        用于在训练集上确定分箱规则后，将规则应用到其他数据集进行PSI计算。

        :param dataset: 数据集标识，默认 'train'
        :param method: 分箱方法，默认 'quantile'
        :param max_n_bins: 最大分箱数，默认 10
        :return: (分箱表, 分箱规则列表)
        """
        from scorecardpipeline.processing import feature_bin_stats

        ds = self._datasets[dataset]
        target_col = "__target__"
        score_col = "__score__"
        df = ds.X.copy()
        df[target_col] = ds.y.values
        df[score_col] = ds.score

        table, rules = feature_bin_stats(
            data=df,
            feature=score_col,
            target=target_col,
            method=method,
            desc="模型评分",
            max_n_bins=max_n_bins,
            margins=False,
            return_cols=['分箱', '样本总数', '好样本数', '坏样本数', '样本占比', '好样本占比', '坏样本占比', '坏样本率', 'LIFT值', '累积LIFT值', '坏账改善', '累积坏账改善', '分档KS值'],
            return_rules=True,
        )
        if isinstance(table, tuple):
            table = table[0]
        return table, rules

    # ---------- 特征重要性 ----------

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """获取特征重要性"""
        if self._importance_cache is None:
            importances = None
            feature_names = None

            # 尝试从模型获取特征重要性
            if hasattr(self.model, "get_feature_importances"):
                try:
                    importances = self.model.get_feature_importances()
                    feature_names = self.model.feature_names_in_ if hasattr(self.model, "feature_names_in_") else None
                except Exception:
                    pass

            # 尝试从 feature_importances_ 属性获取
            if importances is None and hasattr(self.model, "feature_importances_"):
                feature_names = getattr(self.model, "feature_names_in_", None)
                importances = pd.Series(
                    self.model.feature_importances_,
                    index=feature_names if feature_names is not None else range(len(self.model.feature_importances_)),
                )

            # 尝试从 coef_ 属性获取（逻辑回归等线性模型）
            if importances is None and hasattr(self.model, "coef_"):
                try:
                    coefs = np.abs(self.model.coef_)
                    if len(coefs.shape) > 1:
                        coefs = coefs.mean(axis=0)
                    # 尝试从 model 获取 feature_names_in_，如果不存在则尝试获取 inner model
                    feature_names = getattr(self.model, "feature_names_in_", None)
                    if feature_names is None and hasattr(self.model, "model") and hasattr(self.model.model, "feature_names_in_"):
                        feature_names = getattr(self.model.model, "feature_names_in_", None)
                    if feature_names is not None and len(coefs) == len(feature_names):
                        importances = pd.Series(coefs, index=feature_names)
                    else:
                        importances = pd.Series(coefs)
                except Exception:
                    pass

            # 如果无法获取特征重要性，使用空 DataFrame
            if importances is None or len(importances) == 0:
                self._importance_cache = pd.DataFrame(columns=["特征重要性"])
            else:
                importance_df = pd.DataFrame(index=importances.index)
                total = importances.sum()
                importance_df["特征重要性"] = importances.values / total if total else importances.values

                self._importance_cache = importance_df.sort_values("特征重要性", ascending=False)

        df = self._importance_cache.copy()
        if top_n is not None:
            df = df.head(top_n)
        return df

    def _calc_iv(self, y, x):
        """计算 IV 值"""
        try:
            from scorecardpipeline.processing import feature_bin_stats
            df = pd.DataFrame({'y': y, 'x': x})
            result = feature_bin_stats(data=df, feature='x', target='y', return_cols=['IV'])
            if isinstance(result, tuple):
                result = result[0]
            if 'IV' in result.columns:
                return result['IV'].iloc[0] if len(result) > 0 else 0
        except Exception:
            pass
        return np.nan

    # ---------- 特征描述 ----------

    def get_features_describe(self) -> pd.DataFrame:
        """入模变量重要性及描述性统计"""
        if self._features_describe_cache is not None:
            return self._features_describe_cache.copy()

        from scorecardpipeline.utils import feature_summary

        train_ds = self._datasets.get("train") or list(self._datasets.values())[0]
        features = self.feature_names
        if not features:
            features = list(train_ds.X.columns)

        df = train_ds.X[features].copy()
        target_col = self._target_name
        df[target_col] = train_ds.y.values

        val_df = self._datasets.get("test")
        val_X = val_df.X[features] if val_df is not None else None

        summary_df = feature_summary(df, features=features, y=target_col, val_df=val_X)

        coefs = None
        if hasattr(self.model, "coef_"):
            coefs = np.abs(self.model.coef_)
            if len(coefs.shape) > 1:
                coefs = coefs.mean(axis=0)
        elif hasattr(self.model, "feature_importances_"):
            coefs = self.model.feature_importances_

        if coefs is not None:
            feature_names_from_model = getattr(self.model, "feature_names_in_", None)
            if feature_names_from_model is None and hasattr(self.model, "model"):
                feature_names_from_model = getattr(self.model.model, "feature_names_in_", None)
            if feature_names_from_model is not None and len(coefs) == len(feature_names_from_model):
                imp = pd.Series(coefs, index=feature_names_from_model)
            else:
                imp = pd.Series(coefs, index=features[:len(coefs)] if len(coefs) <= len(features) else features)
            total = imp.sum()
            imp = imp / total if total else imp
            imp.name = "特征重要性"

            imp_df = imp.to_frame()
            imp_df.index.name = "特征名"

            summary_df = summary_df.set_index("特征名")
            summary_df = imp_df.join(summary_df, how="right")
            summary_df = summary_df.reset_index()

        # 将特征重要性列插入到 IV 之前
        if "特征重要性" in summary_df.columns:
            iv_idx = None
            for i, col in enumerate(summary_df.columns):
                if col == "IV":
                    iv_idx = i
                    break
            if iv_idx is not None:
                cols = list(summary_df.columns)
                cols.remove("特征重要性")
                cols.insert(iv_idx - 1, "特征重要性")
                summary_df = summary_df[cols]

        self._features_describe_cache = summary_df
        return self._features_describe_cache.copy()

    # ---------- 特征相关性 ----------

    def get_features_corr(self) -> pd.DataFrame:
        """获取特征相关性矩阵"""
        importance = self.get_feature_importance()
        features = importance.index.tolist() if importance is not None and len(importance) > 0 else []
        if not features:
            features = self.feature_names

        train_ds = self._datasets.get("train") or list(self._datasets.values())[0]
        df = train_ds.X[features].select_dtypes(include=[np.number])
        if df.shape[1] == 0:
            return pd.DataFrame()
        return df.corr()

    # ---------- 特征分箱分析 ----------

    def get_feature_bin_table(
        self,
        feature: str,
        dataset: str = "train",
        max_n_bins: int = 10,
        method: str = "quantile",
        margins: bool = True,
        amount_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """单特征分箱效果表

        :param feature: 特征名
        :param dataset: 数据集标识
        :param max_n_bins: 最大分箱数
        :param method: 分箱方法
        :param margins: 是否包含合计行
        :param amount_col: 金额字段名
        """
        from scorecardpipeline.processing import feature_bin_stats

        ds = self._datasets[dataset]
        target_col = "__target__"
        df = ds.X.copy()
        df[target_col] = ds.y.values

        kw: Dict[str, Any] = dict(
            data=df,
            feature=feature,
            target=target_col,
            method=method,
            max_n_bins=max_n_bins,
            margins=margins,
        )
        binner = getattr(self.model, "binner", None)
        if binner is not None:
            kw["combiner"] = binner

        if amount_col and amount_col in df.columns:
            kw["amount"] = amount_col

        table = feature_bin_stats(**kw)
        if isinstance(table, tuple):
            table = table[0]
        return table

    # ---------- TOP n% LIFT ----------

    def _get_top_n_lift_table(
        self,
        percentiles: Tuple[float, ...] = (0.01, 0.03, 0.05, 0.10),
        amount_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """构建 TOP n% 尾部区分能力表

        :param percentiles: TOP n% 的百分位列表
        :param amount_col: 金额字段名（可选），指定时输出金额口径指标
        """
        rows: List[Dict[str, Any]] = []
        for ds_key, ds in self._datasets.items():
            tag = ds.label
            y_arr = ds.y.to_numpy()
            n = len(y_arr)
            overall_bad_rate = float(y_arr.mean())

            sorted_idx = np.argsort(-ds.y_proba)
            sorted_y = y_arr[sorted_idx]

            bad_rates: Dict[str, float] = {}
            lifts: Dict[str, float] = {}
            improvements: Dict[str, float] = {}

            for pct in percentiles:
                top_n = max(1, int(n * pct))
                top_bad_rate = float(sorted_y[:top_n].mean())
                lift = top_bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0.0
                improvement = (top_bad_rate - overall_bad_rate) / overall_bad_rate if overall_bad_rate > 0 else 0.0
                key = f"TOP {int(pct * 100)}%"
                bad_rates[key] = top_bad_rate
                lifts[key] = lift
                improvements[key] = improvement

            bad_rates["TOTAL"] = overall_bad_rate
            lifts["TOTAL"] = 1.0
            improvements["TOTAL"] = 0.0

            rows.append({"数据集": tag, "统计项": "坏样本率", **bad_rates})
            rows.append({"数据集": tag, "统计项": "LIFT值", **lifts})
            rows.append({"数据集": tag, "统计项": "坏账改善", **improvements})

            # 金额口径
            if amount_col and ds.amount is not None:
                amounts_sorted = ds.amount[sorted_idx]
                overall_bad_amount = float(
                    (sorted_y * amounts_sorted).sum() / amounts_sorted.sum()
                ) if amounts_sorted.sum() > 0 else overall_bad_rate

                amt_bad_rates: Dict[str, float] = {}
                amt_lifts: Dict[str, float] = {}
                amt_improvements: Dict[str, float] = {}

                for pct in percentiles:
                    top_n = max(1, int(n * pct))
                    top_amt = amounts_sorted[:top_n]
                    top_y_sorted = sorted_y[:top_n]
                    top_bad_amt = float(
                        (top_y_sorted * top_amt).sum() / top_amt.sum()
                    ) if top_amt.sum() > 0 else 0.0
                    lift_amt = top_bad_amt / overall_bad_amount if overall_bad_amount > 0 else 0.0
                    imp_amt = (top_bad_amt - overall_bad_amount) / overall_bad_amount if overall_bad_amount > 0 else 0.0
                    key = f"TOP {int(pct * 100)}%"
                    amt_bad_rates[key] = top_bad_amt
                    amt_lifts[key] = lift_amt
                    amt_improvements[key] = imp_amt

                amt_bad_rates["TOTAL"] = overall_bad_amount
                amt_lifts["TOTAL"] = 1.0
                amt_improvements["TOTAL"] = 0.0

                rows.append({"数据集": tag, "统计项": "坏样本率", **amt_bad_rates})
                rows.append({"数据集": tag, "统计项": "LIFT值", **amt_lifts})
                rows.append({"数据集": tag, "统计项": "坏账改善", **amt_improvements})

        return pd.DataFrame(rows)

    # ---------- 分月指标 ----------

    def _get_monthly_metrics(self, date_col: str) -> pd.DataFrame:
        """分月计算 KS/AUC"""
        rows: List[Dict[str, Any]] = []
        for ds_key, ds in self._datasets.items():
            if date_col not in ds.X.columns:
                continue
            dates = pd.to_datetime(ds.X[date_col])
            months = dates.dt.to_period(_period_freq("M"))
            for month in sorted(months.unique()):
                mask = months == month
                y_m = ds.y[mask.values]
                proba_m = ds.y_proba[mask.values]
                if len(y_m) < 10 or y_m.nunique() < 2:
                    continue
                try:
                    rows.append({
                        "数据集": ds.label,
                        "月份": str(month),
                        "样本数": len(y_m),
                        "坏样本率": float(y_m.mean()),
                        "KS": _ks(proba_m, y_m),
                        "AUC": _auc(proba_m, y_m),
                    })
                except Exception:
                    pass
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ---------- 分月 PSI 矩阵 ----------

    def _get_monthly_psi_matrix(self, date_col: str) -> pd.DataFrame:
        """分月 PSI 交叉矩阵"""
        month_scores: Dict[str, np.ndarray] = {}
        for ds in self._datasets.values():
            if date_col not in ds.X.columns:
                continue
            dates = pd.to_datetime(ds.X[date_col])
            months = dates.dt.to_period(_period_freq("M"))
            for month in sorted(months.unique()):
                mask = months == month
                key = str(month)
                if key in month_scores:
                    month_scores[key] = np.concatenate([month_scores[key], ds.score[mask.values]])
                else:
                    month_scores[key] = ds.score[mask.values]

        if len(month_scores) < 2:
            return pd.DataFrame()

        labels = sorted(month_scores.keys())
        matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
        for i, m1 in enumerate(labels):
            for j, m2 in enumerate(labels):
                try:
                    matrix.loc[m1, m2] = _psi(month_scores[m1], month_scores[m2])
                except Exception:
                    pass
        return matrix

    # ---------- 图表导出 ----------

    def _export_plots(
        self,
        output_dir: Path,
        n_bins: int = 10,
        bin_method: str = "quantile",
        amount_col: Optional[str] = None,
    ) -> Tuple[Dict[str, List[str]], Dict[str, pd.DataFrame]]:
        """导出所有图表，返回 (图表路径字典, PSI数据表字典)"""
        from scorecardpipeline.utils import ks_plot, bin_plot, corr_plot, psi_plot

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, List[str]] = {}
        tables: Dict[str, pd.DataFrame] = {}

        # 模型级图表
        for ds_key, ds in self._datasets.items():
            tag = ds.label
            model_figs: List[str] = []

            try:
                bt = self.get_bin_table(ds_key, method=bin_method, max_n_bins=n_bins, amount_col=amount_col, margins=True)
                bd = bt.iloc[:-1].reset_index(drop=True) if len(bt) > 1 else bt
                p = str(output_dir / f"bin_{ds_key}.png")
                bin_plot(bd, desc="模型评分", ending=f" {tag}", save=p, figsize=(14, 8))
                _safe_close_figs()
                model_figs.append(p)
            except Exception:
                pass

            try:
                p = str(output_dir / f"ks_{ds_key}.png")
                ks_plot(ds.score, ds.y, title=f"{tag} KS曲线", save=p, figsize=(14, 8))
                _safe_close_figs()
                model_figs.append(p)
            except Exception:
                pass

            if model_figs:
                paths[f"model_{ds_key}"] = model_figs

        # 特征相关性图
        importance = self.get_feature_importance()
        top_features = importance.index.tolist()
        if len(top_features) >= 2:
            try:
                p = str(output_dir / "feature_corr.png")
                corr_plot(self._datasets["train"] if "train" in self._datasets else self._datasets[next(iter(self._datasets))].X[top_features], annot=False, save=p)
                _safe_close_figs()
                paths["feature_corr"] = [p]
            except Exception:
                pass

        # 逐特征图表
        for feat in (top_features or self.feature_names):
            bin_figs: List[str] = []
            for ds_key, ds in self._datasets.items():
                try:
                    ft = self.get_feature_bin_table(feat, ds_key, max_n_bins=n_bins, method=bin_method, amount_col=amount_col, margins=True)
                    fd = ft.iloc[:-1].reset_index(drop=True) if len(ft) > 1 else ft
                    p = str(output_dir / f"bin_{feat}_{ds_key}.png")
                    bin_plot(fd, desc=feat, ending=f" {ds.label}", save=p, figsize=(14, 8))
                    _safe_close_figs()
                    bin_figs.append(p)
                except Exception:
                    pass
            if bin_figs:
                paths[f"feat_bin_{feat}"] = bin_figs

            # PSI 图
            ds_keys = list(self._datasets.keys())
            if len(ds_keys) >= 2:
                try:
                    train_vals = self._datasets[ds_keys[0]].X[feat].dropna()
                    test_vals = self._datasets[ds_keys[1]].X[feat].dropna()
                    p = str(output_dir / f"psi_{feat}.png")
                    psi_result = psi_plot(train_vals, test_vals, desc=feat, save=p, result=True, plot=True, figsize=(15, 8))
                    _safe_close_figs()
                    paths[f"feat_psi_{feat}"] = [p]
                    if isinstance(psi_result, pd.DataFrame):
                        tables[f"feat_psi_{feat}"] = psi_result
                except Exception:
                    pass

        # 评分卡专属图表
        if hasattr(self.model, "scorecard_points"):
            try:
                p = str(output_dir / "plot_weights.png")
                self.model.pretrain_lr.plot_weights(save=p)
                _safe_close_figs()
                paths["model_weights"] = [p]
            except Exception:
                pass

            if len(ds_keys) >= 2:
                try:
                    score_train = self._datasets[ds_keys[0]].score.dropna()
                    score_test = self._datasets[ds_keys[1]].score.dropna()
                    p = str(output_dir / "score_psi.png")
                    score_psi_df = psi_plot(score_train, score_test, desc="模型评分", save=p, result=True, plot=True, figsize=(15, 8))
                    _safe_close_figs()
                    paths["score_psi"] = [p]
                    if isinstance(score_psi_df, pd.DataFrame):
                        tables["score_psi"] = score_psi_df
                except Exception:
                    pass

        return paths, tables

    # ---------- 控制台输出 ----------

    def print_report(self, n_bins: int = 10, amount_col: Optional[str] = None, **kwargs) -> None:
        """打印报告摘要

        :param n_bins: 分箱数，默认 10
        :param amount_col: 金额字段名，用于显示金额口径指标
        """
        print("=" * 72)
        print("模型评估快速报告")
        print("=" * 72)
        print("\n【模型性能指标】")
        print(self.get_metrics().to_string(index=False))

        importance = self.get_feature_importance(top_n=10)
        if not importance.empty:
            print("\n【Top 10 特征重要性】")
            print(importance.to_string())

        for ds_key, ds in self._datasets.items():
            print(f"\n【{ds.label}评分分箱效果】")
            print(self.get_bin_table(ds_key, max_n_bins=n_bins, amount_col=amount_col).to_string(index=False))
            if amount_col:
                print(f"\n【{ds.label}评分分箱效果(金额口径)】")
                print(self.get_bin_table(ds_key, max_n_bins=n_bins, amount_col=amount_col).to_string(index=False))
        print("\n" + "=" * 72)

    # ---------- to_excel ----------

    def to_excel(
        self,
        filepath: str,
        *,
        n_bins: int = 10,
        bin_method: str = "quantile",
        amount_col: Optional[str] = None,
        date_col: Optional[str] = None,
        date_freq: Optional[str] = None,
        group_col: Optional[str] = None,
        with_plots: bool = True,
        model_name: Optional[str] = None,
        project_desc: Optional[str] = None,
        feature_map: Optional[Dict[str, str]] = None,
        feature_info: Optional[pd.DataFrame] = None,
        data_source: Optional[str] = None,
    ) -> str:
        """生成多 Sheet 结构的 Excel 模型报告

        Sheet 结构：
        - 目录
        - 1-基本信息
        - 2-模型性能
        - 3-入模变量分析
        - 4-稳定性分析
        - 5-模型参数
        - 6-模型部署需求

        :param filepath: 保存路径
        :param n_bins: 分箱数，默认 10
        :param bin_method: 分箱方法，默认 'quantile'
        :param amount_col: 金额字段名
        :param date_col: 日期字段名
        :param date_freq: 日期频率
        :param group_col: 分组字段名
        :param with_plots: 是否包含图表
        :param model_name: 模型名称
        :param project_desc: 项目描述
        :param feature_map: 特征含义映射
        :param feature_info: 特征信息表
        :param data_source: 数据来源
        """
        import os
        from scorecardpipeline.excel_writer import ExcelWriter, dataframe2excel

        model_name = model_name or self.model.__class__.__name__
        max_col = 35

        plot_paths: Dict[str, List[str]] = {}
        psi_tables: Dict[str, pd.DataFrame] = {}
        if with_plots:
            plot_dir = Path(filepath).parent / f"{Path(filepath).stem}_assets"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plot_paths, psi_tables = self._export_plots(
                    plot_dir, n_bins=n_bins, bin_method=bin_method, amount_col=amount_col,
                )

        writer = ExcelWriter()

        # ============================================================
        # 目录 Sheet
        # ============================================================
        contents = pd.DataFrame([
            {"序号": 1, "内容": "1-基本信息", "备注": "项目目标、样本选取、样本坏率分布"},
            {"序号": 2, "内容": "2-模型性能", "备注": "模型效果、区分度、稳定性等内容"},
            {"序号": 3, "内容": "3-入模变量分析", "备注": "模型变量有效性及不同数据集分箱情况"},
            {"序号": 4, "内容": "4-稳定性分析", "备注": "评分分布、PSI、CSI等稳定性分析"},
            {"序号": 5, "内容": "5-模型部署需求", "备注": "入模变量信息及测试用例"},
        ])

        ws = writer.get_sheet_by_name("目录")
        end_row, _ = writer.insert_value2sheet(ws, (2, 2), value="模型评估报告", style="header_middle", end_space=(2, max_col))
        end_row, _ = dataframe2excel(contents, writer, sheet_name=ws, start_row=end_row + 1,
                                      left_cols=["内容", "备注"])

        for i, row in contents.iterrows():
            try:
                target_cell = writer.get_cell_space((2, 2))
                writer.insert_hyperlink2sheet(ws, (end_row - len(contents) + i, 3), hyperlink=f"#'{row['内容']}'!{target_cell}")
            except Exception:
                pass

        _, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value="版本号:", style="middle", end_space=(end_row + 2, 2))
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 3), value="V1.0", style="middle", end_space=(end_row + 2, 4))
        _, _ = writer.insert_value2sheet(ws, (end_row, 2), value="创建日期:", style="middle", end_space=(end_row, 2))
        end_row, _ = writer.insert_value2sheet(ws, (end_row, 3), value=date.today().strftime("%Y-%m-%d"), style="middle", end_space=(end_row, 4))
        _, _ = writer.insert_value2sheet(ws, (end_row, 2), value="模型名称:", style="middle", end_space=(end_row, 2))
        end_row, _ = writer.insert_value2sheet(ws, (end_row, 3), value=model_name, style="middle", end_space=(end_row, 4))

        # ============================================================
        # 1-基本信息 Sheet
        # ============================================================
        ws = writer.get_sheet_by_name("1-基本信息")
        end_row, _ = writer.insert_value2sheet(ws, (2, 2), value="一、基本信息", style="header_middle", end_space=(2, max_col))
        try:
            writer.insert_hyperlink2sheet(ws, (2, 2), hyperlink="#'目录'!B2")
        except Exception:
            pass

        # 1.1 项目目标
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value="1、项目目标", style="header_middle", align={"horizontal": "left"})
        desc_text = project_desc or f"使用 {model_name} 模型进行信用风险评估"
        end_row, _ = writer.insert_value2sheet(ws, (end_row, 2), value=desc_text, style="middle", end_space=(end_row, max_col), align={"horizontal": "left"})

        # 1.2 数据样本描述
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value="2、数据样本描述", style="header_middle", align={"horizontal": "left"})

        def _extract_dates(ds, col):
            if col and ds is not None and col in ds.X.columns:
                dates = pd.to_datetime(ds.X[col])
                if not dates.isna().all():
                    return dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")
            return None, None

        # 构建标签描述
        if isinstance(self._target_cfg, dict) and "dpds" in self._target_cfg:
            dpd_val = self._target_cfg["dpds"]
            if "overdue" in self._target_cfg:
                overdue_name = self._target_cfg["overdue"]
                if isinstance(overdue_name, list):
                    overdue_name = overdue_name[0]
                label_text = f"{overdue_name} EVER DPD{dpd_val}+"
            else:
                label_text = f"OVERDUE EVER DPD{dpd_val}+"
        else:
            label_text = "TARGET"

        # 全局日期范围
        global_date_prefix = ""
        if date_col:
            all_dates = []
            for ds in self._datasets.values():
                if ds is not None and date_col in ds.X.columns:
                    all_dates.append(pd.to_datetime(ds.X[date_col]))
            if all_dates:
                all_dates_combined = pd.concat(all_dates, ignore_index=True).dropna()
                if not all_dates_combined.empty:
                    global_date_prefix = (
                        f"{all_dates_combined.min().strftime('%Y-%m-%d')} ~ "
                        f"{all_dates_combined.max().strftime('%Y-%m-%d')}  "
                    )
        sample_interval = global_date_prefix if global_date_prefix else ""

        # 整体描述
        concat_parts = []
        for ds_key, ds in self._datasets.items():
            if ds is None:
                continue
            part = ds.X.copy()
            part["_ds_label_"] = ds.label
            part["_ds_y_"] = ds.y.values
            concat_parts.append(part)
        if concat_parts:
            all_X = pd.concat(concat_parts, ignore_index=True)
            overall_n = len(all_X)
            overall_bad = int(all_X["_ds_y_"].sum())
            overall_bad_rate = overall_bad / overall_n * 100 if overall_n > 0 else 0
            overall_desc = (
                f"{global_date_prefix}样本数: {overall_n}, "
                f"{label_text}: {round(overall_bad_rate, 2)}%"
            )
        else:
            overall_desc = ""

        data_source_str = data_source if data_source else ""
        fixed_rows: List[Dict[str, Any]] = [
            {"统计项": "样本区间", "统计内容": sample_interval or ""},
            {"统计项": "整体样本", "统计内容": overall_desc},
            {"统计项": "模型名称", "统计内容": model_name or ""},
            {"统计项": "取样逻辑", "统计内容": project_desc or ""},
            {"统计项": "数据源", "统计内容": data_source_str},
        ]

        ds_rows: List[Dict[str, Any]] = []
        for ds_key, ds in self._datasets.items():
            if ds is None:
                continue
            n_samples = len(ds.y)
            bad_rate = round(ds.y.mean() * 100, 2)
            content = f"样本数: {n_samples}, {label_text}: {bad_rate}%"
            ds_rows.append({"统计项": ds.label, "统计内容": content})

        desc_df = pd.DataFrame(fixed_rows + ds_rows)
        end_row, _ = dataframe2excel(desc_df, writer, sheet_name=ws, start_row=end_row + 1,
                                     left_cols=["统计项", "统计内容"])

        # 1.3 数据样本统计
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value="3、数据样本统计", style="header_middle", align={"horizontal": "left"})
        sample_rows: List[Dict[str, Any]] = []
        for ds_key, ds in self._datasets.items():
            sample_rows.append({
                "数据集": ds.label,
                "样本数": len(ds.y),
                "好样本数": int((1 - ds.y).sum()),
                "坏样本数": int(ds.y.sum()),
                "坏样本率": float(ds.y.mean()),
            })
        sample_df = pd.DataFrame(sample_rows)
        end_row, _ = dataframe2excel(sample_df, writer, sheet_name=ws, start_row=end_row + 1, percent_cols=["坏样本率"])

        # 1.4 样本分布情况
        freq_label_map = {"D": "日", "W": "周", "M": "月", "ME": "月", "Q": "季度", "Y": "年"}
        if date_col or group_col:
            end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value="4、样本分布情况", style="header_middle", align={"horizontal": "left"})

            if date_col:
                period_labels = {"D": "日期", "W": "周", "M": "月份", "ME": "月份", "Q": "季度", "Y": "年份"}
                _raw_freq = date_freq or "M"
                period_label = period_labels.get(_raw_freq, "周期")
                period_col_name = freq_label_map.get(_raw_freq, _raw_freq)

                for ds_key, ds in self._datasets.items():
                    if date_col in ds.X.columns:
                        dates = pd.to_datetime(ds.X[date_col])
                        try:
                            periods = dates.dt.to_period(_period_freq(_raw_freq))
                        except Exception:
                            periods = dates.dt.to_period("ME")

                        period_stats = ds.y.groupby(periods).agg(["count", "sum", "mean"]).reset_index()
                        period_stats.columns = [period_label, "样本数", "坏样本数", "坏样本率"]
                        period_stats[period_label] = period_stats[period_label].astype(str)
                        period_stats["坏样本数"] = period_stats["坏样本数"].astype(int)
                        end_row, _ = dataframe2excel(
                            period_stats, writer, sheet_name=ws,
                            title=f"{ds.label} {period_col_name}度分布", start_row=end_row + 1,
                            percent_cols=["坏样本率"],
                        )

            if group_col:
                for ds_key, ds in self._datasets.items():
                    if group_col not in ds.X.columns:
                        continue
                    groups = ds.X[group_col]
                    group_stats = pd.DataFrame({
                        "分组": groups,
                        "样本数": 1,
                        "坏样本": ds.y.values,
                    }).groupby("分组").agg({"样本数": "count", "坏样本": "sum"}).reset_index()
                    group_stats["坏样本率"] = group_stats["坏样本"] / group_stats["样本数"]
                    end_row, _ = dataframe2excel(
                        group_stats, writer, sheet_name=ws,
                        title=f"{ds.label} 分组分布", start_row=end_row + 1,
                        percent_cols=["坏样本率"],
                    )

        # ============================================================
        # 2-模型性能 Sheet
        # ============================================================
        ws = writer.get_sheet_by_name("2-模型性能")
        end_row, _ = writer.insert_value2sheet(ws, (2, 2), value="二、模型性能评估", style="header_middle", end_space=(2, max_col))
        try:
            writer.insert_hyperlink2sheet(ws, (2, 2), hyperlink="#'目录'!B2")
        except Exception:
            pass

        section_idx = 1

        # 2.1 性能指标
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value=f"{section_idx}、模型性能验证指标", style="header_middle", align={"horizontal": "left"})
        metrics = self.get_metrics()
        end_row, _ = dataframe2excel(
            metrics, writer, sheet_name=ws,
            start_row=end_row + 1,
            percent_rows=[0, 1, 3, 4],
            condition_rows=[0],
        )
        section_idx += 1

        # 2.2 分月模型效果
        if date_col:
            monthly_metrics = self._get_monthly_metrics(date_col)
            if not monthly_metrics.empty:
                end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value=f"{section_idx}、分月模型效果", style="header_middle", align={"horizontal": "left"})
                end_row, _ = dataframe2excel(
                    monthly_metrics, writer, sheet_name=ws, start_row=end_row + 1,
                    percent_cols=["坏样本率"],
                )
                section_idx += 1

        # 2.3 模型尾部区分能力
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value=f"{section_idx}、模型尾部区分能力（TOP n%）", style="header_middle", align={"horizontal": "left"})
        pct_keys = ["TOP 1%", "TOP 3%", "TOP 5%", "TOP 10%", "TOTAL"]
        table_start = end_row + 1

        if amount_col:
            # 订单口径 + 金额口径左右并排
            lift_table = self._get_top_n_lift_table(percentiles=(0.01, 0.03, 0.05, 0.10))
            lift_amt_raw = self._get_top_n_lift_table(percentiles=(0.01, 0.03, 0.05, 0.10), amount_col=amount_col)
            n_datasets = len(lift_amt_raw) // 6
            amt_rows = []
            for i in range(n_datasets):
                amt_rows.extend(lift_amt_raw.iloc[i * 6 + 3:(i + 1) * 6].values.tolist())
            lift_amt = pd.DataFrame(amt_rows, columns=lift_amt_raw.columns)
            end_row1, end_col1 = dataframe2excel(
                lift_table, writer, sheet_name=ws,
                title="订单口径", start_row=table_start, start_col=2,
                percent_cols=pct_keys,
            )
            end_row2, _ = dataframe2excel(
                lift_amt, writer, sheet_name=ws,
                title="金额口径", start_row=table_start, start_col=end_col1 + 1,
                percent_cols=pct_keys,
            )
            end_row = max(end_row1, end_row2)
        else:
            lift_table = self._get_top_n_lift_table(percentiles=(0.01, 0.03, 0.05, 0.10))
            end_row, _ = dataframe2excel(
                lift_table, writer, sheet_name=ws, start_row=table_start,
                percent_cols=pct_keys,
                auto_filter=True,
            )

        section_idx += 1

        # 2.4 分月PSI矩阵
        if date_col:
            psi_matrix = self._get_monthly_psi_matrix(date_col)
            if not psi_matrix.empty:
                end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value=f"{section_idx}、分月对比PSI", style="header_middle", align={"horizontal": "left"})
                end_row, _ = dataframe2excel(psi_matrix, writer, sheet_name=ws, start_row=end_row + 1, index=True)
                section_idx += 1

        # 2.5 各数据集评分排序性
        for ds_key, ds in self._datasets.items():
            tag = ds.label
            end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value=f"{section_idx}、{tag}评分排序性", style="header_middle", align={"horizontal": "left"})

            figs = plot_paths.get(f"model_{ds_key}", [])

            img_start_row = end_row + 1
            current_col = 2
            max_img_end_row = img_start_row
            for fig in figs:
                try:
                    img_end_row, current_col = writer.insert_pic2sheet(ws, fig, (img_start_row, current_col), figsize=(500, 300))
                    max_img_end_row = max(max_img_end_row, img_end_row)
                except Exception:
                    pass
            if figs:
                end_row = max_img_end_row

            order_table = self.get_bin_table(ds_key, method=bin_method, max_n_bins=n_bins, margins=True)
            pct_cols = [c for c in self._PERCENT_COLS if c in order_table.columns]
            cond_cols = [c for c in self._CONDITION_COLS if c in order_table.columns]
            if amount_col:
                order_amt_table = self.get_bin_table(ds_key, method=bin_method, max_n_bins=n_bins, amount_col=amount_col, margins=True)
                end_row1, end_col1 = dataframe2excel(
                    order_table, writer, sheet_name=ws,
                    title=f"{tag} 评分有效性(订单口径)", start_row=end_row + 1,
                    percent_cols=pct_cols, condition_cols=cond_cols, condition_color="F76E6C",
                )
                _, _ = dataframe2excel(
                    order_amt_table, writer, sheet_name=ws,
                    title=f"{tag} 评分有效性(金额口径)", start_row=end_row + 1, start_col=end_col1 + 1,
                    percent_cols=pct_cols, condition_cols=cond_cols, condition_color="F76E6C",
                )
                end_row = end_row1
            else:
                end_row, _ = dataframe2excel(
                    order_table, writer, sheet_name=ws,
                    title=f"{tag} 评分有效性", start_row=end_row + 1,
                    percent_cols=pct_cols, condition_cols=cond_cols, condition_color="F76E6C",
                )
            section_idx += 1

        # ============================================================
        # 3-入模变量分析 Sheet
        # ============================================================
        ws = writer.get_sheet_by_name("3-入模变量分析")
        end_row, _ = writer.insert_value2sheet(ws, (2, 2), value="三、入模变量分析", style="header_middle", end_space=(2, max_col))
        try:
            writer.insert_hyperlink2sheet(ws, (2, 2), hyperlink="#'目录'!B2")
        except Exception:
            pass

        # 3.1 入模变量重要性及分布情况
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value="1、入模变量重要性及分布情况", style="header_middle", align={"horizontal": "left"})
        features_summary = self.get_features_describe()
        end_row, _ = dataframe2excel(
            features_summary, writer, sheet_name=ws,
            start_row=end_row + 1,
            right_cols=[0],
            percent_cols=['IV', 'KS', 'PSI', '缺失率', '众数占比', '零值率', '负值率', '重复率'],
            condition_cols=['特征重要性'],
        )

        # 3.2 相关性
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value="2、入模变量相关性", style="header_middle", align={"horizontal": "left"})
        corr_df = self.get_features_corr()
        corr_figs = plot_paths.get("feature_corr", [])
        end_row, _ = dataframe2excel(
            corr_df, writer, sheet_name=ws,
            start_row=end_row + 1,
            percent_cols=corr_df.columns.tolist(),
            index=True,
            figures=corr_figs,
            right_cols=[0],
        )

        # 3.3 入模变量有效性分析
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value="3、入模变量有效性分析", style="header_middle", align={"horizontal": "left"})

        importance = self.get_feature_importance()
        feature_list = importance.index.tolist() if not importance.empty else self.feature_names

        for i, feat in enumerate(feature_list):
            end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value=f"3.{i + 1}、{feat} 有效性分析", style="header_middle", align={"horizontal": "left"})

            bin_figs = plot_paths.get(f"feat_bin_{feat}", [])
            img_start_row = end_row + 1
            current_col = 2
            max_img_end_row = img_start_row
            for fig in bin_figs:
                try:
                    img_end_row, current_col = writer.insert_pic2sheet(ws, fig, (img_start_row, current_col), figsize=(500, 300))
                    max_img_end_row = max(max_img_end_row, img_end_row)
                except Exception:
                    pass
            if bin_figs:
                end_row = max_img_end_row

            for ds_key, ds in self._datasets.items():
                try:
                    ft = self.get_feature_bin_table(feat, ds_key, max_n_bins=n_bins, method=bin_method, margins=True)
                    ft_pct = [c for c in self._PERCENT_COLS if c in ft.columns]
                    ft_cond = [c for c in self._CONDITION_COLS if c in ft.columns]
                    if amount_col:
                        ft_amt = self.get_feature_bin_table(feat, ds_key, max_n_bins=n_bins, method=bin_method, amount_col=amount_col, margins=True)
                        end_row1, end_col1 = dataframe2excel(
                            ft, writer, sheet_name=ws,
                            title=f"{ds.label}(订单口径)", start_row=end_row + 1,
                            percent_cols=ft_pct, condition_cols=ft_cond, condition_color="F76E6C",
                        )
                        _, _ = dataframe2excel(
                            ft_amt, writer, sheet_name=ws,
                            title=f"{ds.label}(金额口径)", start_row=end_row + 1, start_col=end_col1 + 1,
                            percent_cols=ft_pct, condition_cols=ft_cond, condition_color="F76E6C",
                        )
                        end_row = end_row1
                    else:
                        end_row, _ = dataframe2excel(
                            ft, writer, sheet_name=ws,
                            title=f"{ds.label}", start_row=end_row + 1,
                            percent_cols=ft_pct, condition_cols=ft_cond, condition_color="F76E6C",
                        )
                except Exception:
                    pass

            # PSI 图表和数据表
            psi_fig_paths = plot_paths.get(f"feat_psi_{feat}", [])
            psi_df = psi_tables.get(f"feat_psi_{feat}")
            if psi_fig_paths:
                for fig_path in psi_fig_paths:
                    try:
                        end_row, _ = writer.insert_pic2sheet(ws, fig_path, (end_row + 1, 2), figsize=(500, 300))
                    except Exception:
                        pass
            if isinstance(psi_df, pd.DataFrame) and not psi_df.empty:
                end_row, _ = dataframe2excel(
                    psi_df, writer, sheet_name=ws,
                    title="PSI稳定性分析", start_row=end_row + 1,
                )

        try:
            writer.set_freeze_panes(ws, (5, 4))
        except Exception:
            pass

        # ============================================================
        # 4-稳定性分析 Sheet
        # ============================================================
        ws = writer.get_sheet_by_name("4-稳定性分析")
        end_row, _ = writer.insert_value2sheet(ws, (2, 2), value="四、模型稳定性分析", style="header_middle", end_space=(2, max_col))
        try:
            writer.insert_hyperlink2sheet(ws, (2, 2), hyperlink="#'目录'!B2")
        except Exception:
            pass

        stab_section = 1

        # 4.1 评分分布统计
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value=f"{stab_section}、评分分布统计", style="header_middle", align={"horizontal": "left"})
        score_dist_rows: List[Dict[str, Any]] = []
        for ds_key, ds in self._datasets.items():
            sc = ds.score
            row: Dict[str, Any] = {"数据集": ds.label}
            row["样本数"] = len(sc)
            row["均值"] = float(np.nanmean(sc))
            row["标准差"] = float(np.nanstd(sc))
            row["最小值"] = float(np.nanmin(sc))
            row["25%分位"] = float(np.nanpercentile(sc, 25))
            row["中位数"] = float(np.nanpercentile(sc, 50))
            row["75%分位"] = float(np.nanpercentile(sc, 75))
            row["最大值"] = float(np.nanmax(sc))
            score_dist_rows.append(row)
        score_dist_df = pd.DataFrame(score_dist_rows)
        end_row, _ = dataframe2excel(
            score_dist_df, writer, sheet_name=ws, start_row=end_row + 1,
        )
        stab_section += 1

        # 4.2 评分PSI矩阵
        if len(self._datasets) >= 2:
            end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value=f"{stab_section}、评分PSI对比矩阵", style="header_middle", align={"horizontal": "left"})
            ds_keys_list = list(self._datasets.keys())
            labels = [self._datasets[k].label for k in ds_keys_list]
            psi_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
            for i, k1 in enumerate(ds_keys_list):
                for j, k2 in enumerate(ds_keys_list):
                    if i == j:
                        psi_matrix.iloc[i, j] = 0.0
                    else:
                        try:
                            psi_matrix.iloc[i, j] = _psi(self._datasets[k1].score, self._datasets[k2].score)
                        except Exception:
                            pass
            end_row, _ = dataframe2excel(psi_matrix, writer, sheet_name=ws, start_row=end_row + 1, index=True)
            end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value="PSI参考标准：<0.1 稳定 | 0.1~0.25 略变 | >0.25 不稳定", style="middle", align={"horizontal": "left"})
            stab_section += 1

        # 4.3 评分漂移分析
        if "train" in self._datasets and len(self._datasets) >= 2:
            end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value=f"{stab_section}、评分漂移分析（vs 训练集）", style="header_middle", align={"horizontal": "left"})
            drift_rows: List[Dict[str, Any]] = []
            base_scores = self._datasets["train"].score if "train" in self._datasets else self._datasets[next(iter(self._datasets))].score
            for ds_key, ds in self._datasets.items():
                if ds_key == "train":
                    continue
                sc = ds.score
                drift = {
                    "数据集": ds.label,
                    "vs": "训练集",
                    "均值偏移": float(np.nanmean(sc) - np.nanmean(base_scores)),
                    "均值偏移%": float((np.nanmean(sc) - np.nanmean(base_scores)) / (np.nanstd(base_scores) + 1e-9)),
                    "中位数偏移": float(np.nanmedian(sc) - np.nanmedian(base_scores)),
                    "好样本(评分>600)占比": float((sc > 600).sum() / len(sc)),
                    "坏样本(评分<500)占比": float((sc < 500).sum() / len(sc)),
                }
                drift_rows.append(drift)
            if drift_rows:
                drift_df = pd.DataFrame(drift_rows)
                pct_cols = [c for c in drift_df.columns if "%" in c or "占比" in c]
                end_row, _ = dataframe2excel(
                    drift_df, writer, sheet_name=ws, start_row=end_row + 1,
                    percent_cols=pct_cols,
                )
            stab_section += 1

        # 4.4 逐特征PSI稳定性表
        if len(self._datasets) >= 2:
            end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value=f"{stab_section}、入模特征PSI稳定性", style="header_middle", align={"horizontal": "left"})
            importance = self.get_feature_importance()
            feat_list = importance.index.tolist() if not importance.empty else self.feature_names
            psi_rows: List[Dict[str, Any]] = []
            base_ds = self._datasets.get("train") or self._datasets[list(self._datasets.keys())[0]]
            other_ds_keys = [k for k in self._datasets if k != "train"]
            if not other_ds_keys:
                other_ds_keys = [k for k in self._datasets if k != list(self._datasets.keys())[0]]

            for feat in feat_list:
                row: Dict[str, Any] = {"特征": feat}
                has_psi = False
                for dk in other_ds_keys:
                    if dk in self._datasets and feat in self._datasets[dk].X.columns:
                        try:
                            psi_val = _psi(base_ds.X[feat], self._datasets[dk].X[feat])
                            row[f"PSI({self._datasets[dk].label})"] = psi_val
                            has_psi = True
                        except Exception:
                            row[f"PSI({self._datasets[dk].label})"] = np.nan
                if has_psi:
                    psi_rows.append(row)
            if psi_rows:
                psi_feat_df = pd.DataFrame(psi_rows)
                end_row, _ = dataframe2excel(
                    psi_feat_df, writer, sheet_name=ws, start_row=end_row + 1,
                )
            stab_section += 1

        # 4.5 模型评分PSI分析（不同数据集之间）
        if len(self._datasets) >= 2 and "train" in self._datasets:
            from scorecardpipeline.utils import psi_plot as _psi_plot

            end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value=f"{stab_section}、模型评分PSI分析", style="header_middle", align={"horizontal": "left"})
            stab_section += 1

            # 获取训练集评分分箱表及其分箱规则
            # 分箱规则在训练集上确定，然后应用到其他数据集，确保PSI计算时使用相同的分箱边界
            train_ds = self._datasets["train"] if "train" in self._datasets else self._datasets[next(iter(self._datasets))]
            other_ds_keys = [k for k in self._datasets if k != "train"]

            # 在训练集上确定分箱规则
            score_train, bin_rules = self.get_bin_table_rules("train", method=bin_method, max_n_bins=n_bins)

            for dk in other_ds_keys:
                test_ds = self._datasets[dk]
                label = test_ds.label

                # 使用训练集的分箱规则对其他数据集进行分箱，确保PSI计算时使用相同的分箱边界
                score_test = self.get_bin_table(dk, method=bin_method, max_n_bins=n_bins, amount_col=amount_col, margins=False, rules=bin_rules)

                # 绘制PSI图
                try:
                    p = str(Path(filepath).parent / f"{Path(filepath).stem}_assets" / f"score_psi_{dk}.png")
                    score_psi_result = _psi_plot(
                        score_train, score_test,
                        labels=["训练集", label],
                        desc=f"模型评分({label})",
                        save=p,
                        result=True,
                        plot=True,
                        figsize=(15, 8)
                    )
                    # 插入图片
                    if os.path.exists(p):
                        end_row, _ = writer.insert_pic2sheet(ws, p, (end_row + 1, 2), figsize=(500, 300))
                    else:
                        import warnings as _warnings
                        _warnings.warn(f"PSI图未生成: {p}")

                    # 插入PSI表格
                    if isinstance(score_psi_result, pd.DataFrame) and not score_psi_result.empty:
                        end_row, _ = dataframe2excel(
                            score_psi_result, writer, sheet_name=ws,
                            start_row=end_row + 1,
                            title=f"评分PSI({label} vs 训练集)"
                        )
                    else:
                        import warnings as _warnings
                        _warnings.warn(f"PSI表格为空或无效")
                except Exception as e:
                    import warnings as _warnings
                    _warnings.warn(f"评分PSI分析失败: {e}")

        # ============================================================
        # 5-模型参数 Sheet
        # ============================================================
        ws = writer.get_sheet_by_name("5-模型参数")
        end_row, _ = writer.insert_value2sheet(ws, (2, 2), value="五、模型选型及参数", style="header_middle", end_space=(2, max_col))
        try:
            writer.insert_hyperlink2sheet(ws, (2, 2), hyperlink="#'目录'!B2")
        except Exception:
            pass

        param_section = 1

        # 5.1 模型选型
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value=f"{param_section}、模型选型", style="header_middle", align={"horizontal": "left"})
        end_row, _ = writer.insert_value2sheet(ws, (end_row, 2), value=model_name, style="middle", align={"horizontal": "left"})
        param_section += 1

        # 5.2 模型参数
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value=f"{param_section}、模型参数", style="header_middle", align={"horizontal": "left"})
        params_str = ""
        if hasattr(self.model, "get_params"):
            try:
                params_str = str(self.model.get_params())
            except Exception:
                pass
        if not params_str and hasattr(self.model, "__dict__"):
            params_str = str({k: v for k, v in self.model.__dict__.items() if not k.startswith("_") and not callable(v)})
        end_row, _ = writer.insert_value2sheet(ws, (end_row, 2), value=params_str or "N/A", style="middle", align={"horizontal": "left"})
        param_section += 1

        # 5.3 入模特征列表
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value=f"{param_section}、入模特征列表", style="header_middle", align={"horizontal": "left"})
        features_df = pd.DataFrame({"序号": range(1, len(self.feature_names) + 1), "变量名": self.feature_names})
        if feature_map:
            features_df["变量含义"] = [feature_map.get(f, "") for f in self.feature_names]
        end_row, _ = dataframe2excel(features_df, writer, sheet_name=ws, start_row=end_row + 1,
                                     left_cols=["变量名", "变量含义"])
        param_section += 1

        # 5.4 评分卡专属内容
        is_scorecard = hasattr(self.model, "scorecard_points")

        if is_scorecard:
            # 逻辑回归拟合结果
            end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value=f"{param_section}、逻辑回归拟合结果", style="header_middle", align={"horizontal": "left"})
            try:
                lr_summary = self.model.pretrain_lr.summary().reset_index(names='Features')
                end_row, _ = dataframe2excel(lr_summary, writer, sheet_name=ws, start_row=end_row + 1, title="逻辑回归系数", figures=plot_paths.get("model_weights", []))
            except Exception:
                pass
            param_section += 1

            # 评分卡刻度配置
            end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value=f"{param_section}、评分卡刻度配置", style="header_middle", align={"horizontal": "left"})
            try:
                scale_df = self.model.scorecard_scale()
                end_row, _ = dataframe2excel(scale_df, writer, sheet_name=ws, start_row=end_row + 1,
                                             right_cols=["刻度项"], left_cols=["备注"])
            except Exception:
                pass
            param_section += 1

            # 评分卡
            end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value=f"{param_section}、评分卡分值表", style="header_middle", align={"horizontal": "left"})
            try:
                sc_points = self.model.scorecard_points(feature_map=feature_map)
                end_row, _ = dataframe2excel(sc_points, writer, sheet_name=ws, start_row=end_row + 1, right_cols=["对应分数", "变量分箱", "变量名称"])
            except Exception:
                pass
            param_section += 1

            # # 评分与 Odds 对照
            # end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value=f"{param_section}、评分与Odds对照表", style="header_middle", align={"horizontal": "left"})
            # try:
            #     odds_ref = self.model.score_odds_reference
            #     end_row, _ = dataframe2excel(odds_ref, writer, sheet_name=ws, start_row=end_row + 1)
            # except Exception:
            #     pass
            # param_section += 1

            # # 评分漂移分析
            # if len(self._datasets) >= 2:
            #     end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value=f"{param_section}、稳定性分析", style="header_middle", align={"horizontal": "left"})
            #     score_psi_figs = plot_paths.get("score_psi", [])
            #     if score_psi_figs:
            #         for fig_path in score_psi_figs:
            #             try:
            #                 end_row, _ = writer.insert_pic2sheet(ws, fig_path, (end_row + 1, 2), figsize=(500, 300))
            #             except Exception:
            #                 pass
            #     score_psi_df = psi_tables.get("score_psi")
            #     if isinstance(score_psi_df, pd.DataFrame) and not score_psi_df.empty:
            #         end_row, _ = dataframe2excel(score_psi_df, writer, sheet_name=ws, start_row=end_row + 1, title="评分PSI")

        # ============================================================
        # 6-模型部署需求 Sheet
        # ============================================================
        ws = writer.get_sheet_by_name("6-模型部署需求")
        end_row, _ = writer.insert_value2sheet(ws, (2, 2), value="六、模型部署需求", style="header_middle", end_space=(2, max_col))
        try:
            writer.insert_hyperlink2sheet(ws, (2, 2), hyperlink="#'目录'!B2")
        except Exception:
            pass

        # 6.1 入模变量信息
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value="1、入模变量信息", style="header_middle", align={"horizontal": "left"})
        if feature_info is not None and isinstance(feature_info, pd.DataFrame) and not feature_info.empty:
            end_row, _ = dataframe2excel(feature_info, writer, sheet_name=ws, start_row=end_row + 1)
        else:
            fi_rows: List[Dict[str, Any]] = []
            for idx, feat in enumerate(self.feature_names):
                fi_rows.append({
                    "序号": idx + 1,
                    "特征名称": feat,
                    "特征含义": (feature_map or {}).get(feat, ""),
                    "字段类型": str(self._datasets["train"] if "train" in self._datasets else self._datasets[next(iter(self._datasets))].X[feat].dtype),
                    "缺失值处理": "默认处理",
                })
            end_row, _ = dataframe2excel(pd.DataFrame(fi_rows), writer, sheet_name=ws, start_row=end_row + 1)

        # 6.2 生产订单测试用例
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value="2、生产订单测试用例", style="header_middle", align={"horizontal": "left"})
        try:
            train_ds = self._datasets["train"] if "train" in self._datasets else self._datasets[next(iter(self._datasets))]
            sample_n = min(5, len(train_ds.X))
            sample_X = train_ds.X[self.feature_names].iloc[:sample_n].copy()
            test_cases = sample_X.reset_index(drop=True)
            test_cases.insert(0, "序号", range(1, sample_n + 1))
            test_cases["模型分数"] = train_ds.score[:sample_n]
            end_row, _ = dataframe2excel(test_cases, writer, sheet_name=ws, start_row=end_row + 1)
        except Exception:
            pass

        # ============================================================
        # 保存
        # ============================================================
        writer.save(filepath)
        return filepath


# ---------------------------------------------------------------------------
# 快捷函数
# ---------------------------------------------------------------------------

def auto_model_report(
    model,
    datasets: Optional[Union[List, Dict]] = None,
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None,
    X_oot=None,
    y_oot=None,
    feature_names: Optional[List[str]] = None,
    target: Optional[Union[str, Dict]] = None,
    overdue: Optional[Union[str, List[str]]] = None,
    dpds: Optional[Union[int, float, List[Union[int, float]]]] = None,
    excel_path: Optional[str] = None,
    verbose: bool = True,
    n_bins: int = 10,
    bin_method: str = "quantile",
    amount_col: Optional[str] = None,
    date_col: Optional[str] = None,
    date_freq: Optional[str] = None,
    group_col: Optional[str] = None,
    with_plots: bool = True,
    model_name: Optional[str] = None,
    project_desc: Optional[str] = None,
    feature_map: Optional[Dict[str, str]] = None,
    feature_info: Optional[pd.DataFrame] = None,
    data_source: Optional[str] = None,
) -> QuickModelReport:
    """一键生成模型报告

    支持三种调用方式：

    1. datasets API（推荐）：传入数据集字典/列表
    2. 兼容 API：传入 X_train/y_train/X_test/y_test/X_oot/y_oot
    3. overdue/dpds 用法（自动从 X 构建二分类标签）

    示例::

        # 方式1: datasets dict
        report = auto_model_report(model, datasets={'train': train_df, 'test': test_df}, excel_path='report.xlsx')

        # 方式2: 兼容 sklearn API
        report = auto_model_report(model, X_train=X, y_train=y, X_test=X_val, y_test=y_val, excel_path='report.xlsx')

    :param model: 训练好的模型
    :param datasets: 数据集字典/列表
    :param X_train: 训练集特征
    :param y_train: 训练集标签
    :param X_test: 测试集特征
    :param y_test: 测试集标签
    :param X_oot: 跨时间验证集特征
    :param y_oot: 跨时间验证集标签
    :param feature_names: 特征名称列表
    :param target: 目标列配置
    :param overdue: 逾期列名
    :param dpds: 逾期天数阈值
    :param excel_path: Excel 保存路径
    :param verbose: 是否打印信息
    :param n_bins: 分箱数
    :param bin_method: 分箱方法
    :param amount_col: 金额字段名
    :param date_col: 日期字段名
    :param date_freq: 日期频率
    :param group_col: 分组字段名
    :param with_plots: 是否包含图表
    :param model_name: 模型名称
    :param project_desc: 项目描述
    :param feature_map: 特征含义映射
    :param feature_info: 特征信息表
    :param data_source: 数据来源
    """
    # 设置 matplotlib 中文字体和图片格式
    try:
        from scorecardpipeline.utils import init_setting
        init_setting()
    except Exception:
        pass

    report = QuickModelReport(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_oot=X_oot,
        y_oot=y_oot,
        feature_names=feature_names,
        target=target,
        datasets=datasets,
        overdue=overdue,
        dpds=dpds,
        amount_col=amount_col,
    )

    if verbose:
        report.print_report(n_bins=n_bins, amount_col=amount_col)

    if excel_path:
        report.to_excel(
            excel_path,
            n_bins=n_bins,
            bin_method=bin_method,
            amount_col=amount_col,
            date_col=date_col,
            date_freq=date_freq,
            group_col=group_col,
            with_plots=with_plots,
            model_name=model_name,
            project_desc=project_desc,
            feature_map=feature_map,
            feature_info=feature_info,
            data_source=data_source,
        )
        if verbose:
            print(f"\n报告已保存至: {excel_path}")

    return report
