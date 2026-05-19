# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/10 10:28
@Author  : itlubber
@Site    : itlubber.art
"""
import numpy as np
import numexpr as ne
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


class NumExprDerive(BaseEstimator, TransformerMixin):
    """基于 numexpr 表达式的特征派生器

    通过传入 (新特征名, 表达式) 元组列表，使用 numexpr 引擎进行向量化计算，快速派生新特征。
    支持 pandas.DataFrame 和 numpy.ndarray 两种输入模式，fit 为空操作（无监督信息），
    符合 sklearn Transformer 规范，可直接放入 Pipeline。

    **支持的表达式语法**

    numexpr 支持丰富的数学和函数操作，包括但不限于：
    - 算术运算: ``+``, ``-``, ``*``, ``/``, ``**``, ``%``
    - 比较运算: ``<``, ``>``, ``<=``, ``>=``, ``==``, ``!=``
    - 逻辑运算: ``&``, ``|``, ``~``（注意表达式中需加括号，如 ``(a > 0) & (b < 10)``）
    - 条件函数: ``where(condition, x, y)``（条件为 True 取 x，否则取 y）
    - 数学函数: ``sin``, ``cos``, ``tan``, ``abs``, ``log``, ``exp``, ``sqrt`` 等
    - 聚合函数: ``sum``, ``mean``, ``max``, ``min``

    **参考样例**

    >>> import pandas as pd
    >>> import numpy as np
    >>> from scorecardpipeline.feature_engineering import NumExprDerive
    >>>
    >>> X = pd.DataFrame({
    >>>     "f0": [2, 1.0, 3],
    >>>     "f1": [1, 2, 3],
    >>>     "f2": [2, 3, 4],
    >>>     "f3": [2.1, 1.4, -6.2]
    >>> })
    >>>
    >>> # 派生新特征
    >>> fd = NumExprDerive(derivings=[
    >>>     ("f4", "where(f1>1, 0, 1)"),      # 条件派生
    >>>     ("f5", "f1+f2"),                  # 加法
    >>>     ("f6", "sin(f3)"),                # 数学函数
    >>>     ("f7", "abs(f3)"),                # 绝对值
    >>> ])
    >>> X_new = fd.fit_transform(X)
    >>>
    >>> # 在 Pipeline 中使用
    >>> from sklearn.pipeline import Pipeline
    >>> pipe = Pipeline([
    >>>     ("derive", NumExprDerive(derivings=[("ratio", "f1/(f2+1)")])),
    >>>     ("model", SomeModel())
    >>> ])
    """
    def __init__(self, derivings=None):
        """
        :param derivings: list，每个元素为 (新特征名, 表达式) 元组，如 [("f4", "where(f1>1, 0, 1)")]
        """
        self.derivings = derivings
        self.fitted_ = False

    def __sklearn_is_fitted__(self):
        """sklearn 检查是否已拟合"""
        return self.fitted_

    def fit(self, X, y=None):
        self._check_keywords()
        self._validate_data(X, dtype=None, ensure_2d=True, force_all_finite=False)
        self.fitted_ = True
        return self

    def _check_keywords(self):
        derivings = self.derivings
        if derivings is None:
            raise ValueError("Deriving rules should not be empty!")
        if not isinstance(derivings, list):
            raise ValueError("Deriving rules should be a list!")
        for i, entry in enumerate(derivings):
            if not isinstance(entry, tuple):
                raise ValueError("The {}-th deriving rule should be a tuple!".format(i))
            if len(entry) != 2:
                raise ValueError("The f}-th deriving rule is not a two-element (drived_name, expression) tuple!".format(i))
            name, expr = entry
            if not isinstance(name, str) or not isinstance(expr, str):
                raise ValueError("The {}-th deriving rule is not a two-string tuple!".format(i))

    @staticmethod
    def _get_context(X, feature_names=None):
        if feature_names is not None:
            return {name: X[:, i] for i, name in enumerate(feature_names)}
        return {"f%d" % i: X[:, i] for i in range(X.shape[1])}

    def _transform_frame(self, X):
        feature_names = X.columns.tolist()
        self.features_names = feature_names
        index = X.index
        X = self._validate_data(X, dtype="numeric", ensure_2d=True, force_all_finite=False)
        context = self._get_context(X, feature_names=feature_names)
        n_derived = len(self.derivings)
        X_derived = np.empty((X.shape[0], n_derived), dtype=np.float64)
        derived_names = []
        for i, (name, expr) in enumerate(self.derivings):
            derived_names.append(name)
            X_derived[:, i] = ne.evaluate(expr, local_dict=context)
        data = np.hstack((X, X_derived))
        columns = feature_names + derived_names
        return DataFrame(data=data, columns=columns, index=index)

    def _transform_ndarray(self, X):
        X = self._validate_data(X, dtype="numeric", ensure_2d=True, force_all_finite=False)
        context = self._get_context(X, feature_names=None)
        n_derived = len(self.derivings)
        X_derived = np.empty((X.shape[0], n_derived), dtype=np.float64)
        derived_names = []
        for i, (name, expr) in enumerate(self.derivings):
            derived_names.append(name)
            X_derived[:, i] = ne.evaluate(expr, local_dict=context)
        return np.hstack((X, X_derived))

    def transform(self, X):
        if isinstance(X, DataFrame):
            return self._transform_frame(X)
        return self._transform_ndarray(X)

    def _more_tags(self):
        return {
            "X_types": ["2darray"],
            "allow_nan": True,
        }
