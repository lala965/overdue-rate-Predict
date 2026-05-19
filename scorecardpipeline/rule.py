# -*- coding: utf-8 -*-
"""
@Time    : 2024/2/26 12:00
@Author  : itlubber
@Site    : itlubber.art
"""
import ast
import numpy as np
import numexpr as ne
from enum import Enum
from functools import reduce
from typing import List

import pandas as pd
from pandas import DataFrame
from sklearn.utils import check_array
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score

from .processing import feature_bin_stats, Combiner
from .excel_writer import dataframe2excel
from .utils import dataframe_plot


def _get_context(X, feature_names):
    return {name: X[:, i] for i, name in enumerate(feature_names)}


def _apply_expr_on_array(expr, X, feature_names):
    ctx = _get_context(X, feature_names)
    return ne.evaluate(expr, local_dict=ctx)


def get_columns_from_query(query_str):
    """获取pandas query语句使用的列

    :param query_str: pandas query 支持的查询语句
    :return: query 语句使用的列名
    """
    tree = ast.parse(query_str, mode='eval')
    columns = set()

    def visit_node(node):
        if isinstance(node, ast.Attribute):
            # 对于属性访问，递归访问其值部分
            visit_node(node.value)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            pass  # 跳过非变量名
        elif isinstance(node, ast.Name) and node.id not in {'and', 'or', 'not'}:
            columns.add(node.id)
        elif isinstance(node, ast.Call):
            # 对于函数调用，访问其函数部分
            visit_node(node.func)

    for node in ast.walk(tree):
        visit_node(node)

    return sorted(columns)


class RuleState(str, Enum):
    INITIALIZED = "initialized"
    APPLIED = "applied"


class RuleStateError(RuntimeError):
    pass


class RuleUnAppliedError(RuleStateError):
    pass


# 中級操作符
op_dict = {"GT": ">", "LT": "<", "EQ": "==", "ADD": "+", "GE": ">=", "LE": "<=", "SUBTRACT": "-", "MULTIPLY": "*", "DIVIDE": "/", "OR": "|", "AND": "&"}
# 数据类型 int, float-->float, 目前不支持 str
value_type_dict = {"int": float, "float": float, "string": str, "bool": bool}
# if_part, then_part, else_part
part_dict = ["if", "then", "else"]


# max_index: 数据的列数     feature_list: 列的名称
def json2expr(data, max_index, feature_list):
    if data.keys()._contains_("operator"):
        op = data.get("operator")
        params = data.get("params")
        if op == "FEATURE_INDEX":  # 取变量，一个值{判断变量，索引是否正常}
            feature = params[0].get("feature")
            if params[0].get("index") >= max_index:  # json中的索引异常: index >= 数据列数
                raise ValueError("index error")
            if feature not in feature_list:  # 变量异常:变量名不在数据的列名中
                raise ValueError("{} do not belong to the data ".format(feature))
            return feature
        elif op in op_dict:  # 两个值，递归
            value_list = [json2expr(params[0], max_index, feature_list), json2expr(params[1], max_index, feature_list)]
            return "(" + str(value_list[0]) + op_dict[op] + str(value_list[1]) + ")"
        else:  # op 不在op_dict报错
            raise TypeError("The operator: {} is invalid".format(op))

    if data.keys().__contains__("value"):
        value_type = data.get("value_type")
        value = data["value"]
        # 对取到值的类型做转换， 不在类型字典中的值报错
        if not value_type_dict.get(value_type):
            raise ValueError("Data type error!")
        return value_type_dict.get(value_type)(value)


class Rule:
    def __init__(self, expr):  # expr 既可以传递字符串，也可以传递dict
        """规则集，基于 pandas.DataFrame.eval 实现布尔表达式求值

        底层使用 ``DataFrame.eval()``（即 pandas.eval）执行表达式求值，支持任意类型的 DataFrame 列：
        数值型、字符串型、类别型（categorical）、日期型等均可。
        表达式最终需要返回布尔值（True / False），用于判断样本是否命中规则。

        支持的表达式语法（均为 pandas eval 支持的语法）：

        - 比较运算: ``<``, ``>``, ``<=``, ``>=``, ``==``, ``!=``
        - 算术运算: ``+``, ``-``, ``*``, ``/``, ``**``, ``%``
        - 逻辑运算: ``&``（与）、``|``（或）、``~``（非）
          - **注意**: 表达式中涉及 ``&`` / ``|`` 时需用括号包裹，如 ``(a > 0) & (b < 10)``
        - 空值判断: ``.isna()`` / ``.isnull()`` / ``.notna()`` / ``.notnull()``
        - 成员判断: ``.isin([values])``（判断是否在给定列表中）
        - 字符串方法: ``.str.contains()``、``.str.startswith()``、``.str.endswith()`` 等
        - 条件选择: ``where(cond, x, y)``

        **空值（NaN / None）行为**: 任何与空值的比较运算均返回 ``False``。例如 ``(年龄 > 30)`` 中若年龄为空，该样本不命中。若需显式判断空值，请使用 ``.isna()`` 或 ``.notna()``。

        **不支持的语法**: Python 原生的 ``in`` 运算符（请使用 ``.isin([...])``）、``is`` / ``is not`` 比较（请使用 ``==`` / ``!=``）。

        :param expr: 规则表达式字符串，类似 ``DataFrame.query`` 的语法。
            支持任意列类型（数值型、字符串型、类别型等），表达式返回值须为布尔型。
            支持中文字段名（需确保字段名不含空格和特殊字符）。
            也支持 dict 格式（JSON 规则树结构，已不推荐）。

        **参考样例**

        >>> from scorecardpipeline import *
        >>> import pandas as pd
        >>> target = "creditability"
        >>> data = germancredit()
        >>> data[target] = data[target].map({"good": 0, "bad": 1})
        >>>
        >>> # 数值型规则
        >>> rule1 = Rule("duration_in_month < 10")
        >>> rule2 = Rule("credit_amount < 500")
        >>> rule1.report(data, target=target)
        >>>
        >>> # 规则组合
        >>> (rule1 | rule2).report(data, target=target)   # 或运算
        >>> (rule1 & rule2).report(data, target=target)   # 与运算
        >>> (~rule1).report(data, target=target)           # 非运算
        >>>
        >>> # 类别型/字符串型规则（使用 .isin()）
        >>> rule3 = Rule("status_of_existing_checking_account.isin(['... < 0 DM', '0 <= ... < 200 DM'])")
        >>> rule3.report(data, target=target)
        >>>
        >>> # 显式判断空值
        >>> rule4 = Rule("(年龄.isna()) | (年龄 > 30)")
        >>> rule4.report(data, target=target)
        >>>
        >>> # 字符串包含规则
        >>> rule5 = Rule("职业.str.contains('管理|技术')")
        >>> rule5.report(data, target=target)
        >>>
        >>> # 成员判断规则
        >>> rule6 = Rule("学历.isin(['本科', '硕士', '博士'])")
        >>> rule6.report(data, target=target)
        """
        self._state = RuleState.INITIALIZED
        self.expr = expr
        self.feature_names_in_ = get_columns_from_query(self.expr)

    def __str__(self):
        return f"Rule({repr(self.expr)})"

    def __repr__(self):
        return f"Rule({repr(self.expr)})"

    def predict(self, X: DataFrame, part=""):  # dict预测对应part_dict 、字符串表达式对应"、"其他情况报错
        if not isinstance(X, DataFrame):
            raise ValueError("Rule can only predict on DataFrame.")
        
        # check_array(X, dtype=None, ensure_2d=True, force_all_finite="allow-nan")
        result = X.eval(self.expr)
        
        # feature_names = X.columns.values.tolist()  # 取数据的列名
        # X = X.select_dtypes("number") # 仅支持数值型变量
        # X = check_array(X, dtype=None, ensure_2d=True, force_all_finite="allow-nan")
        # if isinstance(self.expr, dict):  # dict部分
        #     if part not in part_dict:
        #         raise TypeError("Part : {} not in ['if','then','else']".format(part))
        #     if not self.expr[part]:  # 没有返回值的情况[]
        #         return list()
        #     dict2expr = json2expr(self.expr[part], X.shape[1], feature_names)
        #     if not isinstance(dict2expr, str):  # 返回Value (类型已经做过转换),对其扩充 --> [value] * Len(X)
        #         result = [dict2expr] * len(X)
        #     else:  # 表达式在进行计算
        #         result = ne.evaluate(dict2expr, local_dict={name: X[:, i] for i, name in enumerate(feature_names)})
        #         result = result.tolist()
        #         if not isinstance(result, list):  # result 只有一个数值时，对其扩充 --> [value] * len(X)
        #             result = [result] * len(X)
        # elif isinstance(self.expr, str):  # 字符串表达式部分
        #     if part != "":
        #         raise TypeError('The part of the expression must be ""')
        #     result = ne.evaluate(self.expr, local_dict={name: X[:, i] for i, name in enumerate(feature_names)})
        # else:
        #     raise TypeError("Rule currently only supports dict and expression")

        self.result_ = result

        return result

    def report(self, datasets: pd.DataFrame, target="target", overdue=None, dpd=None, del_grey=False, desc="", filter_cols=None, prior_rules=None, amount=None, **kwargs) -> pd.DataFrame:
        """规则效果报告表格输出

        :param datasets: 数据集，需要包含 目标变量 或 逾期天数，当不包含目标变量时，会通过逾期天数计算目标变量，同时需要传入逾期定义的DPD天数
        :param target: 目标变量名称，默认 target
        :param desc: 规则相关的描述，会出现在返回的表格当中
        :param filter_cols: 指定返回的字段列表，默认不传
        :param prior_rules: 先验规则，可以传入先验规则先筛选数据后再评估规则效果
        :param overdue: 逾期天数字段名称
        :param dpd: 逾期定义方式，逾期天数 > DPD 为 1，其他为 0，仅 overdue 字段起作用时有用
        :param del_grey: 是否删除逾期天数 (0, dpd] 的数据，仅 overdue 字段起作用时有用
        :param amount: 默认为空, 支持传入数值字段（通常为放款金额）, 在分析逾期率时，输出对应的分析结果

        :return: pd.DataFrame，规则效果评估表
        """
        return_cols = ['指标名称', "指标含义", '分箱', '样本总数', '样本占比', '好样本数', '好样本占比', '坏样本数', '坏样本占比', '坏样本率', 'LIFT值', '坏账改善']
        if desc is None or desc == "" and "指标含义" in return_cols:
            return_cols.remove("指标含义")

        rule_expr = self.expr

        def _report_one_rule(data, target, desc='', prior_rules=None):
            if prior_rules:
                prior_tables = prior_rules.report(data, target=target, desc=desc, prior_rules=None)
                prior_tables["规则分类"] = "先验规则"
                temp = data[~prior_rules.predict(data)]
                if amount is not None:
                    rule_result = pd.DataFrame({rule_expr: np.where(self.predict(temp), "命中", "未命中"), amount: temp[amount], "target": temp[target].tolist()})
                else:
                    rule_result = pd.DataFrame({rule_expr: np.where(self.predict(temp), "命中", "未命中"), "target": temp[target].tolist()})
            else:
                prior_tables = pd.DataFrame(columns=return_cols)
                if amount is not None:
                    rule_result = pd.DataFrame({rule_expr: np.where(self.predict(data), "命中", "未命中"), amount: data[amount], "target": data[target].tolist()})
                else:
                    rule_result = pd.DataFrame({rule_expr: np.where(self.predict(data), "命中", "未命中"), "target": data[target].tolist()})

            combiner = Combiner(target=target)
            combiner.load({rule_expr: [["命中"], ["未命中"]]})
            table = feature_bin_stats(rule_result, rule_expr, combiner=combiner, desc=desc, return_cols=return_cols, amount=amount, **kwargs)
            table["风险拒绝比"] = table["坏账改善"] / table["样本占比"]

            # 准确率、精确率、召回率、F1分数
            metrics = pd.DataFrame({
                "分箱": ["命中", "未命中"],
                "准确率": [accuracy_score(rule_result["target"], rule_result[rule_expr].map({"命中": 1, "未命中": 0})), accuracy_score(rule_result["target"], rule_result[rule_expr].map({"命中": 0, "未命中": 1}))],
                "精确率": [precision_score(rule_result["target"], rule_result[rule_expr].map({"命中": 1, "未命中": 0})), precision_score(rule_result["target"], rule_result[rule_expr].map({"命中": 0, "未命中": 1}))],
                "召回率": [recall_score(rule_result["target"], rule_result[rule_expr].map({"命中": 1, "未命中": 0})), recall_score(rule_result["target"], rule_result[rule_expr].map({"命中": 0, "未命中": 1}))],
                "F1分数": [f1_score(rule_result["target"], rule_result[rule_expr].map({"命中": 1, "未命中": 0})), f1_score(rule_result["target"], rule_result[rule_expr].map({"命中": 0, "未命中": 1}))],
            })
            table = table.merge(metrics, on="分箱", how="left")

            if prior_rules:
                # prior_tables.insert(loc=0, column="规则分类", value=["先验规则"] * len(prior_tables))
                table.insert(loc=0, column="规则分类", value=["验证规则"] * len(table))
                table = pd.concat([prior_tables, table]) #.set_index(["规则分类"])
            else:
                table.insert(loc=0, column="规则分类", value=["验证规则"] * len(table))

            return table

        if isinstance(del_grey, bool) and del_grey:
            merge_columns = ["规则分类", "指标名称", "分箱"]
        else:
            merge_columns = ["规则分类", "指标名称", "分箱", "样本总数", "样本占比"]

        if overdue is not None:
            if not isinstance(overdue, list):
                overdue = [overdue]

            if not isinstance(dpd, list):
                dpd = [dpd]

            for i, col in enumerate(overdue):
                for j, d in enumerate(dpd):
                    _datasets = datasets.copy()
                    _datasets[f"{col}_{d}"] = (_datasets[col] > d).astype(int)

                    if isinstance(del_grey, bool) and del_grey:
                        _datasets = _datasets.query(f"({col} > {d}) | ({col} == 0)").reset_index(drop=True)

                    if "指标含义" in return_cols:
                        merge_columns.insert(0, "指标含义")

                    if i == 0 and j == 0:
                        table = _report_one_rule(_datasets, f"{col}_{d}", desc=desc, prior_rules=prior_rules) #.rename(columns={"坏账改善": f"{col} {d}+改善"})
                        table.columns = pd.MultiIndex.from_tuples([("规则详情", c) if c in merge_columns else (f"{col} DPD{d}+", c) for c in table.columns])
                    else:
                        _table = _report_one_rule(_datasets, f"{col}_{d}", desc=desc, prior_rules=prior_rules) #.rename(columns={"坏账改善": f"{col} {d}+改善"})
                        _table.columns = pd.MultiIndex.from_tuples([("规则详情", c) if c in merge_columns else (f"{col} DPD{d}+", c) for c in _table.columns])

                        # table = table.merge(_table[["规则分类", "分箱", f"{col} {d}+改善"]], on=["规则分类", "分箱"])
                        table = table.merge(_table, on=[("规则详情", c) for c in merge_columns])
        else:
            _datasets = datasets.copy()
            table = _report_one_rule(_datasets, target, desc=desc, prior_rules=prior_rules)

        if filter_cols:
            if not isinstance(filter_cols, list):
                filter_cols = [filter_cols]
            return table[[c for c in table.columns if (isinstance(c, tuple) and c[-1] in filter_cols + merge_columns) or (not isinstance(c, tuple) and c in filter_cols + merge_columns)]]

        return table

    def result(self):
        if self._state != RuleState.APPLIED:
            raise RuleUnAppliedError("Invoke `predict` to make a rule applied.")
        return self.result_

    def plot(self, datasets: pd.DataFrame, target="target", overdue=None, dpd=None, del_grey=False, desc="", filter_cols=None, prior_rules=None, amount=None, save=None, **kwargs):
        """将规则效果报告保存为图片

        :param datasets: 数据集，需要包含 目标变量 或 逾期天数，当不包含目标变量时，会通过逾期天数计算目标变量，同时需要传入逾期定义的DPD天数
        :param target: 目标变量名称，默认 target
        :param desc: 规则相关的描述，会出现在返回的表格当中
        :param filter_cols: 指定返回的字段列表，默认不传
        :param prior_rules: 先验规则，可以传入先验规则先筛选数据后再评估规则效果
        :param overdue: 逾期天数字段名称
        :param dpd: 逾期定义方式，逾期天数 > DPD 为 1，其他为 0，仅 overdue 字段起作用时有用
        :param del_grey: 是否删除逾期天数 (0, dpd] 的数据，仅 overdue 字段起作用时有用
        :param amount: 默认为空, 支持传入数值字段（通常为放款金额）, 在分析逾期率时，输出对应的分析结果
        :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
        :param kwargs: dataframe_plot 相关的参数

        :return: Figure
        """
        report = self.report(datasets=datasets, target=target, overdue=overdue, dpd=dpd, del_grey=del_grey, desc=desc, filter_cols=filter_cols, prior_rules=prior_rules, amount=amount, **kwargs)
        return dataframe_plot(report, save=save, **kwargs)

    def __eq__(self, other):
        if not isinstance(other, Rule):
            raise TypeError(f"Input should be of type Rule, got {type(other)} instead.")
        if self._state != other._state:
            raise RuleStateError(f"Input rule should be of the same state.")
        res = self.expr == other.expr
        if self._state == RuleState.INITIALIZED:
            return res
        return res and np.all(self.result() == other.result())

    # rule combinations
    def __or__(self, other):
        if not isinstance(other, Rule):
            raise TypeError(f"Input should be of type Rule, got {type(other)} instead.")
        if self._state != other._state:
            raise RuleStateError(f"Input rule should be of the same state.")
        if isinstance(self.expr, str):
            r = Rule(f"({self.expr}) | ({other.expr})")
            if self._state == RuleState.INITIALIZED:
                return r
            r.result_ = np.logical_or(self.result(), other.result())
            r._state = RuleState.APPLIED
            return r
        elif isinstance(self.expr, dict):
            self.new_dict = {}  # 汇总成新的json
            self.new_dict["name"] = str(self.expr.get("name")) + str(other.expr.get("name"))
            self.new_dict["description"] = str(self.expr.get("description")) + " || " + str(other.expr.get("description"))
            self.new_dict["output"] = self.expr.get("output")

            # if_part
            if_dict = {}
            if_dict["value_type"] = "bool"
            if_dict["operator"] = "OR"
            if_dict["params"] = list()
            if_dict["params"].append(self.expr.get("if"))
            if_dict["params"].append(other.expr.get("if"))
            self.new_dict["if"] = if_dict

            # then_part
            then_part = {}
            if not self.expr.get("then") and not other.expr.get("then"):  # 两条规则的then都为空
                then_part = {}
            elif not self.expr.get("then"):  # 一条规则的then存在
                then_part = other.expr.get("then")
            elif not other.expr.get("then"):
                then_part = self.expr.get("then")
            else:  # 两条规则的then都存在
                if self.expr.get("then").get("value_type") != other.expr.get("then").get("value_type"):
                    raise TypeError("两个规则then_part类型要一致")
                if self.expr.get("then").get("value_type") != "bool":
                    raise TypeError("两个规则之间or运算, 类型需要设置为bool类型")
                then_part["value_type"] = "bool"
                then_part["operator"] = "OR"
                then_part["params"] = list()
                then_part["params"].append(self.expr.get("then"))
                then_part["params"].append(other.expr.get("then"))
            self.new_dict["then"] = then_part

        # else_part
        else_part = {}  # self.else或者other.else存在为空的情况
        if not self.expr.get("else") and not other.expr.get("else"):
            else_part = {}
        elif not self.expr.get("else"):  # 一条规则的then存在
            else_part = other.expr.get("else")
        elif not other.expr.get("else"):
            else_part = self.expr.get("else")
        else:
            if self.expr.get("then").get("value_type") != other.expr.get("then").get("value_type"):
                raise TypeError("两个规则else part类型要一致")
            if self.expr.get("then").get("value_type") != "bool":
                raise TypeError("两个规则之间or运算, 类型需要设置为bool类型")
            else_part["value_type"] = "bool"
            else_part["operator"] = "OR"
            else_part["params"] = list()
            else_part["params"].append(self.expr.get("else"))
            else_part["params"].append(other.expr.get("else"))
        self.new_dict["else"] = else_part

        return Rule(self.new_dict)

    def __and__(self, other):
        if not isinstance(other, Rule):
            raise TypeError(f"Input should be of type Rule, got {type(other)} instead.")
        if self._state != other._state:
            raise RuleStateError(f"Input rule should be of the same state.")
        if isinstance(self.expr, str):  # 表达式
            r = Rule(f"({self.expr}) & ({other.expr})")
            if self._state == RuleState.INITIALIZED:
                return r
            r.result_ = np.logical_and(self.result(), other.result())
            r._state = RuleState.APPLIED
            return r
        elif isinstance(self.expr, dict):  # dict
            self.new_dict = {}  # 汇总成新的json
            self.new_dict["name"] = str(self.expr.get("name")) + str(other.expr.get("name"))
            self.new_dict["description"] = str(self.expr.get("description")) + " && " + str(other.expr.get("description"))
            self.new_dict["output"] = self.expr.get("output")

            # if_part
            if_dict = {}
            if_dict["value_type"] = "bool"
            if_dict["operator"] = "AND"
            if_dict["params"] = list()
            if_dict["params"].append(self.expr.get("if"))
            if_dict["params"].append(other.expr.get("if"))
            self.new_dict["if"] = if_dict

            # then_part
            then_part = {}
            if not self.expr.get("then") and not other.expr.get("then"):  # 两条规则的then都为空
                then_part = {}
            elif not self.expr.get("then"):  # 一条规则的then存在
                then_part = other.expr.get("then")
            elif not other.expr.get("then"):
                then_part = self.expr.get("then")
            else:  # 两条规则的then都存在
                if self.expr["then"].get("value_type") != other.expr["then"].get("value_type"):
                    raise TypeError("两个规则then_part类型要一致")
                if self.expr.get("then").get("value_type") != "bool":
                    raise TypeError("两个规则之间and运算, 类型需要设置为bool类型")
                then_part["value_type"] = "bool"
                then_part["operator"] = "AND"
                then_part["params"] = list()
                then_part["params"].append(self.expr.get("then"))
                then_part["params"].append(other.expr.get("then"))
            self.new_dict["then"] = then_part

            # else_part
            else_part = {}  # self.else 或者other.else 存在为空的情况
            if not self.expr.get("else") and not other.expr.get("else"):
                else_part = {}
            elif not self.expr.get("else"):  # 一条规则的then存在
                else_part = other.expr.get("else")
            elif not other.expr.get("else"):
                else_part = self.expr.get("else")
            else:
                if self.expr.get("else").get("value_type") != other.expr.get("else").get("value_type"):
                    raise TypeError("两个规则else_part类型要一致")
                if self.expr.get("then").get("value_type") != "bool":
                    raise TypeError("两个规则之间and运算, 类型需要设置为bool类型")
                else_part["value_type"] = "bool"
                else_part["operator"] = "AND"
                else_part["params"] = list()
                else_part["params"].append(self.expr.get("else"))
                else_part["params"].append(other.expr.get("else"))
            self.new_dict["else"] = else_part

            return Rule(self.new_dict)

    def __xor__(self, other):
        if not isinstance(other, Rule):
            raise TypeError(f"Input should be of type Rule, got {type(other)} instead.")
        if self._state != other._state:
            raise RuleStateError(f"Input rule should be of the same state.")
        r = Rule(f"({self.expr}) ^ ({other.expr})")
        if self._state == RuleState.INITIALIZED:
            return r
        r.result_ = np.logical_xor(self.result(), other.result())
        r._state = RuleState.APPLIED
        return r

    def __mul__(self, other):
        return self.__or__(other)

    def __invert__(self):
        r = Rule(f"~({self.expr})")
        if self._state == RuleState.INITIALIZED:
            return r
        r.result_ = np.logical_not(self.result())
        r._state = RuleState.APPLIED
        return r

    @staticmethod
    def save(report, excel_writer, sheet_name=None, merge_column=None, percent_cols=None, condition_cols=None, custom_cols=None, custom_format="#,##0", color_cols=None, start_col=2, start_row=2, **kwargs):
        """保存规则结果至excel中，参数与 https://scorecardpipeline.itlubber.art/scorecardpipeline.html#scorecardpipeline.dataframe2excel 一致
        """
        if merge_column:
            merge_column = [c for c in report.columns if (isinstance(c, tuple) and c[-1] in merge_column) or (not isinstance(c, tuple) and c in merge_column)]

        if percent_cols:
            percent_cols = [c for c in report.columns if (isinstance(c, tuple) and c[-1] in percent_cols) or (not isinstance(c, tuple) and c in percent_cols)]

        if condition_cols:
            condition_cols = [c for c in report.columns if (isinstance(c, tuple) and c[-1] in condition_cols) or (not isinstance(c, tuple) and c in condition_cols)]
        
        if custom_cols:
            custom_cols = [c for c in report.columns if (isinstance(c, tuple) and c[-1] in custom_cols) or (not isinstance(c, tuple) and c in custom_cols)]
        
        if color_cols:
            color_cols = [c for c in report.columns if (isinstance(c, tuple) and c[-1] in color_cols) or (not isinstance(c, tuple) and c in color_cols)]
        
        end_row, end_col = dataframe2excel(report, excel_writer, sheet_name=sheet_name, merge_column=merge_column, percent_cols=percent_cols, condition_cols=condition_cols, custom_cols=custom_cols, custom_format=custom_format, color_cols=color_cols, start_col=start_col, start_row=start_row, **kwargs)
        return end_row, end_col


def ruleset_report(datasets: pd.DataFrame, rules: List[Rule], target="target", overdue=None, dpd=None, filter_cols=None, save=None, **kwargs) -> pd.DataFrame:
    """批量评估规则集效果，逐条统计每条规则的命中情况及汇总

    该函数对传入的规则集进行批量评估：
    1. 计算所有规则并集命中的汇总坏账情况
    2. 按规则传入顺序逐条统计每条规则的命中样本及剩余样本
    3. 最终输出包含原始样本、各规则逐条结果及汇总的三段式报告

    :param datasets: 需要评估规则效果的数据集
    :param rules: Rule 列表，按传入顺序逐步累加统计
    :param target: 目标变量名称，默认 "target"
    :param overdue: 逾期天数字段名称，当传入时优先于 target 使用多逾期标签分析
    :param dpd: 逾期定义方式，逾期天数 > dpd 为 1，其他为 0，需与 overdue 配合使用
    :param filter_cols: 规则详情中需要保留的列名列表，默认 None（保留全部）
    :param save: 图片保存路径，默认 None，不保存图片
    :param kwargs: 透传至 Rule.report 方法的参数，如 combiner、method 等

    :return: pd.DataFrame，规则评估报告，列结构取决于标签类型：
        - 单标签：["分箱", "样本总数", "好样本数", "坏样本数", ...] 等
        - 多逾期标签：MultiIndex 列结构，包含各逾期标签的统计指标

    **参考样例**

    >>> # 单一 target 分析
    >>> rules = [
    >>>     Rule("(年龄 < 30) & (收入 > 5000)"),
    >>>     Rule("(历史逾期次数 < 2)"),
    >>> ]
    >>> report = ruleset_report(data, rules, target="target")
    >>>
    >>> # 多逾期标签分析
    >>> report_multi = ruleset_report(
    >>>     data, rules,
    >>>     overdue=["MOB1", "MOB2"],
    >>>     dpd=[15, 7, 3],
    >>>     filter_cols=["样本总数", "好样本数", "坏样本数", "坏样本率"]
    >>> )
    """
    datasets = datasets.copy()

    feature_names_missing = set([f for rule in rules for f in rule.feature_names_in_]) - set(datasets.columns)
    if len(feature_names_missing) > 0:
        raise ValueError(f"数据集字段缺少以下字段: {feature_names_missing}")

    report = pd.DataFrame()

    table_total = reduce(lambda r1, r2: r1 | r2, rules).report(datasets, target=target, overdue=overdue, dpd=dpd, filter_cols=filter_cols, margins=True, **kwargs)

    if target is not None and (overdue is None or dpd is None):
        table_total["分箱"] = ["汇总", "剩余样本", "原始样本"]
        table_total = table_total.drop(columns=["规则分类", "指标名称"])

        report = pd.concat([report, table_total.loc[table_total["分箱"] == "原始样本", :]])

        for rule in rules:
            table = rule.report(datasets, target=target, overdue=overdue, dpd=dpd, filter_cols=filter_cols, margins=False, **kwargs)
            table["分箱"] = [rule.expr, "剩余样本"]
            table = table.drop(columns=["规则分类", "指标名称"])

            report = pd.concat([report, table])

            datasets = datasets[~rule.predict(datasets)]

        report = pd.concat([report, table_total.loc[table_total["分箱"] == "汇总", :]]).reset_index(drop=True)

    else:
        table_total[("规则详情", "分箱")] = ["汇总", "剩余样本", "原始样本"]
        table_total = table_total.drop(columns=[("规则详情", "规则分类"), ("规则详情", "指标名称")])

        report = pd.concat([report, table_total.loc[table_total[("规则详情", "分箱")] == "原始样本", :]])

        for rule in rules:
            table = rule.report(datasets, target=target, overdue=overdue, dpd=dpd, filter_cols=filter_cols, margins=False, **kwargs)
            table[("规则详情", "分箱")] = [rule.expr, "剩余样本"]
            table = table.drop(columns=[("规则详情", "规则分类"), ("规则详情", "指标名称")])

            report = pd.concat([report, table])

            datasets = datasets[~rule.predict(datasets)]

        report = pd.concat([report, table_total.loc[table_total[("规则详情", "分箱")] == "汇总", :]]).reset_index(drop=True)

    if save:
        dataframe_plot(report, save=save, **kwargs)

    return report


def bin_table_badrate_prediction(group, amount=None):
    """分箱坏账预测的聚合函数，用于 groupby 后的统计

    :param group: groupby 后的数据组，预期包含 BAD_RATE 列（分箱内坏样本率）
    :param amount: 金额字段名称，传入后按金额加权计算坏账指标

    :return: pd.Series，包含样本总数、坏样本数、坏样本率三个指标

    **参考样例**

    >>> # 样本维度（不使用金额字段）
    >>> grouped = df.groupby("分箱")
    >>> result = grouped.apply(bin_table_badrate_prediction)
    >>>
    >>> # 金额维度（使用金额字段）
    >>> result_amount = grouped.apply(bin_table_badrate_prediction, amount="放款金额")
    """
    if amount is None:
        return pd.Series(dict(
            样本总数=len(group),
            坏样本数=group["BAD_RATE"].sum(),
            坏样本率=group["BAD_RATE"].mean(),
        ))
    else:
        return pd.Series(dict(
            样本总数=group[amount].sum(),
            坏样本数=(group["BAD_RATE"] * group[amount]).sum(),
            坏样本率=(group["BAD_RATE"] * group[amount]).sum() / group[amount].sum(),
        ))


def sawpin_badrate_prediction_by_score(base: pd.DataFrame, test: pd.DataFrame, swap_in_ruleset, feature, target="target", overdue=None, dpd=None, rules={}, method="quantile", max_n_bins=10, amount=None, **kwargs):
    """SWAP IN 分析，基于 base 分箱预测 test 的逾期率

    该函数在 base 数据集上对 feature 进行分箱，映射 test 数据集的逾期预测值。
    通过 base 的分箱规则，将 test 中命中规则的样本映射到对应分箱的坏样本率。
    支持单 target 和多逾期标签（overdue+dpds 组合）分析。

    :param base: pd.DataFrame，有表现的数据集，用于建立分箱规则和计算基准坏账率
    :param test: pd.DataFrame，测试数据集（包含置入样本），需要预测风险
    :param swap_in_ruleset: Rule 或 list[Rule]，置入规则列表
    :param feature: str，特征名称，用于在 base 上建立分箱规则并映射 test 的坏样本率
    :param target: str，目标变量名称，默认 "target"
    :param overdue: list or str，逾期字段名称，当传入时优先于 target 使用多逾期标签分析
    :param dpd: list or int，逾期天数阈值，逾期天数 > dpd 为 1，需与 overdue 配合使用
    :param rules: dict，分箱规则，格式为 {feature: [bins]}，用于指定分箱切点
    :param method: str，分箱方法，默认 "quantile"，可选项参考 toad.Combiner
    :param max_n_bins: int，最大分箱数，默认 10
    :param amount: str，金额字段名称，传入后额外输出按金额加权的分析报告
    :param kwargs: 透传至 feature_bin_stats 的其他参数

    :return:
        + 样本维度报告: pd.DataFrame，包含分箱信息、样本统计、坏样本率、LIFT 等
        + 金额维度报告: pd.DataFrame，当 amount 不为空时返回，按金额加权的分析报告

    **参考样例**

    >>> # 使用示例
    >>> swap_in_ruleset = [Rule("(当前履约机构数 < 22) & (自营资质分 >= 0) & (自营资质分 < 555)")]
    >>> # 单一target分析
    >>> swap_in_data, swap_in_data_amount = sawpin_badrate_prediction_by_score(
    >>>     swap_data, swap_data, swap_in_ruleset, "自营资质分",
    >>>     target="target", amount="放款金额"
    >>> )
    >>> # 多逾期标签分析
    >>> overdue_columns = ["MOB1"]  # 根据实际数据调整
    >>> dpd_thresholds = [7, 3, 0]  # 逾期天数阈值
    >>> swap_in_data_multi, swap_in_data_amount_multi = sawpin_badrate_prediction_by_score(
    >>>     swap_data, swap_data, swap_in_ruleset, "自营资质分",
    >>>     overdue=overdue_columns, dpd=dpd_thresholds, amount="放款金额"
    >>> )
    >>> print("单一标签分析结果:")
    >>> print(swap_in_data)
    >>> print("多标签分析结果:")
    >>> print(swap_in_data_multi)
    """
    test = test.copy()

    if isinstance(swap_in_ruleset, list):
        rule_swap_in = reduce(lambda r1, r2: r1 | r2, swap_in_ruleset)
        test["SWAPIN"] = rule_swap_in.predict(test).astype(int)
    else:
        rule_swap_in = swap_in_ruleset
        test["SWAPIN"] = rule_swap_in.predict(test).astype(int)

    # 如果没有指定overdue和dpd，使用单一target进行分析
    if overdue is None or dpd is None:
        base_table, rules = feature_bin_stats(base, feature, target=target, overdue=overdue, dpd=dpd, rules=rules, method=method, max_n_bins=max_n_bins, return_rules=True, **kwargs)

        combiner = Combiner().load({feature: rules})
        test["BINS"] = combiner.transform(test[feature])
        test["BAD_RATE"] = test["BINS"].map(dict(zip(base_table.index, base_table["坏样本率"])))
        swap_in_data = test.groupby("SWAPIN").apply(lambda x: bin_table_badrate_prediction(x)).sort_index(ascending=False)
        swap_in_data.index = [rule_swap_in.expr, "原始样本"]
        swap_in_data.index.name = "规则详情"
        swap_in_data.loc["置换样本", :] = pd.Series(dict(
                样本总数=len(test),
                坏样本数=test["BAD_RATE"].sum(),
                坏样本率=test["BAD_RATE"].mean(),
            ))
        swap_in_data = swap_in_data.assign(
            样本占比=lambda x: x["样本总数"] / swap_in_data.loc["置换样本", "样本总数"],
            LIFT值=lambda x: x["坏样本率"]  / swap_in_data.loc["置换样本", "坏样本率"],
            坏账改善=lambda x: (swap_in_data.loc["置换样本", "坏样本率"] - x["坏样本率"]) / swap_in_data.loc["置换样本", "坏样本率"],
            风险拒绝比=lambda x: x["坏账改善"] / x["样本占比"],
        )

        if amount is not None:
            swap_in_data_amount = test.groupby("SWAPIN").apply(lambda x: bin_table_badrate_prediction(x, amount=amount)).sort_index(ascending=False)
            swap_in_data_amount.index = [rule_swap_in.expr, "原始样本"]
            swap_in_data_amount.index.name = "规则详情"
            swap_in_data_amount.loc["置换样本", :] = pd.Series(dict(
                    样本总数=test[amount].sum(),
                    坏样本数=(test[amount] * test["BAD_RATE"]).sum(),
                    坏样本率=(test[amount] * test["BAD_RATE"]).sum() / test[amount].sum(),
                ))
            swap_in_data_amount = swap_in_data_amount.assign(
                样本占比=lambda x: x["样本总数"] / swap_in_data_amount.loc["置换样本", "样本总数"],
                LIFT值=lambda x: x["坏样本率"]  / swap_in_data_amount.loc["置换样本", "坏样本率"],
                坏账改善=lambda x: (swap_in_data_amount.loc["置换样本", "坏样本率"] - x["坏样本率"]) / swap_in_data_amount.loc["置换样本", "坏样本率"],
                风险拒绝比=lambda x: x["坏账改善"] / x["样本占比"],
            )
        else:
            swap_in_data_amount = swap_in_data.copy()

        swap_in_data = swap_in_data[['样本总数', '样本占比', '坏样本数', '坏样本率', 'LIFT值', '坏账改善', '风险拒绝比']].sort_index(key=lambda idx: idx.map(lambda x: {"原始样本": 0, "置换样本": 2}.get(x, 1)))
        swap_in_data_amount = swap_in_data_amount[['样本总数', '样本占比', '坏样本数', '坏样本率', 'LIFT值', '坏账改善', '风险拒绝比']].sort_index(key=lambda idx: idx.map(lambda x: {"原始样本": 0, "置换样本": 2}.get(x, 1)))

        return swap_in_data, swap_in_data_amount

    else:
        # 处理多个逾期标签的情况
        merge_columns = ["样本总数", "样本占比"]
        swap_in_data_final, swap_in_data_amount_final = None, None

        if not isinstance(overdue, list):
            overdue = [overdue]

        if not isinstance(dpd, list):
            dpd = [dpd]

        amount_feature = [amount] if amount is not None else []

        # 遍历所有逾期标签组合
        for i, col in enumerate(overdue):
            for j, d in enumerate(dpd):
                _target = f"{col} {d}+"

                # 在base数据上创建新的目标变量
                _datasets = base[list(set([feature] + amount_feature + [col]))].copy()
                _datasets[_target] = (_datasets[col] > d).astype(int)

                # 递归调用处理当前逾期标签
                _swap_in_data, _swap_in_data_amount = sawpin_badrate_prediction_by_score(_datasets, test, swap_in_ruleset, feature, target=_target, overdue=None, dpd=None, rules=rules, amount=amount, **kwargs)

                # 重命名列名为多级索引，确保规则详情相关字段在最前面
                _swap_in_data.columns = pd.MultiIndex.from_tuples(
                    [("规则详情", c) if c in merge_columns else (_target, c) for c in _swap_in_data.columns]
                )
                _swap_in_data_amount.columns = pd.MultiIndex.from_tuples(
                    [("规则详情", c) if c in merge_columns else (_target, c) for c in _swap_in_data_amount.columns]
                )

                # 合并结果
                if swap_in_data_final is None:
                    swap_in_data_final = _swap_in_data
                    swap_in_data_amount_final = _swap_in_data_amount
                else:
                    # 确保索引一致后合并
                    swap_in_data_final = pd.concat([swap_in_data_final, _swap_in_data.drop(columns=[("规则详情", c) for c in merge_columns])], axis=1)
                    swap_in_data_amount_final = pd.concat([swap_in_data_amount_final, _swap_in_data_amount.drop(columns=[("规则详情", c) for c in merge_columns])], axis=1)

        # 重新排列列的顺序，确保规则详情相关字段在最前面
        def reorder_columns(df):
            # 提取规则详情相关的列
            rule_columns = [col for col in df.columns if col[0] == "规则详情"]
            # 提取其他列
            other_columns = [col for col in df.columns if col[0] != "规则详情"]
            # 重新组合：规则详情列在前，其他列在后
            reordered_columns = rule_columns + other_columns
            return df[reordered_columns]

        if swap_in_data_final is not None:
            swap_in_data_final = reorder_columns(swap_in_data_final)
            swap_in_data_amount_final = reorder_columns(swap_in_data_amount_final)

        return swap_in_data_final, swap_in_data_amount_final


swapout_report = ruleset_report
swapout_report.__doc__ = ruleset_report.__doc__


def swapin_report(base, test, swap_in_ruleset, feature, target="target", overdue=None, dpd=None, amount=None, origin_mask=None, tmp_col="swapinstep", **kwargs):
    """SWAPIN 报告，逐步累加置入样本，评估每步的坏账改善效果

    该函数按顺序将 swap_in_ruleset 中的每条规则逐步置入样本，逐步累加计算每步置入后的坏账率改善效果。
    通过逐步累加的方式，可以观察每条规则对整体坏账改善的边际贡献。

    :param base: pd.DataFrame, 有表现的数据集，用于建立分箱规则和计算基准坏账率
    :param test: pd.DataFrame, 测试数据集（包含置入样本），需要预测风险
    :param swap_in_ruleset: Rule 或 list[Rule], 置入规则列表，按传入顺序逐步置入样本
    :param feature: str, 特征名称，用于在 base 上建立分箱规则并映射 test 的坏样本率
    :param target: str, 目标变量名称，默认 "target"
    :param overdue: list or str, 逾期天数字段名称，当传入时优先于 target 使用多逾期标签分析
    :param dpd: list or int, 逾期定义方式，逾期天数 > dpd 为 1，其他为 0，需与 overdue 配合使用
    :param amount: str, 金额字段名称，传入后额外输出按金额加权的分析报告
    :param origin_mask: pd.Series(bool), 自定义原始样本掩码，True 表示该样本视为原始样本，默认 None（未命中任意规则的样本为原始样本）
    :param tmp_col: str, 内部使用的临时列名，默认 "swapinstep"，避免与数据列名冲突
    :param kwargs: sawpin_badrate_prediction_by_score 的其他参数，如 method、max_n_bins 等

    :return:
        当 amount=None 时返回 (swapin_data,)，否则返回 (swapin_data, swapin_data_amount)：

        - swapin_data: pd.DataFrame，每行对应一条规则的置入结果，
          列包含样本总数、样本占比、坏样本数、坏样本率、LIFT值、坏账改善、风险拒绝比
        - swapin_data_amount: pd.DataFrame，按金额加权的分析报告，结构同上

    **参考样例**

    >>> # 单一 target 分析
    >>> swap_in_ruleset = [
    >>>     Rule("(当前履约机构数 < 22) & (自营资质分 >= 0) & (自营资质分 < 555)"),
    >>>     Rule("(近6个月审批查询次数 < 3)"),
    >>> ]
    >>> swapin_data, swapin_data_amount = swapin_report(
    >>>     base=swap_data,
    >>>     test=swap_data,
    >>>     swap_in_ruleset=swap_in_ruleset,
    >>>     feature="自营资质分",
    >>>     target="target",
    >>>     amount="放款金额"
    >>> )
    >>> # 多逾期标签分析
    >>> overdue_columns = ["MOB1"]  # 根据实际数据调整
    >>> dpd_thresholds = [7, 3, 0]  # 逾期天数阈值
    >>> swapin_data_multi, swapin_data_amount_multi = swapin_report(
    >>>     base=swap_data,
    >>>     test=swap_data,
    >>>     swap_in_ruleset=swap_in_ruleset,
    >>>     feature="自营资质分",
    >>>     overdue=overdue_columns,
    >>>     dpd=dpd_thresholds,
    >>>     amount="放款金额"
    >>> )
    """
    rules = swap_in_ruleset if isinstance(swap_in_ruleset, (list, tuple)) else [swap_in_ruleset]

    if len(rules) == 0:
        return pd.DataFrame(), pd.DataFrame()

    def _cols(x):
        if x is None:
            return set()
        if isinstance(x, str):
            return {x}
        try:
            return set(x)
        except TypeError:
            return {x}

    def _predict(rule, df):
        if df.empty:
            return pd.Series(False, index=df.index)

        r = rule.predict(df)
        return (
            r if isinstance(r, pd.Series)
            else pd.Series(r, index=df.index)
        ).fillna(False).astype(bool)

    def _check_missing(df, cols, name):
        miss = set(cols) - set(df.columns)
        if miss:
            raise ValueError(f"{name} missing columns: {sorted(miss)}")

    def _safe_div(num, den):
        return num / den if pd.notna(den) and den != 0 else pd.NA

    def _recalc_metrics(df):
        if df.empty or "原始样本" not in df.index:
            return df

        df = df.copy()
        origin_pos = list(df.index).index("原始样本")

        if isinstance(df.columns, pd.MultiIndex):
            n_col = ("规则详情", "样本总数")
            p_col = ("规则详情", "样本占比")

            if n_col not in df.columns:
                return df

            origin_n = df.iloc[origin_pos][n_col]

            if p_col in df.columns:
                df[p_col] = _safe_div(df[n_col], origin_n)

            for top in df.columns.get_level_values(0).unique():
                if top == "规则详情":
                    continue

                bad_col = (top, "坏样本数")
                br_col = (top, "坏样本率")
                lift_col = (top, "LIFT值")
                improve_col = (top, "坏账改善")
                reject_col = (top, "风险拒绝比")

                if bad_col not in df.columns or br_col not in df.columns:
                    continue

                origin_bad = df.iloc[origin_pos][bad_col]
                origin_br = df.iloc[origin_pos][br_col]

                if lift_col in df.columns:
                    df[lift_col] = _safe_div(df[br_col], origin_br)

                if improve_col in df.columns:
                    den = df[n_col] + origin_n
                    merged_br = (df[bad_col] + origin_bad) / den.mask(den.eq(0))
                    df[improve_col] = _safe_div(origin_br - merged_br, origin_br)

                if reject_col in df.columns and improve_col in df.columns and p_col in df.columns:
                    df[reject_col] = df[improve_col] / df[p_col].mask(df[p_col].eq(0))

        else:
            origin_n = df.iloc[origin_pos]["样本总数"]
            origin_bad = df.iloc[origin_pos]["坏样本数"]
            origin_br = df.iloc[origin_pos]["坏样本率"]

            df["样本占比"] = _safe_div(df["样本总数"], origin_n)
            df["LIFT值"] = _safe_div(df["坏样本率"], origin_br)

            den = df["样本总数"] + origin_n
            merged_br = (df["坏样本数"] + origin_bad) / den.mask(den.eq(0))
            df["坏账改善"] = _safe_div(origin_br - merged_br, origin_br)

            df["风险拒绝比"] = df["坏账改善"] / df["样本占比"].mask(df["样本占比"].eq(0))

        return df

    def _rename_last_swap_sample(df):
        if df.empty or "置换样本" not in df.index:
            return df

        df = df.copy()
        idx = list(df.index)
        last_pos = len(idx) - 1 - idx[::-1].index("置换样本")
        idx[last_pos] = "通过样本"
        df.index = idx
        df.index.name = "规则详情"

        return df

    # base：只检查 target / overdue / amount
    base_cols = _cols(amount)
    if overdue is not None and dpd is not None:
        base_cols |= _cols(overdue)
    else:
        base_cols |= _cols(target)

    # test：只检查 amount / rule 入模变量
    rule_cols = set()
    for rule in rules:
        rule_cols |= _cols(getattr(rule, "feature_names_in_", []))

    test_cols = _cols(amount) | rule_cols

    _check_missing(base, base_cols, "base")
    _check_missing(test, test_cols, "test")

    if tmp_col in base.columns or tmp_col in test.columns:
        raise ValueError(f"{tmp_col} already exists in base/test.")

    # 关键修复：
    # 先在同一个 test 全量样本上计算每条规则命中结果；
    # 原始样本 = 未命中任意规则的样本，和规则顺序无关。
    rule_masks = [_predict(rule, test) for rule in rules]

    if origin_mask is None:
        hit_any = reduce(lambda x, y: x | y, rule_masks)
        origin_mask = ~hit_any
        candidate_mask = hit_any
    else:
        origin_mask = pd.Series(origin_mask, index=test.index).fillna(False).astype(bool)
        candidate_mask = ~origin_mask
        rule_masks = [m & candidate_mask for m in rule_masks]

    used = pd.Series(False, index=test.index)

    report = []
    report_amount = []

    for rule, rule_mask in zip(rules, rule_masks):
        hit = rule_mask & ~used

        if not hit.any():
            continue

        current_mask = origin_mask | used

        step_test = pd.concat([
            test.loc[current_mask].assign(**{tmp_col: 0}),
            test.loc[hit].assign(**{tmp_col: 1}),
        ], axis=0)

        step_rule = Rule(f"{tmp_col} == 1")

        r, ra = sawpin_badrate_prediction_by_score(
            base=base,
            test=step_test,
            swap_in_ruleset=step_rule,
            feature=feature,
            target=target,
            overdue=overdue,
            dpd=dpd,
            amount=amount,
            **kwargs
        )

        r = r.rename(index={step_rule.expr: rule.expr})
        ra = ra.rename(index={step_rule.expr: rule.expr})

        if report:
            r = r.drop(index="原始样本", errors="ignore")
            ra = ra.drop(index="原始样本", errors="ignore")

        report.append(r)
        report_amount.append(ra)

        used = used | hit

    report = pd.concat(report, axis=0) if report else pd.DataFrame()
    report_amount = pd.concat(report_amount, axis=0) if report_amount else pd.DataFrame()

    report = _rename_last_swap_sample(_recalc_metrics(report))
    report_amount = _rename_last_swap_sample(_recalc_metrics(report_amount))

    return report, report_amount
