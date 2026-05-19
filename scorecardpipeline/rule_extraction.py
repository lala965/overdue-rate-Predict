# -*- coding: utf-8 -*-
"""
@Time    : 2024/2/29 13:29
@Author  : itlubber
@Site    : itlubber.art
"""
import warnings
import os
import re
import graphviz
import dtreeviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from IPython.display import display
from openpyxl.worksheet.worksheet import Worksheet

import category_encoders as ce
from optbinning import OptimalBinning
from sklearn.tree import DecisionTreeClassifier

from .rule import Rule
from .utils import init_setting
from .excel_writer import ExcelWriter, dataframe2excel


class DecisionTreeRuleExtractor:
    """循环决策树规则挖掘器

    循环训练多棵决策树，每次训练后剔除特征重要性最高的特征，
    挖掘满足 LIFT 条件的特征组合策略。支持将规则报告输出至 Excel。

    **核心方法**

    - ``fit()``: 循环训练决策树，挖掘高 LIFT 组合策略
    - ``transform()``: 在新数据集上评估已有规则的效果
    - ``report()``: 生成组合策略报告并写入 Excel

    **参考样例**

    >>> from scorecardpipeline.rule_extraction import DecisionTreeRuleExtractor
    >>> extractor = DecisionTreeRuleExtractor(target="target", max_iter=10)
    >>> extractor.fit(df, max_depth=2, lift=1.5)
    >>> extractor.report(save="rules_report.xlsx")
    """

    def __init__(self, target="target", labels=["positive", "negative"], feature_map={}, nan=-1., max_iter=128, writer=None, seed=None, theme_color="2639E9", decimal=4):
        """决策树自动规则挖掘工具包

        :param target: 数据集中好坏样本标签列名称，默认 target
        :param labels: 好坏样本标签名称，传入一个长度为2的列表，第0个元素为好样本标签，第1个元素为坏样本标签，默认 ["positive", "negative"]
        :param feature_map: 变量名称及其含义，在后续输出报告和策略信息时增加可读性，默认 {}
        :param nan: 在决策树策略挖掘时，默认空值填充的值，默认 -1
        :param max_iter: 最多支持在数据集上训练多少颗树模型，每次生成一棵树后，会剔除特征重要性最高的特征后，再生成树，默认 128
        :param writer: 在之前程序运行时生成的 ExcelWriter，可以支持传入一个已有的writer，后续所有内容将保存至该workbook中，默认 None
        :param seed: 随机种子，保证结果可复现使用，默认为 None
        :param theme_color: 主题色，默认 2639E9 克莱因蓝，可设置位其他颜色
        :param decimal: 精度，决策树分裂节点阈值的精度范围，默认 4，即保留4位小数
        """
        self.decimal = decimal
        self.seed = seed
        self.nan = nan
        self.target = target
        self.labels = labels
        self.theme_color = theme_color
        self.feature_map = feature_map
        self.decision_trees = []
        self.max_iter = max_iter
        self.target_enc = None
        self.feature_names = None
        self.dt_rules = pd.DataFrame()
        self.end_row = 2
        self.start_col = 2
        self.describe_columns = ["组合策略", "命中数", "命中率", "好样本数", "好样本占比", "坏样本数", "坏样本占比", "坏样本率", "LIFT值", "坏账改善", "准确率", "精确率", "召回率", "F1分数", "样本整体坏率"]

        init_setting()

        if writer:
            self.writer = writer
        else:
            self.writer = ExcelWriter(theme_color=self.theme_color)

    def encode_cat_features(self, X, y):
        """对类别型特征进行 Target Encoding

        使用 category_encoders.TargetEncoder 对类别特征进行编码，
        编码值基于目标变量的条件均值。编码映射保存在 self.target_enc 中。

        :param X: 原始特征 DataFrame
        :param y: 目标变量（pd.Series）
        :return: pd.DataFrame，编码后的特征
        """
        cat_features = list(set(X.select_dtypes(include=[object, pd.CategoricalDtype]).columns))
        cat_features_index = [i for i, f in enumerate(X.columns) if f in cat_features]

        if len(cat_features) > 0:
            if self.target_enc is None:
                self.target_enc = ce.TargetEncoder(cols=cat_features)
                self.target_enc.fit(X[cat_features], y)
                self.target_enc.target_mapping = {}
                X_TE = X.join(self.target_enc.transform(X[cat_features]).add_suffix('_target'))
                for col in cat_features:
                    mapping = X_TE[[col, f"{col}_target"]].drop_duplicates()
                    self.target_enc.target_mapping[col] = dict(zip(mapping[col], mapping[f"{col}_target"]))
            else:
                X_TE = X.join(self.target_enc.transform(X[cat_features]).add_suffix('_target'))

            X_TE = X_TE.drop(columns=cat_features)
            return X_TE.rename(columns={f"{c}_target": c for c in cat_features})
        else:
            return X

    def get_dt_rules(self, tree):
        """从训练好的决策树中提取所有叶子节点的路径规则

        递归遍历决策树的每个节点，生成形如 ``"feature <= threshold"`` 或
        ``"feature > threshold"`` 的规则表达式，最终返回每个叶子节点对应的组合规则。

        :param tree: 训练好的 DecisionTreeClassifier 模型
        :return: list[Rule]，每个叶子节点对应的 Rule 列表
        """
        rules = dict()

        def recurse(node=0, parent=None):  # 搜每个节点的规则
            if node == 0 or tree.tree_.children_left[node] != -1:  # 非叶子节点,搜索每个节点的规则
                name = tree.feature_names_in_[tree.tree_.feature[node]]
                threshold = np.round(tree.tree_.threshold[node], self.decimal)
                if parent:
                    recurse(tree.tree_.children_left[node], parent & Rule("{} <= {}".format(name, threshold)))
                    recurse(tree.tree_.children_right[node], parent & Rule("{} > {}".format(name, threshold)))
                else:
                    recurse(tree.tree_.children_left[node], Rule("{} <= {}".format(name, threshold)))
                    recurse(tree.tree_.children_right[node], Rule("{} > {}".format(name, threshold)))
            else:
                rules[node] = parent

        recurse()

        return list(rules.values())

    def select_dt_rules(self, decision_tree, x, y, lift=0., max_samples=1., save=None, verbose=False, drop=False):
        """评估并筛选决策树的叶子节点规则

        从决策树中提取所有规则，过滤出 LIFT >= lift 且命中率 <= max_samples 的策略，
        绘制决策树可视化图（可选），并返回规则评估报告。

        :param decision_tree: 训练好的 DecisionTreeClassifier 模型
        :param x: 特征数据
        :param y: 目标变量
        :param lift: LIFT 阈值，默认 0
        :param max_samples: 最大样本占比阈值，默认 1.0
        :param save: 决策树图片保存路径，默认 None
        :param verbose: 是否打印报告，默认 False
        :param drop: 是否返回待剔除特征，默认 False
        :return: pd.DataFrame，规则评估报告；str/int，特征名或规则数
        """
        rules = self.get_dt_rules(decision_tree)

        rules_reports = pd.DataFrame()
        for rule in rules:
            rules_reports = pd.concat([rules_reports, rule.report(x.join(y), target=y.name).query("分箱 == '命中'")])

        rules_reports = rules_reports.rename(columns={"指标名称": "组合策略", "样本总数": "命中数", "样本占比": "命中率"}).drop(columns=["分箱"])
        rules_reports["样本整体坏率"] = round(y.mean(), self.decimal)
        rules_reports = rules_reports.query(f"LIFT值 >= {lift} & 命中率 <= {max_samples}").reset_index(drop=True)

        if len(rules_reports) > 0:
            try:
                if verbose:
                    if self.feature_map is not None and len(self.feature_map) > 0:
                        display(rules_reports.replace(self.feature_map, regex=True))
                    else:
                        display(rules_reports)

                viz_model = dtreeviz.model(decision_tree,
                                           X_train=x,
                                           y_train=y,
                                           feature_names=decision_tree.feature_names_in_,
                                           target_name=self.target,
                                           class_names=self.labels,
                                           )

                # font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matplot_chinese.ttf')
                # font_manager.fontManager.addfont(font_path)
                # plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
                # plt.rcParams['axes.unicode_minus'] = False

                decision_tree_viz = viz_model.view(
                    scale=1.5,
                    orientation='LR',
                    colors={
                        "classes": [None, None, ["#2639E9", "#F76E6C"], ["#2639E9", "#F76E6C", "#FE7715", "#FFFFFF"]],
                        "arrow": "#2639E9",
                        'text_wedge': "#F76E6C",
                        "pie": "#2639E9",
                        "tile_alpha": 1,
                        "legend_edge": "#FFFFFF",
                    },
                    ticks_fontsize=10,
                    label_fontsize=10,
                    fontname=plt.rcParams['font.family'],
                )

                if verbose:
                    display(decision_tree_viz)
                if save:
                    if os.path.dirname(save) and not os.path.exists(os.path.dirname(save)):
                        os.makedirs(os.path.dirname(save))

                    try:
                        decision_tree_viz.save("combine_rules_cache.svg")
                    except graphviz.backend.execute.ExecutableNotFound:
                        print("请确保您已安装 graphviz 程序并且正确配置了 PATH 路径。可参考: https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-your-systems-path-aft")

                    # 尝试使用 CairoSVG 或 svglib+reportlab 将 SVG 转换为 PNG
                    _svg_to_png_success = False
                    try:
                        import cairosvg
                        cairosvg.svg2png(url="combine_rules_cache.svg", write_to=save, dpi=240)
                        _svg_to_png_success = True
                    except ImportError:
                        pass
                    except Exception:
                        pass

                    if not _svg_to_png_success:
                        try:
                            from reportlab.graphics import renderPDF
                            from svglib.svglib import svg2rlg
                            drawing = svg2rlg("combine_rules_cache.svg")
                            if drawing is not None:
                                renderPDF.drawToFile(drawing, save, dpi=240, fmt="PNG")
                                _svg_to_png_success = True
                            else:
                                raise ImportError("svglib failed to parse SVG")
                        except ImportError:
                            pass
                        except Exception:
                            pass

                    if not _svg_to_png_success:
                        print("警告: 保存决策树图片失败。如需保存 PNG 格式图片，请安装可选依赖:")
                        print("  方式1 (推荐): pip install CairoSVG>=2.7.0")
                        print("    - Linux: sudo apt-get install libcairo2-dev")
                        print("    - macOS: brew install cairo")
                        print("  方式2: pip install reportlab svglib>=1.5.0")
                        print("  或一次性安装所有可选依赖: pip install scorecardpipeline[all]")

            except AttributeError:
                print("请检查 dtreeviz、graphviz 等依赖库是否正确安装")
            except:
                print("请检查 dtreeviz、graphviz 等依赖库是否正确安装")

        if os.path.isfile("combine_rules_cache.svg"):
            os.remove("combine_rules_cache.svg")

        if os.path.isfile("combine_rules_cache"):
            os.remove("combine_rules_cache")

        if drop:
            if len(rules_reports) > 0:
                return rules_reports, decision_tree.feature_names_in_[list(decision_tree.feature_importances_).index(max(decision_tree.feature_importances_))], len(rules_reports)
            else:
                return rules_reports, decision_tree.feature_names_in_[list(decision_tree.feature_importances_).index(min(decision_tree.feature_importances_))], len(rules_reports)
        else:
            return rules_reports, len(rules_reports)

    def query_dt_rules(self, x, y, parsed_rules=None):
        """在新数据集上评估已有规则的效果

        将已挖掘的规则应用于新数据集，计算每条规则的命中情况及坏样本率、LIFT 等指标。

        :param x: 特征数据
        :param y: 目标变量
        :param parsed_rules: 已解析的规则 DataFrame 或 Rule 列表，默认 None
        :return: pd.DataFrame，各规则的命中评估报告
        """
        if isinstance(parsed_rules, pd.DataFrame):
            parsed_rules = [Rule(r) for r in parsed_rules["组合策略"].unique()]

        rules_reports = pd.DataFrame()
        for rule in parsed_rules:
            rules_reports = pd.concat([rules_reports, rule.report(x.join(y), target=y.name).query("分箱 == '命中'")])

        rules_reports = rules_reports.rename(columns={"指标名称": "组合策略", "样本总数": "命中数", "样本占比": "命中率"}).drop(columns=["分箱"])
        rules_reports["样本整体坏率"] = round(y.mean(), self.decimal)

        return rules_reports

    def insert_dt_rules(self, parsed_rules, end_row, start_col, save=None, sheet=None, figsize=(500, 350)):
        """将规则报告写入 Excel 工作表

        将规则评估报告 DataFrame 写入指定的 Excel sheet，并可选插入决策树图片。

        :param parsed_rules: 规则评估报告 DataFrame
        :param end_row: 起始写入行号
        :param start_col: 起始写入列号
        :param save: 决策树图片保存路径，默认 None
        :param sheet: 工作表名称，默认 None（使用默认 sheet）
        :param figsize: 图片尺寸，默认 (500, 350)
        :return: tuple(int, int)，更新后的结束行和结束列
        """
        if isinstance(sheet, Worksheet):
            worksheet = sheet
        else:
            worksheet = self.writer.get_sheet_by_name(sheet or "决策树组合策略挖掘")

        end_row, end_col = dataframe2excel(parsed_rules, self.writer, sheet_name=worksheet, start_row=end_row + 1, start_col=start_col, percent_cols=['好样本占比', '坏样本占比', '命中率', '坏样本率', '样本整体坏率', 'LIFT值', '坏账改善', '准确率', '精确率', '召回率', 'F1分数'], condition_cols=["坏样本率", "LIFT值"])

        if save is not None and os.path.isfile(save):
            end_row, end_col = self.writer.insert_pic2sheet(worksheet, save, (end_row + 1, start_col), figsize=figsize)

        return end_row, end_col

    def fit(self, x, y=None, max_depth=2, lift=0., max_samples=1., min_score=None, verbose=False, *args, **kwargs):
        """组合策略挖掘

        :param x: 包含标签的数据集
        :param max_depth: 决策树最大深度，即最多组合的特征个数，默认 2
        :param lift: 组合策略最小的lift值，默认 0.，即全部组合策略
        :param max_samples: 每条组合策略的最大样本占比，默认 1.0，即全部组合策略
        :param min_score: 决策树拟合时最小的auc，如果不满足则停止后续生成决策树
        :param verbose: 是否调试模式，仅在 jupyter 环境有效
        :param kwargs: DecisionTreeClassifier 参数，参考 https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        """
        worksheet = self.writer.get_sheet_by_name("策略详情")

        y = x[self.target]
        X_TE = self.encode_cat_features(x.drop(columns=[self.target]), y)
        X_TE = X_TE.fillna(self.nan)

        self.feature_names = list(X_TE.columns)

        for i in range(self.max_iter):
            decision_tree = DecisionTreeClassifier(max_depth=max_depth, *args, **kwargs)
            decision_tree = decision_tree.fit(X_TE, y)

            if (min_score is not None and decision_tree.score(X_TE, y) < min_score) or len(X_TE.columns) < max_depth:
                break

            try:
                parsed_rules, remove, total_rules = self.select_dt_rules(decision_tree, X_TE, y, lift=lift, max_samples=max_samples, verbose=verbose, save=f"model_report/auto_mining_rules/combiner_rules_{i}.png", drop=True)

                if len(parsed_rules) > 0:
                    self.dt_rules = pd.concat([self.dt_rules, parsed_rules]).reset_index(drop=True)

                    if self.writer is not None:
                        if self.feature_map is not None and len(self.feature_map) > 0:
                            parsed_rules["组合策略"] = parsed_rules["组合策略"].replace(self.feature_map, regex=True)
                        self.end_row, _ = self.insert_dt_rules(parsed_rules, self.end_row, self.start_col, save=f"model_report/auto_mining_rules/combiner_rules_{i}.png", figsize=(500, 100 * total_rules), sheet=worksheet)

                X_TE = X_TE.drop(columns=remove)
                self.decision_trees.append(decision_tree)
            except:
                import traceback
                traceback.print_exc()

        if len(self.dt_rules) <= 0:
            print(f"未挖掘到有效策略, 可以考虑适当调整预设的筛选参数, 降低 lift / 提高 max_samples, 当前筛选标准为: 提取 lift >= {lift} 且 max_samples <= {max_samples} 的策略")

        return self

    def transform(self, x, y=None):
        """在新数据集上评估已有规则的效果

        使用 fit 阶段挖掘的规则，在新数据集上进行评估，返回各规则的命中情况及指标。

        :param x: 包含标签的数据集
        :param y: 目标变量（如果 x 中不含标签列则需要传入），默认 None
        :return: pd.DataFrame，规则命中评估报告
        """
        y = x[self.target]
        X_TE = self.encode_cat_features(x.drop(columns=[self.target]), y)
        X_TE = X_TE.fillna(self.nan)
        if self.dt_rules is not None and len(self.dt_rules) > 0:
            parsed_rules = self.query_dt_rules(X_TE, y, parsed_rules=self.dt_rules)
            if self.feature_map is not None and len(self.feature_map) > 0:
                parsed_rules["组合策略"] = parsed_rules["组合策略"].replace(self.feature_map, regex=True)
            return parsed_rules
        else:
            return pd.DataFrame(columns=self.describe_columns)

    def report(self, valid=None, sheet="组合策略汇总", save=None):
        """生成组合策略报告并写入 Excel

        将训练集和验证集（可选）上的规则命中情况汇总写入 Excel 报告。

        :param valid: 验证数据集，支持 pd.DataFrame、list[DataFrame] 或 dict，默认 None
        :param sheet: 保存组合策略汇总的 sheet 名称，默认 "组合策略汇总"
        :param save: 保存报告的文件路径，默认 None（不保存）
        :return: tuple(pd.DataFrame, ...)，每个数据集的规则命中报告
        """
        worksheet = self.writer.get_sheet_by_name(sheet or "决策树组合策略挖掘")

        if sheet:
            self.writer.workbook.move_sheet(sheet, -1)

        parsed_rules_train = self.dt_rules.copy()

        if self.feature_map is not None and len(self.feature_map) > 0:
            parsed_rules_train["组合策略"] = parsed_rules_train["组合策略"].replace(self.feature_map, regex=True)

        self.end_row, _ = self.writer.insert_value2sheet(worksheet, (2 if sheet else self.end_row + 2, self.start_col), value="组合策略: 训练集", style="header_middle", end_space=(2 if sheet else self.end_row + 2, self.start_col + len(parsed_rules_train.columns) - 1))
        self.end_row, _ = self.insert_dt_rules(parsed_rules_train, self.end_row, self.start_col, sheet=worksheet)
        outputs = (parsed_rules_train,)

        if valid is not None:
            if isinstance(valid, pd.DataFrame) and len(valid) > 0:
                parsed_rules_val = self.transform(valid)
                self.end_row, _ = self.writer.insert_value2sheet(worksheet, (self.end_row + 2, self.start_col), value="组合策略: 验证集", style="header_middle", end_space=(self.end_row + 2, self.start_col + len(parsed_rules_val.columns) - 1))
                self.end_row, _ = self.insert_dt_rules(parsed_rules_val, self.end_row, self.start_col, sheet=worksheet)
                outputs = outputs + (parsed_rules_val,)

            elif isinstance(valid, (list, tuple)):
                for i, dataset in enumerate(valid):
                    if isinstance(dataset, pd.DataFrame) and len(dataset) > 0:
                        parsed_rules_val = self.transform(dataset)
                        self.end_row, _ = self.writer.insert_value2sheet(worksheet, (self.end_row + 2, self.start_col), value=f"组合策略: 验证集 {i + 1}", style="header_middle", end_space=(self.end_row + 2, self.start_col + len(parsed_rules_val.columns) - 1))
                        self.end_row, _ = self.insert_dt_rules(parsed_rules_val, self.end_row, self.start_col, sheet=worksheet)
                        outputs = outputs + (parsed_rules_val,)

            elif isinstance(valid, dict):
                for k, dataset in valid.items():
                    if isinstance(dataset, pd.DataFrame) and len(dataset) > 0:
                        parsed_rules_val = self.transform(dataset)
                        self.end_row, _ = self.writer.insert_value2sheet(worksheet, (self.end_row + 2, self.start_col), value=f"组合策略: {k}", style="header_middle", end_space=(self.end_row + 2, self.start_col + len(parsed_rules_val.columns) - 1))
                        self.end_row, _ = self.insert_dt_rules(parsed_rules_val, self.end_row, self.start_col, sheet=worksheet)
                        outputs = outputs + (parsed_rules_val,)

        if save:
            self.writer.save(save)

        return outputs
