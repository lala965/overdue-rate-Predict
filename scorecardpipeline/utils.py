# -*- coding: utf-8 -*-
"""
@Time    : 2023/05/21 16:23
@Author  : itlubber
@Site    : itlubber.art
"""
import warnings
from typing import List, Optional, Union, Any

warnings.filterwarnings("ignore")

import os
import re
import six
import pickle
import random
import joblib
import warnings
import numpy as np
import pandas as pd


# ==================== 日期频率兼容性辅助函数 ====================
def _normalize_freq(freq: str) -> str:
    """将旧的 pandas 频率别名转换为新版本支持的格式。

    适用于 resample() / to_period() / to_offset() 等需要频率参数的场景。
    同时兼容旧版 pandas（如 2.x）和新版 pandas（3.x+）。

    :param freq: 频率别名字符串，支持带数字前缀的格式（如 '3M', '5H', '2D'）
    :return: 标准化后的频率别名

    >>> _normalize_freq("M")    # pandas < 3.0 的月份别名
    'ME'
    >>> _normalize_freq("ME")   # 新版保持不变
    'ME'
    >>> _normalize_freq("H")    # pandas < 3.0 的小时别名
    'h'
    >>> _normalize_freq("h")   # 新版保持不变
    'h'
    >>> _normalize_freq("3M")   # 每 3 个月
    '3ME'
    >>> _normalize_freq("5H")   # 每 5 小时
    '5h'
    >>> _normalize_freq("2D")   # 每 2 天（无变化）
    '2D'
    >>> _normalize_freq("Q")    # 季度别名（Q 在新版仍可用）
    'Q'
    >>> _normalize_freq("D")    # 日别名（无变化）
    'D'
    """
    import re
    _freq = str(freq).strip()
    # 月: M -> ME（pandas 3.x+ 不再支持 M）
    # 处理带数字前缀的格式，如 '3M' -> '3ME'
    if re.match(r'^\d*M$', _freq):
        return _freq[:-1] + 'ME'
    if _freq == 'M':
        return 'ME'
    # 小时: H -> h（pandas 3.x+ 不再支持 H）
    if re.match(r'^\d*H$', _freq):
        return _freq[:-1] + 'h'
    if _freq == 'H':
        return 'h'
    return _freq


def _normalize_freq_for_period(freq: str) -> str:
    """将旧的频率别名转换为 to_period() 支持的格式。

    与 _normalize_freq 的区别：
    - to_period() 不接受 'h'，只接受 'H'（新版 pandas 也仍用 'H'）
    - to_period() 接受 'ME'（新版）也接受 'M'（旧版），新版优先 'ME'
    - 所以这里只处理 M -> ME，H 保持 'H' 不变

    :param freq: 频率别名字符串，支持带数字前缀的格式（如 '3M'）
    :return: 标准化后的频率别名

    >>> _normalize_freq_for_period("M")
    'ME'
    >>> _normalize_freq_for_period("3M")
    '3ME'
    >>> _normalize_freq_for_period("H")
    'H'
    """
    import re
    _freq = str(freq).strip()
    if re.match(r'^\d*M$', _freq):
        return _freq[:-1] + 'ME'
    if _freq == 'M':
        return 'ME'
    return freq


from functools import partial
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter, FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import seaborn as sns
from joblib import Parallel, delayed
from optbinning import OptimalBinning
from sklearn.metrics import roc_curve, auc, roc_auc_score

from .logger import init_logger


def get_system_font_name(font_path=None, fallback="楷体"):
    """获取可用于 openpyxl Font 的系统字体名称。

    优先级：1) 传入路径对应字体已在系统字体库中  2) font_path 文件注册后得到的字体名  3) fallback

    :param font_path: 字体文件路径或系统字体名，默认使用 scorecardpipeline 的 matplot_chinese.ttf
    :param fallback: 所有方案都失败时使用的字体名，默认 楷体（Windows/Linux/跨平台均可用）
    :return: str，可用于 openpyxl.styles.Font(name=...) 的字体名
    """
    import sys

    default_font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matplot_chinese.ttf")
    font_path = font_path or default_font_path

    # 1. 直接是系统已存在的字体名
    existing = [f.name.lower() for f in font_manager.fontManager.ttflist]
    if font_path.lower() in existing:
        return font_path

    # 2. 字体文件注册后取字体名
    if os.path.isfile(font_path):
        font_manager.fontManager.addfont(font_path)
        registered = [f for f in font_manager.fontManager.ttflist if os.path.normpath(os.path.abspath(f.fname)) == os.path.normpath(os.path.abspath(font_path))]
        if registered:
            return registered[0].name
        # addfont 成功但 name 提取失败，尝试从文件路径提取（部分平台有效）
        name = os.path.splitext(os.path.basename(font_path))[0]
        if name and name.lower() in existing:
            return name

    # 3. fallback
    return fallback


# 全局字体名，供 excel_writer.py 使用（延迟初始化，调用 init_font_for_excel 后有效）
_excel_font_name: str = None


def _install_system_font(font_path=None, fallback="楷体"):
    """跨平台安装字体到系统字体库（权限最小），返回注册后的字体名。

    安装策略（按顺序尝试）：
    1. Windows: AddFontResourceW 注册到当前进程（无需管理员权限）+ 复制到用户字体目录
    2. macOS: 复制到 ~/Library/Fonts
    3. Linux: 复制到 ~/.local/share/fonts + fc-cache -f
    4. 全部失败: 返回 fallback

    :param font_path: 字体文件路径
    :param fallback: fallback 字体名
    :return: str，注册成功后的字体名，失败则返回 fallback
    """
    import sys
    import shutil

    default_font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matplot_chinese.ttf")
    font_path = font_path or default_font_path
    if not os.path.isfile(font_path):
        return fallback

    font_name = os.path.splitext(os.path.basename(font_path))[0]
    platform = sys.platform

    # ---- Windows ----
    if platform == "win32":
        # 方案1: AddFontResourceW 仅注册到当前进程
        try:
            import ctypes
            GWL_USER = -4
            # 确保字体文件路径是绝对路径
            abs_path = os.path.abspath(font_path)
            if ctypes.windll.gdi32.AddFontResourceW(abs_path):
                # 注册成功后通知所有窗口刷新字体
                ctypes.windll.user32.SendMessageTimeoutW(
                    0xFFFF, 0x001D, 0, 0, 0x0002, 500, None
                )
                return font_name
        except Exception:
            pass

        # 方案2: 复制到用户字体目录（无需管理员权限）
        user_font_dir = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts")
        if not user_font_dir or user_font_dir == "Microsoft\\Windows\\Fonts":
            user_font_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Microsoft", "Windows", "Fonts")
        try:
            os.makedirs(user_font_dir, exist_ok=True)
            dest = os.path.join(user_font_dir, os.path.basename(font_path))
            shutil.copy2(font_path, dest)
            # 再次 AddFontResourceW
            try:
                import ctypes
                if ctypes.windll.gdi32.AddFontResourceW(os.path.abspath(dest)):
                    ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001D, 0, 0, 0x0002, 500, None)
                    return font_name
            except Exception:
                pass
        except Exception:
            pass

    # ---- macOS ----
    elif platform == "darwin":
        user_font_dir = os.path.join(os.path.expanduser("~"), "Library", "Fonts")
        try:
            os.makedirs(user_font_dir, exist_ok=True)
            dest = os.path.join(user_font_dir, os.path.basename(font_path))
            shutil.copy2(font_path, dest)
            return font_name
        except Exception:
            pass

    # ---- Linux ----
    elif platform.startswith("linux"):
        user_font_dir = os.path.join(os.path.expanduser("~"), ".local", "share", "fonts")
        try:
            os.makedirs(user_font_dir, exist_ok=True)
            dest = os.path.join(user_font_dir, os.path.basename(font_path))
            shutil.copy2(font_path, dest)
            # 刷新字体缓存
            import subprocess
            subprocess.run(["fc-cache", "-f", user_font_dir], capture_output=True)
            return font_name
        except Exception:
            pass

    return fallback


def init_font_for_excel(font_path=None, fallback="楷体"):
    """初始化中文字体，供 ExcelWriter 使用（权限最小方案，跨平台：Windows / Linux / macOS）。

    :param font_path: 字体文件路径，默认使用 scorecardpipeline 的 matplot_chinese.ttf
    :param fallback: 所有方案都失败时使用的字体名，默认 楷体
    :return: str，实际使用的字体名
    """
    global _excel_font_name
    _excel_font_name = _install_system_font(font_path, fallback)
    return _excel_font_name


def get_excel_font_name():
    """获取已注册的 Excel 字体名，未初始化时自动使用默认值初始化。"""
    global _excel_font_name
    if _excel_font_name is None:
        _excel_font_name = _install_system_font()
    return _excel_font_name


def seed_everything(seed: int, freeze_torch=False):
    """
    固定当前环境随机种子，以保证后续实验可重复

    :param seed: 随机种子
    :param freeze_torch: 是否固定 pytorch 的随机种子
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    if freeze_torch:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def init_setting(font_path=None, seed=None, freeze_torch=False, logger=False, **kwargs):
    """
    初始化环境配置，去除警告信息、修改 pandas 默认配置、固定随机种子、日志记录

    :param seed: 随机种子，默认为 None
    :param freeze_torch: 是否固定 pytorch 环境
    :param font_path: 画图时图像使用的字体，支持系统字体名称、本地字体文件路径，默认为 scorecardppeline 提供的中文字体
    :param logger: 是否需要初始化日志器，默认为 False ，当参数为 True 时返回 logger
    :param kwargs: 日志初始化传入的相关参数

    :return: 当 logger 为 True 时返回 logging.Logger
    """
    warnings.filterwarnings("ignore")

    pd.options.display.float_format = '{:.4f}'.format
    pd.set_option("display.max_colwidth", 300)
    pd.set_option('expand_frame_repr', False)

    if "seaborn-ticks" in plt.style.available:
        plt.style.use('seaborn-ticks')
    else:
        plt.style.use('seaborn-v0_8-ticks')

    if font_path is not None and font_path.lower() in [font.fname.lower() for font in font_manager.fontManager.ttflist]:
        plt.rcParams['font.family'] = font_path
    else:
        font_path = font_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matplot_chinese.ttf')

        if not os.path.isfile(font_path):
            import wget
            font_path = wget.download(
                "https://itlubber.art/upload/matplot_chinese.ttf"
                , os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matplot_chinese.ttf')
            )

        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()

    plt.rcParams['axes.unicode_minus'] = False

    # 同时注册字体供 ExcelWriter(openpyxl) 使用
    init_font_for_excel(font_path)

    if seed:
        seed_everything(seed, freeze_torch=freeze_torch)

    if logger:
        return init_logger(**kwargs)


def load_pickle(file, engine="joblib"):
    """
    导入 pickle 文件

    :param file: pickle 文件路径

    :return: pickle 文件的内容
    """
    if engine == "joblib":
        return joblib.load(file)
    elif engine == "dill":
        import dill
        with open(file, "rb") as f:
            return dill.load(f)
    elif engine == "pickle":
        with open(file, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"engine 目前只支持 [joblib, dill, pickle], 不支持 {engine}")


def save_pickle(obj, file, engine="joblib"):
    """
    保持数据至 pickle 文件

    :param obj: 需要保存的数据
    :param file: 文件路径
    """
    if engine == "joblib":
        return joblib.dump(obj, file)
    elif engine == "dill":
        import dill
        with open(file, "wb") as f:
            return dill.dump(obj, f)
    elif engine == "pickle":
        with open(file, "wb") as f:
            return pickle.dump(obj, f)
    else:
        raise ValueError(f"engine 目前只支持 [joblib, dill, pickle], 不支持 {engine}")


def feature_describe(data, feature=None, percentiles=None, missing=None, cardinality=None):
    """特征描述性统计，计算特征的基础统计指标

    :param data: pd.DataFrame 或 pd.Series，需要统计的数据
    :param feature: str，特征名称，传入后只统计该列，不传则统计整个 DataFrame
    :param percentiles: list，分位数点列表，默认 [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99]
    :param missing: 缺失值标记，如 -999、np.nan 等，传入后会将这些值替换为 np.nan 再统计
    :param cardinality: int，小于此基数的类别型特征进行频数统计而非分位数统计

    :return: pd.Series，统计指标序列，包含样本数、非空数、查得率以及（数值型）最小值/均值/最大值/各分位点

    **参考样例**

    >>> # 单特征统计
    >>> stats = feature_describe(data, feature="年龄")
    >>> # 指定分位数
    >>> stats = feature_describe(data, feature="收入", percentiles=[0.25, 0.5, 0.75])
    >>> # 类别型特征（基数小于阈值时返回频数分布）
    >>> stats = feature_describe(data, feature="学历", cardinality=10)
    >>> # 处理缺失值标记
    >>> stats = feature_describe(data, feature="收入", missing=-999)
    """
    if feature and feature not in data.columns:
        raise ValueError(f"feature {feature} must in columns.")

    if cardinality and cardinality < 1:
        raise ValueError(f"cardinality must grater 1")

    if percentiles is None:
        percentiles = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99]

    if feature:
        series = data[feature]
    else:
        series = data.copy()

    if missing:
        series = series.replace(missing, np.nan)

    if (cardinality and series.nunique() <= cardinality) or not pd.api.types.is_numeric_dtype(series):
        describe = {
            "样本数": len(series),
            "非空数": len(series) - series.isnull().sum(),
            "查得率": 1 - series.isnull().mean(),
        }
        describe.update((series.replace(np.nan, '缺失值').value_counts(dropna=False) / len(series)).to_dict())
        return pd.Series(describe, name=feature)
    else:
        describe = {
            "样本数": len(series),
            "非空数": len(series) - series.isnull().sum(),
            "查得率": 1 - series.isnull().mean(),
            "最小值": series.min(),
            "平均值": series.mean(),
            "最大值": series.max(),
            # "众数": series.mode()[0],
        }
        quantile = series.quantile(percentiles)
        quantile.index = [f"{int(i * 100)}%" for i in percentiles]
        describe.update(quantile.to_dict())
        return pd.Series(describe, name=feature).reindex(['样本数', '非空数', '查得率', '最小值', '平均值'] + [f"{int(i * 100)}%" for i in percentiles] + ['最大值'])


def groupby_feature_describe(data, by=None, n_jobs=-1, save=None, **kwargs):
    """按指定字段分组后对各数值列进行统计描述

    :param data: 数据集，pd.DataFrame
    :param by: 分组字段，支持单字段或字段列表，默认 None
    :param n_jobs: 并行任务数，默认 -1（使用全部 CPU）
    :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
    :param kwargs: dataframe_plot 相关的参数
    :return: pd.DataFrame，统计描述结果；如果传入了 save，则同时保存图片
    """
    if not isinstance(by, (tuple, list, np.ndarray)):
        by = [by]

    describe = pd.DataFrame()

    def __feature_describe(group, _p, f, **kwargs):
        temp = feature_describe(group[f], **kwargs)
        temp.index = pd.MultiIndex.from_product([[f], temp.index])
        temp = pd.DataFrame(temp, columns=[_p])
        return temp

    def _feature_describe(group, _p, _by=None, **kwargs):
        if len(_p) <= 1:
            _p = _p[0]

        __describe = partial(lambda f: __feature_describe(group, _p, f, **kwargs))
        return _p, pd.concat(Parallel(n_jobs=n_jobs)(delayed(__describe)(f) for f in group.columns if f not in _by))[_p]

    for info in Parallel(n_jobs=n_jobs)(delayed(_feature_describe)(group, p, _by=by, **kwargs) for p, group in data.groupby(by=by)):
        describe[info[0]] = info[1]

    if len(by) > 1:
        describe.columns = pd.MultiIndex.from_tuples(describe.columns)

    describe.index.names = ["特征名称", "统计指标"]

    if save:
        dataframe_plot(describe, save=save, **kwargs)

    return describe


def germancredit():
    """
    加载德国信贷数据集 German Credit Data

    数据来源：https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

    :return: pd.DataFrame
    """
    from pandas.api.types import CategoricalDtype

    data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'germancredit.csv'))

    cate_levels = {
        "status_of_existing_checking_account": ['... < 0 DM', '0 <= ... < 200 DM', '... >= 200 DM / salary assignments for at least 1 year', 'no checking account'],
        "credit_history": ["no credits taken/ all credits paid back duly", "all credits at this bank paid back duly", "existing credits paid back duly till now", "delay in paying off in the past", "critical account/ other credits existing (not at this bank)"],
        "savings_account_and_bonds": ["... < 100 DM", "100 <= ... < 500 DM", "500 <= ... < 1000 DM", "... >= 1000 DM", "unknown/ no savings account"],
        "present_employment_since": ["unemployed", "... < 1 year", "1 <= ... < 4 years", "4 <= ... < 7 years", "... >= 7 years"],
        "personal_status_and_sex": ["male : divorced/separated", "female : divorced/separated/married", "male : single", "male : married/widowed", "female : single"],
        "other_debtors_or_guarantors": ["none", "co-applicant", "guarantor"],
        "property": ["real estate", "building society savings agreement/ life insurance", "car or other, not in attribute Savings account/bonds", "unknown / no property"],
        "other_installment_plans": ["bank", "stores", "none"],
        "housing": ["rent", "own", "for free"],
        "job": ["unemployed/ unskilled - non-resident", "unskilled - resident", "skilled employee / official", "management/ self-employed/ highly qualified employee/ officer"],
        "telephone": ["none", "yes, registered under the customers name"],
        "foreign_worker": ["yes", "no"]}

    def cate_type(levels):
        return CategoricalDtype(categories=levels, ordered=True)

    for i in cate_levels.keys():
        data[i] = data[i].astype(cate_type(cate_levels[i]))

    return data


def round_float(num, decimal=4):
    """
    调整数值分箱的上下界小数点精度，如未超出精度保持原样输出

    :param num: 分箱的上界或者下界
    :param decimal: 小数点保留的精度

    :return: 精度调整后的数值
    """
    if ~pd.isnull(num) and isinstance(num, float):
        return float(str(num).split(".")[0] + "." + str(num).split(".")[1][:decimal])
    else:
        return num


def feature_bins(bins, decimal=4):
    """
    根据 Combiner 的规则生成分箱区间，并生成区间对应的索引

    :param bins: Combiner 的规则
    :param decimal: 区间上下界需要保留的精度，默认小数点后4位

    :return: dict ，key 为区间的索引，value 为区间
    """
    if len(bins) == 0:
        return {0: "全部样本"}
    if isinstance(bins, list): bins = np.array(bins)
    EMPTYBINS = len(bins) if not isinstance(bins[0], (set, list, np.ndarray)) else -1

    l = []
    if not isinstance(bins[0], (set, list, np.ndarray)):
        has_empty = len(bins) > 0 and pd.isnull(bins[-1])
        if has_empty: bins = bins[:-1]
        sp_l = ["负无穷"] + [round_float(b, decimal=decimal) for b in bins] + ["正无穷"]
        for i in range(len(sp_l) - 1): l.append('[' + str(sp_l[i]) + ' , ' + str(sp_l[i + 1]) + ')')
        if has_empty: l.append('缺失值')
    else:
        for keys in bins:
            keys_update = set()
            for key in keys:
                if pd.isnull(key) or (isinstance(key, str) and key == "nan"):
                    keys_update.add("缺失值")
                elif isinstance(key, str) and key.strip() == "":
                    keys_update.add("空字符串")
                else:
                    keys_update.add(key)
            label = ','.join(keys_update)
            l.append(label)

    return {i if b != "缺失值" else EMPTYBINS: b for i, b in enumerate(l)}


def extract_feature_bin(bin_var):
    """
    根据单个区间提取的分箱的上下界

    :param bin_var: 区间字符串

    :return: list or tuple
    """
    pattern = re.compile(r"^(\[|\()(-inf|负无穷|[-+]?\d+(\.\d+)?)(\s*,\s*|\s*~\s*)(inf|正无穷|[-+]?\d+(\.\d+)?)?(\]|\))$")
    match = pattern.match(bin_var)

    if match:
        start = -np.inf if match.group(2) in ["-inf", "负无穷"] else float(match.group(2))
        end = np.inf if match.group(5) in ["inf", "正无穷"] else float(match.group(5))
        return start, end
    else:
        return [np.nan if b == "缺失值" else ("" if b == "空字符串" else b) for b in bin_var.split("%,%" if "%,%" in bin_var else ",")]


def inverse_feature_bins(feature_table, bin_col="分箱"):
    """
    根据变量分箱表得到 Combiner 的规则

    :param feature_table: 变量分箱表
    :param bin_col: 变量分箱表中分箱对应的列名，默认 分箱

    :return: list
    """
    if isinstance(feature_table, pd.DataFrame):
        bin_vars = feature_table[bin_col].tolist()
    elif isinstance(feature_table, pd.Series):
        bin_vars = feature_table.tolist()
    else:
        bin_vars = feature_table

    has_empty = ("缺失值" in bin_vars) | (np.nan in bin_vars)

    bin_vars = [b for b in bin_vars if b not in ["缺失值", np.nan]]
    extract_bin_vars = [extract_feature_bin(bin_var) for bin_var in bin_vars]

    if isinstance(extract_bin_vars[0], tuple):
        inverse_bin_vars = sorted({s for b in extract_bin_vars for s in b})[1:-1]
        inverse_bin_vars += [np.nan] if has_empty else []
    else:
        inverse_bin_vars = extract_bin_vars
        inverse_bin_vars += [[np.nan]] if has_empty else []

    return inverse_bin_vars


# 默认颜色主题
DEFAULT_COLORS = ["#2639E9", "#F76E6C", "#FE7715"]


def _sort_stability_bins(data: pd.DataFrame, bin_col: str = "分箱") -> pd.DataFrame:
    """按分箱区间顺序重排 PSI/CSI 明细，并生成排序号。"""
    if bin_col not in data.columns:
        result = data.copy()
        result["排序号"] = np.arange(1, len(result) + 1)
        return result

    result = data.copy()
    result["_origin_order"] = np.arange(len(result))

    def _sort_key(label, origin_order):
        if _is_missing_bin_label(label):
            return 3, np.inf, np.inf, origin_order
        if _is_special_bin_label(label):
            return 2, np.inf, np.inf, origin_order

        try:
            bounds = extract_feature_bin(str(label).strip())
        except Exception:
            bounds = None

        if isinstance(bounds, tuple):
            lower, upper = bounds
            return 0, lower, upper, origin_order

        return 1, origin_order, origin_order, origin_order

    sort_keys = result.apply(
        lambda row: _sort_key(row[bin_col], row["_origin_order"]),
        axis=1,
        result_type="expand",
    )
    sort_keys.columns = ["_sort_group", "_sort_lower", "_sort_upper", "_sort_origin"]
    result = pd.concat([result, sort_keys], axis=1)
    result = (
        result
        .sort_values(
            ["_sort_group", "_sort_lower", "_sort_upper", "_sort_origin"],
            kind="mergesort",
        )
        .drop(columns=["_origin_order", "_sort_group", "_sort_lower", "_sort_upper", "_sort_origin"])
        .reset_index(drop=True)
    )
    result["排序号"] = np.arange(1, len(result) + 1)
    return result


def _is_missing_bin_label(label: Any) -> bool:
    """判断是否为缺失分箱标签。"""
    if pd.isna(label):
        return True
    text = str(label).strip().lower()
    return text in {'missing', 'nan', 'none', 'null', '缺失', '缺失值'}


def _is_special_bin_label(label: Any) -> bool:
    """判断是否为特殊值分箱标签。"""
    if pd.isna(label):
        return False
    text = str(label).strip().lower()
    return text in {'special', '特殊', '特殊值'}


def _is_interval_like_label(label: Any) -> bool:
    """判断分箱标签是否像数值区间。"""
    if pd.isna(label):
        return False
    text = str(label).strip()
    if _is_missing_bin_label(text):
        return False
    return bool(re.match(r'^[\(\[].*,.*[\)\]]$', text))


def _infer_numeric_feature_table(feature_table: pd.DataFrame) -> bool:
    """根据分箱标签粗略判断是否为数值型特征分箱表。"""
    label_col = '分箱标签' if '分箱标签' in feature_table.columns else '分箱'
    labels = feature_table[label_col].dropna() if label_col in feature_table.columns else pd.Series(dtype=object)
    labels = labels[~labels.apply(_is_missing_bin_label)]
    if labels.empty:
        return False
    interval_hits = labels.apply(_is_interval_like_label)
    return bool(interval_hits.any())


def _detect_bad_rate_trend(feature_table: pd.DataFrame) -> str:
    """判断坏样本率趋势，排除缺失值和特殊值分箱。"""
    if '坏样本率' not in feature_table.columns:
        return '未知'

    working = feature_table.copy()

    # 通过分箱索引排除缺失值(-1)和特殊值(-2)分箱
    if '分箱' in working.columns:
        bin_idx = pd.to_numeric(working['分箱'], errors='coerce')
        working = working[bin_idx.isna() | (bin_idx >= 0)]

    # 通过标签排除缺失值和特殊值分箱
    label_col = '分箱标签' if '分箱标签' in working.columns else '分箱'
    if label_col in working.columns:
        working = working[
            ~working[label_col].apply(_is_missing_bin_label)
            & ~working[label_col].apply(_is_special_bin_label)
        ]

    rates = pd.to_numeric(working['坏样本率'], errors='coerce').dropna().to_numpy()
    if len(rates) <= 1:
        return '未知'

    diffs = np.diff(rates)
    tol = 1e-6
    diffs = diffs[np.abs(diffs) > tol]
    if len(diffs) == 0:
        return '平稳'
    if np.all(diffs >= 0):
        return '上升'
    if np.all(diffs <= 0):
        return '下降'

    signs = np.sign(diffs)
    non_zero = signs[signs != 0]
    sign_changes = 0 if len(non_zero) <= 1 else int(np.sum(non_zero[1:] != non_zero[:-1]))
    if sign_changes == 1:
        return 'U型' if non_zero[0] < 0 < non_zero[-1] else '倒U型'
    return '波动'


def _build_bin_metric_summary(feature_table: pd.DataFrame) -> str:
    """构建分箱图角标摘要（紧凑两列表达）。"""
    items = []

    if '指标IV值' in feature_table.columns:
        iv_values = pd.to_numeric(feature_table['指标IV值'], errors='coerce').dropna()
        if not iv_values.empty:
            items.append(f"IV {iv_values.iloc[-1]:.4f}")

    if _infer_numeric_feature_table(feature_table) and '分档KS值' in feature_table.columns:
        ks_values = pd.to_numeric(feature_table['分档KS值'], errors='coerce').dropna()
        if not ks_values.empty:
            items.append(f"KS {ks_values.max():.4f}")

    if 'LIFT值' in feature_table.columns:
        lift_values = pd.to_numeric(feature_table['LIFT值'], errors='coerce').dropna()
        if not lift_values.empty:
            items.append(f"LIFT {lift_values.min():.2f}~{lift_values.max():.2f}")

    trend = _detect_bad_rate_trend(feature_table)
    if trend != '未知':
        items.append(f"趋势 {trend}")

    if not items:
        return ''

    rows = []
    for i in range(0, len(items), 2):
        rows.append('    '.join(items[i:i + 2]))

    return '\n'.join(rows)


def format_bin_label(label: Any, max_len: int = 35) -> str:
    """格式化分箱标签，截断过长文本。"""
    if pd.isna(label):
        return '缺失值'
    text = str(label).strip()
    if _is_missing_bin_label(text):
        return '缺失值'
    if len(text) > max_len:
        return text[:max_len] + '..'
    return text


def save_figure(fig, save: Optional[str] = None):
    """保存图像到文件。"""
    if save:
        if os.path.dirname(save):
            os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=240, format="png", bbox_inches="tight")
        plt.close('all')


def setup_axis_style(ax, colors=None, hide_top_right=True):
    """设置坐标轴样式。"""
    if colors is None:
        colors = DEFAULT_COLORS
    theme_color = colors[0] if colors else "#2639E9"

    ax.spines['top'].set_color(theme_color)
    ax.spines['bottom'].set_color(theme_color)
    ax.spines['right'].set_color(theme_color)
    ax.spines['left'].set_color(theme_color)
    ax.tick_params(axis='both', colors=theme_color)


def _is_feature_table(data) -> bool:
    """判断是否为特征分箱统计表。"""
    if not isinstance(data, pd.DataFrame):
        return False
    # 必须包含样本统计列
    stat_cols = ['好样本数', '坏样本数', '样本总数', '坏样本率']
    if not all(col in data.columns for col in stat_cols):
        return False
    # 分箱标识列：'分箱' 或 '分箱标签' 至少有一个
    return '分箱' in data.columns or '分箱标签' in data.columns


def bin_plot(feature_table, desc="", figsize=(14, 8), colors=None, save=None, anchor=0.935, max_len=35, fontdict=None, hatch=True, ending="分箱图", title=None, ax=None, orientation='horizontal', show_data_points=True, show_overall_bad_rate=True, return_frame=False):
    """特征分箱可视化图（增强版）。

    支持横向和纵向两种布局，显示好/坏样本堆叠柱状图、坏样本率折线、整体坏样本率参考线，
    以及 IV/KS/LIFT/趋势等统计指标。

    :param feature_table: 特征分箱的统计信息表，由 feature_bin_stats 运行得到
    :param desc: 特征中文含义或者其他相关信息
    :param figsize: 图像尺寸大小，默认 (14, 8)
    :param colors: 图片主题颜色，默认 ["#2639E9", "#F76E6C", "#FE7715"]
    :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
    :param anchor: 图例在图中的位置，通常 0.935 左右
    :param max_len: 分箱显示的最大长度，防止分类变量分箱过多文本过长导致图像显示区域很小，默认最长 35 个字符
    :param fontdict: 柱状图上的文字内容格式设置
    :param hatch: 柱状图是否显示斜杠，默认显示
    :param ending: 分箱图标题显示的后缀，标题格式为: f'{desc}{ending}'
    :param title: 完整标题（优先级高于 desc + ending）
    :param ax: 可选的 matplotlib Axes 对象，用于在已有画布上绘图
    :param orientation: 图表方向，'horizontal'/'h'（横向，默认）或 'vertical'/'v'（纵向）
    :param show_data_points: 是否显示数据点标记
    :param show_overall_bad_rate: 是否显示整体坏样本率参考线
    :param return_frame: 是否返回分箱统计表

    :return: matplotlib Figure 或 (Figure, DataFrame)，如果传入 ax 则返回 ax

    **参考样例**

    >>> from scorecardpipeline import bin_plot, feature_bin_stats
    >>>
    >>> # 生成分箱统计表
    >>> bin_table = feature_bin_stats(data, 'age', target='target')
    >>>
    >>> # 绘制分箱图
    >>> bin_plot(bin_table, desc="年龄", save="age_bin_plot.png")
    >>>
    >>> # 绘制纵向分箱图
    >>> bin_plot(bin_table, desc="年龄", orientation='vertical')
    """
    if colors is None:
        colors = DEFAULT_COLORS
    if fontdict is None:
        fontdict = {"color": "#000000"}

    feature_table = feature_table.copy()

    # 兼容 feature_bin_stats 返回的格式（有 分箱标签 但无 分箱 列）
    if '分箱' not in feature_table.columns and '分箱标签' in feature_table.columns:
        feature_table['分箱'] = feature_table['分箱标签']

    # 处理分箱标签：优先显示具体分箱标签
    plot_labels = None
    if '分箱标签' in feature_table.columns:
        candidate_labels = feature_table['分箱标签']
        if candidate_labels.notna().any():
            plot_labels = candidate_labels.astype(str)

    if plot_labels is None:
        plot_labels = feature_table['分箱'].astype(str)

    feature_table = feature_table.copy()
    feature_table['_plot_bin_label'] = plot_labels.apply(
        lambda x: format_bin_label(x, max_len)
    )

    # 判断方向
    orientation_key = orientation.lower()
    if orientation_key not in ['horizontal', 'h', '横向', 'vertical', 'v', '纵向']:
        raise ValueError("orientation 仅支持 'horizontal'/'h'/'横向' 或 'vertical'/'v'/'纵向'")
    is_horizontal = orientation_key in ['horizontal', 'h', '横向']

    # 统一排序：分离缺失值/特殊值分箱，数值型按区间下界升序，类别型保持原顺序
    label_col = '分箱标签' if '分箱标签' in feature_table.columns else '分箱'
    missing_mask = feature_table[label_col].apply(_is_missing_bin_label)
    special_mask = feature_table[label_col].apply(_is_special_bin_label)
    normal_rows = feature_table[~missing_mask & ~special_mask].copy()
    special_rows = feature_table[special_mask].copy()
    missing_rows = feature_table[missing_mask].copy()

    # 仅对数值型区间分箱排序
    is_numeric = _infer_numeric_feature_table(feature_table)
    if len(normal_rows) > 0 and is_numeric:
        def extract_lower_bound(bin_label):
            try:
                bounds = extract_feature_bin(str(bin_label).strip())
                if isinstance(bounds, tuple):
                    return bounds[0]
            except Exception:
                pass
            return float('inf')

        normal_rows['_sort_key'] = normal_rows[label_col].apply(extract_lower_bound)
        normal_rows = normal_rows.sort_values('_sort_key').drop(columns=['_sort_key'])

    # 重新组合：普通分箱(升序) + 特殊值分箱 + 缺失值分箱(最后)
    feature_table = pd.concat([normal_rows, special_rows, missing_rows], ignore_index=True)

    # 保存排序后的表用于 return_frame
    _sorted_table = feature_table.copy()

    if is_horizontal:
        # barh 第一行在底部、最后一行在顶部，反转使视觉从上到下为升序
        feature_table = feature_table.iloc[::-1].reset_index(drop=True)

    overall_bad_rate = float(feature_table['坏样本率'].mul(feature_table['样本总数']).sum() / feature_table['样本总数'].sum())
    axis_theme = colors[0]
    line_color = '#E85D4A'
    reference_color = '#4C8DFF'
    rate_fontdict = {
        'color': line_color,
        'fontsize': 10,
        'fontweight': 'semibold',
        'bbox': dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor=line_color, linewidth=0.6, alpha=0.92)
    }

    # 获取或创建 Axes
    if ax is not None:
        ax1 = ax
        fig = ax.figure
        return_ax = True
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        return_ax = False

    if is_horizontal:
        # 横向柱状图
        y_pos = np.arange(len(feature_table))
        ax1.barh(y_pos, feature_table['好样本数'], color=colors[0], label='好样本',
                 hatch="/" if hatch else None, edgecolor='white' if hatch else None, alpha=0.92)
        ax1.barh(y_pos, feature_table['坏样本数'], left=feature_table['好样本数'], color=colors[1],
                 label='坏样本', hatch="\\" if hatch else None, edgecolor='white' if hatch else None, alpha=0.92)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(feature_table['_plot_bin_label'])
        ax1.set_xlabel('样本数', color=axis_theme)

        ax2 = ax1.twiny()
        ax2.plot(feature_table['坏样本率'], y_pos, color=line_color, label='坏样本率', linestyle=(0, (4, 3)), linewidth=2.1,
                 marker='o' if show_data_points else None, markersize=5.5, markerfacecolor='white',
                 markeredgecolor=line_color, markeredgewidth=1.4)
        ax2.set_xlabel('坏样本率', color=axis_theme)
        ax2.set_xlim(left=0.)

        if show_overall_bad_rate:
            ax2.axvline(overall_bad_rate, color=reference_color, linestyle=(0, (2, 2)), linewidth=1.8, alpha=0.9,
                        label='整体坏样本率')

        x_right = max(ax2.get_xlim()[1], float(feature_table['坏样本率'].max()) * 1.15 if len(feature_table) > 0 else 0.1)
        ax2.set_xlim(right=x_right)
        x_offset = max((ax2.get_xlim()[1] - ax2.get_xlim()[0]) * 0.012, 0.003)
        for i, rate in enumerate(feature_table['坏样本率']):
            ax2.text(rate + x_offset, i, f'{rate:.2%}', va='center', ha='left', fontdict=rate_fontdict, clip_on=False)

        ax2.xaxis.set_major_formatter(PercentFormatter(1, decimals=0, is_latex=True))
    else:
        # 纵向柱状图
        x_pos = np.arange(len(feature_table))
        width = 0.6

        ax1.bar(x_pos, feature_table['好样本数'], width, color=colors[0], label='好样本',
                hatch="/" if hatch else None, edgecolor='white' if hatch else None, alpha=0.92)
        ax1.bar(x_pos, feature_table['坏样本数'], width, bottom=feature_table['好样本数'], color=colors[1],
                label='坏样本', hatch="\\" if hatch else None, edgecolor='white' if hatch else None, alpha=0.92)
        ax1.set_ylabel('样本数', color=axis_theme)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(feature_table['_plot_bin_label'], rotation=45, ha='right')

        ax2 = ax1.twinx()
        ax2.plot(x_pos, feature_table['坏样本率'], color=line_color, label='坏样本率', linestyle=(0, (4, 3)), linewidth=2.1,
                 marker='o' if show_data_points else None, markersize=5.5, markerfacecolor='white',
                 markeredgecolor=line_color, markeredgewidth=1.4)
        ax2.set_ylabel('坏样本率', color=axis_theme)
        ax2.set_ylim(bottom=0.)

        if show_overall_bad_rate:
            ax2.axhline(overall_bad_rate, color=reference_color, linestyle=(0, (2, 2)), linewidth=1.8, alpha=0.9,
                        label='整体坏样本率')

        y_top = max(float(feature_table['坏样本率'].max()) if len(feature_table) > 0 else 0.0, overall_bad_rate)
        ax2.set_ylim(top=max(ax2.get_ylim()[1], y_top * 1.18 if y_top > 0 else 0.1))
        y_offset = max(ax2.get_ylim()[1] * 0.015, 0.003)
        for i, rate in enumerate(feature_table['坏样本率']):
            ax2.text(i, rate + y_offset, f'{rate:.2%}', ha='center', va='bottom', fontdict=rate_fontdict, clip_on=False)

        ax2.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, is_latex=True))

    setup_axis_style(ax1, [axis_theme], hide_top_right=False)
    setup_axis_style(ax2, [axis_theme], hide_top_right=False)
    ax1.tick_params(axis='both', colors=axis_theme)
    ax2.tick_params(axis='both', colors=axis_theme)
    ax1.grid(False)
    ax2.grid(False)

    metric_summary = _build_bin_metric_summary(feature_table.drop(columns=['_plot_bin_label'], errors='ignore'))
    if not return_ax:
        if title is not None:
            fig.suptitle(f'{title}\n\n')
        else:
            fig.suptitle(f'{desc}{ending}\n\n')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        legend = fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center',
                            ncol=len(labels1 + labels2), bbox_to_anchor=(0.5, anchor), frameon=False)

        plt.tight_layout()
        if metric_summary:
            fig.canvas.draw()
            ax_pos = ax1.get_position()
            legend_bbox = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
            fig.text(ax_pos.x0, legend_bbox.y0, metric_summary, ha='left', va='bottom',
                     fontsize=10, color=axis_theme,
                     bbox=dict(boxstyle='round,pad=0.28', facecolor='white', edgecolor=axis_theme, alpha=0.9, linewidth=0.8))
        save_figure(fig, save)

        if return_frame:
            return fig, _sorted_table.drop(columns=['_plot_bin_label'], errors='ignore')
        return fig
    else:
        if metric_summary:
            ax2.text(0.0, 1.02, metric_summary, transform=ax2.transAxes, ha='left', va='bottom',
                     fontsize=10, color=axis_theme,
                     bbox=dict(boxstyle='round,pad=0.28', facecolor='white', edgecolor=axis_theme, alpha=0.9, linewidth=0.8),
                     clip_on=False)
        if title is not None:
            ax1.set_title(title)
        else:
            ax1.set_title(f'{desc}{ending}')
        return ax1


def corr_plot(data, figure_size=(16, 8), fontsize=16, mask=False, save=None, annot=True, max_len=35, linewidths=0.1, fmt='.2f', step=2 * 5 + 1, linecolor='white', **kwargs):
    """
    特征相关图

    :param data: 原始数据
    :param figure_size: 图片大小，默认 (16, 8)
    :param fontsize: 字体大小，默认 16
    :param mask: 是否只显示下三角部分内容，默认 False
    :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
    :param annot: 是否在图中显示相关性的数值，默认 True
    :param max_len: 特征显示的最大长度，防止特征名称过长导致图像区域非常小，默认 35，可以传 None 表示不限制
    :param fmt: 数值显示格式，当 annot 为 True 时该参数生效，默认显示两位小数点
    :param step: 色阶的步数，以 0 为中心，默认 2（以0为中心对称） * 5（划分五个色阶） + 1（0一档单独显示）= 11
    :param linewidths: 相关图之间的线条宽度，默认 0.1 ，如果设置为 None 则不现实线条
    :param linecolor: 线的颜色，当 linewidths 大于 0 时生效，默认为 white
    :param kwargs: sns.heatmap 函数其他参数，参考：https://seaborn.pydata.org/generated/seaborn.heatmap.html
    :return: Figure
    """
    if max_len is None:
        corr = data.corr()
    else:
        corr = data.rename(columns={c: c if len(str(c)) <= max_len else f"{str(c)[:max_len]}..." for c in data.columns}).corr()

    corr_mask = np.zeros_like(corr, dtype=bool)
    corr_mask[np.triu_indices_from(corr_mask)] = True

    fig, ax = plt.subplots(figsize=figure_size)
    map_plot = sns.heatmap(corr
                           , cmap=sns.diverging_palette(267, 267, n=step, s=100, l=40)
                           , vmax=1
                           , vmin=-1
                           , center=0
                           , square=True
                           , linewidths=linewidths
                           , annot=annot
                           , fmt=fmt
                           , linecolor=linecolor
                           , robust=True
                           , cbar=True
                           , ax=ax
                           , mask=corr_mask if mask else None
                           , **kwargs
                           )

    map_plot.tick_params(axis='x', labelrotation=270, labelsize=fontsize)
    map_plot.tick_params(axis='y', labelrotation=0, labelsize=fontsize)

    if save:
        if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
            os.makedirs(os.path.dirname(save), exist_ok=True)

        fig.savefig(save, dpi=240, format="png", bbox_inches="tight")
        plt.close('all')

    return fig


def ks_plot(score, target, title="", fontsize=14, figsize=(16, 8), save=None, colors=["#2639E9", "#F76E6C", "#FE7715"], anchor=0.945):
    """
    数值特征 KS曲线 & ROC曲线

    :param score: 数值特征，通常为评分卡分数
    :param target: 标签值
    :param title: 图像标题
    :param fontsize: 字体大小，默认 14
    :param figsize: 图像大小，默认 (16, 8)
    :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
    :param colors: 图片主题颜色，默认即可
    :param anchor: 图例显示的位置，默认 0.945，根据实际显示情况进行调整即可，0.95 附近小范围调整
    :return: Figure
    """
    auc_value = roc_auc_score(target, score)

    if auc_value < 0.5:
        warnings.warn('评分AUC指标小于50%, 推断数据值越大, 正样本率越高, 将数据值转为负数后进行绘图')
        score = -score
        auc_value = 1 - auc_value

    # if np.mean(score) < 0 or np.mean(score) > 1:
    #     warnings.warn('Since the average of pred is not in [0,1], it is treated as credit score but not probability.')
    #     score = -score

    df = pd.DataFrame({'label': target, 'pred': score})

    def n0(x):
        return sum(x == 0)

    def n1(x):
        return sum(x == 1)

    df_ks = df.sort_values('pred', ascending=False).reset_index(drop=True) \
        .assign(group=lambda x: np.ceil((x.index + 1) / (len(x.index) / len(df.index)))) \
        .groupby('group')['label'].agg([n0, n1]) \
        .reset_index().rename(columns={'n0': 'good', 'n1': 'bad'}) \
        .assign(
        group=lambda x: (x.index + 1) / len(x.index),
        cumgood=lambda x: np.cumsum(x.good) / sum(x.good),
        cumbad=lambda x: np.cumsum(x.bad) / sum(x.bad)
    ).assign(ks=lambda x: abs(x.cumbad - x.cumgood))

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # KS曲线
    dfks = df_ks.loc[lambda x: x.ks == max(x.ks)].sort_values('group').iloc[0]

    ax[0].plot(df_ks.group, df_ks.ks, color=colors[0], label="KS曲线")
    ax[0].plot(df_ks.group, df_ks.cumgood, color=colors[1], label="累积好客户占比")
    ax[0].plot(df_ks.group, df_ks.cumbad, color=colors[2], label="累积坏客户占比")
    ax[0].fill_between(df_ks.group, df_ks.cumbad, df_ks.cumgood, color=colors[0], alpha=0.25)

    ax[0].plot([dfks['group'], dfks['group']], [0, dfks['ks']], 'r--')
    ax[0].text(dfks['group'], dfks['ks'], f"KS: {round(dfks['ks'], 4)} at: {dfks.group:.2%}", horizontalalignment='center', fontsize=fontsize)

    ax[0].spines['top'].set_color(colors[0])
    ax[0].spines['bottom'].set_color(colors[0])
    ax[0].spines['right'].set_color(colors[0])
    ax[0].spines['left'].set_color(colors[0])
    ax[0].set_xlabel('% of Population', fontsize=fontsize)
    ax[0].set_ylabel('% of Total Bad / Good', fontsize=fontsize)

    ax[0].set_xlim((0, 1))
    ax[0].set_ylim((0, 1))

    handles1, labels1 = ax[0].get_legend_handles_labels()

    # ax[0].legend(loc='upper center', ncol=len(labels1), bbox_to_anchor=(0.5, 1.1), frameon=False)

    # ROC 曲线
    fpr, tpr, thresholds = roc_curve(target, score)

    ax[1].plot(fpr, tpr, color=colors[0], label="ROC Curve")
    ax[1].stackplot(fpr, tpr, color=colors[0], alpha=0.25)
    ax[1].plot([0, 1], [0, 1], color=colors[1], lw=2, linestyle=':')
    # ax[1].tick_params(axis='x', labelrotation=0, grid_color="#FFFFFF", labelsize=fontsize)
    # ax[1].tick_params(axis='y', labelrotation=0, grid_color="#FFFFFF", labelsize=fontsize)
    ax[1].text(0.5, 0.5, f"AUC: {auc_value:.4f}", fontsize=fontsize, horizontalalignment="center", transform=ax[1].transAxes)

    ax[1].spines['top'].set_color(colors[0])
    ax[1].spines['bottom'].set_color(colors[0])
    ax[1].spines['right'].set_color(colors[0])
    ax[1].spines['left'].set_color(colors[0])
    ax[1].set_xlabel("False Positive Rate", fontsize=fontsize)
    ax[1].set_ylabel('True Positive Rate', fontsize=fontsize)

    ax[1].set_xlim((0, 1))
    ax[1].set_ylim((0, 1))

    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")

    handles2, labels2 = ax[1].get_legend_handles_labels()

    if title: title += " "
    fig.suptitle(f"{title}K-S & ROC CURVE\n", fontsize=fontsize, fontweight="bold")

    fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', ncol=len(labels1 + labels2), bbox_to_anchor=(0.5, anchor), frameon=False)

    plt.tight_layout()

    if save:
        if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
            os.makedirs(os.path.dirname(save), exist_ok=True)

        plt.savefig(save, dpi=240, format="png", bbox_inches="tight")

    return fig


def hist_plot(score, y_true=None, figsize=(15, 10), bins=30, save=None, labels=["好样本", "坏样本"], desc="", anchor=1.11, fontsize=14, kde=False, **kwargs):
    """
    数值特征分布图

    :param score: 数值特征，通常为评分卡分数
    :param y_true: 标签值
    :param figsize: 图像大小，默认 (15, 10)
    :param bins: 分箱数量大小，默认 30
    :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
    :param labels: 字典或列表，图例显示的分类名称，默认 ["好样本", "坏样本"]，按照目标变量顺序对应即可，从0开始
    :param anchor: 图例显示的位置，默认 1.1，根据实际显示情况进行调整即可，1.1 附近小范围调整
    :param fontsize: 字体大小，默认 14
    :param kwargs: sns.histplot 函数其他参数，参考：https://seaborn.pydata.org/generated/seaborn.histplot.html

    :return: Figure
    """
    target_unique = 1 if y_true is None else len(np.unique(y_true))

    if y_true is not None:
        if isinstance(labels, dict):
            y_true = y_true.map(labels)
            hue_order = list(labels.values())
        else:
            y_true = y_true.map({i: v for i, v in enumerate(labels)})
            hue_order = labels
    else:
        y_true = None
        hue_order = None

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    palette = sns.diverging_palette(340, 267, n=target_unique, s=100, l=40)

    sns.histplot(
        x=score, hue=y_true, element="step", stat="probability", bins=bins, common_bins=True, common_norm=True, palette=palette, hue_order=hue_order[::-1], ax=ax, kde=kde, **kwargs
    )

    # sns.despine()

    ax.spines['top'].set_color("#2639E9")
    ax.spines['bottom'].set_color("#2639E9")
    ax.spines['right'].set_color("#2639E9")
    ax.spines['left'].set_color("#2639E9")

    ax.set_xlabel("值域范围", fontsize=fontsize)
    ax.set_ylabel("样本占比", fontsize=fontsize)

    ax.yaxis.set_major_formatter(PercentFormatter(1))

    ax.set_title(f"{desc + ' ' if desc else '特征'}分布情况\n\n", fontsize=fontsize)

    if y_true is not None:
        ax.legend([t for t in hue_order for _ in range(2)] if kde else hue_order, loc='upper center', ncol=target_unique * 2 if kde else target_unique, bbox_to_anchor=(0.5, anchor), frameon=False, fontsize=fontsize)

    fig.tight_layout()

    if save:
        if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
            os.makedirs(os.path.dirname(save), exist_ok=True)

        plt.savefig(save, dpi=240, format="png", bbox_inches="tight")
        plt.close('all')

    return fig


def psi_plot(expected, actual, labels=["预期", "实际"], desc="", save=None, colors=["#2639E9", "#F76E6C", "#FE7715"], figsize=(15, 8), anchor=0.94, width=0.35, result=False, plot=True, max_len=None, hatch=True):
    """
    特征 PSI 图

    :param expected: 期望分布情况，传入需要验证的特征分箱表
    :param actual: 实际分布情况，传入需要参照的特征分箱表
    :param labels: 期望分布和实际分布的名称，默认 ["预期", "实际"]
    :param desc: 标题前缀显示的名称，默认为空，推荐传入特征名称或评分卡名字
    :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
    :param colors: 图片主题颜色，默认即可
    :param figsize: 图像大小，默认 (15, 8)
    :param anchor: 图例显示的位置，默认 0.94，根据实际显示情况进行调整即可，0.94 附近小范围调整
    :param width: 预期分布与实际分布柱状图之间的间隔，默认 0.35
    :param result: 是否返回 PSI 统计表，默认 False
    :param plot: 是否画 PSI图，默认 True
    :param max_len: 特征显示的最大长度，防止特征名称过长导致图像区域非常小，默认 None 表示不限制
    :param hatch: 是否显示柱状图上的斜线，默认为 True

    :return: 当 result 为 True 时，返回 pd.DataFrame
    """
    expected = expected.rename(columns={"样本总数": f"{labels[0]}样本数", "样本占比": f"{labels[0]}样本占比", "坏样本率": f"{labels[0]}坏样本率"})
    actual = actual.rename(columns={"样本总数": f"{labels[1]}样本数", "样本占比": f"{labels[1]}样本占比", "坏样本率": f"{labels[1]}坏样本率"})
    df_psi = expected.merge(actual, on="分箱", how="outer").replace(np.nan, 0)
    df_psi[f"{labels[1]}% - {labels[0]}%"] = df_psi[f"{labels[1]}样本占比"] - df_psi[f"{labels[0]}样本占比"]
    df_psi[f"ln({labels[1]}% / {labels[0]}%)"] = np.log(df_psi[f"{labels[1]}样本占比"] / df_psi[f"{labels[0]}样本占比"])
    df_psi["分档PSI值"] = (df_psi[f"{labels[1]}% - {labels[0]}%"] * df_psi[f"ln({labels[1]}% / {labels[0]}%)"])
    df_psi = df_psi.fillna(0).replace(np.inf, 0).replace(-np.inf, 0)
    df_psi["总体PSI值"] = df_psi["分档PSI值"].sum()
    df_psi["指标名称"] = desc
    df_psi = _sort_stability_bins(df_psi, bin_col="分箱")

    if plot:
        x = df_psi.apply(lambda row: f"{int(row['排序号'])}. {row['分箱']}", axis=1)
        x = x.apply(lambda l: l if max_len is None or len(str(l)) < max_len else f"{str(l)[:max_len]}...")
        x_indexes = np.arange(len(x))
        fig, ax1 = plt.subplots(figsize=figsize)

        ax1.bar(x_indexes - width / 2, df_psi[f'{labels[0]}样本占比'], width, label=f'{labels[0]}样本占比', color=colors[0], hatch="/" if hatch else None)
        ax1.bar(x_indexes + width / 2, df_psi[f'{labels[1]}样本占比'], width, label=f'{labels[1]}样本占比', color=colors[1], hatch="\\" if hatch else None)

        ax1.set_ylabel('样本占比: 分箱内样本数 / 样本总数')
        ax1.set_xticks(x_indexes)
        ax1.set_xticklabels(x)
        ax1.tick_params(axis='x', labelrotation=90)

        ax2 = ax1.twinx()
        ax2.plot(x_indexes, df_psi[f"{labels[0]}坏样本率"], color=colors[0], label=f"{labels[0]}坏样本率", linestyle=(5, (10, 3)))
        ax2.plot(x_indexes, df_psi[f"{labels[1]}坏样本率"], color=colors[1], label=f"{labels[1]}坏样本率", linestyle=(5, (10, 3)))

        ax2.scatter(x_indexes, df_psi[f"{labels[0]}坏样本率"], marker=".")
        ax2.scatter(x_indexes, df_psi[f"{labels[1]}坏样本率"], marker=".")

        ax2.set_ylabel('坏样本率: 坏样本数 / 样本总数')

        fig.suptitle(f"{desc + ' ' if desc else ''}{labels[0]} vs {labels[1]} 群体稳定性指数(PSI): {df_psi['分档PSI值'].sum():.4f}\n\n")

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', ncol=len(labels1 + labels2), bbox_to_anchor=(0.5, anchor), frameon=False)

        fig.tight_layout()

        if save:
            if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
                os.makedirs(os.path.dirname(save), exist_ok=True)

            fig.savefig(save, dpi=240, format="png", bbox_inches="tight")
            plt.close('all')

    if result:
        return df_psi[["指标名称", "排序号", "分箱", f"{labels[0]}样本数", f"{labels[0]}样本占比", f"{labels[0]}坏样本率", f"{labels[1]}样本数", f"{labels[1]}样本占比", f"{labels[1]}坏样本率", f"{labels[1]}% - {labels[0]}%", f"ln({labels[1]}% / {labels[0]}%)", "分档PSI值", "总体PSI值"]]


def csi_plot(expected, actual, score_bins, labels=["预期", "实际"], desc="", save=None, colors=["#2639E9", "#F76E6C", "#FE7715"], figsize=(15, 8), anchor=0.94, width=0.35, result=False, plot=True, max_len=None, hatch=True):
    """
    特征 CSI 图

    :param expected: 期望分布情况，传入需要验证的特征分箱表
    :param actual: 实际分布情况，传入需要参照的特征分箱表
    :param score_bins: 逻辑回归模型评分表
    :param labels: 期望分布和实际分布的名称，默认 ["预期", "实际"]
    :param desc: 标题前缀显示的名称，默认为空，推荐传入特征名称或评分卡名字
    :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
    :param colors: 图片主题颜色，默认即可
    :param figsize: 图像大小，默认 (15, 8)
    :param anchor: 图例显示的位置，默认 0.94，根据实际显示情况进行调整即可，0.94 附近小范围调整
    :param width: 预期分布与实际分布柱状图之间的间隔，默认 0.35
    :param result: 是否返回 CSI 统计表，默认 False
    :param plot: 是否画 CSI图，默认 True
    :param max_len: 特征显示的最大长度，防止特征名称过长导致图像区域非常小，默认 None 表示不限制
    :param hatch: 是否显示柱状图上的斜线，默认为 True
    :return: 当 result 为 True 时，返回 pd.DataFrame
    """
    expected = expected.rename(columns={"样本总数": f"{labels[0]}样本数", "样本占比": f"{labels[0]}样本占比", "坏样本率": f"{labels[0]}坏样本率"})
    actual = actual.rename(columns={"样本总数": f"{labels[1]}样本数", "样本占比": f"{labels[1]}样本占比", "坏样本率": f"{labels[1]}坏样本率"})
    df_csi = expected.merge(actual, on="分箱", how="outer").replace(np.nan, 0)
    df_csi[f"{labels[1]}% - {labels[0]}%"] = df_csi[f"{labels[1]}样本占比"] - df_csi[f"{labels[0]}样本占比"]
    df_csi = df_csi.merge(pd.DataFrame({"分箱": feature_bins(score_bins["bins"]).values(), "对应分数": score_bins["scores"]}), on="分箱", how="left").replace(np.nan, 0)
    df_csi["分档CSI值"] = (df_csi[f"{labels[1]}% - {labels[0]}%"] * df_csi["对应分数"])
    df_csi = df_csi.fillna(0).replace(np.inf, 0).replace(-np.inf, 0)
    df_csi["总体CSI值"] = df_csi["分档CSI值"].sum()
    df_csi["指标名称"] = desc
    df_csi = _sort_stability_bins(df_csi, bin_col="分箱")

    if plot:
        x = df_csi.apply(lambda row: f"{int(row['排序号'])}. {row['分箱']}", axis=1)
        x = x.apply(lambda l: str(l) if max_len is None or len(str(l)) < max_len else f"{str(l)[:max_len]}...")
        x_indexes = np.arange(len(x))
        fig, ax1 = plt.subplots(figsize=figsize)

        ax1.bar(x_indexes - width / 2, df_csi[f'{labels[0]}样本占比'], width, label=f'{labels[0]}样本占比', color=colors[0], hatch="/" if hatch else None)
        ax1.bar(x_indexes + width / 2, df_csi[f'{labels[1]}样本占比'], width, label=f'{labels[1]}样本占比', color=colors[1], hatch="\\" if hatch else None)

        ax1.set_ylabel('样本占比: 分箱内样本数 / 样本总数')
        ax1.set_xticks(x_indexes)
        ax1.set_xticklabels(x)
        ax1.tick_params(axis='x', labelrotation=90)

        ax2 = ax1.twinx()
        ax2.plot(x_indexes, df_csi[f"{labels[0]}坏样本率"], color=colors[0], label=f"{labels[0]}坏样本率", linestyle=(5, (10, 3)))
        ax2.plot(x_indexes, df_csi[f"{labels[1]}坏样本率"], color=colors[1], label=f"{labels[1]}坏样本率", linestyle=(5, (10, 3)))

        ax2.scatter(x_indexes, df_csi[f"{labels[0]}坏样本率"], marker=".")
        ax2.scatter(x_indexes, df_csi[f"{labels[1]}坏样本率"], marker=".")

        ax2.set_ylabel('坏样本率: 坏样本数 / 样本总数')

        fig.suptitle(f"{desc + ' ' if desc else ''}{labels[0]} vs {labels[1]} 特征稳定性指标(CSI): {df_csi['分档CSI值'].sum():.4f}\n\n")

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', ncol=len(labels1 + labels2), bbox_to_anchor=(0.5, anchor), frameon=False)

        fig.tight_layout()

        if save:
            if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
                os.makedirs(os.path.dirname(save), exist_ok=True)

            fig.savefig(save, dpi=240, format="png", bbox_inches="tight")
            plt.close('all')

    if result:
        return df_csi[["指标名称", "排序号", "分箱", f"{labels[0]}样本数", f"{labels[0]}样本占比", f"{labels[0]}坏样本率", f"{labels[1]}样本数", f"{labels[1]}样本占比", f"{labels[1]}坏样本率", f"{labels[1]}% - {labels[0]}%", "对应分数", "分档CSI值", "总体CSI值"]]


def dataframe_plot(df, row_height=0.4, font_size=14, header_color='#2639E9', row_colors=['#dae3f3', 'w'], edge_color='w', bbox=[0, 0, 1, 1], header_columns=0, ax=None, save=None, **kwargs):
    """
    将 dataframe 转换为图片，推荐行和列都不多的数据集使用该方法

    :param df: 需要画图的 dataframe 数据
    :param row_height: 行高，默认 0.4
    :param font_size: 字体大小，默认 14
    :param header_color: 标题颜色，默认 #2639E9
    :param row_colors: 行颜色，默认 ['#dae3f3', 'w']，交替使用两种颜色
    :param edge_color: 表格边框颜色，默认白色
    :param bbox: 边的显示情况，[左，右，上，下]，即仅显示上下两条边框
    :param header_columns: 标题行数，默认仅有一个标题行，即 0
    :param ax: 如果需要在某张画布的子图中显示，那么传入对应的 ax 即可
    :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
    :param kwargs: plt.table 相关的参数，参考：https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.table.html

    :return: Figure
    """
    data = df.copy()
    for col in data.select_dtypes('datetime'):
        data[col] = data[col].dt.strftime("%Y-%m-%d")

    for col in data.select_dtypes('float'):
        data[col] = data[col].apply(lambda x: np.nan if pd.isnull(x) else round(x, 4))

    cols_width = [max(data[col].apply(lambda x: len(str(x).encode())).max(), len(str(col).encode())) / 8. for col in data.columns]

    if ax is None:
        size = (sum(cols_width), (len(data) + 1) * row_height)
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, colWidths=cols_width, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    fig.tight_layout()

    if save:
        if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
            os.makedirs(os.path.dirname(save))

        fig.savefig(save, dpi=240, format="png", bbox_inches="tight")
        plt.close('all')

    return fig


def distribution_plot(data, date="date", target="target", save=None, figsize=(10, 6), colors=["#2639E9", "#F76E6C", "#FE7715"], freq="M", anchor=0.94, result=False, hatch=True):
    """
    样本时间分布图

    :param data: 数据集
    :param date: 日期列名称，如果格式非日期，会尝试自动转为日期格式，默认 date，替换为数据中对应的日期列（如申请时间、授信时间、放款时间等）
    :param target: 数据集中标签列的名称，默认 target
    :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
    :param figsize: 图像大小，默认 (10, 6)
    :param colors: 图片主题颜色，默认即可
    :param freq: 汇总统计的日期格式，按年、季度、月、周、日等统计，参考：https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    :param anchor: 图例显示的位置，默认 0.94，根据实际显示情况进行调整即可，0.94 附近小范围调整
    :param result: 是否返回分布表，默认 False
    :param hatch: 是否显示柱状图上的斜线，默认为 True
    :return:
    """
    df = data.copy()

    if 'time' not in str(df[date].dtype):
        df[date] = pd.to_datetime(df[date])

    _freq = _normalize_freq(freq)
    temp = df.set_index(date).assign(
        好样本=lambda x: (x[target] == 0).astype(int),
        坏样本=lambda x: (x[target] == 1).astype(int),
    ).resample(_freq).agg({"好样本": sum, "坏样本": sum})

    temp.index = [i.strftime("%Y-%m-%d") for i in temp.index]

    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    temp.plot(kind='bar', stacked=True, ax=ax1, color=colors[:2], hatch="/" if hatch else None, legend=False)
    ax1.tick_params(axis='x', labelrotation=-90)
    ax1.set(xlabel=None)
    ax1.set_ylabel('样本数')
    ax1.set_title('不同时点数据集样本分布情况\n\n')

    ax2 = plt.twinx()
    (temp["坏样本"] / temp.sum(axis=1)).plot(ax=ax2, color=colors[-1], style="--", linewidth=2, label="坏样本率")
    # sns.despine()

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', ncol=len(labels1 + labels2), bbox_to_anchor=(0.5, anchor), frameon=False)

    fig.tight_layout()

    if save:
        if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
            os.makedirs(os.path.dirname(save))

        fig.savefig(save, dpi=240, format="png", bbox_inches="tight")
        plt.close('all')

    if result:
        temp = temp.reset_index().rename(columns={date: "日期", "index": "日期", 0: "好样本", 1: "坏样本"})
        temp["样本总数"] = temp["坏样本"] + temp["好样本"]
        temp["样本占比"] = temp["样本总数"] / temp["样本总数"].sum()
        temp["好样本占比"] = temp["好样本"] / temp["好样本"].sum()
        temp["坏样本占比"] = temp["坏样本"] / temp["坏样本"].sum()
        temp["坏样本率"] = temp["坏样本"] / temp["样本总数"]

        return temp[["日期", "样本总数", "样本占比", "好样本", "好样本占比", "坏样本", "坏样本占比", "坏样本率"]]


def sample_lift_transformer(df, rule, target='target', sample_rate=0.7):
    """采取好坏样本 sample_rate:1 的抽样方式时，计算抽样样本和原始样本上的 lift 指标

    该函数计算在好坏样本抽样比（sample_rate:1）下，规则命中的拒绝人群在抽样样本和原始样本上的 LIFT 值。
    主要用于在存在抽样偏差的样本上评估规则的区分能力，并推算到原始全量样本。

    :param df: 原始数据，需包含规则涉及的字段和目标变量
    :param rule: Rule，规则表达式对象
    :param target: 目标变量名称，默认 "target"
    :param sample_rate: 好样本采样比例，默认 0.7，即好样本按 0.7:1 的比例抽样（实际模型评估时常用 7:1 等比例抽样以平衡样本）

    :return: tuple，(lift_sam, lift_ori)
        - lift_sam: float，抽样样本上拒绝人群的 LIFT 值
        - lift_ori: float，原始样本上拒绝人群的 LIFT 值

    LIFT 值含义：
        - LIFT > 1：拒绝人群的坏样本率高于整体平均，规则有区分能力
        - LIFT = 1：规则无区分能力
        - LIFT < 1：拒绝人群的坏样本率低于整体平均，规则反向

    **参考样例**

    >>> from scorecardpipeline import Rule, sample_lift_transformer
    >>>
    >>> # 定义规则
    >>> rule = Rule("(年龄 < 30) & (历史逾期次数 >= 2)")
    >>>
    >>> # 计算 LIFT（好样本7:1抽样）
    >>> lift_sam, lift_ori = sample_lift_transformer(data, rule, target="target", sample_rate=7)
    >>> print(f"抽样样本 LIFT: {lift_sam:.4f}")
    >>> print(f"原始样本 LIFT: {lift_ori:.4f}")
    """
    rj_df = df[rule.predict(df)]
    ps_df = df[~rule.predict(df)]

    # 拒绝样本好坏样本数
    rj = len(rj_df)
    bad_rj = rj_df[target].sum()
    good_rj = rj - bad_rj

    # 通过样本好坏样本数
    ps = len(ps_df)
    bad_ps = ps_df[target].sum()
    good_ps = ps - bad_ps

    # 抽样样本上的lift
    lift_sam = (bad_rj / rj) / ((bad_rj + bad_ps) / (rj + ps))

    # 原始样本上的lift
    lift_ori = bad_rj / (bad_rj + bad_ps) * (1 + (sample_rate * bad_ps + good_ps) / (sample_rate * bad_rj + good_rj))

    return lift_sam, lift_ori


def tasks_executor(tasks, n_jobs=-1, pool="thread"):
    """多进程或多线程任务执行

    基于 concurrent.futures 实现的任务并行执行器，支持线程池和进程池两种模式。

    :param tasks: 可调用对象（函数）列表，每个任务应为一个不带参数或带可选参数的函数
    :param n_jobs: 并行 worker 数量，默认 -1（使用全部 CPU），当 pool="thread" 时为线程数，当 pool="process" 时为进程数
    :param pool: 并行模式，默认 "thread"（线程池），可选 "process"（进程池）

    :return: list，各任务执行结果的列表

    **参考样例**

    >>> # 使用线程池并行执行任务
    >>> def task1():
    >>>     return "task1 done"
    >>> def task2():
    >>>     return "task2 done"
    >>> results = tasks_executor([task1, task2], n_jobs=4, pool="thread")
    >>>
    >>> # 使用进程池并行执行
    >>> def heavy_task(x):
    >>>     return x * 2
    >>> results = tasks_executor([lambda: heavy_task(i) for i in range(10)], n_jobs=-1, pool="process")
    """
    if len(tasks) <= 0:
        raise ValueError("执行任务数必须大于0")

    if pool == "joblib":
        pass

    from concurrent.futures import wait, ALL_COMPLETED
    if pool == "thread":
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else 1)
    elif pool == "process":
        from concurrent.futures import ProcessPoolExecutor
        executor = ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else 1)

    _tasks = []
    for task in tasks:
        executor.submit(task)

    wait(_tasks, return_when=ALL_COMPLETED)

    return [t.result() for t in _tasks]


def monotonic_bad_rate_binning(df, feature, target, target_rates, greater_is_better=True):
    """根据目标违约率寻找最佳分箱切点，并确保逾期率单调

    :param df: 包含特征和目标的DataFrame
    :param feature: 要分箱的特征列名
    :param target: 目标变量列名
    :param target_rates: 目标违约率列表(从高到低或从低到高取决于greater_is_better)
    :param greater_is_better: 评分越高是否越好(违约率越低)

    :return: 分箱切点列表(已排序且唯一)
    """
    df = pd.DataFrame({"score": df[feature], "target": df[target]}).dropna()

    # 根据评分方向决定排序方式
    ascending_order = not greater_is_better
    df = df.sort_values("score", ascending=ascending_order)

    # 初始化变量
    cutpoints = []
    remaining_df = df.copy()
    last_bad_rate = None

    for rate in target_rates[:-1]:  # 最后一个箱处理剩余部分
        if len(remaining_df) == 0:
            break

        # 使用二分法寻找满足目标违约率的分割点
        low = remaining_df["score"].min()
        high = remaining_df["score"].max()
        best_cut = None

        for _ in range(100):
            if low >= high:
                break

            mid = (low + high) / 2

            # 根据评分方向决定分箱逻辑
            if greater_is_better:
                temp_df = remaining_df[remaining_df["score"] >= mid]
            else:
                temp_df = remaining_df[remaining_df["score"] <= mid]

            if len(temp_df) == 0:
                break

            temp_bad_rate = temp_df["target"].mean()

            if abs(temp_bad_rate - rate) < 0.001:  # 足够接近目标
                best_cut = mid
                break
            elif temp_bad_rate > rate:
                if greater_is_better:
                    low = mid + 1  # 需要更高的分数（更低的违约率）
                else:
                    high = mid - 1  # 需要更低的分数（更高的违约率）
            else:
                if greater_is_better:
                    high = mid - 1  # 需要更低的分数（更高的违约率）
                else:
                    low = mid + 1  # 需要更高的分数（更低的违约率）

        if best_cut is None:
            best_cut = (low + high) / 2

        # 根据评分方向获取当前箱
        if greater_is_better:
            current_bin = remaining_df[remaining_df["score"] >= best_cut]
            remaining_df = remaining_df[remaining_df["score"] < best_cut]
        else:
            current_bin = remaining_df[remaining_df["score"] <= best_cut]
            remaining_df = remaining_df[remaining_df["score"] > best_cut]

        # 计算当前箱的坏样本率
        current_bad_rate = current_bin["target"].mean()

        # 检查单调性
        if last_bad_rate is not None:
            if (greater_is_better and current_bad_rate >= last_bad_rate) or (not greater_is_better and current_bad_rate <= last_bad_rate):
                # 不满足单调性，跳过当前切点
                continue

        # 满足条件，记录切点
        cutpoints.append(best_cut)
        last_bad_rate = current_bad_rate

    # 确保切点唯一且正确排序
    cutpoints = sorted(list(set(cutpoints)), reverse=not greater_is_better)

    # 后处理：合并不满足单调性的箱
    while True:
        # 创建分箱
        bins = [-np.inf] + cutpoints + [np.inf] if len(cutpoints) > 0 else [-np.inf, np.inf]
        try:
            bin_labels = pd.cut(df["score"], bins=bins)
            bin_stats = df.groupby(bin_labels)["target"].agg(["count", "mean"])
            bin_stats.columns = ["样本数", "坏样本率"]

            # 检查单调性
            bad_rates = bin_stats["坏样本率"].values
            monotonic = True

            for i in range(1, len(bad_rates)):
                if (greater_is_better and bad_rates[i] >= bad_rates[i - 1]) or (not greater_is_better and bad_rates[i] <= bad_rates[i - 1]):
                    monotonic = False
                    break

            if monotonic or len(cutpoints) <= 1:
                break

            # 找到需要合并的切点
            merge_index = i - 1 if greater_is_better else i
            if merge_index < len(cutpoints):
                # 移除中间的切点
                cutpoints.pop(merge_index)
        except ValueError:
            # 如果分箱边界不是单调的，调整排序
            cutpoints = sorted(cutpoints)
            continue

    return cutpoints


def feature_summary(
    df: pd.DataFrame,
    features: List[str] = None,
    y: Optional[str] = None,
    val_df: Optional[pd.DataFrame] = None,
    max_n_bins: int = 5,
    psi_method: str = 'random_split',
    psi_group_col: Optional[str] = None,
    psi_date_col: Optional[str] = None,
    psi_freq: str = 'M',
    psi_test_size: float = 0.3,
    percentiles: List[float] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """综合特征描述统计.

    整合特征基础统计、IV、KS、PSI，快速获取数据集特征详情。
    参考 toad.detect + IV + KS + PSI 的功能组合。

    :param df: 训练/基准数据集
    :param features: 特征列表，None则分析全部（排除y列）
    :param y: 目标变量列名，传入则计算IV/KS/趋势
    :param val_df: 验证集，用于计算PSI，不传则使用psi_method指定的方式
    :param max_n_bins: IV计算分箱数，默认5
    :param psi_method: PSI计算方式
        - 'random_split': 随机拆分两份数据计算PSI（默认）
        - 'group_col': 按psi_group_col指定的分组列计算PSI
        - 'date_col': 按psi_date_col指定的日期列分组计算PSI
    :param psi_group_col: 分组列名（当psi_method='group_col'时使用）
    :param psi_date_col: 日期列名（当psi_method='date_col'时使用）
    :param psi_freq: 时间频率，'D'/'W'/'M'/'Q'，默认'M'
    :param psi_test_size: 随机拆分比例（当psi_method='random_split'时使用），默认0.3
    :param percentiles: 分位数点，默认[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    :param random_state: 随机种子
    :return: 综合特征描述DataFrame

    Example:
        >>> summary = feature_summary(df)
        >>> summary = feature_summary(df, y='target')
        >>> summary = feature_summary(df, y='target', psi_method='random_split')
        >>> summary = feature_summary(df, y='target', psi_method='date_col', psi_date_col='apply_date')
    """
    if percentiles is None:
        percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

    if features is None:
        if y is not None and y in df.columns:
            features = [c for c in df.columns if c != y]
        else:
            features = df.columns.tolist()
        if len(features) == 0:
            features = df.columns.tolist()

    # 确保features在df中
    features = [f for f in features if f in df.columns]
    if len(features) == 0:
        return pd.DataFrame()

    total = len(df)
    results = []

    for feat in features:
        series = df[feat]
        non_null = series.notna().sum()
        missing_rate = (total - non_null) / total
        is_numeric = pd.api.types.is_numeric_dtype(series)

        result = {
            '特征名': feat,
            '字段类型': 'numeric' if is_numeric else 'categorical',
            '样本数': total,
            '缺失数': total - non_null,
            '缺失率': round(missing_rate * 100, 2),
            '唯一值数': series.nunique(),
        }

        # 众数
        if non_null > 0:
            mode_value = series.mode()
            result['众数'] = mode_value.iloc[0] if len(mode_value) > 0 else None
            result['众数频数'] = (series == result['众数']).sum() if result['众数'] is not None else 0
            result['众数占比'] = result['众数频数'] if non_null > 0 else 0
        else:
            result['众数'] = None
            result['众数频数'] = 0
            result['众数占比'] = 0

        # 零值率、负值率（仅数值型）
        if is_numeric:
            non_null_series = series.dropna()
            result['零值数'] = (non_null_series == 0).sum()
            result['零值率'] = result['零值数'] if non_null > 0 else 0
            result['负值数'] = (non_null_series < 0).sum()
            result['负值率'] = result['负值数'] if non_null > 0 else 0
        else:
            result['零值数'] = 0
            result['零值率'] = 0
            result['负值数'] = 0
            result['负值率'] = 0

        # 重复率
        if non_null > 0:
            unique_count = series.nunique()
            result['重复数'] = non_null - unique_count
            result['重复率'] = result['重复数'] / total
        else:
            result['重复数'] = 0
            result['重复率'] = 0

        # 分布统计
        if is_numeric:
            desc = series.describe()
            result['最小值'] = round(float(desc.get('min', np.nan)), 4) if not pd.isna(desc.get('min')) else None
            result['最大值'] = round(float(desc.get('max', np.nan)), 4) if not pd.isna(desc.get('max')) else None
            result['平均值'] = round(float(desc.get('mean', np.nan)), 4) if not pd.isna(desc.get('mean')) else None
            result['标准差'] = round(float(desc.get('std', np.nan)), 4) if not pd.isna(desc.get('std')) else None

            for p in percentiles:
                col_name = f'{int(p * 100)}%'
                result[col_name] = round(float(series.quantile(p)), 4)
        else:
            result['最小值'] = None
            result['最大值'] = None
            result['平均值'] = None
            result['标准差'] = None

            value_counts = series.value_counts()
            if len(value_counts) > 0:
                sorted_categories = value_counts.index.tolist()
                total_count = len(series.dropna())
                percentile_categories = {}

                for p in percentiles:
                    target_count = int(total_count * p)
                    cumulative_count = 0
                    selected_cat = None

                    for cat in sorted_categories:
                        cat_count = value_counts[cat]
                        cumulative_count += cat_count
                        if cumulative_count >= target_count:
                            selected_cat = cat
                            break

                    if selected_cat is None:
                        selected_cat = sorted_categories[-1] if sorted_categories else None

                    percentile_categories[p] = selected_cat

                for p in percentiles:
                    col_name = f'{int(p * 100)}%'
                    result[col_name] = percentile_categories.get(p)
            else:
                for p in percentiles:
                    col_name = f'{int(p * 100)}%'
                    result[col_name] = None

        results.append(result)

    results_df = pd.DataFrame(results).set_index('特征名')

    # 计算IV、KS和趋势
    if y is not None and y in df.columns:
        y_series = df[y]
        numeric_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]

        iv_values = {f: np.nan for f in features}
        ks_values = {f: np.nan for f in features}
        trend_values = {f: 'categorical' if not pd.api.types.is_numeric_dtype(df[f]) else 'unknown' for f in features}

        if numeric_features:
            try:
                from .processing import Combiner

                for feat in numeric_features:
                    try:
                        data_subset = df[[feat, y]].dropna()
                        if len(data_subset) < 10:
                            continue

                        bin_table = Combiner.feature_bin_stats(
                            data_subset, feat, target=y,
                            max_n_bins=max_n_bins,
                            ks=True,
                            empty_separate=True,
                        )

                        if not bin_table.empty:
                            if '分档IV值' in bin_table.columns:
                                total_iv = bin_table['分档IV值'].sum()
                                if not np.isnan(total_iv) and not np.isinf(total_iv):
                                    iv_values[feat] = round(float(total_iv), 4)

                            if '分档KS值' in bin_table.columns:
                                max_ks = bin_table['分档KS值'].abs().max()
                                if not np.isnan(max_ks):
                                    ks_values[feat] = round(float(max_ks), 4)

                            # 检测单调性趋势
                            bad_rates = bin_table['坏样本率'].dropna().values
                            if len(bad_rates) >= 2:
                                diffs = np.diff(bad_rates)
                                all_positive = np.all(diffs >= 0)
                                all_negative = np.all(diffs <= 0)

                                if all_positive:
                                    trend_values[feat] = 'ascending'
                                elif all_negative:
                                    trend_values[feat] = 'descending'
                                else:
                                    peak_idx = np.argmax(bad_rates)
                                    valley_idx = np.argmin(bad_rates)
                                    if peak_idx > 0 and peak_idx < len(bad_rates) - 1:
                                        trend_values[feat] = 'peak'
                                    elif valley_idx > 0 and valley_idx < len(bad_rates) - 1:
                                        trend_values[feat] = 'valley'
                                    else:
                                        trend_values[feat] = 'unknown'
                    except Exception:
                        pass
            except ImportError:
                pass

        results_df['IV'] = pd.Series(iv_values)
        results_df['KS'] = pd.Series(ks_values)
        results_df['趋势'] = pd.Series(trend_values)

    # 计算PSI
    psi_values = {f: np.nan for f in features}

    def _compute_psi_from_bins(data1, data2, n_bins=10):
        """从两个Series计算PSI（基于等频分箱）"""
        try:
            combined = pd.concat([data1, data2]).dropna()
            if len(combined) < 10 or len(data1.dropna()) < 5 or len(data2.dropna()) < 5:
                return np.nan

            bins = np.quantile(combined, np.linspace(0, 1, n_bins + 1))
            bins = np.unique(bins)
            if len(bins) < 3:
                bins = np.linspace(combined.min(), combined.max(), n_bins + 1)
                bins = np.unique(bins)

            p1 = np.histogram(data1.dropna(), bins=bins, density=True)[0]
            p2 = np.histogram(data2.dropna(), bins=bins, density=True)[0]

            p1 = p1 / (p1.sum() + 1e-10)
            p2 = p2 / (p2.sum() + 1e-10)

            p1 = np.clip(p1, 1e-10, None)
            p2 = np.clip(p2, 1e-10, None)

            psi = np.sum((p2 - p1) * np.log(p2 / p1))
            return round(float(psi), 4)
        except Exception:
            return np.nan

    if val_df is not None:
        for feat in features:
            if feat not in df.columns or feat not in val_df.columns:
                psi_values[feat] = np.nan
                continue
            psi_values[feat] = _compute_psi_from_bins(df[feat], val_df[feat], n_bins=max_n_bins + 1)

    elif psi_method == 'random_split' and len(df) >= 100:
        try:
            from sklearn.model_selection import train_test_split

            df_copy = df.dropna(subset=features, how='all').copy()
            if len(df_copy) >= 100:
                df1, df2 = train_test_split(df_copy, test_size=psi_test_size, random_state=random_state)
                for feat in features:
                    if feat not in df.columns:
                        psi_values[feat] = np.nan
                        continue
                    psi_values[feat] = _compute_psi_from_bins(df1[feat], df2[feat], n_bins=max_n_bins + 1)
        except Exception:
            pass

    elif psi_method == 'group_col' and psi_group_col is not None and psi_group_col in df.columns:
        groups = df[psi_group_col].dropna().unique()
        if len(groups) >= 2:
            for feat in features:
                if feat not in df.columns:
                    psi_values[feat] = np.nan
                    continue

                psi_list = []
                for i, g1 in enumerate(groups):
                    for g2 in groups[i + 1:]:
                        data1 = df[df[psi_group_col] == g1][feat].dropna()
                        data2 = df[df[psi_group_col] == g2][feat].dropna()
                        if len(data1) > 10 and len(data2) > 10:
                            psi = _compute_psi_from_bins(data1, data2, n_bins=max_n_bins + 1)
                            if not np.isnan(psi):
                                psi_list.append(psi)

                if len(psi_list) > 0:
                    psi_values[feat] = round(float(np.mean(psi_list)), 4)

    elif psi_method == 'date_col' and psi_date_col is not None and psi_date_col in df.columns:
        try:
            df_copy = df.copy()
            df_copy[psi_date_col] = pd.to_datetime(df_copy[psi_date_col])

            _p_freq = _normalize_freq_for_period(psi_freq)
            if _p_freq in ('ME', 'M'):
                df_copy['_period'] = df_copy[psi_date_col].dt.to_period('ME').astype(str)
            elif _p_freq in ('W',):
                df_copy['_period'] = df_copy[psi_date_col].dt.to_period('W').astype(str)
            elif _p_freq in ('Q',):
                df_copy['_period'] = df_copy[psi_date_col].dt.to_period('Q').astype(str)
            else:
                df_copy['_period'] = df_copy[psi_date_col].dt.date.astype(str)

            periods = sorted(df_copy['_period'].dropna().unique())
            if len(periods) >= 2:
                for feat in features:
                    if feat not in df.columns:
                        psi_values[feat] = np.nan
                        continue

                    psi_list = []
                    for i, p1 in enumerate(periods):
                        for p2 in periods[i + 1:]:
                            data1 = df_copy[df_copy['_period'] == p1][feat].dropna()
                            data2 = df_copy[df_copy['_period'] == p2][feat].dropna()
                            if len(data1) > 10 and len(data2) > 10:
                                psi = _compute_psi_from_bins(data1, data2, n_bins=max_n_bins + 1)
                                if not np.isnan(psi):
                                    psi_list.append(psi)

                    if len(psi_list) > 0:
                        psi_values[feat] = round(float(np.mean(psi_list)), 4)
        except Exception:
            pass

    if psi_values:
        results_df['PSI'] = pd.Series(psi_values)

    # 重置索引，使特征名成为列
    results_df = results_df.reset_index()

    # 调整列顺序：基础统计 -> IV/KS/趋势/PSI -> 分布统计
    base_cols = ['特征名', '字段类型', '样本数', '趋势', 'IV', 'KS', 'PSI', '缺失数', '缺失率', '唯一值数', '众数', '众数频数', '众数占比']
    quality_cols = ['零值数', '零值率', '负值数', '负值率', '重复数', '重复率']
    dist_cols = ['最小值', '最大值', '平均值', '标准差'] + [f'{int(p * 100)}%' for p in percentiles]

    ordered_cols = base_cols + quality_cols + dist_cols
    ordered_cols = [c for c in ordered_cols if c in results_df.columns]
    results_df = results_df[ordered_cols]

    return results_df


# ==================== 分箱趋势分析图 ====================


def bin_trend_plot(
    data: pd.DataFrame,
    feature: str,
    target: str,
    dimension_cols: Optional[Union[str, List[str]]] = None,
    date_col: Optional[str] = None,
    date_freq: str = 'M',
    method: str = 'step',
    max_n_bins: int = 10,
    min_bin_size: float = 0.02,
    rules: Optional[dict] = None,
    special_values: Optional[List] = None,
    shared_bins: Optional[Union[str, bool]] = 'max_samples',
    sort_by: Optional[str] = None,
    sort_order: str = 'asc',
    max_groups: Optional[int] = None,
    figsize: Optional[tuple] = None,
    colors: Optional[List[str]] = None,
    title: Optional[str] = None,
    show_overall: bool = True,
    show_stats: bool = True,
    orientation: str = 'vertical',
    dpi: int = 150,
    save: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """绘制特征分箱风险趋势图。

    该图表集成了特征在不同维度下的样本分布、坏率走势、统计指标等信息。
    支持按时间维度（自动聚合）或指定维度列进行分组展示。

    :param data: 输入数据
    :param feature: 特征列名
    :param target: 目标变量列名（0/1）
    :param dimension_cols: 维度列名（单维或多维），用于分组展示
    :param date_col: 日期列名，如提供则按日期分组
    :param date_freq: 日期聚合频率，'D'/'W'/'M'/'Q'，默认'M'
    :param method: 分箱方法，默认 'step'，可选 'chi', 'dt', 'quantile', 'step', 'kmeans', 'cart', 'mdlp', 'uniform'
    :param max_n_bins: 最大分箱数，默认10
    :param min_bin_size: 最小箱占比，默认0.02
    :param rules: 预定义分箱规则 {特征名: 分箱边界列表}
    :param special_values: 特殊值列表
    :param shared_bins: 各分组是否共享同一切分点，默认 'max_samples'
        - 'first': 使用第一个分组（最早时间/第一个维度值）的切分点
        - 'last': 使用最后一个分组（最近时间/最后一个维度值）的切分点
        - 'max_samples': 使用样本量最多的分组的切分点（默认）
        - False 或 None: 每个分组独立计算切分点
    :param sort_by: 排序列名，None表示不排序，默认按维度值排序
    :param sort_order: 排序方向，'asc'/'desc'
    :param max_groups: 最大展示分组数，None表示全部展示
    :param figsize: 图像尺寸，None时自动计算
    :param colors: 配色方案，默认 ["#2639E9", "#F76E6C", "#FE7715"]
    :param title: 图表标题
    :param show_overall: 是否显示整体样本面板
    :param show_stats: 是否显示统计指标
    :param orientation: 图表方向，'vertical'（纵向，默认）或 'horizontal'
    :param dpi: 图像分辨率
    :param save: 保存路径
    :param kwargs: 其他参数
    :return: matplotlib Figure

    **参考样例**

    >>> from scorecardpipeline import bin_trend_plot
    >>>
    >>> # 按月份查看特征趋势
    >>> fig = bin_trend_plot(df, feature='age', target='bad', date_col='apply_date')
    >>>
    >>> # 按客群维度查看
    >>> fig = bin_trend_plot(df, feature='score', target='bad', dimension_cols='customer_type')
    >>>
    >>> # 多维度交叉
    >>> fig = bin_trend_plot(df, feature='income', target='bad', dimension_cols=['region', 'channel'])
    """
    from .processing import feature_bin_stats

    if colors is None:
        colors = DEFAULT_COLORS

    orientation_key = orientation.lower()
    is_horizontal = orientation_key in ['horizontal', 'h', '横向']

    # 处理维度列
    if dimension_cols is not None:
        if isinstance(dimension_cols, str):
            dimension_cols = [dimension_cols]
    else:
        dimension_cols = []

    # 处理日期列
    if date_col is not None:
        data = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col])

        try:
            if date_freq == 'D':
                data['_date_group'] = data[date_col].dt.strftime('%Y-%m-%d')
            else:
                _p_freq = _normalize_freq_for_period(date_freq)
                data['_date_group'] = data[date_col].dt.to_period(_p_freq).astype(str)
        except Exception:
            warnings.warn(f"无法识别 date_freq={date_freq}，已回退为按月分组")
            data['_date_group'] = data[date_col].dt.to_period('ME').astype(str)

        dimension_cols.append('_date_group')

    # 创建组合维度列
    if len(dimension_cols) > 0:
        data = data.copy()
        data['_group_key'] = data[dimension_cols].astype(str).agg('_'.join, axis=1)
        group_col = '_group_key'
    else:
        group_col = None

    # 处理 shared_bins：从指定分组提取切分点，统一应用到所有分组
    if shared_bins and group_col is not None and rules is None:
        groups = data[group_col].unique()
        _sort_by = sort_by if (sort_by is not None and sort_by in data.columns) else None
        if _sort_by is not None:
            _group_order = data.groupby(group_col)[_sort_by].first().sort_values(
                ascending=(sort_order == 'asc')
            ).index.tolist()
        else:
            _group_order = sorted(groups)

        _shared_bins = str(shared_bins).lower()
        if _shared_bins == 'first':
            ref_group = _group_order[0] if _group_order else None
        elif _shared_bins == 'last':
            ref_group = _group_order[-1] if _group_order else None
        else:  # 'max_samples' 或其他真值
            group_sizes = data.groupby(group_col).size()
            ref_group = group_sizes.idxmax()

        if ref_group is not None:
            from .processing import Combiner
            ref_data = data[data[group_col] == ref_group]
            _valid = ~(pd.isna(ref_data[feature]) | pd.isna(ref_data[target]))
            X_ref = ref_data.loc[_valid, feature]
            y_ref = ref_data.loc[_valid, target]
            if len(X_ref) > 0:
                try:
                    combiner = Combiner(method=method, max_n_bins=max_n_bins, min_bin_size=min_bin_size)
                    combiner.fit(X_ref, y_ref)
                    splits = combiner.export()["rule"]
                    if feature in splits and splits[feature]:
                        rules = {feature: splits[feature]}
                except Exception:
                    pass  # 回退到独立分箱

    # 计算整体分箱统计
    def _compute_stats(df_subset, feat, tgt, _method, _max_bins, _min_bin, _rules, _special):
        """计算单个分组的分箱统计表。"""
        if len(df_subset) == 0:
            return pd.DataFrame()

        X = df_subset[feat].copy()
        y = df_subset[tgt].copy()

        valid_mask = ~(pd.isna(X) | pd.isna(y))
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        if len(X_valid) < 10:
            return pd.DataFrame()

        try:
            return feature_bin_stats(
                pd.concat([X_valid, y_valid], axis=1),
                feat,
                target=tgt,
                method=_method,
                max_n_bins=_max_bins,
                min_bin_size=_min_bin,
                rules=_rules,
                empty_separate=True,
            )
        except Exception:
            return pd.DataFrame()

    overall_stats = _compute_stats(data, feature, target, method, max_n_bins, min_bin_size, rules, special_values)

    if overall_stats.empty:
        raise ValueError(f"无法计算特征 '{feature}' 的分箱统计")

    panel_stats = [('Overall', overall_stats.copy())] if show_overall else []

    if group_col is not None:
        groups = data[group_col].unique()
        if sort_by is not None and sort_by in data.columns:
            group_order = data.groupby(group_col)[sort_by].first().sort_values(
                ascending=(sort_order == 'asc')
            ).index.tolist()
        else:
            group_order = sorted(groups)

        if max_groups is not None and len(group_order) > max_groups:
            group_order = group_order[:max_groups]

        for group_val in group_order:
            df_group = data[data[group_col] == group_val]
            stats = _compute_stats(df_group, feature, target, method, max_n_bins, min_bin_size, rules, special_values)
            if not stats.empty:
                panel_stats.append((group_val, stats.copy()))

    if not panel_stats:
        raise ValueError("没有可用的分箱统计数据")

    n_panels = len(panel_stats)
    if is_horizontal:
        n_cols = 1
        n_rows = n_panels
    else:
        n_cols = min(3, n_panels)
        n_rows = int(np.ceil(n_panels / n_cols))

    if figsize is None:
        if is_horizontal:
            figsize = (10.5, max(4.8 * n_rows, 5.2))
        else:
            figsize = (max(5.2 * n_cols, 10.5), max(5.4 * n_rows, 5.2))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    if title is None:
        title = f"{feature} - Risk Trend Analysis"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    legend_handles = [
        Patch(facecolor=colors[0], edgecolor='white', label='好样本'),
        Patch(facecolor=colors[1], edgecolor='white', label='坏样本'),
        Line2D([0], [0], color='#E85D4A', linestyle=(0, (4, 3)), linewidth=2.1, marker='o', markersize=5, markerfacecolor='white', label='坏样本率'),
        Line2D([0], [0], color='#4C8DFF', linestyle=(0, (2, 2)), linewidth=1.8, label='整体坏样本率'),
    ]

    summary_cols = ['指标IV值', '分档KS值', 'LIFT值']
    panel_max_len = 22 if is_horizontal else 18

    for idx, (group_name, group_df) in enumerate(panel_stats):
        ax = axes_flat[idx]
        group_total = group_df['样本总数'].sum()
        group_bad = group_df['坏样本数'].sum()
        group_bad_rate = group_bad / group_total if group_total > 0 else 0.0
        panel_title = f"{group_name}\n({int(group_bad)}/{int(group_total)}, {group_bad_rate:.1%})"

        panel_df = group_df.copy()
        if not show_stats:
            panel_df = panel_df.drop(columns=summary_cols, errors='ignore')

        try:
            bin_plot(
                data=panel_df,
                ax=ax,
                title=panel_title,
                colors=colors,
                orientation='horizontal' if is_horizontal else 'vertical',
                max_len=panel_max_len,
                show_overall_bad_rate=True,
            )
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(panel_title)

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].axis('off')

    fig.legend(
        handles=legend_handles,
        loc='upper center',
        ncol=4,
        bbox_to_anchor=(0.5, 0.94),
        frameon=False,
        fontsize=9,
    )

    fig.subplots_adjust(
        top=0.84 if n_rows > 1 else 0.80,
        bottom=0.08,
        left=0.06,
        right=0.98,
        hspace=0.62 if n_rows > 1 else 0.42,
        wspace=0.28,
    )

    if save:
        save_figure(fig, save)

    return fig


def batch_bin_trend_plot(
    data: pd.DataFrame,
    features: List[str],
    target: str,
    dimension_cols: Optional[Union[str, List[str]]] = None,
    date_col: Optional[str] = None,
    date_freq: str = 'M',
    sort_by: str = 'iv',
    max_features: int = 10,
    figsize_per_feature: tuple = (12, 4),
    save_dir: Optional[str] = None,
    **kwargs
) -> dict:
    """批量绘制多个特征的风险趋势图。

    :param data: 输入数据
    :param features: 特征列表
    :param target: 目标变量列名
    :param dimension_cols: 维度列名
    :param date_col: 日期列名
    :param date_freq: 日期聚合频率
    :param sort_by: 排序指标，'iv'/'ks'
    :param max_features: 最大绘制特征数
    :param figsize_per_feature: 每个特征的图尺寸
    :param save_dir: 保存目录
    :param kwargs: 其他参数传递给 bin_trend_plot
    :return: 特征名到 Figure 的字典

    **参考样例**

    >>> from scorecardpipeline import batch_bin_trend_plot
    >>>
    >>> results = batch_bin_trend_plot(df, features=['age', 'income', 'score'], target='bad')
    >>> for feat, fig in results.items():
    ...     fig.savefig(f"{feat}_trend.png")
    """
    from .processing import feature_bin_stats

    results = {}

    # 计算特征排序
    feature_scores = []
    for feat in features:
        try:
            stats_df = feature_bin_stats(data[[feat, target]].dropna(), feat, target=target, method='step')
            if not stats_df.empty:
                iv_val = stats_df['指标IV值'].sum() if '指标IV值' in stats_df.columns else 0
                ks_val = stats_df['分档KS值'].max() if '分档KS值' in stats_df.columns else 0
                score = iv_val if sort_by == 'iv' else ks_val
                feature_scores.append({'feature': feat, 'score': score, 'iv': iv_val, 'ks': ks_val})
        except Exception:
            pass

    if feature_scores:
        score_df = pd.DataFrame(feature_scores).sort_values('score', ascending=False)
        sorted_features = score_df['feature'].tolist()[:max_features]
    else:
        sorted_features = features[:max_features]

    # 批量绘制
    for i, feat in enumerate(sorted_features):
        print(f"[{i+1}/{len(sorted_features)}] Plotting {feat}...")

        try:
            fig = bin_trend_plot(
                data, feature=feat, target=target,
                dimension_cols=dimension_cols,
                date_col=date_col, date_freq=date_freq,
                figsize=figsize_per_feature,
                **kwargs
            )

            results[feat] = fig

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{feat}_trend.png")
                fig.savefig(save_path, dpi=150, bbox_inches='tight')

        except Exception as e:
            warnings.warn(f"绘制特征 {feat} 失败: {e}")

    return results


# ==================== 多逾期天数分箱图 ====================


def _is_multiindex_bin_table(df: pd.DataFrame) -> bool:
    """检查是否为多级表头的分箱表（来自 feature_bin_stats）。"""
    return isinstance(df.columns, pd.MultiIndex)


def _extract_target_names_from_bin_table(bin_table: pd.DataFrame) -> List[str]:
    """从多级表头分箱表中提取目标名称列表。"""
    level_0_names = bin_table.columns.get_level_values(0).unique()
    target_names = [name for name in level_0_names if name != '分箱详情']
    return target_names


def _get_stats_for_target(bin_table: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """从多级表头分箱表中提取指定目标的统计信息。

    :param bin_table: 多级表头分箱表
    :param target_name: 目标名称
    :return: 单目标的分箱统计表（标准格式）
    """
    common_cols = []
    target_cols = []

    for col_tuple in bin_table.columns:
        if col_tuple[0] == '分箱详情':
            common_cols.append(col_tuple[1])
        elif col_tuple[0] == target_name:
            target_cols.append(col_tuple[1])

    stats_df = pd.DataFrame()

    # 添加公共列
    col_mapping = {
        '分箱标签': '分箱',
        '样本总数': '样本总数',
        '样本占比': '样本占比',
    }

    for orig_col, std_col in col_mapping.items():
        if orig_col in common_cols:
            stats_df[std_col] = bin_table[('分箱详情', orig_col)].values

    # 添加目标列
    target_col_mapping = {
        '好样本数': '好样本数',
        '坏样本数': '坏样本数',
        '坏样本率': '坏样本率',
    }

    for orig_col, std_col in target_col_mapping.items():
        if orig_col in target_cols:
            stats_df[std_col] = bin_table[(target_name, orig_col)].values

    return stats_df


def bin_overdues_plot(
    data: pd.DataFrame,
    feature: Optional[str] = None,
    overdue: Optional[List[str]] = None,
    dpds: Optional[List[int]] = None,
    bin_table: Optional[pd.DataFrame] = None,
    method: str = 'step',
    max_n_bins: int = 10,
    min_bin_size: float = 0.02,
    rules: Optional[dict] = None,
    shared_bins: Optional[Union[str, bool]] = 'max_samples',
    figsize: Optional[tuple] = None,
    colors: Optional[List[str]] = None,
    title: Optional[str] = None,
    show_stats: bool = True,
    max_cols: int = 3,
    save: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """绘制多个逾期天数的分箱图（横向展示）。

    支持两种输入方式：
    1. 原始数据 + overdue + dpds：根据原始数据计算分箱并绘图
    2. 分箱表（来自 feature_bin_stats）：直接解析多级表头分箱表并绘图

    :param data: 输入数据（原始数据模式）或分箱表（当传入 bin_table 时忽略）
    :param feature: 特征列名（原始数据模式需要）
    :param overdue: 逾期天数列名列表，如 ['dpd7', 'dpd15', 'dpd30']
    :param dpds: 逾期阈值列表，与 overdue 一一对应，如 [1, 1, 1]
        表示逾期天数>=该阈值时视为坏样本
    :param bin_table: 分箱表（来自 feature_bin_stats 的多级表头 DataFrame）
        传入后将直接使用分箱表绘图，忽略 data/overdue/dpds 参数
    :param method: 分箱方法，默认 'step'，可选 'chi', 'dt', 'quantile', 'step', 'kmeans', 'cart', 'mdlp', 'uniform'
    :param max_n_bins: 最大分箱数，默认10
    :param min_bin_size: 最小箱占比，默认0.02
    :param rules: 预定义分箱规则 {特征名: 分箱边界列表}
    :param shared_bins: 各逾期目标是否共享同一切分点，默认 'max_samples'
        - 'first': 使用第一个逾期定义的切分点
        - 'last': 使用最后一个逾期定义的切分点
        - 'max_samples': 使用有效样本量最多的逾期定义的切分点（默认）
        - False 或 None: 每个逾期定义独立计算切分点
    :param figsize: 图像尺寸，None时自动计算
    :param colors: 配色方案，默认 ["#2639E9", "#F76E6C", "#FE7715"]
    :param title: 图表总标题
    :param show_stats: 是否显示统计指标
    :param max_cols: 每行最多显示几个子图
    :param save: 保存路径
    :param kwargs: 其他参数
    :return: matplotlib Figure

    **参考样例**

    >>> from scorecardpipeline import bin_overdues_plot, feature_bin_stats
    >>>
    >>> # 方式1：使用原始数据
    >>> fig = bin_overdues_plot(
    ...     df,
    ...     feature='score',
    ...     overdue=['MOB1', 'MOB3'],
    ...     dpds=[0, 7],
    ... )
    >>>
    >>> # 方式2：使用 feature_bin_stats 生成的分箱表
    >>> bin_table = feature_bin_stats(
    ...     df,
    ...     feature='score',
    ...     overdue=['MOB1', 'MOB3'],
    ...     dpds=[0, 7]
    ... )
    >>> fig = bin_overdues_plot(bin_table=bin_table)
    """
    from .processing import Combiner

    if colors is None:
        colors = DEFAULT_COLORS

    # 分箱表模式：直接解析多级表头分箱表
    if bin_table is not None:
        if not _is_multiindex_bin_table(bin_table):
            raise ValueError("bin_table 必须是多级表头的分箱表（来自 feature_bin_stats）")

        target_names = _extract_target_names_from_bin_table(bin_table)

        if len(target_names) == 0:
            raise ValueError("分箱表中没有找到目标列（除了 '分箱详情'）")

        # 从分箱详情中提取特征名
        if ('分箱详情', '指标名称') in bin_table.columns:
            feature = bin_table[('分箱详情', '指标名称')].iloc[0]
        else:
            feature = 'Feature'

        n_plots = len(target_names)

        # 计算行列数
        n_cols = min(max_cols, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        # 自动计算图像尺寸
        if figsize is None:
            figsize = (4 * n_cols, 4 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # 处理单个子图的情况
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_plots > 1 else [axes]

        # 绘制每个目标的分箱图
        for idx, target_name in enumerate(target_names):
            ax = axes[idx]

            try:
                stats_df = _get_stats_for_target(bin_table, target_name)

                if stats_df.empty:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(target_name)
                    continue

                # 格式化分箱标签
                if '分箱' in stats_df.columns:
                    stats_df['分箱'] = stats_df['分箱'].apply(lambda x: format_bin_label(x, 35))

                bin_plot(
                    data=stats_df,
                    ax=ax,
                    title=target_name,
                    colors=colors,
                    orientation='vertical'
                )

                # 添加统计信息
                if show_stats:
                    stats_parts = []

                    # 计算整体坏样本率
                    if '坏样本数' in stats_df.columns and '样本总数' in stats_df.columns:
                        total_bad = stats_df['坏样本数'].sum()
                        total_samples = stats_df['样本总数'].sum()
                        if total_samples > 0:
                            bad_rate = total_bad / total_samples
                            stats_parts.append(f"BadRate: {bad_rate:.2%}")

                    if stats_parts:
                        stats_text = ", ".join(stats_parts)
                        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                               ha='right', va='top', fontsize=8,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(target_name)

        # 隐藏多余的子图
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')

        # 设置总标题
        if title is None:
            title = f"{feature} - Multi DPD Binning Analysis"
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        if save:
            save_figure(fig, save)

        return fig

    # 原始数据模式
    if feature is None:
        raise ValueError("原始数据模式需要提供 feature 参数")
    if overdue is None or dpds is None:
        raise ValueError("原始数据模式需要提供 overdue 和 dpds 参数")

    if len(overdue) != len(dpds):
        raise ValueError("overdue 和 dpds 长度必须一致")

    n_plots = len(overdue)

    # 计算行列数
    n_cols = min(max_cols, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    # 自动计算图像尺寸
    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # 处理单个子图的情况
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_plots > 1 else [axes]

    # 计算全局分箱规则
    global_rules = None
    if shared_bins and rules is None:
        _shared = str(shared_bins).lower()
        if _shared == 'first':
            ref_idx = 0
        elif _shared == 'last':
            ref_idx = len(overdue) - 1
        else:  # 'max_samples' 或其他真值
            valid_counts = []
            for dpd_col, threshold in zip(overdue, dpds):
                y_tmp = (data[dpd_col] >= threshold).astype(int)
                valid_counts.append((~(pd.isna(data[feature]) | pd.isna(y_tmp))).sum())
            ref_idx = int(np.argmax(valid_counts))

        dpd_col = overdue[ref_idx]
        threshold = dpds[ref_idx]
        y = (data[dpd_col] >= threshold).astype(int)
        valid_mask = ~(pd.isna(data[feature]) | pd.isna(y))
        X_valid = data.loc[valid_mask, feature]
        y_valid = y[valid_mask]

        try:
            combiner = Combiner(method=method, max_n_bins=max_n_bins, min_bin_size=min_bin_size)
            combiner.fit(X_valid, y_valid)
            splits = combiner.export()["rule"]
            if feature in splits and splits[feature]:
                global_rules = {feature: splits[feature]}
        except Exception:
            pass
    elif rules is not None:
        global_rules = rules

    # 绘制每个逾期定义的分箱图
    from .processing import feature_bin_stats

    for idx, (dpd_col, threshold) in enumerate(zip(overdue, dpds)):
        ax = axes[idx]

        try:
            # 创建二元目标变量
            y = (data[dpd_col] >= threshold).astype(int)
            target_name = f"{dpd_col}>={threshold}"

            # 构建临时数据
            temp_df = data[[feature]].copy()
            temp_df['_target'] = y

            # 计算分箱统计
            stats_df = feature_bin_stats(
                temp_df,
                feature,
                target='_target',
                method=method,
                max_n_bins=max_n_bins,
                min_bin_size=min_bin_size,
                rules=global_rules.get(feature, None) if global_rules else None,
                empty_separate=True,
            )

            if stats_df.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(target_name)
                continue

            # 格式化分箱标签
            if '分箱' in stats_df.columns:
                stats_df['分箱'] = stats_df['分箱'].apply(lambda x: format_bin_label(x, 35))

            bin_plot(
                data=stats_df,
                ax=ax,
                title=target_name,
                colors=colors,
                orientation='vertical'
            )

            # 添加统计信息
            if show_stats:
                valid_mask = ~(pd.isna(data[feature]) | pd.isna(y))
                X_valid = data.loc[valid_mask, feature]
                y_valid = y[valid_mask]

                bad_rate = y_valid.mean()
                stats_text = f"BadRate: {bad_rate:.2%}"
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                       ha='right', va='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{dpd_col} (>= {threshold})")

    # 隐藏多余的子图
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    # 设置总标题
    if title is None:
        title = f"{feature} - Multi DPD Binning Analysis"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, save)

    return fig
