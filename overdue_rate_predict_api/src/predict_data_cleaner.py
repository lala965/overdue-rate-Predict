'''
 # @ Author: Wanyh
 # @ Create Time: 2026/5/7 15:16
 # @ Modified by: Wanyh
 # @ Modified time: 2026/5/7 15:16
 # @ Description:
 '''

import math
import numpy as np


# 1. 分箱解析
def parse_bin(bin_str):

    bin_str = str(bin_str).strip()

    if 'NaN' in bin_str or '缺失' in bin_str:
        return None, None, None, None

    left_flag = bin_str[0]
    right_flag = bin_str[-1]

    nums = bin_str[1:-1].split(',')

    left = nums[0].strip()
    right = nums[1].strip()

    if left in ('-inf', '-infinity', '负无穷'):
        left = -math.inf
    else:
        left = float(left)

    if right in ('inf', 'infinity', '正无穷'):
        right = math.inf
    else:
        right = float(right)

    return left, right, left_flag, right_flag


# 2. 单变量映射
def map_score(value, bin_map, missing_to_nan=False):

    # 缺失值处理
    if (
            value is None
            or (isinstance(value, float) and np.isnan(value))
            or value == ''
            or value == -9999
    ):

        if missing_to_nan:
            return np.nan

        nan_keys = [
            k for k in bin_map.keys()
            if 'NaN' in str(k) or '缺失' in str(k)
        ]

        if nan_keys:
            return bin_map[nan_keys[0]]

        valid_scores = [
            v for k, v in bin_map.items()
            if ('缺失' not in str(k))
               and ('NaN' not in str(k))
        ]

        return min(valid_scores) if valid_scores else np.nan

    # 正常分箱匹配
    for bin_str, score in bin_map.items():

        if ('NaN' in str(bin_str)) or ('缺失' in str(bin_str)):
            continue

        left, right, left_flag, right_flag = parse_bin(bin_str)

        if left is None:
            continue

        left_ok = value >= left if left_flag == '[' else value > left
        right_ok = value <= right if right_flag == ']' else value < right

        if left_ok and right_ok:
            return score

    return np.nan


# todo 3. 单订单打分
def handle_single_order(
        df,
        scorecard_dict,
        missing_to_nan=False
):

    score_cols = []

    # 遍历所有评分变量
    for feat, feat_bin_map in scorecard_dict.items():
        # 特征不存在
        if feat not in df.columns:
            print(f"特征不存在: {feat}")
            continue

        score_col = feat + '_score'
        # 取值
        original_value = df[feat].iloc[0]

        # 映射分数
        mapped_score = map_score(
            original_value,
            feat_bin_map,
            missing_to_nan=missing_to_nan
        )

        # 写入分数列
        df[score_col] = mapped_score
        score_cols.append(score_col)

        # print(f"{feat}: {original_value} -> {mapped_score}")
        # logging.info(f"{feat}: {original_value} -> {mapped_score}")

    # 总分
    if score_cols:
        df['total_score'] = df[score_cols].sum(axis=1)
    else:
        df['total_score'] = np.nan

    # 返回结果
    return {
        "order_sn": df['order_sn'].iloc[0],
        "order_time": df['order_time'].iloc[0].strftime('%Y-%m-%d %H:%M:%S'),
        "total_score": round(float(df['total_score'].iloc[0]), 2),
        "feature_score_detail": {
            col: float(df[col].iloc[0])
            for col in score_cols
        }
    }


# todo 多订单打分
def handle_multi_order(
        df,
        scorecard_dict,
        missing_to_nan=False
):

    result_list = []

    # 遍历每一行
    for idx in range(len(df)):

        # 取单行
        row_df = df.iloc[[idx]].copy()

        # 调用单订单逻辑
        single_result = handle_single_order(
            row_df,
            scorecard_dict,
            missing_to_nan=missing_to_nan
        )

        result_list.append(single_result)

    return result_list


# todo 全量订单打分
# def handle_full_data(
#         df,
#         scorecard_dict,
#         missing_to_nan=False
# ):
#
#     return handle_multi_order(
#         df,
#         scorecard_dict,
#         missing_to_nan=missing_to_nan
#     )


# todo 全量订单打分（只返回总分）
def handle_full_data(
        df,
        scorecard_dict,
        missing_to_nan=False
):

    result_list = []
    # 遍历每一行
    for idx in range(len(df)):
        row_df = df.iloc[[idx]].copy()
        score_cols = []

        # 所有变量打分
        for feat, feat_bin_map in scorecard_dict.items():
            if feat not in row_df.columns:
                continue

            score_col = feat + '_score'
            original_value = row_df[feat].iloc[0]

            mapped_score = map_score(
                original_value,
                feat_bin_map,
                missing_to_nan=missing_to_nan
            )
            row_df[score_col] = mapped_score
            score_cols.append(score_col)

        # 总分
        if score_cols:
            row_df['total_score'] = row_df[score_cols].sum(axis=1)
        else:
            row_df['total_score'] = np.nan

        # 只返回核心字段
        result_list.append({
            "order_sn": row_df['order_sn'].iloc[0],
            "order_time": row_df['order_time'].iloc[0].strftime(
                '%Y-%m-%d %H:%M:%S'
            ),
            "total_score": round(
                float(row_df['total_score'].iloc[0]),
                2
            )
        })

    return result_list