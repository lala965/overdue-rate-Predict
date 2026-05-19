'''
 # @ Author: Wanyh
 # @ Create Time: 2026/5/12 15:26
 # @ Modified by: Wanyh
 # @ Modified time: 2026/5/12 15:26
 # @ Description:
 '''

import pandas as pd
import hashlib


def md5_id_card(value):
    if pd.isna(value):
        return pd.NA
    return hashlib.md5(str(value).strip().encode('utf-8')).hexdigest()


def match_nearest_report(
        df_user,
        df_br_order,
        join_col='order_sn',
        left_time_col='order_time',
        right_time_col='report_time',
        max_days=3
):
    """
    按订单号匹配报告：只保留 report_time >= order_time 且 max_days 天内最近的一条报告。
    返回的 df_merge_all 已经带上选中的整条报告字段；没有有效报告的订单保留为空。
    如果左表已经有同名报告字段，则只补空值，不覆盖左表已有值。
    """
    left = df_user.copy()
    right = df_br_order.copy()

    required_left_cols = {join_col, left_time_col}
    required_right_cols = {join_col, right_time_col}
    missing_left_cols = required_left_cols - set(left.columns)
    missing_right_cols = required_right_cols - set(right.columns)
    if missing_left_cols:
        raise KeyError(f"左表缺少字段: {sorted(missing_left_cols)}")
    if missing_right_cols:
        raise KeyError(f"右表缺少字段: {sorted(missing_right_cols)}")

    left[join_col] = left[join_col].astype('string').str.strip()
    right[join_col] = right[join_col].astype('string').str.strip()
    left[left_time_col] = pd.to_datetime(left[left_time_col], errors='coerce')
    right[right_time_col] = pd.to_datetime(right[right_time_col], errors='coerce')

    left = left.reset_index(drop=True).reset_index(names='_left_row_id')
    right_cols = [col for col in right.columns if col != join_col]
    right_rename = {col: f'{col}_match' for col in right_cols}
    right_match = right.rename(columns=right_rename)
    match_time_col = right_rename[right_time_col]
    match_cols = [right_rename[col] for col in right_cols]

    df_match_all = left.merge(
        right_match,
        on=join_col,
        how='left'
    )

    df_valid = df_match_all[df_match_all[match_time_col].notna()].copy()
    df_valid = df_valid[df_valid[match_time_col] >= df_valid[left_time_col]].copy()
    df_valid['time_diff'] = df_valid[match_time_col] - df_valid[left_time_col]
    df_valid = df_valid[df_valid['time_diff'] <= pd.Timedelta(days=max_days)].copy()

    df_best_report = (
        df_valid
        .sort_values(['_left_row_id', 'time_diff'])
        .drop_duplicates(subset=['_left_row_id'], keep='first')
    )

    # 只把右表报告字段和 time_diff 拼回用户表。
    df_best_report = df_best_report[['_left_row_id'] + match_cols + ['time_diff']]
    df_merge_all = left.merge(df_best_report, on='_left_row_id', how='left')

    for right_col, match_col in right_rename.items():
        if right_col in df_merge_all.columns:
            df_merge_all[right_col] = df_merge_all[right_col].combine_first(df_merge_all[match_col])
        else:
            df_merge_all[right_col] = df_merge_all[match_col]

    temp_cols = [col for col in match_cols if col in df_merge_all.columns]
    df_merge_all = df_merge_all.drop(columns=temp_cols)
    df_merge_all = df_merge_all.drop(columns=['_left_row_id'])

    df_final = (
        df_best_report
        .merge(left[['_left_row_id', join_col, left_time_col]], on='_left_row_id', how='left')
        [[join_col, left_time_col, match_time_col, 'time_diff']]
        .rename(columns={match_time_col: right_time_col})
        .reset_index(drop=True)
    )

    return df_merge_all, df_final


def match_nearest_report_by_idcard(
        df_user,
        df_br_card,
        join_col='id_card_no_md5',
        left_time_col='order_time',
        right_time_col='report_time',
        max_days=3
):
    """
    按身份证 md5 直接匹配报告：只保留 report_time >= order_time 且 max_days 天内最近的一条报告。
    适用于报告表没有 order_sn，或希望完全按用户身份证匹配报告的场景。
    """
    left = df_user.copy()
    right = df_br_card.copy()

    left[join_col] = left[join_col].astype('string').str.strip().str.lower()
    right[join_col] = right[join_col].astype('string').str.strip().str.lower()

    return match_nearest_report(
        df_user=left,
        df_br_order=right,
        join_col=join_col,
        left_time_col=left_time_col,
        right_time_col=right_time_col,
        max_days=max_days
    )



def fill_report_by_idcard(
        df_user,
        df_br_order,
        join_col='id_card_no_md5',
        left_time_col='order_time',
        right_time_col='report_time',
        max_days=3
):
    """
    对订单号未匹配到有效报告的样本，用身份证 md5 补全最近报告。
    时间规则与订单号匹配一致：report_time >= order_time，且 time_diff <= max_days。
    命中后补全整条报告字段，不只补 report_time。
    """
    df = df_user.copy()
    right = df_br_order.copy()

    df[join_col] = df[join_col].astype('string').str.strip().str.lower()
    right[join_col] = right[join_col].astype('string').str.strip().str.lower()
    df[left_time_col] = pd.to_datetime(df[left_time_col], errors='coerce')
    right[right_time_col] = pd.to_datetime(right[right_time_col], errors='coerce')

    need_fill = df[
        df[right_time_col].isna()
        & df[join_col].notna()
    ].copy()

    need_fill_unique = need_fill.drop_duplicates(subset=[join_col, left_time_col])
    right_use = right.dropna(subset=[join_col, right_time_col]).drop_duplicates()

    df_match_all = need_fill_unique.merge(
        right_use,
        on=join_col,
        how='left',
        suffixes=('', '_fill')
    )

    fill_time_col = f'{right_time_col}_fill'
    df_valid = df_match_all[df_match_all[fill_time_col].notna()].copy()
    df_valid = df_valid[df_valid[fill_time_col] >= df_valid[left_time_col]].copy()
    df_valid['time_diff_fill'] = df_valid[fill_time_col] - df_valid[left_time_col]
    df_valid = df_valid[df_valid['time_diff_fill'] <= pd.Timedelta(days=max_days)].copy()

    df_fill_key = (
        df_valid
        .sort_values([join_col, left_time_col, 'time_diff_fill'])
        .drop_duplicates(subset=[join_col, left_time_col], keep='first')
        .reset_index(drop=True)
    )

    # 记录右表字段在 merge 后对应的临时列名，后面统一回填到原字段。
    fill_source_map = {}
    for col in right_use.columns:
        if col == join_col:
            continue
        source_col = f'{col}_fill' if col in need_fill_unique.columns else col
        if source_col in df_fill_key.columns:
            fill_source_map[col] = source_col

    keep_cols = [join_col, left_time_col, 'time_diff_fill'] + list(fill_source_map.values())
    df_fill_key = df_fill_key[keep_cols].copy()

    df_result = df.merge(
        df_fill_key,
        on=[join_col, left_time_col],
        how='left'
    )

    for target_col, source_col in fill_source_map.items():
        if target_col not in df_result.columns:
            df_result[target_col] = pd.NA
        df_result[target_col] = df_result[target_col].combine_first(df_result[source_col])

    df_result['fill_type'] = None
    df_result.loc[df_result[fill_time_col].notna(), 'fill_type'] = join_col

    temp_cols = [col for col in fill_source_map.values() if col in df_result.columns]
    df_result = df_result.drop(columns=temp_cols, errors='ignore')

    return df_result, df_fill_key


def match_report_order_then_idcard(
        df_user,
        df_report_order,
        df_report_idcard=None,
        order_col='order_sn',
        id_col='id_card_no_md5',
        left_time_col='order_time',
        right_time_col='report_time',
        max_days=3,
        match_type_col='report_match_type'
):
    """
    同一个报告产品的标准拼接流程：
    1. 先用 order_sn 匹配 report_time >= order_time 且 max_days 天内最近的一条报告；
    2. order_sn 未匹配到有效报告的样本，再用 id_card_no_md5 按同样时间规则补全；
    3. 返回补全后的主表、订单号命中键、身份证补全键。

    df_report_idcard 为空时，默认使用 df_report_order 作为身份证补全表。
    布尔这种拆成两张表的产品，分别传 df_report_order 和 df_report_idcard；
    百融这种同一张表里部分有订单号、全部有身份证的产品，两张表传同一个 DataFrame 即可。
    """
    if df_report_idcard is None:
        df_report_idcard = df_report_order

    df_order_matched, df_order_key = match_nearest_report(
        df_user=df_user,
        df_br_order=df_report_order,
        join_col=order_col,
        left_time_col=left_time_col,
        right_time_col=right_time_col,
        max_days=max_days
    )

    df_result, df_idcard_key = fill_report_by_idcard(
        df_user=df_order_matched,
        df_br_order=df_report_idcard,
        join_col=id_col,
        left_time_col=left_time_col,
        right_time_col=right_time_col,
        max_days=max_days
    )

    df_result[match_type_col] = pd.NA

    if len(df_order_key) > 0:
        order_match_flag = (
            df_order_key[[order_col, left_time_col]]
            .drop_duplicates()
            .assign(_order_match=True)
        )
        df_result = df_result.merge(
            order_match_flag,
            on=[order_col, left_time_col],
            how='left'
        )
        df_result.loc[df_result['_order_match'].eq(True), match_type_col] = order_col
        df_result = df_result.drop(columns=['_order_match'])

    if len(df_idcard_key) > 0:
        idcard_match_flag = (
            df_idcard_key[[id_col, left_time_col]]
            .drop_duplicates()
            .assign(_idcard_match=True)
        )
        df_result = df_result.merge(
            idcard_match_flag,
            on=[id_col, left_time_col],
            how='left'
        )
        df_result.loc[df_result['_idcard_match'].eq(True), match_type_col] = id_col
        df_result = df_result.drop(columns=['_idcard_match'])

    return df_result, df_order_key, df_idcard_key
