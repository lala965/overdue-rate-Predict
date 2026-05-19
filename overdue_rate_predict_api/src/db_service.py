'''
 # @ Author: Wanyh
 # @ Create Time: 2026/5/7 14:14
 # @ Modified by: Wanyh
 # @ Modified time: 2026/5/7 14:14
 # @ Description:
 '''

import pymysql
import pandas as pd
from datetime import datetime

class CONFIG:
    username = 'yjia'
    password = 'Yjia@123'
    host = 'am-t4ne09l1kjusi1nae153470o.singapore.ads.aliyuncs.com'
    database = 'erp'
    port = 3306


def get_conn():
    return pymysql.connect(
        host=CONFIG.host,
        user=CONFIG.username,
        password=CONFIG.password,
        database=CONFIG.database,
        port=CONFIG.port,
        charset='utf8mb4'
    )


def validate_time_filter(use_time_filter, start_time, end_time):
    if not use_time_filter:
        return None, None

    if not start_time or not end_time:
        raise ValueError("开启时间筛选时必须传 start_time 和 end_time")

    try:
        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise ValueError("时间格式错误，必须是 YYYY-MM-DD HH:MM:SS")

    if start_dt > end_dt:
        raise ValueError("start_time 不能大于 end_time")

    return start_time, end_time

def query_order_interest(
        mode=1,
        order_sn=None,
        page=1,
        page_size=1000,
        use_time_filter=False,
        start_time=None,
        end_time=None
):
    """
    mode=1：查全量
    mode=2：查单个订单，order_sn='订单号'
    mode=3：查多个订单，order_sn='订单1,订单2,订单3'
    """

    base_sql = """
        select a.order_sn,a.order_time,
               c.flag_score,
               ifnull(c.scorecust2,-9999) as scorecust2,
               ifnull(c.`scoreywbase_V3_0`,-9999) as `scoreywbase_V3_0`,
               ifnull(c.scorertstdd1_V1_2,-9999) as `scorertstdd1_V1_2`,
               ifnull(c.scoreywpro_V1_3,-9999) as `scoreywpro_V1_3`,
               ifnull(c.scoreysstd_V3_3_1,-9999) as `scoreysstd_V3_3_1`,
                b.flag_interestprefer,
               ifnull(b.ip_m24_id_ra_orgnum,0) as ip_m24_id_ra_orgnum,
               ifnull(b.ip_m6_id_rate_mean,0) as ip_m6_id_rate_mean,
               ifnull(b.ip_m3_id_rc_orgnum_ratio,0) as ip_m3_id_rc_orgnum_ratio,
               ifnull(b.ip_m24_id_rc_allnum_ratio,0) as ip_m24_id_rc_allnum_ratio,
               ifnull(b.ip_m2_cell_rc_orgnum_ratio,0) as ip_m2_cell_rc_orgnum_ratio,
               ifnull(b.ip_m24_cell_rc_orgnum_ratio,0) as ip_m24_cell_rc_orgnum_ratio
        from dwd_rent_order a
        inner join dwd_backtrack_bairong_interest_prefer_v3_risk_report b on a.platform_order_sn = b.order_sn
        inner join dwd_backtrack_bairong_score_risk_report c  on a.order_sn = c.order_sn

        where a.is_revenue_valid=1
          and b.flag_interestprefer=1
          
    """



    if mode == 1:
        params = None

        page = int(page)
        page_size = int(page_size)

        offset = (page - 1) * page_size

        sql = base_sql
        params = []

        if use_time_filter:
            sql += """
                and a.order_time >= %s
                and a.order_time < %s
            """
            params.extend([start_time, end_time])

        sql += """
            order by a.order_time
            limit %s, %s
        """

        params.extend([offset, page_size])
        params = tuple(params)

    elif mode == 2:
        if not order_sn:
            raise ValueError("mode=2 时必须传入 order_sn")
        sql = base_sql + " and a.order_sn = %s order by a.order_time"
        params = (order_sn,)

    elif mode == 3:
        if not order_sn:
            raise ValueError("mode=3 时必须传入多个订单号，用英文逗号分隔")

        order_list = [x.strip() for x in order_sn.split(",") if x.strip()]
        placeholders = ",".join(["%s"] * len(order_list))

        sql = base_sql + f" and a.order_sn in ({placeholders}) order by a.order_time"
        params = tuple(order_list)

    else:
        raise ValueError("mode 只能是 1、2、3")

    conn = get_conn()

    try:
        df = pd.read_sql(sql, conn, params=params)
        return df
    finally:
        conn.close()