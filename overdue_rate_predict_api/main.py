'''
 # @ Author: Wanyh
 # @ Create Time: 2026/5/7 14:45
 # @ Modified by: Wanyh
 # @ Modified time: 2026/5/7 14:45
 # @ Description:
 '''

import json
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from flask import Flask, request, jsonify
from src.db_service import query_order_interest,validate_time_filter
from src.predict_data_cleaner import handle_single_order,handle_multi_order,handle_full_data
import logging


app = Flask(__name__)


logger = logging.getLogger("score_logger")
logger.setLevel(logging.INFO)
# 防止重复添加 handler
if not logger.handlers:
    file_handler = logging.FileHandler(
        "modelFile.log",
        encoding='utf-8'
    )
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

try:
    with open(r'./src/LR_mob6ever15_Backtracking_binning.json', 'r', encoding='utf-8') as f:
        LR_mob6ever15_Backtracking_binning = json.load(f)
except Exception as e:
    logger.error(f"加载模型时出现问题: {str(e)}")
    raise RuntimeError("加载模型时出现问题，请检查模型文件的有效性。") from e


@app.route('/predict', methods=['POST'])
def scoreCard():

    try:
        data = request.get_json() or {}

        model_name = data.get("model_name")
        model_version = data.get("model_version", "default")
        params = data.get("params", {})

        logger.info(f"请求参数: {data}")

        if not model_name:
            return jsonify({
                "status": "failed",
                "error": "缺少 model_name"
            }), 400

        if not isinstance(params, dict):
            return jsonify({
                "status": "failed",
                "error": "params 必须是 JSON 对象"
            }), 400

        # 模型路由
        if model_name == "LR_mob6ever15":
            scorecard_dict = LR_mob6ever15_Backtracking_binning
        else:
            return jsonify({
                "status": "failed",
                "error": f"模型未识别: {model_name}"
            }), 400

        # 从 params 取业务参数
        mode = params.get("mode")
        order_sn = params.get("order_sn")

        use_time_filter = params.get("use_time_filter", False)
        start_time = params.get("start_time")
        end_time = params.get("end_time")

        if isinstance(use_time_filter, str):
            use_time_filter = use_time_filter.lower() == "true"

        if mode not in [1, 2, 3]:
            return jsonify({
                "status": "failed",
                "error": "mode 只能是 1、2、3"
            }), 400

        # mode=2/3 不允许传分页和时间筛选
        if mode in [2, 3]:

            if "page" in params or "page_size" in params:
                return jsonify({
                    "status": "failed",
                    "error": "mode=2/3 不支持分页参数"
                }), 400

            if use_time_filter:
                return jsonify({
                    "status": "failed",
                    "error": "mode=2/3 不支持时间筛选"
                }), 400

            if "start_time" in params or "end_time" in params:
                return jsonify({
                    "status": "failed",
                    "error": "mode=2/3 不支持 start_time/end_time"
                }), 400

        page = None
        page_size = None

        # mode=1 全量模式
        if mode == 1:

            page = int(params.get("page", 1))
            page_size = int(params.get("page_size", 1000))

            if page < 1:
                return jsonify({
                    "status": "failed",
                    "error": "page 必须大于等于 1"
                }), 400

            if page_size < 1:
                return jsonify({
                    "status": "failed",
                    "error": "page_size 必须大于等于 1"
                }), 400

            start_time, end_time = validate_time_filter(
                use_time_filter,
                start_time,
                end_time
            )

        # mode=2 单订单
        elif mode == 2:

            if not order_sn:
                return jsonify({
                    "status": "failed",
                    "error": "mode=2 时必须传 order_sn"
                }), 400

        # mode=3 多订单
        elif mode == 3:
            if not order_sn:
                return jsonify({
                    "status": "failed",
                    "error": "mode=3 时必须传多个订单号"
                }), 400

        # 查询数据
        df = query_order_interest(
            mode=mode,
            order_sn=order_sn,
            page=page,
            page_size=page_size,
            use_time_filter=use_time_filter,
            start_time=start_time,
            end_time=end_time
        )

        logger.info(f"查询结果: {df.to_dict()}")

        if df.empty:

            response = {
                "status": "success",
                "model_name": model_name,
                "model_version": model_version,
                "logic": mode,
                "order_count": 0,
                "data": []
            }

            if mode == 1:
                response["page"] = page
                response["page_size"] = page_size
                response["use_time_filter"] = use_time_filter
                response["start_time"] = start_time
                response["end_time"] = end_time

            return jsonify(response)

        # 打分逻辑
        if mode == 1:
            result_df = handle_full_data(df, scorecard_dict)
            logger.info(f"全量打分完成，订单数: {len(result_df)}")

        elif mode == 2:
            result_df = handle_single_order(df, scorecard_dict)
            logger.info(f"单订单打分完成: {result_df}")

        elif mode == 3:
            result_df = handle_multi_order(df, scorecard_dict)
            logger.info(f"多订单打分完成，订单数: {len(result_df)}")

        if isinstance(result_df, dict):
            count = 1
        elif isinstance(result_df, list):
            count = len(result_df)
        else:
            count = 0

        response = {
            "status": "success",
            "model_name": model_name,
            "model_version": model_version,
            "logic": mode,
            "order_count": count,
            "data": result_df
        }

        if mode == 1:
            response["page"] = page
            response["page_size"] = page_size
            response["use_time_filter"] = use_time_filter
            response["start_time"] = start_time
            response["end_time"] = end_time

        return jsonify(response)

    except ValueError as e:
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 400

    except Exception as e:
        logger.exception("接口异常")
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500


if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False
    )