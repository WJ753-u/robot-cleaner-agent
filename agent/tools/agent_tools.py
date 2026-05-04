import os
from rag.rag_service import RagSummarizeService
import random
import csv
import json
from langchain_core.tools import tool
from agent.tools.mcp_tools import get_amap_weather
from utils.config_handler import agent_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
rag = RagSummarizeService()
user_ids = ["1001","1002","1003"]
month_arr = ["2025-08", "2025-09", "2025-10"]
external_data = {}
@tool
def rag_summarize(query:str) -> str:
    """
    从向量数据库中检索与查询相关的资料文档，然后使用RAG模型生成总结。
    """
    return rag.rag_summarize(query)
@tool
def get_weather(city:str) -> str:
    """
    通过高德 MCP 获取城市实时天气信息，以消息字符串的形式返回。
    """
    return get_amap_weather(city)
@tool
def get_user_location() -> str:
    """
    获取用户的当前位置信息,以消息字符串的形式返回
    """
    return random.choice(["北京市","沈阳市","加利福尼亚州"])#随机选择一个城市作为用户当前位置
@tool
def get_user_id() -> str:
    """
    获取用户的ID,以消息字符串的形式返回
    """
    return random.choice(user_ids)
@tool
def get_current_month() -> str:
    """
    获取当前月份,以消息字符串的形式返回
    """
    return random.choice(month_arr)

def generate_external_data():
    """
    {
        "user_id":{
            "month": {"特征": xxx, "效率": xxx, ...}
            "month": {"特征": xxx, "效率": xxx, ...}
            "month": {"特征": xxx, "效率": xxx, ...}
        },
        "user_id":{
            "month": {"特征": xxx, "效率": xxx, ...}
            "month": {"特征": xxx, "效率": xxx, ...}
            "month": {"特征": xxx, "效率": xxx, ...}
        },
        "user_id":{
            "month": {"特征": xxx, "效率": xxx, ...}
            "month": {"特征": xxx, "效率": xxx, ...}
            "month": {"特征": xxx, "效率": xxx, ...}
        },
    }
    :return:
    """
    if not external_data:
        external_data_path = get_abs_path(agent_config["external_data_path"])
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件不存在: {external_data_path}")
        with open(external_data_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_id:str = row["用户ID"]
                feature:str = row["特征"]
                efficiency:str = row["清洁效率"]
                consumables:str = row["耗材"]
                comparison:str = row["对比"]
                time:str = row["时间"]
                if user_id not in external_data:
                    external_data[user_id] = {}
                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "比较": comparison,                    
                }
@tool
def fetch_external_data(user_id:str, month:str) -> str:
    """
    获取外部数据,得到指定用户在指定月份的使用记录，以消息字符串的形式返回
    """
    generate_external_data()
    try:
        return json.dumps(external_data[user_id][month], ensure_ascii=False)
    except KeyError:
        logger.warning(f"[fetch_external_data]检测到用户{user_id}在{month}没有使用记录")
        return ""
@tool
def fill_context_for_report():
    """
    无入参无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息
    """
    return "fill_context_for_report已调用"
