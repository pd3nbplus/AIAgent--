# src\Mini_Agent\tools\base_tools.py

from langchain_core.tools import tool
from datetime import datetime

@tool
def get_current_date() -> str:
    """获取当前的日期和星期几。适用于询问今天周几、日期的问题。"""
    now = datetime.now()
    # 直接返回包含星期的字符串，杜绝幻觉
    return f"日期：{now.strftime('%Y-%m-%d')}, 星期：{now.strftime('%A')}"

@tool
def get_current_time() -> str:
    """获取当前的具体时间（时:分:秒）。适用于询问几点、剩余时间的问题。"""
    return f"时间：{datetime.now().strftime('%H:%M:%S')}"

# 导出列表
base_tools = [get_current_date, get_current_time]