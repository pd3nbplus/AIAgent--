# src\Mini_Agent\state.py
# --- 2. 定义状态 ---
# 这是 LangGraph 的核心。我们需要定义一个“容器”，用来在节点之间传递数据。
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
import operator

# 定义状态结构
class AgentState(TypedDict):
    # messages 是一个消息列表，operator.add 表示每次更新状态时，新消息会追加到列表后面（而不是覆盖）
    messages: Annotated[Sequence[BaseMessage], operator.add]