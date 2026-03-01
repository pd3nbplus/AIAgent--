# src\Mini_Agent\graph.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage

from src.core.config import settings
from .state import AgentState
from .tools import ALL_TOOLS
from src.utils.xml_parser import extract_tool_calls_from_content
from src.core.redis_client import RedisClient

# 1. 初始化模型
llm = ChatOpenAI(
    model=settings.llm.model_name,
    base_url=settings.llm.base_url,
    api_key=settings.llm.api_key,
    temperature=settings.llm.temperature
)

llm_with_tools = llm.bind_tools(ALL_TOOLS)

# 2. 定义节点
def call_model(state: AgentState):
    messages = state["messages"]
    raw_response = llm_with_tools.invoke(messages)
    
    extracted_tool_calls = extract_tool_calls_from_content(raw_response.content)
    
    response = AIMessage(
        content=raw_response.content,
        tool_calls=extracted_tool_calls
    )
    return {"messages": [response]}

tool_node = ToolNode(ALL_TOOLS)

# 3. 路由逻辑
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END

# 4. 构建图
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    END: END
})
workflow.add_edge("tools", "agent")

# 5. 编译 (带 Checkpointer)
redis_client_instance = RedisClient()
memory = redis_client_instance.redis_saver
memory.setup()

app = workflow.compile(checkpointer=memory)