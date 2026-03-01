# --- 1. 定义工具集 ---
# 不再需要手动写 JSON Schema，框架会自动从函数签名和文档字符串生成。

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

tools = [get_current_date, get_current_time]

# # --- 调试：打印工具信息 ---
# print("\n🔍 已注册的工具列表:")
# for t in tools:
#     print(f"  - 名称：{t.name}")
#     print(f"  - 描述：{t.description}")
#     print(f"  - 参数：{t.args_schema}")
#     print("-" * 30)
# # -----------------------
# --- 2. 定义状态 ---
# 这是 LangGraph 的核心。我们需要定义一个“容器”，用来在节点之间传递数据。
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
import operator

# 定义状态结构
class AgentState(TypedDict):
    # messages 是一个消息列表，operator.add 表示每次更新状态时，新消息会追加到列表后面（而不是覆盖）
    messages: Annotated[Sequence[BaseMessage], operator.add]


# --- 3. 构建节点 (Nodes) ---
# 节点就是具体的执行逻辑。我们需要两个主要节点：模型节点 和 工具执行节点。
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

# 1. 初始化模型 (指向你的本地 Qwen)
# base_url 指向你的 Docker 服务
llm = ChatOpenAI(
    model="qwen-3-4b",  # 替换为你之前查到的真实模型名
    base_url="http://localhost:7575/v1", # 注意加上 /v1
    api_key="not-needed", # 本地通常不需要 key
    temperature=0.05
)

# 将工具绑定到模型上，这样模型就知道自己有这些能力了
llm_with_tools = llm.bind_tools(tools)

import re,json
from langchain_core.messages import AIMessage

def extract_tool_calls_from_content(content: str):
    """从 content 中提取所有被<tool_call>包围的工具调用 JSON"""
    tool_calls = []
    matches = re.findall(r"<tool_call>\s*({.*?})\s*</tool_call>", content, re.DOTALL)
    for i, match in enumerate(matches):
        try:
            data = json.loads(match.strip())
            name = data.get("name")
            arguments = data.get("arguments", {})
            if name:
                # 注意：LangChain 内部使用 'args' 而不是 'arguments'
                tool_calls.append({
                    "name": name,
                    "args": arguments,          # ⚠️ 关键：用 'args'
                    "id": f"call_{i}_{name}",   # 必须有唯一 id
                    "type": "tool_call"
                })
        except Exception as e:
            print(f"❌ 解析工具调用失败: {e}")
    return tool_calls
# 2. 定义“思考”节点
def call_model(state: AgentState):
    messages = state["messages"]
    raw_response = llm_with_tools.invoke(messages)
    # 提取工具调用
    extracted_tool_calls = extract_tool_calls_from_content(raw_response.content)
    
    # 创建新的 AIMessage，注入 tool_calls
    response = AIMessage(
        content=raw_response.content,      # 保留原始思考过程（可选）
        tool_calls=extracted_tool_calls    # 👈 核心：手动注入
    )
    
    # print("🔧 注入的 tool_calls:", extracted_tool_calls)  # 调试用

    return {"messages": [response]}

# 3. 定义“工具执行”节点
# LangGraph 内置了 ToolNode，自动处理工具查找和参数解析，不用我们自己写 registry 了！
tool_node = ToolNode(tools)


# --- 4. 构建图 (Graph) & 路由逻辑 --- 
# 这是替代 while 循环的部分。我们需要定义流程怎么走。

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

import sys
import os
# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 现在可以直接导入
from core.redis_client import RedisClient

# 路由函数：决定下一步是去“调工具”还是“直接结束”
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    
    # 如果最后一条消息是 AI 发出的，且包含工具调用请求
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools" # 去工具节点
    return END # 否则结束

# 1. 初始化图
workflow = StateGraph(AgentState)

# 2. 添加节点
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# 3. 添加边 (Edges)
workflow.add_edge(START, "agent") # 从开始直接进入 agent 思考
workflow.add_conditional_edges(
    "agent", 
    should_continue, # 根据思考结果决定去向
    {
        "tools": "tools", # 如果有工具调用，去 tools 节点
        END: END          # 如果没有，结束
    }
)


workflow.add_edge("tools", "agent") # 工具执行完后，必须回到 agent 节点进行下一轮思考（形成闭环！）

# 4. 编译成可执行的应用
# 内存初始化
# memory = MemorySaver()
# app = workflow.compile(checkpointer=memory)
# print("✅ 记忆系统已加载，内存记忆")

# redis记忆初始化
memory = RedisClient().redis_saver
memory.setup()
print("✅ redis记忆系统已加载")
app = workflow.compile(checkpointer=memory)


# --- 5. 运行 Agent --- 
from langchain_core.messages import HumanMessage

def run_framework_agent(query: str):
    print(f"👤 用户：{query}")
    inputs = {"messages": [HumanMessage(content=query)]}
    
    # stream 模式可以实时看到每一步的执行（思考 -> 工具 -> 结果）
    for event in app.stream(inputs, stream_mode="values"):
        last_msg = event["messages"][-1]
        role = last_msg.type
        content = last_msg.content
        
        # 简单格式化输出
        if role == "ai":
            if last_msg.tool_calls:
                print(f"🧠 思考: 准备调用工具 {[t['name'] for t in last_msg.tool_calls]}")
            else:
                print(f"🤖 Agent: {content}")
        elif role == "tool":
            print(f"📝 工具结果: {content}")

# 如果不传 thread_id，LangGraph 会认为每次都是新对话。
def run_framework_agent_memory(query: str, thread_id: str = "user_123"):
    print(f"\n👤 用户 (Thread: {thread_id}): {query}")
    
    # 配置运行时参数，必须包含 thread_id
    config = {"configurable": {"thread_id": thread_id}}
    
    inputs = {"messages": [HumanMessage(content=query)]}
    
    # 传入 config 参数
    for event in app.stream(inputs, config, stream_mode="values"):
        last_msg = event["messages"][-1]
        role = last_msg.type
        content = last_msg.content
        
        # 简单格式化输出
        if role == "ai":
            if last_msg.tool_calls:
                print(f"🧠 思考: 准备调用工具 {[t['name'] for t in last_msg.tool_calls]}")
            else:
                # 匹配思考部分 输出内容去掉思考
                think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                # 去掉思考内容 去掉黄行
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                content = re.sub(r'\n', '', content, flags=re.DOTALL)
                print(f"🤖 Agent: {content}")
        elif role == "tool":
            print(f"📝 工具结果: {content}")

if __name__ == "__main__":
    # query = "现在几点了？顺便告诉我今天是星期几。"
    # run_framework_agent(query)
    # 第一轮对话
    # run_framework_agent_memory("我叫张三，记住我的名字。", thread_id="session_003")
    
    # print("\n--- 模拟程序重启，开启新对话 ---\n")
    
    # # 第二轮对话 (使用相同的 thread_id)
    run_framework_agent_memory("我叫什么名字？", thread_id="session_003")
    
    # # 第三轮对话 (使用不同的 thread_id，应该是失忆状态)
    # run_framework_agent_memory("我叫什么名字？", thread_id="session_002")
