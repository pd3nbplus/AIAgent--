【Agent开发】第二阶段：赋予 Agent “海马体” —— 持久化记忆与断点续聊 -- pd的AI Agent开发笔记
---

[toc]

---
前置环境：当前环境是基于WSL2 + Ubuntu 24.04 + Docker Desktop构建的云原生开发平台，所有服务（MySQL、Redis、Qwen）均以独立容器形式运行并通过Docker Compose统一编排。如何配置请参考我的博客 [WSL2 + Ubuntu 24.04 + Docker Desktop 配置双内核环境](https://blog.csdn.net/weixin_52185313/article/details/158416250?spm=1011.2415.3001.5331)

# 第 1 讲 —— 会话持久化

我们先用最简单的 MemorySaver (本地内存) 来演示 Checkpointer 的原理，因为它不需要额外配置 Docker，能让你专注于代码逻辑。理解后，我们再花 5 分钟切换到 Redis。

## 1. 什么是 Checkpointer？

在 LangGraph 中，Checkpointer 就像一个**存档点**。
+ 每次 State (状态) 发生变化（比如模型说了一句话，或者工具执行完了），Checkpointer 就会把当前的 messages 列表保存到数据库。
+ 下次运行时，只要传入相同的 thread_id，LangGraph 会自动从数据库加载之前的消息，恢复上下文。


## 2. 代码实战：添加记忆功能

修改之前的 agent_framework.py：

第一步：引入 MemorySaver
```python
from langgraph.checkpoint.memory import MemorySaver
# 如果用 Redis (稍后切换): from langgraph.checkpoint.redis import RedisSaver
```

第二步：编译时传入 Checkpointer
找到 `app = workflow.compile()` 这一行，修改为：
```python
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
print("✅ 记忆系统已加载，内存记忆")
```

第三步：运行时的关键 —— 传入 thread_id
这是最重要的一步！如果不传 thread_id，LangGraph 会认为每次都是新对话。

```python
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
                print(f"🤖 Agent: {content}")
        elif role == "tool":
            print(f"📝 工具结果: {content}")
```

完整代码：
```python
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
    
    print("🔧 注入的 tool_calls:", extracted_tool_calls)  # 调试用

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
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
print("✅ 记忆系统已加载，内存记忆")

# --- 5. 运行 Agent --- 
from langchain_core.messages import HumanMessage

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
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        elif role == "tool":
            print(f"📝 工具结果: {content}")

if __name__ == "__main__":
    # 第一轮对话
    run_framework_agent_memory("我叫张三，记住我的名字。", thread_id="session_001")
    
    print("\n--- 模拟程序重启，开启新对话 ---\n")
    
    # 第二轮对话 (使用相同的 thread_id)
    run_framework_agent_memory("我叫什么名字？", thread_id="session_001")
    
    # 第三轮对话 (使用不同的 thread_id，应该是失忆状态)
    run_framework_agent_memory("我叫什么名字？", thread_id="session_002")
```

输出：
```text
✅ 记忆系统已加载，内存记忆

👤 用户 (Thread: session_001): 我叫张三，记住我的名字。
🤖 Agent:已记住您的名字是张三。请问还有其他需要帮助的吗？

--- 模拟程序重启，开启新对话 ---


👤 用户 (Thread: session_001): 我叫什么名字？
🤖 Agent: 您叫张三。

👤 用户 (Thread: session_002): 我叫什么名字？
🤖 Agent:我无法获取您的姓名信息。您需要直接告诉我是谁在提问哦！
```

## 3. 进阶：切换到 Redis

安装依赖,langgraph-checkpoint-redis会自动安装Redis依赖,注意：Redis版本需要>=8.0
```bash
pip install langgraph-checkpoint-redis
```

写一个基类设置Redis连接
```python
# src/service/redis_client.py
import redis
from langgraph.checkpoint.redis import RedisSaver

# 单例模式
class RedisClient:
    _instance = None
    def __new__(cls, *args, **kwargs):
        # 如果实例不存在，则创建新实例
        if not cls._instance:
            cls._instance = super().__new__(cls)
            print("创建新的单例实例")
        else:
            print("返回已存在的单例实例")
        return cls._instance
    def __init__(self):
        self.host = "127.0.0.1"
        self.port = 6379
        self.db = 0
        # TTL 配置，包含：
        # default_ttl：检查点存活时间（分钟），默认 60 分钟。
        # refresh_on_read：读取时是否刷新 TTL，默认 True。
        self.ttl_config = {"default_ttl": 120, "refresh_on_read": False}
        self.redis = redis.Redis(host=self.host, port=self.port, db=self.db)
        self.redis_saver = RedisSaver(redis_client=self.redis,ttl=self.ttl_config)
```

修改记忆注入：
```python
# src/Mini-Agent/agent_framework.py
import sys
import os
# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 现在可以直接导入
from service.redis_client import RedisClient

# redis记忆初始化
memory = RedisClient().redis_saver
memory.setup()
print("✅ redis记忆系统已加载")
app = workflow.compile(checkpointer=memory)
```

输出：
```text
创建新的单例实例
✅ redis记忆系统已加载

👤 用户 (Thread: session_001): 我叫张三，记住我的名字。
🤖 Agent: 已记住您的名字是张三。请问还有其他需要帮助的吗？

--- 模拟程序重启，开启新对话 ---


👤 用户 (Thread: session_001): 我叫什么名字？
🤖 Agent: 您叫张三。

👤 用户 (Thread: session_002): 我叫什么名字？
🤖 Agent: 我无法获取您的姓名信息。您需要直接告诉我是谁在提问哦！
```

# 🕵️‍♂️ 实战演练：Agent 记忆追踪行动

如果上面的对话成功了，现在我们要看看数据到底存在哪了。

## 连接上redis，并查看数据

```json
// checkpoint > checkpoint:session_001 > checkpoint:session_001:__empty__:1f113db2-0629-6e28-8001-35b18e366769
[
  {
    "thread_id": "session_001",
    "run_id": "",
    "checkpoint_ns": "__empty__",
    "checkpoint_id": "1f113db2-0629-6e28-8001-35b18e366769",
    "parent_checkpoint_id": "1f113db1-c5a9-63de-8000-bf8851e7566b",
    "checkpoint_ts": 1772196728729.553,
    "checkpoint": {
      "type": "json",
      "v": 4,
      "ts": "2026-02-27T12:52:08.729552+00:00",
      "id": "1f113db2-0629-6e28-8001-35b18e366769",
      "channel_values": {
        "messages": [
          {
            "lc": 1,
            "type": "constructor",
            "id": [
              "langchain",
              "schema",
              "messages",
              "HumanMessage"
            ],
            "kwargs": {
              "content": "我叫张三，记住我的名字。",
              "type": "human"
            }
          },
          {
            "lc": 1,
            "type": "constructor",
            "id": [
              "langchain",
              "schema",
              "messages",
              "AIMessage"
            ],
            "kwargs": {
              "content": "<think>\n好的，用户说“我叫张三，记住我的名字。”首先，我需要确认用户的要求。用户希望我记住他的名字，张三。接下来，我需要检查是否有需要调用的工具来处理这个请求。\n\n查看提供的工具，有get_current_date和get_current_time，这两个函数分别用于获取当前日期和时间。但用户的需求是记住名字，这并不涉及日期或时间的获取。因此，这两个工具在这里并不适用。\n\n用户可能是在测试我是否能处理简单的指令，或者希望后续对话中能使用这个名字。作为AI，我需要确认是否已经正确记录了用户的名字。根据我的设计，我应该能够记住对话中的关键信息，如用户提到的名字。因此，不需要调用任何工具，直接回应用户即可。\n\n接下来，我应该回复用户，确认已经记住了他的名字，并询问是否有其他需要帮助的地方。这样既符合用户的要求，也保持了对话的流畅性。不需要使用任何工具函数，因为任务已经完成，只需自然语言回应即可。\n</think>\n\n已记住您的名字是张三。请问还有其他需要帮助的吗？",
              "type": "ai",
              "tool_calls": [],
              "invalid_tool_calls": []
            }
          }
        ]
      },
      "channel_versions": {
        "__start__": "00000000000000000000000000000002.0.35019554440721345",
        "messages": "00000000000000000000000000000003.0.1628457894508898",
        "branch:to:agent": "00000000000000000000000000000003.0.1628457894508898"
      },
      "versions_seen": {
        "__input__": {},
        "__start__": {
          "__start__": "00000000000000000000000000000001.0.3573929798569724"
        },
        "agent": {
          "branch:to:agent": "00000000000000000000000000000002.0.35019554440721345"
        }
      },
      "updated_channels": [
        "messages"
      ],
      "pending_sends": []
    },
    "metadata": "{\"source\":\"loop\",\"step\":1,\"parents\":{}}",
    "has_writes": false,
    "source": "loop",
    "step": 1
  }
]
```


## 使用 Python 脚本查看 (更直观)

```python
import redis
import json

# 连接 Redis
r = redis.Redis(host="127.0.0.1", port=6379, db=0, decode_responses=True)

# 查找所有包含 session_001 的 Key
pattern = "*session_001*"
keys = r.keys(pattern)

print(f"🔍 找到 {len(keys)} 个相关 Key:")
for key in keys:
    print(f"\n📂 Key: {key}")
    
    # 获取类型
    k_type = r.type(key)
    print(f"   类型：{k_type}")
    
    # 获取 TTL
    ttl = r.ttl(key)
    print(f"   剩余寿命：{ttl} 秒")
    
    # 获取内容
    if k_type == 'string':
        value = r.get(key)
        # 尝试格式化 JSON
        try:
            data = json.loads(value)
            # 简单打印消息内容
            if 'channel_values' in data and 'messages' in data['channel_values']:
                msgs = data['channel_values']['messages']
                print(f"   📝 内容预览 (最后一条): {msgs[-1]['content'][:50]}...")
            else:
                print(f"   📝 内容预览：{value[:100]}...")
        except:
            print(f"   📝 内容：{value[:100]}...")
```

## 🧠 深度解析：LangGraph Redis 存储结构揭秘

能注意到Redis中总共有4个储存部分
+ checkpoint
+ checkpoint_latest
+ checkpoint_write
+ write_keys_zset

然后里面才是session_001、session_002的数据

### 四大存储组件的作用

看到的四个 Key 前缀，分别承担了不同的职责，共同构成了一个高可用、可回溯的记忆系统：

| Key 前缀 | 角色比喻 | 核心作用 |
| :--- | :--- | :--- |
| `checkpoint` | 📚 历史档案库 | 存储每一个步骤的完整状态快照。就像游戏的每一个存档点。如果程序崩溃，可以从这里恢复。 |
| `checkpoint_latest` | 🚀 快速传送门 | 指向当前线程最新的那个 Checkpoint ID。避免每次都要遍历所有历史记录去找最新状态，极大提升读取速度。 |
| `checkpoint_writes` | ✍️ 增量日志本 | 存储单次执行中产生的新数据（比如刚生成的回复）。采用“写时复制”策略，只有当一步彻底成功后，才合并到主 checkpoint 中。保证原子性。 |
| `write_keys_zset` | 🗂️ 索引目录 | (ZSet = Sorted Set) 维护所有待处理写入操作的有序列表，确保在并发环境下，写入操作按顺序执行，不丢数据。 |

> 💡 架构智慧：
> 这种设计采用了 CQRS (命令查询职责分离) 的思想。
> + 写路径：先写 checkpoint_writes (轻量)，成功后再合并。
> + 读路径：通过 checkpoint_latest 快速定位，读取 checkpoint 中的完整状态。

🎬 情景设定：用户 "IronMan" 的对话流

这是一个非常深刻的问题！你触及到了 **LangGraph Checkpoint Redis** 实现的核心架构。

很多开发者只把 Redis 当作一个“存字符串的字典”，但 LangGraph 团队在这里利用了 Redis 的高级特性（ZSet、原子写）来实现**分布式状态机**最关键的三个需求：
1.  **并发安全**：多个用户同时对话，或者同一个用户快速发送多条消息，数据不能乱。
2.  **断点续传**：程序崩了，重启后能精确知道哪一步没做完。
3.  **时间旅行**：随时回滚到之前的任意一步状态。

你看到的四个 Key 结构，正是为了实现这些目标而设计的"**写时复制 (Copy-On-Write) + 日志追加**"模式。

我们用"**银行记账**"和"**游戏存档**"的例子来通俗解释这四个部分。

---

### 🎬 情景设定：用户 "IronMan" 的对话流

假设用户 "IronMan" 正在进行多轮对话：
1.  **Step 0**: 用户说：“你好”。
2.  **Step 1**: Agent 回复：“你好，我是助手”。
3.  **Step 2**: 用户说：“我叫 IronMan”。
4.  **Step 3**: Agent 思考中... (此时程序突然**崩溃/断电**了！)
5.  **Step 3 (恢复)**: 程序重启，继续执行 Step 3，回复：“好的，记住了”。

我们来看看在这个过程中，Redis 里的四个 Key 是如何协作的。

---

#### 1. `checkpoint`: 完整的历史快照 (Full Snapshots)

> **类比**：**游戏的“存档文件” (.sav)**
> 每一个存档文件都记录了那一刻游戏的**完整状态**（位置、血量、背包）。

*   **结构**：`checkpoint:{thread_id}:{ns}:{checkpoint_id}`
*   **内容**：包含该时刻完整的 `channel_values` (所有消息列表)。
*   **行为**：
    *   当 Step 1 完成时，生成 `checkpoint_..._id_1`，里面存了 `[Msg0, Msg1]`。
    *   当 Step 2 完成时，生成 `checkpoint_..._id_2`，里面存了 `[Msg0, Msg1, Msg2]`。
*   **特点**：
    *   **不可变 (Immutable)**：一旦写入，永不修改。这是“写时复制”的体现——新的状态是旧状态的**副本 + 增量**，而不是直接覆盖旧文件。
    *   **作用**：支持“时间旅行”。如果你想回滚到 Step 1，只需读取 `id_1` 的那个 Key 即可。


#### 2. `checkpoint_writes`: 增量日志 (Write-Ahead Log / Delta)

> **类比**：**银行的“交易流水单”**
> 银行不会每发生一笔交易就重写一次你的总余额文件。它只是记一笔流水：“+100 元”。

*   **结构**：`checkpoint_writes:{thread_id}:{ns}:{checkpoint_id}:{task_id}:{write_index}`
    *   注意这里多了 `task_id` 和 `write_index`，说明它是更细粒度的。
*   **内容**：**只存变化的部分 (Delta)**。比如 Step 3 中，Agent 调用工具产生的那个新消息。
*   **行为**：
    *   在 Step 3 执行过程中，Agent 生成了回复。在它准备更新主存档之前，先把这条新消息写到 `checkpoint_writes` 里。
    *   **关键点**：这个写入是**原子性**的。即使写完这一条就断电了，这条数据也是安全的。
*   **为什么需要它？(解决崩溃问题)**：
    *   如果 Step 3 执行到一半断电了：
        *   `checkpoint_id_2` (Step 2 的存档) 是完整的，没坏。
        *   `checkpoint_writes` 里可能有一条未提交的记录。
    *   **重启后**：LangGraph 会检查 `checkpoint_writes`。发现有一条属于 `checkpoint_id_2` 的未完成写入。它会重放这条日志，把数据补全，然后再生成新的 `checkpoint_id_3`。
    *   **如果没有它**：断电瞬间，内存里的新消息丢了，且没有落盘，Agent 就会丢失刚才的思考结果，甚至陷入死循环。


#### 3. `checkpoint_latest`: 指针 (The Pointer)

> **类比**：**游戏主菜单上的“当前存档槽位”**
> 你硬盘里有 10 个存档文件 (`checkpoint_1` ... `checkpoint_10`)，但主菜单只需要知道**哪一个**是最新的，以便你点击“继续游戏”。

*   **结构**：`checkpoint_latest:{thread_id}:{ns}`
*   **内容**：存储的是**最新那个 Checkpoint 的 ID** (例如 `"1f113db2-..."`)。
*   **行为**：
    *   每次成功生成一个新的 Checkpoint，就更新这个 Key 的值。
    *   这是一个**可变**的 Key。
*   **作用**：**O(1) 快速定位**。
    *   如果不存这个指针，每次启动 Agent，你都得去遍历所有 `checkpoint:*` 的 Key，比较它们的时间戳来找最新的，效率极低。
    *   有了它，LangGraph 只要读这一个 Key，拿到 ID，再去读对应的 `checkpoint` 大文件，瞬间恢复状态。

---

#### 4. `write_keys_zset`: 调度器与垃圾回收 (The Scheduler & GC)

> **类比**：**机场的“航班起飞时间表” + “过期行李清理清单”**
> 这是一个 **Sorted Set (ZSet)**。ZSet 的特点是：每个成员都有一个 **Score (分数)**，通常是时间戳。

*   **结构**：`write_keys_zset:{thread_id}` (或者是全局的，取决于配置)
*   **内容**：
    *   Member: `checkpoint_writes` 的 Key 名称。
    *   Score: 该写入发生的**时间戳**。
*   **核心作用 1：崩溃恢复的“待办清单”**
    *   当系统重启时，它不仅看 `checkpoint_latest`，还会查这个 ZSet。
    *   它会问：“有没有哪些 `checkpoint_writes` 是发生在最新 Checkpoint 之后，但还没被合并进主存档的？”
    *   通过 Score (时间戳) 排序，它能按顺序重放这些未完成的写入。

*   **核心作用 2：TTL 与 垃圾回收 (GC)** ⭐ **这是 ZSet 最妙的地方**
    *   你之前配置了 `ttl=3600` (1 小时过期)。
    *   Redis 的 Key 过期是异步的，有时候不靠谱。而且 `checkpoint` 是一串链表，删了一个中间的，后面的怎么办？
    *   LangGraph 利用 ZSet 的 Score 来做**主动清理**：
        1.  定期（或每次启动时）查询 ZSet：`ZRANGEBYSCORE ... 0 (当前时间 - TTL)`。
        2.  找出所有**超过 1 小时**的 `checkpoint_writes` Key。
        3.  批量删除这些 Key，并从 ZSet 中移除它们。
        4.  同时，如果某个旧的 `checkpoint` 不再被任何未完成的 write 引用，也可以被安全删除。
    *   **为什么不用 Redis 原生 TTL？** 因为我们需要**事务性**。删除数据和更新索引必须在逻辑上保持一致，ZSet 提供了这种有序的管理能力。

---

#### 🔄 综合流程演示：一次完整的“写时复制”

让我们把刚才的 **Step 2 -> Step 3** 过程串联起来：

1.  **初始状态**：
    *   `checkpoint_latest` 指向 `CP_2` (Step 2 完成)。
    *   `checkpoint:CP_2` 存有 `[Msg0, Msg1, Msg2]`。
    *   `write_keys_zset` 是空的 (假设之前的都清理了)。

2.  **开始 Step 3 (Agent 思考并生成回复)**：
    *   LangGraph 读取 `CP_2` 到内存。
    *   Agent 生成新消息 `Msg3`。

3.  **预写日志 (Write-Ahead)**：
    *   系统生成一个唯一的 `write_id`。
    *   将 `Msg3` 写入 `checkpoint_writes:...:write_id`。
    *   **关键**：同时将这个 Key 加入 `write_keys_zset`，Score 设为当前时间戳。
    *   *(此时如果断电，数据已在 Redis，下次可恢复)*

4.  **创建新快照 (Copy-On-Write)**：
    *   系统在内存中合并 `CP_2` 的数据 + `Msg3`，形成新状态 `State_3`。
    *   生成新的 ID `CP_3`。
    *   将 `State_3` 写入新的 Key `checkpoint:CP_3`。
    *   *(注意：`checkpoint:CP_2` 依然原封不动地留在那里，这就是“写时复制”)*

5.  **提交与清理 (Commit)**：
    *   更新 `checkpoint_latest` 指向 `CP_3`。
    *   从 `write_keys_zset` 中移除刚才那个 `write_id` (因为已经合并进主存档了，不需要再作为“待办事项”了)。
    *   (可选) 删除 `checkpoint_writes:...:write_id` 以节省空间。

6.  **垃圾回收 (GC)**：
    *   后台进程扫描 `write_keys_zset`。
    *   发现 `CP_1` 相关的 writes 已经是 2 小时前的了 (Score 很小)。
    *   删除 `CP_1` 的存档和相关 writes，释放内存。

---

#### 💡 为什么要这么复杂？(设计哲学)

你可能会想：“直接 `SET messages [...]` 覆盖不就行了吗？”

1.  **并发冲突**：
    *   如果是微服务架构，两个容器同时处理同一个用户的请求（虽然少见，但在重试机制下可能发生）。
    *   直接覆盖会导致**丢失更新** (Lost Update)。
    *   使用 `checkpoint_writes` + `ZSet` + 版本号机制，可以检测到冲突，确保只有合法的写入被合并。

2.  **Exactly-Once 语义**：
    *   在分布式系统中，网络抖动可能导致请求重发。
    *   通过 `write_index` 和 ZSet 的记录，LangGraph 能保证同一条消息**只被处理一次**，不会重复回复用户。

3.  **人类介入 (Human-in-the-loop)**：
    *   因为保留了所有历史 `checkpoint` (直到 GC)，你可以随时让时间倒流。
    *   比如用户说“不对，重来”，你可以直接把 `checkpoint_latest` 指回 `CP_2`，Agent 就会忘记 `CP_3` 发生的一切，重新生成。

| Key 前缀 | 角色 | 可变性 | 核心作用 | 类比 |
| :--- | :--- | :--- | :--- | :--- |
| **`checkpoint`** | **存档文件** | ❌ 不可变 | 存储完整状态快照，支持时间旅行 | 游戏 `.sav` 文件 |
| **`checkpoint_writes`** | **事务日志** | ✅ 追加写 | 防止崩溃丢数据，实现原子写入 | 银行流水单 |
| **`checkpoint_latest`** | **指针** | ✅ 可变 | 快速找到最新存档，O(1) 加载 | 游戏“继续”按钮 |
| **`write_keys_zset`** | **调度器** | ✅ 增删改 | 管理待处理写入，基于时间戳做 GC | 航班时刻表 + 清理清单 |

### JSON 数据结构逐字段拆解

聚焦checkpoint内容，这是最核心的部分。

**A. 身份标识 (Identity)**
```json
"thread_id": "session_001",
"checkpoint_ns": "__empty__", 
"checkpoint_id": "1f113db2-0629-6e28-8001-35b18e366769",
"parent_checkpoint_id": "1f113db1-c5a9-63de-8000-bf8851e7566b"
```
+ `thread_id`: 你的会话 ID。这是隔离不同用户数据的钥匙。
+ `checkpoint_ns`: 命名空间。__empty__ 表示这是主流程。如果你以后搞多 Agent 协作（子图），这里会出现子图的名字（如 agent_router），实现嵌套状态管理。
+ `checkpoint_id`: 当前这一步的唯一 UUID。
+ `parent_checkpoint_id`: 关键！ 它指向上一步的 ID。这就形成了一条链表。
   + 作用：支持时间旅行。你可以拿着这个 ID 去加载“上一秒”的状态，实现“撤销”、“回滚”或“分支剧情”功能。

**B. 核心数据：channel_values (记忆的真相)**
```json
"channel_values": {
  "messages": [
    {
      "lc": 1,
      "type": "constructor",
      "id": ["langchain", "schema", "messages", "HumanMessage"],
      "kwargs": {
        "content": "我叫张三，记住我的名字。",
        "type": "human"
      }
    },
    {
      // ... AIMessage ...
      "kwargs": {
        "content": "<think>...</think>\n\n已记住您的名字...",
        "type": "ai",
        "tool_calls": [], 
        "invalid_tool_calls": []
      }
    }
  ]
}
```

+ `channel_values`: 这里存放着你定义的 `AgentState` 中的所有变量。因为我只定义了 `messages`，所以这里只有 `messages` 键。如果你定义了 `user_name: str`，这里也会多一个 `user_name` 键。
+ LangChain 序列化格式 (`lc`, `type`, `id`):
   + 这不是普通的 JSON，这是 LangChain 特定的序列化协议。
   + `lc: 1`: 协议版本。
   + `id`: 告诉反序列化器，这个对象对应 Python 里的哪个类 (langchain.schema.messages.HumanMessage)。
   + `kwargs`: 真正的构造函数参数。
   + 意义：无论你的 Python 代码怎么变，只要遵循这个协议，存进去的对象就能原样还原出来，包括复杂的对象方法（如果支持的话）。

**C. 版本控制：channel_versions & versions_seen**
```json
"channel_versions": {
  "messages": "00000000000000000000000000000003.0.1628457894508898",
  "branch:to:agent": "..."
},
"versions_seen": {
  "agent": {
    "branch:to:agent": "..."
  }
}
```

+ 这是什么？ 这是 LangGraph 状态机引擎的“心跳”。
+ 作用：
   + 每个 Channel (如 `messages`) 都有一个版本号。每次有新消息加入，版本号递增。
   + `versions_seen` 记录每个节点（Node）最后看到的版本。
   + 核心逻辑：当 Agent 运行时，它会检查：“我上次处理到版本 3 了，现在最新版本是 4 吗？如果是，我就需要再次运行；如果还是 3，说明没新事，我可以休息。”
   + 这防止了无限循环和重复处理，是 ReAct 循环能自动停止的底层保障。

**D. 更新标记：updated_channels**

```json
"updated_channels": ["messages"]
```
+ 告诉系统：在这一步操作中，只有 messages 通道发生了变化。其他通道（如果有）保持不变。这有助于优化增量更新。


## 🔍 实战验证：利用结构特性“篡改”记忆

```python
if __name__ == "__main__":
    run_framework_agent_memory("我叫张三，记住我的名字。", thread_id="session_003")

    # 然后手动去修改相关记忆内容 checkpoint:session_003:__empty__:{checkpoint_id} 把"张三"改成"pd"，然后再运行下面

    # 第二轮对话 (使用相同的 thread_id)
    run_framework_agent_memory("我叫什么名字？", thread_id="session_003")
```

运行结果如下：
```text
创建新的单例实例
✅ redis记忆系统已加载

👤 用户 (Thread: session_003): 我叫什么名字？
🤖 Agent: 您叫pd。
```


# 构建“长期记忆” —— MySQL 用户画像系统

本讲目标：利用 MySQL 存储用户的关键事实（如名字、偏好、职业），实现跨会话的永久记忆。

核心区别：
+ Redis (短期)：存完整的对话流，有 TTL，会过期，适合上下文理解。
+ MySQL (长期)：存结构化的“事实”，永久保存，适合用户画像和个性化服务。

目录结构：
```bash
./src
├── Mini_Agent
│   ├── __init__.py
│   ├── agent_framework.py
│   ├── agent_v1.py
│   ├── graph.py
│   ├── state.py
│   └── tools
├── core
│   ├── __init__.py
│   ├── config.py
│   ├── db_session.py
│   └── models.py
├── main.py
├── service
│   ├── __init__.py
│   └── redis_client.py
├── test
│   ├── __init__.py
│   └── redis_look.py
└── utils
    ├── __init__.py
    └── xml_parser.py
```
## 1. 数据库设计：用户画像表 (user_profiles)

我们需要一张表来存储 Key-Value 形式的用户信息。为了支持扩展，我们设计得灵活一些。

**🗄️ SQL 建表语句**

执行以下 SQL：
```sql
-- 创建数据库（如果还没创建）
CREATE DATABASE IF NOT EXISTS agent_memory CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE agent_memory;

-- 创建用户画像表
CREATE TABLE IF NOT EXISTS user_profiles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    thread_id VARCHAR(255) NOT NULL,       -- 关联会话 ID (可选，也可以按 user_id)
    user_key VARCHAR(100) NOT NULL,        -- 记忆的键，如 "name", "favorite_food"
    user_value TEXT NOT NULL,              -- 记忆的值，如 "IronMan", "Pizza"
    confidence_score FLOAT DEFAULT 1.0,    -- 置信度 (0.0-1.0)，Agent 不确定时可以存低分
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- 索引优化：快速查找某个会话的所有记忆，或特定键
    UNIQUE KEY uk_thread_key (thread_id, user_key), 
    INDEX idx_thread (thread_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

💡 设计思路解析
+ `thread_id`：我们将记忆绑定到会话。未来可以扩展为 user_id（如果做了用户登录系统）。
+ `user_key` / `user_value`：采用 EAV (Entity-Attribute-Value) 模型的简化版。这样不需要为每个属性（名字、年龄、喜好）单独建列，扩展性极强。
+ `UNIQUE KEY`：确保同一个会话下，同一个 key（如 "name"）只有一条记录。更新时直接 `ON DUPLICATE KEY UPDATE`。
+ `confidence_score`：这是一个高级字段。如果 Agent 听不清或不确定用户说的是不是事实，可以存 0.5。查询时只查 > 0.8 的高置信度数据。

## 2. 开发记忆工具：Save & Get

我们需要两个工具让 Agent 操作数据库。为了代码整洁，我们先封装一个 DB 连接类。

### 🛠️ 步骤 A：安装依赖、定义模型 (Model)

```bash
pip install pymysql sqlalchemy
```
(使用 SQLAlchemy 可以让代码更通用，避免原生 SQL 注入风险)

我们需要一个 Python 类来映射数据库表。
```py
# src/core/models.py
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, UniqueConstraint, Index
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class UserProfile(Base):
    __tablename__ = 'user_profiles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(String(255), nullable=False, index=True)  # 会话ID
    user_key = Column(String(100), nullable=False)               # 键：name, preference
    user_value = Column(Text, nullable=False)                    # 值：IronMan, Pizza
    confidence_score = Column(Float, default=1.0)                # 置信度
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 定义唯一约束：同一个 session 下，key 不能重复
    __table_args__ = (
        UniqueConstraint('thread_id', 'user_key', name='uk_thread_key'),
        Index('idx_thread', 'thread_id'),
    )

    def __repr__(self):
        return f"<UserProfile(thread_id={self.thread_id}, key={self.user_key}, value={self.user_value})>"
```

💡 优势：
+ 如果你以后想换 PostgreSQL，只需改连接字符串，这个类完全不用动。
+ `__table_args__` 直接在代码里定义了索引和约束，不需要去数据库手动敲 SQL。
+ `datetime.utcnow` 自动处理时间戳。

设定配置文件
```py
import os
from dotenv import load_dotenv

load_dotenv() # 加载 .env 文件

class Settings:
    # LLM 配置
    LLM_MODEL_NAME = os.getenv("LLM_MODEL", "qwen-3-4b")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:7575/v1")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMP", "0.05"))
    
    # Redis 配置
    REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    
    # MySQL 配置
    DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:xxxxxx@localhost:3307/agent_memory")

settings = Settings()
```

### 🛠️ 步骤 B：封装数据库会话 (Session Manager)

为了保证连接池的高效利用，我们用一个依赖注入风格的上下文管理器。

```py
# src/core/db_session.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from src.core.models import Base
from src.core.config import settings

# 创建引擎
engine = create_engine(
    settings.DATABASE_URL, 
    pool_pre_ping=True,          # 自动检测断连并重连
    pool_size=10,                # 连接池大小
    max_overflow=20,             # 最大溢出连接数
    echo=False                   # 生产环境关闭 SQL 日志，调试时设为 True
)

# 创建 Session 工厂
# scoped_session 确保在多线程/异步环境下每个线程有独立的 Session
SessionLocal = scoped_session(sessionmaker(bind=engine))

def init_db():
    """初始化数据库：如果表不存在则自动创建"""
    Base.metadata.create_all(bind=engine)
    print("✅ 数据库表结构已同步 (ORM 模式)")

def get_db_session():
    """获取数据库会话"""
    return SessionLocal()
```


### 🛠️ 步骤 C：编写 memory_tools.py

```py
# src\Mini_Agent\tools\memory_tools.py
# src\Mini_Agent\tools\memory_tools.py
from langchain_core.tools import tool
from src.core.db_session import get_db_session
from src.core.models import UserProfile
from sqlalchemy import select
from typing import List

@tool
def save_user_memory(key: str, value: str, thread_id: str) -> str:
    """
    保存用户的关键信息到长期记忆。
    参数：
        key: 信息类别 (如 'name', 'job')
        value: 具体内容
        thread_id: 当前会话 ID
    """
    session = get_db_session()
    try:
        # 1. 尝试查询是否已存在
        stmt = select(UserProfile).where(
            UserProfile.thread_id == thread_id,
            UserProfile.user_key == key
        )
        existing_profile = session.execute(stmt).scalars().first()

        if existing_profile:
            # 2. 如果存在，更新值
            existing_profile.user_value = value
            existing_profile.confidence_score = 1.0
            # updated_at 会自动由 onupdate 触发
        else:
            # 3. 如果不存在，插入新记录
            new_profile = UserProfile(
                thread_id=thread_id,
                user_key=key,
                user_value=value,
                confidence_score=1.0
            )
            session.add(new_profile)

        session.commit()
        return f"✅ 成功记住：{key} = {value}"
    
    except Exception as e:
        session.rollback()
        return f"❌ 保存记忆失败：{str(e)}"
    finally:
        session.close()

@tool
def get_user_memory(keys: List[str], thread_id: str) -> str:
    """
    查询用户的长期记忆。
    参数：
        keys: 想要查询的键列表 (如 ['name', 'job'])。为空则查所有。
        thread_id: 当前会话 ID
    """
    session = get_db_session()
    try:
        query = select(UserProfile).where(UserProfile.thread_id == thread_id)
        
        if keys:
            query = query.where(UserProfile.user_key.in_(keys))
        
        # 按创建时间排序，确保拿到最新的（虽然 unique 约束保证只有一个）
        query = query.order_by(UserProfile.created_at.desc())
        
        results = session.execute(query).scalars().all()
        
        if not results:
            return "ℹ️ 未找到相关记忆。"
        
        # 格式化为易读字符串
        memory_list = [f"{p.user_key}: {p.user_value}" for p in results]
        return "🧠 长期记忆加载：" + "; ".join(memory_list)

    except Exception as e:
        return f"❌ 查询记忆失败：{str(e)}"
    finally:
        session.close()

memory_tools = [save_user_memory, get_user_memory]
```

✨ 代码亮点解析
1. 类型安全：
   + `UserProfile` 是强类型的 Python 对象。IDE 会自动提示 user_value, thread_id 等属性，写错字段名直接报错，不用等到运行时。
2. 防 SQL 注入：
   + `session.execute(stmt)` 内部会自动处理参数转义。你再也不用担心 `'; DROP TABLE users; --` 这种攻击了。
3. 逻辑清晰：
   + `select(...).where(...)` 的链式调用非常像自然语言，可读性远超拼接 SQL 字符串。
   + `scalars().first()` 和 `scalars().all()` 是直接取出模型对象，不需要再手动解析元组。
4. 事务管理：
   + `commit()` 和 `rollback()` 显式控制事务，确保数据一致性。

把之前的获取时间的函数也放到tools下
```py
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
```

收集导出的工具列表
```python
# src\Mini_Agent\tools\__init__.py
from .base_tools import base_tools
from .memory_tools import memory_tools

# 统一导出所有工具
ALL_TOOLS = base_tools + memory_tools
```

### 🛠️ 步骤 D：重构智能体编排

分层架构 (Layered Architecture) 进行重构：
+ Tools Layer: 独立管理所有工具。
+ State Layer: 独立定义状态结构。
+ Graph Layer: 专注构建工作流（节点、边、编译）。
+ Entry Point: 只负责运行和交互。

**状态定义 (AgentState)**
```py
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
```

**LangGraph 构建逻辑 (Nodes, Edges, Compile)**

```py
# src\Mini_Agent\graph.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage

from src.core.config import settings
from .state import AgentState
from .tools import ALL_TOOLS
from src.utils.xml_parser import extract_tool_calls_from_content
from src.service.redis_client import RedisClient

# 1. 初始化模型
llm = ChatOpenAI(
    model=settings.LLM_MODEL_NAME,
    base_url=settings.LLM_BASE_URL,
    api_key=settings.LLM_API_KEY,
    temperature=settings.LLM_TEMPERATURE
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
```

#### 🛠️ 步骤 E：编写main.py

```python
import re
from langchain_core.messages import HumanMessage
from src.Mini_Agent.graph import app

def run_agent(query: str, thread_id: str = "default_session"):
    print(f"\n👤 用户 (Thread: {thread_id}): {query}")
    
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=query)]}
    
    for event in app.stream(inputs, config, stream_mode="values"):
        last_msg = event["messages"][-1]
        role = last_msg.type
        content = last_msg.content
        
        if role == "ai":
            if last_msg.tool_calls:
                print(f"🧠 思考：准备调用工具 {[t['name'] for t in last_msg.tool_calls]}")
            else:
                # 清理输出
                clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                clean_content = re.sub(r'\n+', '\n', clean_content).strip()
                print(f"🤖 Agent: {clean_content}")
        elif role == "tool":
            print(f"📝 工具结果：{content}")

# python -m src.main
if __name__ == "__main__":
    # 示例运行
    run_agent("我叫pd，记住我。", thread_id="session_004")
    run_agent("我是谁？", thread_id="session_004")
```

在控制台中输入：
```bash
python -m src.main
```

控制台输出：
```py
用户 (Thread: session_004): 我叫pd，记住我。
🧠 思考：准备调用工具 ['save_user_memory']
📝 工具结果：✅ 成功记住：name = pd
🤖 Agent: 已记住您的名字是pd。如果您有其他需要保存的信息，也可以随时告诉我哦！😊

👤 用户 (Thread: session_004): 我是谁？
🧠 思考：准备调用工具 ['get_user_memory']
📝 工具结果：🧠 长期记忆加载：name: pd
🤖 Agent: 您是pd。如果您有其他问题或需要帮助，随时告诉我哦！😊
```

可以在mysql数据库中查看结果:

![alt text](./img/mysql存储结果.png)


## ✅ 现在的架构全貌

你的 AI Agent 平台现在已经非常完善了：

| 组件 | 技术栈 | 作用 | 数据形式 |
| :--- | :--- | :--- | :--- |
| 短期记忆 | Redis + LangGraph Checkpointer | 对话上下文、状态快照、断点续传 | JSON Blob (Checkpoints) |
| 长期记忆 | MySQL + SQLAlchemy ORM | 用户画像、永久事实、偏好设置 | Structured Rows (ORM Objects) |
| 大脑 | Qwen (Docker) + LangGraph | 推理、规划、工具调用 | Prompts & Tool Calls |
| 工具链 | Python Functions | 时间、数据库读写、API 调用 | Functions |

这才是真正的云原生、工程化的 AI Agent 架构！

