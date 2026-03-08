import json
import re
import requests
from datetime import datetime
from typing import Optional, Dict, Any
from src.core.prompt_registry import PROMPT_KEYS, core_prompt_registry

# --- 1. 定义工具集 ---

def get_current_date(*args, **kwargs) -> str:
    """获取当前的日期（包含年月日和星期几）"""
    now = datetime.now()
    # %Y-%m-%d: 2026-02-26
    # %A: 完整的星期名称 (Thursday), %a: 缩写 (Thu)
    # 为了保险，我们同时返回中文星期（如果系统 locale 支持）或英文
    # 这里为了通用性，返回明确的格式
    date_str = now.strftime("%Y-%m-%d")
    weekday_str = now.strftime("%A") # 例如 "Thursday"
    
    # 简单映射到中文，防止模型对英文星期反应迟钝（可选，视模型语言能力强弱）
    weekday_map = {
        "Monday": "星期一", "Tuesday": "星期二", "Wednesday": "星期三",
        "Thursday": "星期四", "Friday": "星期五", "Saturday": "星期六", "Sunday": "星期日"
    }
    cn_weekday = weekday_map.get(weekday_str, weekday_str)
    
    return f"日期：{date_str}, 星期：{cn_weekday} ({weekday_str})"

def get_current_time(*args, **kwargs) -> str:
    """获取当前的具体时间（时:分:秒）"""
    now = datetime.now()
    return f"时间：{now.strftime('%H:%M:%S')}"

# 更新注册表
TOOLS_REGISTRY = {
    "get_current_date": {
        "description": "当用户询问日期、今天几号、今天是星期几时使用。不需要参数。",
        "function": get_current_date
    },
    "get_current_time": {
        "description": "当用户询问具体几点、当前时刻时使用。不需要参数。",
        "function": get_current_time
    }
}

# 更新工具定义 (Schema)
TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "获取当前的日期（包含年月日和星期几）",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的具体时间（时:分:秒）",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]

# --- 2. LLM 交互层 ---
# 检查模型名
# curl http://localhost:7575/v1/models
# "id": "qwen-3-4b",  <-- 🎯 这就是你要的 model name！

MODEL_NAME = "qwen-3-4b"
LLM_API_URL = "http://localhost:7575/v1/chat/completions"

def call_llm(messages: list, tools_desc: str) -> Dict[str, Any]:
    full_system_prompt = core_prompt_registry.get(PROMPT_KEYS.MINI_AGENT_V1_SYSTEM).format(
        tools_desc=tools_desc
    )
    
    payload = {
        "model": MODEL_NAME, # 模型名需与你容器内一致
        "messages": [
            {"role": "system", "content": full_system_prompt},
            *messages
        ],
        "temperature": 0.05, # Agent 需要低温度以保证逻辑稳定
        "stream": False
    }
    
    try:
        response = requests.post(LLM_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        content = data['choices'][0]['message']['content']
        
        # 2. 【新增】去除 Qwen 的思维链标签 (<think> ... </think>)
        # 使用正则匹配 <think> 开头到 </think> 结尾的内容，并替换为空
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        # 清理可能存在的 markdown 标记
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        
        # 3. 去除首尾空白字符
        content = content.strip()

        return json.loads(content.strip())
        
    except Exception as e:
        print(f"❌ LLM 调用失败: {e}")
        # 如果是连接错误，提示用户检查 URL
        if "Connection refused" in str(e):
            print("💡 提示：请检查 LLM_API_URL 是否正确。如果在 WSL2 直接运行，尝试改用 http://localhost:<端口>")
        return {"thought": "系统错误", "action": None}

# --- 3. ReAct 引擎 ---

def run_agent(user_query: str):
    messages = [{"role": "user", "content": user_query}]
    print(f"👤 用户: {user_query}")
    
    max_steps = 5  # 防止死循环
    step = 0
    
    while step < max_steps:
        step += 1
        print(f"\n--- 🔄 第 {step} 轮思考 ---")
        
        # 1. 请求 LLM
        result = call_llm(messages, json.dumps(TOOLS_DEFINITION, ensure_ascii=False))
        thought = result.get("thought", "")
        action = result.get("action")
        action_input = result.get("action_input", {})
        
        print(f"🧠 思考: {thought}")
        
        # 2. 判断是否结束
        if not action:
            print(f"🤖 Agent: {thought}") # 此时 thought 里通常包含最终回答
            break
        
        # 3. 执行工具
        if action in TOOLS_REGISTRY:
            tool_func = TOOLS_REGISTRY[action]["function"]
            print(f"🛠️ 执行工具: {action}")
            try:
                # 简单起见，这里假设所有工具都返回字符串
                observation = tool_func(**action_input)
                print(f"📝 观察结果: {observation}")
                
                # 将结果反馈给 LLM
                messages.append({"role": "assistant", "content": json.dumps(result, ensure_ascii=False)})
                messages.append({"role": "user", "content": f"工具 {action} 执行完毕，结果是：{observation}。请根据结果回答用户。"})
                
            except Exception as e:
                error_msg = f"工具执行出错: {str(e)}"
                print(f"❌ {error_msg}")
                messages.append({"role": "user", "content": error_msg})
        else:
            print(f"⚠️ 未知工具: {action}")
            messages.append({"role": "user", "content": f"错误：找不到工具 {action}。请重试。"})
            
    if step >= max_steps:
        print("⚠️ 达到最大思考步数，停止。")

# --- 4. 启动 ---
if __name__ == "__main__":
    # 测试问题
    query = "现在几点了？顺便告诉我今天是星期几。"
    run_agent(query)
