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
    # run_agent("我叫pd，记住我。", thread_id="session_004")
    # run_agent("我是谁？", thread_id="session_004")
    run_agent("“根据知识库，你们最好的产品叫什么？”", thread_id="session_007")