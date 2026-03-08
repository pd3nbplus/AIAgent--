import asyncio
import re
from typing import Optional

from langchain_core.messages import HumanMessage

from src.agent.orchestrator import RoutedAgentExecutor
from src.Mini_Agent.graph import app


def run_mini_agent(query: str, thread_id: str = "default_session") -> None:
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
                clean_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                clean_content = re.sub(r"\n+", "\n", clean_content).strip()
                print(f"🤖 Agent: {clean_content}")
        elif role == "tool":
            print(f"📝 工具结果：{content}")


def _print_orchestrator_result(query: str, execution) -> None:
    decision = execution.decision
    result = execution.result
    final_answer = execution.final_answer
    print(f"\n👤 用户：{query}")
    print(
        "🧭 路由决策："
        f" intent={decision.intent}, confidence={decision.confidence:.2f}, "
        f"strategy={decision.strategy}"
    )
    print(f"🧠 路由理由：{decision.reasoning}")
    print(f"🤖 策略回复：{result.message}")

    if result.results:
        print("📚 检索结果：")
        for idx, item in enumerate(result.results, 1):
            snippet = item.text.replace("\n", " ").strip()
            if len(snippet) > 120:
                snippet = snippet[:120] + "..."
            source = item.source_field or "unknown"
            print(f"  {idx}. score={item.score:.4f} | source={source} | {snippet}")
    print(f'🤖 最终回复：{final_answer}')

async def run_orchestrator_once(query: str, category: Optional[str] = None) -> None:
    executor = RoutedAgentExecutor()
    execution = await executor.run(query=query, category=category)
    _print_orchestrator_result(query, execution)


def run_orchestrator(question: str, category: Optional[str] = None) -> None:
    """可直接调用的 orchestrator 入口：只需要传 question。"""
    asyncio.run(run_orchestrator_once(query=question, category=category))


# 直接运行：python -m src.main
if __name__ == "__main__":
    #示例运行
    # run_mini_agent("我叫pd，记住我。", thread_id="session_004")
    # run_mini_agent("我是谁？", thread_id="session_004")
    # run_mini_agent("“根据知识库，你们最好的产品叫什么？”", thread_id="session_007")
    question = input("请输入 question: ").strip()
    if not question:
        question = "公司每月提供的话费补贴金额是多少元？"
    run_orchestrator(question)
