# augmented/router.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from langchain_openai import ChatOpenAI
from src.core.config import settings
from src.core.prompt_registry import PROMPT_KEYS, core_prompt_registry
from src.utils.xml_parser import remove_think_and_n
# from .llm_router import get_llm_client # 假设你有一个获取 LLM 的工厂函数

# 1. 定义输出结构 (Schema)
class RouteDecision(BaseModel):
    """路由决策对象：给出意图、策略、置信度以及可选澄清问题。"""

    intent: Literal["CHIT_CHAT", "FACT_LOOKUP", "HOW_TO", "COMPARISON", "CODE_SEARCH", "UNKNOWN"] = Field(
        description="用户问题的意图分类"
    )
    confidence: float = Field(
        description="分类的置信度 (0.0 - 1.0)"
    )
    reasoning: str = Field(
        description="简短的分类理由，用于调试"
    )
    strategy: Literal["direct_reply", "fast_retrieval", "standard_retrieval", "deep_search", "code_search", "fallback", "clarify_needed"] = Field(
        description="建议的后续处理策略名称"
    )
    clarification_questions: Optional[List[str]] = Field(
        default=None,
        description="低置信度时用于澄清需求的问题列表；高置信度时为 null"
    )

# 2. 统一从 core prompt registry 读取模板
ROUTER_PROMPT_TEMPLATE = core_prompt_registry.get(PROMPT_KEYS.AGENT_INTENT_ROUTER)

class IntentRouter:
    """意图路由器：使用 LLM 将用户问题映射到可执行策略。"""

    def __init__(self, llm_model: str = "gpt-3.5-turbo"): 
        # 注意：路由任务很简单，可以用小模型 (gpt-3.5/haiku) 以节省成本和延迟
        self.llm = ChatOpenAI(
            base_url=settings.llm.base_url,
            model=settings.llm.model_name,
            api_key=settings.llm.api_key,
            temperature=float(settings.llm.temperature),
        )

        def extract_and_clean(message):
            # message 可能是 AIMessage 对象，也可能是字符串 (取决于版本和配置)
            content = message.content if hasattr(message, 'content') else str(message)
            # 移除 think 和 n 标记
            return remove_think_and_n(content)

        clean_chain = RunnableLambda(extract_and_clean)
        self.parser = PydanticOutputParser(pydantic_object=RouteDecision)
        
        prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT_TEMPLATE)
        
        # 构建链：Prompt -> LLM -> clean_chain -> Parser
        self.chain = prompt | self.llm | clean_chain | self.parser

    async def route(self, question: str) -> RouteDecision:
        """
        对单个问题进行路由决策
        """
        try:
            decision = await self.chain.ainvoke({"question": question})
            return decision
        except Exception as e:
            # 解析失败时的降级策略：再次确认
            print(f"⚠️ 路由解析失败：{e}，默认降级为确认问题")
            # 降级时也可以返回一个特殊的 clarify_needed，或者直接走标准检索
            return RouteDecision(
                intent="UNKNOWN",
                confidence=0.0,
                reasoning="Parsing failed or model error",
                strategy="clarify_needed",
                clarification_questions=["您的问题似乎有些复杂，能再详细描述一下具体场景吗？", "您是想查询政策、操作流程还是技术文档？"]
            )
# 测试脚本
if __name__ == "__main__":
    # python -m src.agent.router
    import asyncio
    
    async def test_router():
        router = IntentRouter()
        queries = [
            "你好，在吗？",
            "公司的年假政策是怎样的？",
            "如何重置我的登录密码？",
            "对比一下 v1.0 和 v2.0 版本的 API 响应速度。",
            "Python 里怎么用 pandas 读取 csv？",
            "我有如下需求："
        ]
        
        for q in queries:
            print(f"\n❓ Q: {q}")
            res = await router.route(q)
            print(f"🎯 Intent: {res.intent}")
            print(f"🙋 Confidence: {res.confidence}")
            print(f"💡 Strategy: {res.strategy}")
            print(f"🧠 Reasoning: {res.reasoning}")

            if res.clarification_questions:
                print("❓ 需要澄清，建议询问用户:")
                for i, cq in enumerate(res.clarification_questions, 1):
                    print(f"   {i}. {cq}")
            else:
                print("✅ 意图清晰，直接执行策略。")
            print("-" * 30)
    asyncio.run(test_router())
