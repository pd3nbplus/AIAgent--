from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from src.agent.router import IntentRouter, RouteDecision
from src.common.llm_adapter import LLMAdapter
from src.core.prompt_registry import PROMPT_KEYS, core_prompt_registry
from src.agent.strategies import (
    BaseAgentStrategy,
    ClarifyNeededStrategy,
    CodeSearchStrategy,
    DeepSearchStrategy,
    DirectReplyStrategy,
    FastRetrievalStrategy,
    FallbackStrategy,
    StandardRetrievalStrategy,
    StrategyContext,
    StrategyName,
    StrategyResult,
)

# intent 到执行策略的固定映射（用于稳定线上行为）
INTENT_TO_STRATEGY: Dict[str, StrategyName] = {
    "CHIT_CHAT": "direct_reply",
    "FACT_LOOKUP": "fast_retrieval",
    "HOW_TO": "standard_retrieval",
    "COMPARISON": "deep_search",
    "CODE_SEARCH": "code_search",
    "UNKNOWN": "fallback",
}


@dataclass(slots=True)
class RoutedExecution:
    """一次路由执行的完整结果：包含路由决策与策略执行产物。"""

    decision: RouteDecision
    result: StrategyResult
    final_answer: str


class AgentGraphState(TypedDict, total=False):
    """LangGraph 运行状态。

    字段约定（按执行顺序）：
    1) run() 初始化时写入 `query/category`；
    2) route 节点写入 `decision/selected_strategy`；
    3) strategy 节点写入 `result`；
    4) compose 节点写入 `final_answer`。
    """

    # 原始用户问题：所有节点都会用到
    query: str
    # 可选业务分类（例如产品线），用于策略检索过滤
    category: Optional[str]
    # route 节点产出的结构化路由结果（意图/置信度/建议策略）
    decision: RouteDecision
    # 由 orchestrator.resolve_strategy_name 决定的最终策略名（用于 conditional_edges 分流）
    selected_strategy: StrategyName
    # 具体策略执行后的统一产物（文本消息 + 可选检索结果）
    result: StrategyResult
    # compose 节点最终写回的用户可见答案
    final_answer: str


class AgentStrategyOrchestrator:
    """策略编排器：负责将 RouteDecision 分发到具体策略实现。"""

    def __init__(self, registry: Optional[Dict[StrategyName, BaseAgentStrategy]] = None):
        self.registry = registry or self._build_default_registry()

    @staticmethod
    def _build_default_registry() -> Dict[StrategyName, BaseAgentStrategy]:
        return {
            "direct_reply": DirectReplyStrategy(),
            "fast_retrieval": FastRetrievalStrategy(),
            "standard_retrieval": StandardRetrievalStrategy(),
            "deep_search": DeepSearchStrategy(),
            "code_search": CodeSearchStrategy(),
            "fallback": FallbackStrategy(),
            "clarify_needed": ClarifyNeededStrategy(),
        }

    def resolve_strategy_name(self, decision: RouteDecision) -> StrategyName:
        """解析最终执行策略。

        规则：
        1) 明确的 clarify_needed 优先；
        2) 低置信度自动转 clarify_needed；
        3) 否则按 intent 强映射；
        4) 最后才回退到 decision.strategy/fallback。
        """
        if decision.strategy == "clarify_needed" and "clarify_needed" in self.registry:
            return "clarify_needed"

        if decision.confidence < 0.6 and "clarify_needed" in self.registry:
            return "clarify_needed"

        # 优先按 intent 做强映射，避免 LLM 输出 strategy 偏移导致执行不稳定。
        by_intent = INTENT_TO_STRATEGY.get(decision.intent)
        if by_intent and by_intent in self.registry:
            return by_intent

        by_strategy = decision.strategy.lower().strip()
        if by_strategy in self.registry:
            return by_strategy  # type: ignore[return-value]
        return "fallback"

    async def execute(
        self,
        query: str,
        decision: RouteDecision,
        category: Optional[str] = None,
    ) -> StrategyResult:
        strategy_name = self.resolve_strategy_name(decision)
        strategy = self.registry[strategy_name]
        return await strategy.execute(
            StrategyContext(
                query=query,
                category=category,
                intent=decision.intent,
                route_strategy=decision.strategy,
                clarification_questions=decision.clarification_questions,
            )
        )


class RoutedAgentExecutor:
    """高层执行入口：使用 LangGraph 实现 route->strategy->compose。"""

    _DIRECT_REPLY_PROMPT = (
        "你是一个简洁、准确的中文助手。请直接回答用户问题，不要输出思维过程。\n\n"
        "问题：\n{query}"
    )

    def __init__(
        self,
        router: Optional[IntentRouter] = None,
        orchestrator: Optional[AgentStrategyOrchestrator] = None,
        llm: Optional[LLMAdapter] = None,
    ):
        self.router = router or IntentRouter()
        self.orchestrator = orchestrator or AgentStrategyOrchestrator()
        self.llm = llm or LLMAdapter()
        self.rag_answer_prompt = core_prompt_registry.get(PROMPT_KEYS.SELF_RAG_GENERATE)
        self.app = self._build_graph().compile()

    def _build_graph(self):
        # 图结构固定为：START -> route -> (strategy_x) -> compose -> END
        workflow = StateGraph(AgentGraphState)
        workflow.add_node("route", self._route_node)
        workflow.add_node("compose", self._compose_node)

        for strategy_name in self.orchestrator.registry:
            # 每个策略是一个独立节点，便于后续按节点做观测/重试/限流
            workflow.add_node(strategy_name, self._build_strategy_node(strategy_name))
            workflow.add_edge(strategy_name, "compose")

        workflow.add_edge(START, "route")
        workflow.add_conditional_edges(
            "route",
            # route 节点写入 selected_strategy 后，在这里决定下一跳
            self._select_strategy_from_state,
            {name: name for name in self.orchestrator.registry},
        )
        workflow.add_edge("compose", END)
        return workflow

    async def _route_node(self, state: AgentGraphState) -> AgentGraphState:
        # 只读取 query，产出 decision + selected_strategy
        query = state["query"]
        decision = await self.router.route(query)
        selected_strategy = self.orchestrator.resolve_strategy_name(decision)
        return {
            "decision": decision,
            "selected_strategy": selected_strategy,
        }

    @staticmethod
    def _select_strategy_from_state(state: AgentGraphState) -> StrategyName:
        selected = state.get("selected_strategy")
        if selected:
            return selected
        return "fallback"

    def _build_strategy_node(self, strategy_name: StrategyName):
        async def _run_strategy(state: AgentGraphState) -> AgentGraphState:
            # 读取 route 阶段产物，执行对应策略，写回 result
            query = state["query"]
            decision = state["decision"]
            category = state.get("category")
            strategy = self.orchestrator.registry[strategy_name]
            result = await strategy.execute(
                StrategyContext(
                    query=query,
                    category=category,
                    intent=decision.intent,
                    route_strategy=decision.strategy,
                    clarification_questions=decision.clarification_questions,
                )
            )
            return {"result": result}

        return _run_strategy

    @staticmethod
    def _build_context_text(result: StrategyResult, max_items: int = 3, max_chars: int = 1500) -> str:
        chunks = []
        for item in result.results[:max_items]:
            text = (item.text or "").strip()
            if not text:
                continue
            chunks.append(text[:max_chars])
        return "\n\n".join(chunks) if chunks else "（无可用上下文）"

    async def _compose_final_answer(self, query: str, result: StrategyResult) -> str:
        no_llm_strategies = {"clarify_needed", "fallback", "code_search"}
        if result.strategy in no_llm_strategies:
            return result.message

        try:
            if result.strategy == "direct_reply":
                text = await self.llm.generate_text(
                    template=self._DIRECT_REPLY_PROMPT,
                    payload={"query": query},
                )
                text = text.strip()
                return text or result.message

            context_text = self._build_context_text(result)
            text = await self.llm.generate_text(
                template=self.rag_answer_prompt,
                payload={"query": query, "contexts": context_text},
            )
            text = text.strip()
            return text or result.message
        except Exception:
            # 生成失败时回退到策略消息，保证主链路可用
            return result.message

    async def _compose_node(self, state: AgentGraphState) -> AgentGraphState:
        # 基于 query + result 组装最终答案，写回 final_answer
        query = state["query"]
        result = state["result"]
        final_answer = await self._compose_final_answer(query=query, result=result)
        return {"final_answer": final_answer}

    async def run(self, query: str, category: Optional[str] = None) -> RoutedExecution:
        """对外统一调用方法。"""
        final_state = await self.app.ainvoke(
            {
                "query": query,
                "category": category,
            }
        )
        decision = final_state["decision"]
        result = final_state["result"]
        final_answer = final_state["final_answer"]
        return RoutedExecution(decision=decision, result=result, final_answer=final_answer)
