from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class PromptKeys:
    AGENT_INTENT_ROUTER: str = "agent.intent_router"

    RAG_REWRITER_STANDARD_SYSTEM: str = "rag.rewriter.standard_system"
    RAG_REWRITER_HYDE_SYSTEM: str = "rag.rewriter.hyde_system"
    RAG_REWRITER_HUMAN_QUERY: str = "rag.rewriter.human_query"
    RAG_INGESTION_ENHANCE_SYSTEM: str = "rag.ingestion.enhance_system"
    RAG_INGESTION_ENHANCE_HUMAN: str = "rag.ingestion.enhance_human"

    AUGMENTED_STANDARD: str = "augmented.standard"
    AUGMENTED_ADVERSARIAL: str = "augmented.adversarial"
    AUGMENTED_MIXED_PAIR: str = "augmented.mixed_pair"
    AUGMENTED_STRICT_NEGATIVE_FIRST: str = "augmented.strict_negative_first"
    AUGMENTED_EVALUATOR_ANSWER: str = "augmented.evaluator.answer"
    AUGMENTED_ANALYST_DIAGNOSIS: str = "augmented.analyst.diagnosis"

    SELF_RAG_GENERATE: str = "self_rag.generate"
    SELF_RAG_JUDGE_RELEVANCE: str = "self_rag.judge_relevance"
    SELF_RAG_JUDGE_GROUNDING: str = "self_rag.judge_grounding"
    SELF_RAG_JUDGE_UTILITY: str = "self_rag.judge_utility"
    SELF_RAG_REWRITE_QUERY: str = "self_rag.rewrite_query"

    MINI_AGENT_V1_SYSTEM: str = "mini_agent.v1.system"


PROMPT_KEYS = PromptKeys()


PROMPT_FILE_MAP: Dict[str, str] = {
    PROMPT_KEYS.AGENT_INTENT_ROUTER: "agent/intent_router.txt",

    PROMPT_KEYS.RAG_REWRITER_STANDARD_SYSTEM: "rag/rewriter_standard_system.txt",
    PROMPT_KEYS.RAG_REWRITER_HYDE_SYSTEM: "rag/rewriter_hyde_system.txt",
    PROMPT_KEYS.RAG_REWRITER_HUMAN_QUERY: "rag/rewriter_human_query.txt",
    PROMPT_KEYS.RAG_INGESTION_ENHANCE_SYSTEM: "rag/ingestion_enhance_system.txt",
    PROMPT_KEYS.RAG_INGESTION_ENHANCE_HUMAN: "rag/ingestion_enhance_human.txt",

    PROMPT_KEYS.AUGMENTED_STANDARD: "augmented/standard.txt",
    PROMPT_KEYS.AUGMENTED_ADVERSARIAL: "augmented/adversarial.txt",
    PROMPT_KEYS.AUGMENTED_MIXED_PAIR: "augmented/mixed_pair.txt",
    PROMPT_KEYS.AUGMENTED_STRICT_NEGATIVE_FIRST: "augmented/strict_negative_first.txt",
    PROMPT_KEYS.AUGMENTED_EVALUATOR_ANSWER: "augmented/evaluator_answer.txt",
    PROMPT_KEYS.AUGMENTED_ANALYST_DIAGNOSIS: "augmented/analyst_diagnosis.txt",

    PROMPT_KEYS.SELF_RAG_GENERATE: "self_rag/generate.txt",
    PROMPT_KEYS.SELF_RAG_JUDGE_RELEVANCE: "self_rag/judge_relevance.txt",
    PROMPT_KEYS.SELF_RAG_JUDGE_GROUNDING: "self_rag/judge_grounding.txt",
    PROMPT_KEYS.SELF_RAG_JUDGE_UTILITY: "self_rag/judge_utility.txt",
    PROMPT_KEYS.SELF_RAG_REWRITE_QUERY: "self_rag/rewrite_query.txt",

    PROMPT_KEYS.MINI_AGENT_V1_SYSTEM: "mini_agent/v1_system.txt",
}


class CorePromptRegistry:
    """统一 Prompt 注册中心（Prompt 文本统一存放于 core/prompt/*.txt）。"""

    def __init__(self, prompt_file_map: Dict[str, str], base_dir: Path | None = None):
        self._prompt_file_map = dict(prompt_file_map)
        self._base_dir = base_dir or (Path(__file__).resolve().parent / "prompt")
        self._cache: Dict[str, str] = {}

    def get(self, key: str) -> str:
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        relpath = self._prompt_file_map.get(key)
        if relpath is None:
            raise KeyError(f"Prompt key not found: {key}")

        path = self._base_dir / relpath
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found for key={key}: {path}")

        content = path.read_text(encoding="utf-8").strip()
        self._cache[key] = content
        return content


core_prompt_registry = CorePromptRegistry(PROMPT_FILE_MAP)
