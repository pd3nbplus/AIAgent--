# LLM 路由模块：
# 从一个 JSON 文件加载多个 endpoint，并按顺序调用（失败则降级到下一个）。
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.augmented.config import GeneratorConfig
from src.utils.xml_parser import remove_think_and_n

logger = logging.getLogger(__name__)


@dataclass
class LLMEndpoint:
    # 对应 JSON 里的单个 LLM 节点配置。
    url: str
    model: str
    api_key: str
    temperature: float = 0.7


class LLMRouter:
    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.endpoints = self._load_endpoints(config.llm_json_path)
        # 实例级降级游标：
        # 一旦第 k 个失败，后续调用将从第 k+1 个开始，不再尝试前 k 个。
        self._degrade_start_idx = 0

    def _load_endpoints(self, json_path: str) -> List[LLMEndpoint]:
        # 支持两种 JSON 形态：
        # 1) {"llms": [...]} 2) [...]
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        records = payload.get("llms", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list) or not records:
            raise ValueError(f"LLM JSON 配置无效或为空: {json_path}")

        endpoints: List[LLMEndpoint] = []
        for item in records:
            endpoints.append(
                LLMEndpoint(
                    url=item["url"],
                    model=item["model"],
                    api_key=item.get("api_key", "not-needed"),
                    temperature=float(item.get("temperature", 0.7)),
                )
            )
        return endpoints

    def invoke(self, prompt: ChatPromptTemplate, payload: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        # 顺序降级：第一个 endpoint 失败后切到下一个。
        last_error: Optional[Exception] = None
        for idx in range(self._degrade_start_idx, len(self.endpoints)):
            ep = self.endpoints[idx]
            try:
                llm = ChatOpenAI(
                    model=ep.model,
                    base_url=ep.url,
                    api_key=ep.api_key if ep.api_key else "not-needed",
                    temperature=ep.temperature,
                )
                chain = prompt | llm
                res = chain.invoke(payload)
                text = remove_think_and_n(getattr(res, "content", "") or "")
                if text:
                    return text, ep.model
            except Exception as e:
                last_error = e
                # 触发实例级熔断：失败的当前节点及其之前节点都不再尝试。
                self._degrade_start_idx = max(self._degrade_start_idx, idx + 1)
                logger.warning("⚠️ LLM 调用失败，降级到下一个 endpoint。model=%s err=%s", ep.model, e)
                continue

        raise RuntimeError(f"所有 LLM endpoint 调用失败: {last_error}")
