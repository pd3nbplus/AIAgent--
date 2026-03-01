# src/rag/strategies/retrievers/vector_text.py
from src.rag.strategies.base import BaseRetrievalStrategy, SearchResult
from src.core.milvus_client import get_milvus_client
import logging
from typing import List

logger = logging.getLogger(__name__)

class VectorTextRetriever(BaseRetrievalStrategy):
    """
    主路: 标准向量检索
    目标字段：text
    输入查询：原始查询
    """
    def __init__(self):
        self.milvus = get_milvus_client()
        logger.info("🔌 加载插件：VectorTextRetriever (原文向量检索)")

    def search(self, query: str, top_k: int, filter_expr: str = None, **kwargs) -> List[SearchResult]:
        logger.debug(f"🔍 [VectorText] 检索：{query}")
        hits = self.milvus.search(
            query=query,
            top_k=top_k,
            filter_expr=filter_expr,
            output_fields=["text", "metadata"]
        )
        
        return [
            SearchResult(
                text=h['text'],
                score=h['score'],
                metadata=h['metadata'],
                source_field="vector_text"
            ) for h in hits
        ]