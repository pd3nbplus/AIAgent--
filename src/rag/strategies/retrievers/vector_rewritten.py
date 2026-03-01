# src/rag/strategies/retrievers/vector_rewritten.py
from src.rag.strategies.base import BaseRetrievalStrategy, SearchResult
from src.core.milvus_client import get_milvus_client
from src.rag.rewriter import rewriter_instance # 复用之前的重写器
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class VectorRewrittenRetriever(BaseRetrievalStrategy):
    """
    插件 2: 变体向量检索
    策略：先让 LLM 重写查询 (Query Rewriting)，再用重写后的句子进行向量搜索。
    """
    def __init__(self):
        self.milvus = get_milvus_client()
        self.rewriter = rewriter_instance
        logger.info("🔌 [Plugin] 加载变体向量检索插件 (Rewritten)")

    def search(self, query: str, top_k: int, filter_expr: Optional[str] = None, **kwargs) -> List[SearchResult]:
        # 1. 生成变体查询
        try:
            rewritten_query = self.rewriter.rewrite(query)
            if rewritten_query == query:
                logger.debug("⚠️ [Vector-Rewritten] 重写后无变化，跳过此路以避免重复")
                return []
            logger.info(f"🔄 [Vector-Rewritten] 变体查询：{rewritten_query}")
        except Exception as e:
            logger.error(f"❌ [Vector-Rewritten] 重写失败：{e}")
            return []
        
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