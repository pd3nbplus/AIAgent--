# src/mini_agent/tools/rag_tools.py
from langchain_core.tools import tool
from src.rag.pipeline import pipeline_instance
import uuid
import logging

logger = logging.getLogger(__name__)


@tool
def add_knowledge(text: str, category: str = "general") -> str:
    """
    向知识库中添加新的知识片段。
    适用于：用户上传文档内容、记录重要事实、补充背景信息。
    参数：
        text: 要存储的具体文本内容。
        category: 分类标签，如 "product_info", "user_manual", "company_policy"。
    """
    try:
        doc_id = str(uuid.uuid4())[:8]
        pipeline_instance.milvus.insert_data(
            id=doc_id,
            text=text,
            metadata={"category": category}
        )
        return f"✅ 知识已存入 (ID: {doc_id}, 分类：{category})"
    except Exception as e:
        logger.error(f"添加知识失败：{e}")
        return f"❌ 存入失败：{str(e)}"

@tool
def search_knowledge(query: str, top_k: int = 3) -> str:
    """
    在知识库中搜索与查询最相关的信息。
    适用于：回答基于文档的问题、查找特定政策、回忆用户提供的背景资料。
    参数：
        query: 用户的自然语言查询。
        top_k: 返回最相关的几条结果，默认 3 条。
    """
    try:
        results = pipeline_instance.run(query, top_k=top_k)
        if not results:
            return "ℹ️ 未找到相关知识。"
        
        relevant_texts = []
        for r in results:
            # 这里不再硬编码 0.5，而是直接使用过滤后的结果
            relevant_texts.append(f"- {r['text']} (置信度：{r['score']:.2f})")
        
        if not relevant_texts:
            return "ℹ️ 找到了匹配项，但相关性较低 (低于阈值)。"
            
        return "📚 相关知识:\n" + "\n".join(relevant_texts)
    except Exception as e:
        logger.error(f"搜索知识失败：{e}")
        return f"❌ 搜索失败：{str(e)}"

rag_tools = [add_knowledge, search_knowledge]