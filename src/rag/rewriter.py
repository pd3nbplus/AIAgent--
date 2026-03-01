# src/rag/rewriter.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.core.config import settings
from src.utils.xml_parser import remove_think_and_n
import logging

logger = logging.getLogger(__name__)

class QueryRewriter:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm.model_name,
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
            temperature=settings.llm.rewrite_temperature
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            你是一个高级 RAG 系统的查询优化专家。
            任务：将用户的自然语言问题改写为更适合向量数据库检索的陈述句。
            
            策略：
            1. **语义对齐**：将口语化词汇转换为专业术语（例："最好的" -> "旗舰、核心、最佳"）。
            2. **句式转换**：将疑问句转换为陈述句（向量库通常存储事实陈述）。
            3. **去噪**：去除礼貌用语和无关上下文。
            4. **扩展**：如果原句太短，适当补充隐含的主语或背景。
            
            约束：
            - 只输出改写后的句子，不要包含任何解释、引号或额外文本。
            - 如果原句已经非常清晰，可保持原意但微调措辞。
            
            示例：
            User: 你们最好的产品叫什么？
            Output: 该公司的旗舰产品的名称。
            
            User: 入职三年有几天年假？
            Output: 员工入职满三年后的年假天数政策规定。
            """),
            ("human", "{original_query}")
        ])

    def rewrite(self, original_query: str) -> str:
        """执行查询重写"""
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"original_query": original_query})
            rewritten_query = response.content.strip()
            # 移除 "Think:" 和 "\n"
            rewritten_query = remove_think_and_n(rewritten_query)

            logger.info(f"🔄 [Rewriter] 原始：{original_query}")
            logger.info(f"✨ [Rewriter] 重写：{rewritten_query}")
            
            return rewritten_query
        except Exception as e:
            logger.error(f"❌ [Rewriter] 失败，降级使用原查询：{e}")
            return original_query

# 单例
rewriter_instance = QueryRewriter()