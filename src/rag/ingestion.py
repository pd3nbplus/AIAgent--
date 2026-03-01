# src/rag/ingestion.py
import os
import logging
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredMarkdownLoader
from src.core.milvus_client import get_milvus_client
from src.core.es_client import es_client_instance
from src.core.config import settings
from src.utils.xml_parser import remove_think_and_n
from src.rag.factories import ChunkerFactory # 新增导入
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import uuid
import json

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    def __init__(self):
        self.milvus = get_milvus_client()
        # 1. 初始化 LLM (用于元数据增强)
        self.llm_enhancer = ChatOpenAI(
            model=settings.llm.model_name,
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
            temperature=settings.llm.temperature
        )
        # 配置分块策略 (Advanced RAG 核心参数)
        # 可根据文档类型调整，这里使用通用配置
        self.text_splitter = ChunkerFactory.get_chunker()
        logger.info("✅ 数据摄入管道初始化完成 (递归分块模式)")

    def load_document(self, file_path: str) -> List:
        """加载文档并解析为 LangChain Document 对象"""
        ext = os.path.splitext(file_path)[1].lower()
        logger.info(f"📂 正在加载文档：{file_path}")
        
        try:
            if ext == ".pdf":
                loader = PyMuPDFLoader(file_path)
            elif ext == ".md":
                loader = TextLoader(file_path, encoding='utf-8')
            elif ext == ".txt":
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                logger.warning(f"⚠️ 不支持的文件类型：{ext}")
                return []
            
            docs = loader.load()
            logger.info(f"📄 文档加载成功，共 {len(docs)} 页/部分")
            return docs
        except Exception as e:
            logger.error(f"❌ 文档加载失败：{e}")
            return []

    def enhance_metadata(self, chunk_text: str, source: str) -> Dict:
        """利用 LLM 为 Chunk 生成假设性问题和摘要 (元数据增强)"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            你是一个知识库索引专家。请阅读以下文本片段，并生成：
            1. 一个简短的摘要 (summary)。
            2. 3 个用户可能用来查询该片段的“假设性问题” (questions)，用逗号分隔。
            
            只返回 JSON 格式，不要其他解释。
            示例格式：{{"summary": "...", "questions": "问题 1, 问题 2, 问题 3"}}
            """),
            ("human", "文本片段：{text}")
        ])
        
        try:
            chain = prompt | self.llm_enhancer
            response = chain.invoke({"text": chunk_text})

            content = response.content.strip()
            # 去掉think等
            content = remove_think_and_n(content)
            
            # 清理可能的 markdown 标记
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
                
            meta = json.loads(content)
            return meta
        except Exception as e:
            logger.warning(f"⚠️ 元数据增强失败：{e}，使用默认值")
            return {"summary": "", "questions": ""}

    def process_file(self, file_path: str, category: str = "general"):
        """处理单个文件：加载 -> 分块 -> 增强 -> 入库"""
        docs = self.load_document(file_path)
        if not docs:
            return

        all_chunks = []
        
        # 分块
        for doc in docs:
            # doc.page_content 是文本，doc.metadata 包含页码等信息
            splits = self.text_splitter.split_documents([doc])
            all_chunks.extend(splits)
        
        logger.info(f"✂️ 分块完成，共生成 {len(all_chunks)} 个 chunks")

        # 入库
        success_count = 0
        for i, chunk in enumerate(all_chunks):
            text = chunk.page_content
            
            # 跳过过短的块（可能是噪点）
            if len(text.strip()) < 20:
                continue

            # 元数据增强 (可选：为了速度，生产环境可异步或批量处理)
            enhanced_meta = self.enhance_metadata(text, chunk.metadata.get("source", ""))
            summary_str = enhanced_meta.get("summary", "")
            questions_str = enhanced_meta.get("questions", "")

            final_metadata = {
                "source": os.path.basename(file_path),
                "page": chunk.metadata.get("page", 0),
                "category": category,
                "summary": summary_str,
                "questions": questions_str
            }
            
            doc_id = f"{os.path.basename(file_path)}_{i}_{uuid.uuid4().hex[:6]}"
            
            # 2. 存入 Milvus (向量库)
            try:
                self.milvus.insert_data(
                    id=doc_id,
                    text=text,
                    metadata=final_metadata
                )
                success_count += 1
            except Exception as e:
                logger.error(f"❌ Milvus 插入失败：{e}")

            # 3. 存入 Elasticsearch (关键词库) - 双管齐下
            if es_client_instance.is_available():
                # A. 写入问题索引 (包含 summary)
                if questions_str:
                    es_client_instance.indexing_question(
                        doc_id=doc_id,
                        questions=questions_str,
                        text=text,
                        metadata=final_metadata
                    )
                
                # B. 写入摘要索引 (新增！)
                if summary_str:
                    es_client_instance.indexing_summary(
                        doc_id=doc_id,
                        summary=summary_str,
                        text=text,
                        metadata=final_metadata
                    )

        logger.info(f"✅ 文件 {file_path} 处理完毕，成功入库 {success_count}/{len(all_chunks)} 条记录")

    def process_directory(self, folder_path: str, category: str = "general"):
        """批量处理文件夹下的所有支持文件"""
        supported_exts = ['.pdf', '.txt', '.md']
        files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in supported_exts]
        
        if not files:
            logger.warning(f"⚠️ 目录 {folder_path} 下未找到支持的文件")
            return
            
        logger.info(f"🚀 开始批量处理目录：{folder_path}, 共 {len(files)} 个文件")
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            self.process_file(file_path, category)
            
        logger.info("🎉 所有文件处理完成！")

# 单例
ingestion_pipeline = DataIngestionPipeline()