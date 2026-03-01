# src/rag/chunkers.py
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class BaseChunker:
    """分块器基类"""
    def split_documents(self, docs: List[Document]) -> List[Document]:
        raise NotImplementedError

class RecursiveChunker(BaseChunker):
    """递归字符分块策略 (当前默认)"""
    def __init__(self, chunk_size: int, chunk_overlap: int, separators: List[str]):
        logger.info(f"✂️ 初始化递归分块器：Size={chunk_size}, Overlap={chunk_overlap}")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators
        )

    def split_documents(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)

class FixedChunker(BaseChunker):
    """固定长度分块策略 (简单粗暴)"""
    def __init__(self, chunk_size: int, chunk_overlap: int):
        logger.info(f"✂️ 初始化固定分块器：Size={chunk_size}, Overlap={chunk_overlap}")
        self.splitter = CharacterTextSplitter(
            separator="\n", # 简单按行切
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_documents(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)

# 可以在这里继续添加 SentenceChunker, SemanticChunker 等