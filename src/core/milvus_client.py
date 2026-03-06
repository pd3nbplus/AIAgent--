# src/core/milvus_client.py
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
from src.core.config import settings  # 👈 导入配置
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class MilvusClient:
    def __init__(self):
        # 从配置读取连接信息
        self.host = settings.db.milvus_host
        self.port = settings.db.milvus_port
        self.collection_name = settings.db.milvus_collection
        self.metric_type = settings.db.milvus_metric_type
        self.index_type = settings.db.milvus_index_type
        self.index_m =  settings.db.milvus_index_m
        self.index_ef_construction = settings.db.milvus_index_ef_construction
        self.search_ef = settings.db.milvus_search_ef

        self.collection = None
        
        # 初始化 Embedding 模型
        self.model_name = settings.embedding.model_name
        logger.info(f"🔄 加载 Embedding 模型：{self.model_name} (首次运行会下载)...")
        
        try:
            self.embedding_model = SentenceTransformer(self.model_name)
            self.dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"✅ 模型加载完成，向量维度：{self.dim}")
        except Exception as e:
            logger.error(f"❌ 加载 Embedding 模型失败：{e}")
            raise e
        

        self._connect()
        self._init_collection()

    def _connect(self):
        """连接 Milvus 服务"""
        try:
            connections.connect(host=self.host, port=self.port)
            logger.info(f"✅ 成功连接到 Milvus ({self.host}:{self.port})")
        except Exception as e:
            logger.error(f"❌ 连接 Milvus 失败：{e}")
            raise e

    def _init_collection(self):
        """初始化集合（如果不存在）"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info(f"ℹ️ 集合 {self.collection_name} 已存在，已加载")
        else:
            # 定义 Schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            schema = CollectionSchema(fields, "RAG Knowledge Base for AI Agent")
            
            self.collection = Collection(self.collection_name, schema)
            
            # 使用配置中的索引参数
            index_params = {
                "metric_type": self.metric_type,
                "index_type": self.index_type,
                "params": {
                    "M": self.index_m, 
                    "efConstruction": self.index_ef_construction
                }
            }
            
            self.collection.create_index("vector", index_params)
            self.collection.load()
            logger.info(f"✅ 集合 {self.collection_name} 创建成功，索引类型：{self.index_type}")

    def embed_text(self, text: str) -> list[float]:
        """将文本转换为向量"""
        return self.embedding_model.encode(text, normalize_embeddings=True).tolist()

    def insert_data(self, id: str, text: str, metadata: dict = None):
        """插入一条知识"""
        vector = self.embed_text(text)
        data = [{
            "id": id,
            "vector": vector,
            "text": text,
            "metadata": metadata or {}
        }]
        self.collection.insert(data)
        logger.debug(f"📝 插入数据：{id}")

    def search(self, query: str, top_k: int = 3, filter_expr: str = None, output_fields: List[str] = None) -> List[dict]:
        """支持元数据过滤和指定输出字段的搜索"""
        query_vector = self.embed_text(query)
        
        search_params = {
            "metric_type": self.metric_type,
            "params": {"ef": self.search_ef}
        }
        
        # 默认输出
        if not output_fields:
            output_fields = ["text", "metadata"]
            
        try:
            results = self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=filter_expr,  # 👈 支持过滤
                output_fields=output_fields
            )
        except Exception as e:
            logger.error(f"❌ Milvus 搜索失败 (Expr: {filter_expr}): {e}")
            return []
        
        hits = []
        for hit in results[0]:
            # 注意：这里先不过滤阈值，交给上层策略处理，以便 RRF 能看到更多候选
            hits.append({
                "text": hit.entity.get("text"),
                "score": hit.score,
                "metadata": hit.entity.get("metadata")
            })
        return hits

    def drop_collection(self):
        """删除集合（慎用）"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.warning(f"🗑️ 集合 {self.collection_name} 已删除")

    def scan_collection(self, limit: int = 500, offset: int = 0) -> List[Dict]:
        """
        扫描集合中的数据 (用于全量同步)
        注意：Milvus 深度分页性能有限，仅适用于离线同步任务。
        生产环境建议使用 Milvus 2.3+ 的 Iterator API。
        """
        try:
            # 使用空表达式查询所有，配合 limit 和 offset
            # output_fields 必须包含 id, text, metadata
            results = self.collection.query(
                expr="",  # 空表达式表示所有
                output_fields=["text", "metadata"],
                limit=limit,
                offset=offset
            )
            return results
        except Exception as e:
            logger.error(f"❌ Milvus 扫描失败：{e}")
            return []

# 为了方便工具调用，可以提供一个全局单例实例
# 注意：在多线程/多进程环境下可能需要更复杂的单例管理
milvus_client_instance = None

def get_milvus_client() -> MilvusClient:
    global milvus_client_instance
    if milvus_client_instance is None:
        milvus_client_instance = MilvusClient()
    return milvus_client_instance

if __name__ == "__main__":
    client = get_milvus_client()
