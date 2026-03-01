
【Agent开发】第三阶段：RAG 实战 —— 赋予 Agent “外脑” -- pd的AI Agent开发笔记
---

[toc]

---

前置环境：当前环境是基于WSL2 + Ubuntu 24.04 + Docker Desktop构建的云原生开发平台，所有服务（MySQL、Redis、Qwen）均以独立容器形式运行并通过Docker Compose统一编排。如何配置请参考我的博客 [WSL2 + Ubuntu 24.04 + Docker Desktop 配置双内核环境](https://blog.csdn.net/weixin_52185313/article/details/158416250?spm=1011.2415.3001.5331)

我们将分三步走：
1. 环境准备：在 docker-compose.yml 中加入 Milvus 服务。
2. 知识库构建 (ETL)：编写脚本将本地文档切片、向量化并存入 Milvus。
3. Agent 集成：开发 RAG 工具，让 Agent 能“查阅”知识库回答问题。

# 环境准备：部署 Milvus

🚀 修改方案概览
1. 新增 Milvus 服务组：包含 milvus-standalone (核心服务) 和 etcd (元数据管理，Milvus 强依赖)。
2. 统一命名规范：所有 Milvus 相关容器前缀设为 milvus-，方便一键管理。
3. 启动策略：设置 restart: "no"，确保它们不会随 Docker 守护进程自动启动，只有你手动 docker-compose up milvus-standalone 时才运行。
4. 网络整合：全部加入 ai-net，确保 Agent 能同时访问 Qwen, MySQL, Redis 和 Milvus。
5. 数据持久化：配置 volumes，防止重启后向量数据丢失。
6. 用Attu 图形化界面进行数据管理、索引管理、监控等

Milvus 依赖较多（Etcd, MinIO），手动部署很麻烦，但用 Docker Compose 就很简单。

新建一个文件 docker-compose-milvus.yml，内容如下：

```yml
services:
  # ================= 基础服务 (默认开机自启) =================
  [mysql、redis、qwen等....]
# ================= Milvus 向量数据库服务 (默认不启动) =================
# 一键启停命令：docker-compose up -d attu 或 source ~/.bashrc >> milvus-up
# 停止命令：source ~/.bashrc >> milvus-stop

  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    container_name: milvus-etcd
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ./data/milvus/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3
    networks:
      - ai-net
    restart: "no"  # 跟随 Milvus 策略，不自动启动

  milvus-standalone:
    image: milvusdb/milvus:v2.4.10
    container_name: milvus-standalone
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      # 重新修正：官方 standalone 必须配 MinIO。下面会补充 MinIO 服务。
      MINIO_ADDRESS: minio:9000
      MILVUS_LOG_LEVEL: info
    volumes:
      - ./data/milvus/milvus:/var/lib/milvus
    ports:
      - "19530:19530"  # Milvus SDK 端口
      - "9091:9091"    # Prometheus 监控端口 (可选)
    depends_on:
      - "etcd"
      - "minio"
    networks:
      - ai-net
    restart: "no"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    container_name: milvus-minio
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - ./data/milvus/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
    networks:
      - ai-net
    restart: "no"
  # ================= Attu 图形化客户端 =================
  # 访问地址：http://localhost:3000

  attu:
    image: zilliz/attu:v2.4
    container_name: milvus-attu
    environment:
      MILVUS_URL: http://milvus-standalone:19530 # 指向 Milvus 容器内部地址
      # 如果需要在浏览器直接连接，可能需要暴露端口或配置跨域，但通常 Attu 作为代理运行
    ports:
      - "3000:3000"
    depends_on:
      - "milvus-standalone"
    networks:
      - ai-net
    restart: "no"
```

启动和停止
```bash
# 创建数据目录
mkdir -p ./data/milvus/{etcd,milvus,minio}
```

将启动和关闭命令写入`~/.bashrc`文件中
```bash
# 添加到 ~/.bashrc
alias milvus-up='docker-compose up -d milvus-standalone attu'
alias milvus-stop='docker-compose stop milvus-standalone attu etcd minio'
```

```bash
# 启动 Milvus
source ~/.bashrc
milvus-up
# 验证状态
docker-compose ps
docker-compose logs -f milvus-standalone

# 停止 Milvus
source ~/.bashrc
milvus-stop
```

![alt text](./img/milvus启动.png)

![alt text](./img/attu页面.png)

## Milvus 的整体架构和数据流

### 1. Milvus：向量数据库系统

+ 职责：提供向量相似性搜索、索引构建、数据插入/查询等核心功能。
+ 架构特点：采用微服务架构（从 v2.0 起），将不同功能拆分为多个组件（如 Proxy、RootCoord、DataNode、IndexNode 等）。
依赖外部存储：
+ 对象存储（如 MinIO 或 AWS S3）：用于持久化存储原始向量数据、索引文件等大块二进制数据。
+ 元数据存储（如 etcd）：用于存储集群状态、集合（collection）schema、数据分片（segment）信息、索引状态等轻量级结构化元数据。

### 2. MinIO（或兼容 S3 的对象存储）：持久化数据存储

+ 作用：作为 Milvus 的“数据湖”或“持久层”。
+ 存储内容：
   + 原始向量数据（raw vectors）
   + 构建好的索引文件（如 IVF_PQ、HNSW 等）
   + Binlog（用于数据回放和容错）
   + 快照（snapshots）等
+ 特点：
   + 高吞吐、高可用的对象存储
   + 不处理复杂查询，只做“读/写/删”对象操作
   + 数据以文件形式（如 .parquet、.bin）存储

> 💡 MinIO 是 Milvus 推荐的本地对象存储方案；在云上可直接使用 S3、GCS、Azure Blob 等。

### 3. etcd：分布式键值存储，用于元数据管理

+ 作用：Milvus 的“协调中枢”和“状态注册中心”。
+ 存储内容：
   + 集合（Collection）和分区（Partition）的元信息
   + Segment（数据分片）与 Channel（消息通道）的映射关系
   + 各组件（如 DataNode、IndexNode）的注册与心跳
   + 任务队列状态（如建索引任务）
   + DDL 操作（如 create collection）的事务日志
+ 特点：
   + 强一致性、低延迟
   + 支持 Watch 机制，用于组件间状态同步
   + 存储的是“小而关键”的元数据，不是向量本身


### 三者协作流程示例（插入数据）：

1. 用户通过 Milvus Client 插入向量 → 请求到达 Proxy。
2. Proxy 将数据写入 消息队列（如 Pulsar/Kafka，Milvus 2.x 也依赖消息队列，standalone除外）。
3. DataNode 消费消息，将向量批量写入 MinIO（生成 segment 文件）。
4. DataNode 将新 segment 的元信息（如 ID、位置、大小、时间戳）注册到 etcd。
5. 查询时，QueryNode 从 etcd 获取 segment 元数据，再从 MinIO 加载实际数据到内存进行搜索

```text
+-------------------+
|     Milvus        | ← 用户接口 & 计算逻辑
+-------------------+
        │
        ├── 读写元数据 ──→ etcd（轻量、结构化、强一致）
        │
        └── 读写向量/索引 ──→ MinIO（大文件、持久化存储）
```

| 部署模式       | 是否需要手动部署消息队列？ | 内部是否包含？ | 适用场景 |
|----------------|----------------------------|----------------|--------|
| Standalone（单机版） | ❌ 不需要                  | ✅ 内置一个嵌入式 RocksMQ（基于 RocksDB） | 开发、测试、小规模使用 |
| Cluster（集群版）    | ✅ 需要                    | ❌ 必须外接 Pulsar 或 Kafka | 生产环境、高吞吐、高可用 |

🧩 三者 + 消息队列的关系（完整版）
在 集群模式 下，Milvus 依赖四大外部组件：

| 组件        | 作用                     | 是否必需 |
|-------------|--------------------------|--------|
| etcd    | 存储元数据、协调服务状态 | ✅ 是   |
| MinIO/S3| 存储向量数据和索引文件   | ✅ 是   |
| Pulsar/Kafka | 流式数据管道，缓冲写入、触发异步任务 | ✅ 是（集群模式） |
| Milvus  | 核心服务                 | —      |

而在 Standalone 模式(2.5版本后，但**目前的最新稳定版 LTS是v2.4.10**)下：
+ etcd、MinIO、RocksMQ 都由 Milvus 自动启动（通过 milvus run standalone 或 Helm chart 的 standalone 配置）。
+ 你只需运行一个命令，无需单独部署这三个组件（但生产不推荐）。

# 应用Milvus

## 📦 第一步：安装依赖

```bash
pip install pymilvus sentence-transformers
```

## 🏗️ 第二步：封装 Milvus 操作类

### 1. 更新 config.py
我们在 core/config.py 中定义了 Milvus 的配置项

```python
import os
from dotenv import load_dotenv

load_dotenv() # 加载 .env 文件

class Settings:
    # LLM 配置
    LLM_MODEL_NAME = os.getenv("LLM_MODEL", "qwen-3-4b")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:7575/v1")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMP", "0.05"))
    
    # Redis 配置
    REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    
    # MySQL 配置
    DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:mysql2002@localhost:3307/agent_memory")

    # ================= Milvus 配置 =================
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "knowledge_base")
    
    # Embedding 模型配置
    # 推荐中文：BAAI/bge-small-zh-v1.5, shibing624/text2vec-base-chinese
    # 推荐英文：sentence-transformers/all-MiniLM-L6-v2
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    
    # Milvus 索引参数
    MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "HNSW")
    MILVUS_METRIC_TYPE = os.getenv("MILVUS_METRIC_TYPE", "COSINE")
    MILVUS_INDEX_M = int(os.getenv("MILVUS_INDEX_M", "8"))
    MILVUS_INDEX_EF_CONSTRUCTION = int(os.getenv("MILVUS_INDEX_EF", "200"))
    MILVUS_SEARCH_EF = int(os.getenv("MILVUS_SEARCH_EF", "64"))
    
    # 相似度阈值 (低于此分数的结果将被过滤)
    RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.5"))

settings = Settings()
```

### 2. 创建 MilvusClient 类

新建 `src/core/milvus_client.py`

```python
# src/core/milvus_client.py
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
from src.core.config import settings  # 👈 导入配置
import logging

logger = logging.getLogger(__name__)

class MilvusClient:
    def __init__(self):
        # 从配置读取连接信息
        self.host = settings.MILVUS_HOST
        self.port = settings.MILVUS_PORT
        self.collection_name = settings.MILVUS_COLLECTION_NAME
        
        self.collection = None
        
        # 初始化 Embedding 模型
        self.model_name = settings.EMBEDDING_MODEL_NAME
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
                "metric_type": settings.MILVUS_METRIC_TYPE,
                "index_type": settings.MILVUS_INDEX_TYPE,
                "params": {
                    "M": settings.MILVUS_INDEX_M, 
                    "efConstruction": settings.MILVUS_INDEX_EF_CONSTRUCTION
                }
            }
            
            self.collection.create_index("vector", index_params)
            self.collection.load()
            logger.info(f"✅ 集合 {self.collection_name} 创建成功，索引类型：{settings.MILVUS_INDEX_TYPE}")

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

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """搜索最相关的知识片段"""
        query_vector = self.embed_text(query)
        
        # 使用配置中的搜索参数
        search_params = {
            "metric_type": settings.MILVUS_METRIC_TYPE,
            "params": {"ef": settings.MILVUS_SEARCH_EF}
        }
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["text", "metadata"]
        )
        
        # 格式化返回结果
        hits = []
        for hit in results[0]:
            # 使用配置中的阈值过滤
            if hit.score >= settings.RAG_SCORE_THRESHOLD:
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

# 为了方便工具调用，可以提供一个全局单例实例
# 注意：在多线程/多进程环境下可能需要更复杂的单例管理
milvus_client_instance = None

def get_milvus_client() -> MilvusClient:
    global milvus_client_instance
    if milvus_client_instance is None:
        milvus_client_instance = MilvusClient()
    return milvus_client_instance
```

#### 📦 Milvus 向量数据库模块解析

这个模块是你的 **AI Agent 长期记忆库** 的核心引擎。它的作用是将非结构化的文本（如文档、对话记录）转化为计算机可理解的**数学向量**，并实现毫秒级的**语义检索**。

我们可以把它拆解为四个核心功能板块：

1. **核心组件：Embedding 模型 (文本翻译官)**
**代码对应**：`__init__` 中的 `SentenceTransformer` 和 `embed_text` 方法。

*   **作用**：将人类语言翻译成“向量语言”。
*   **算法策略**：**稠密向量表示 (Dense Vector Representation)**
    *   **原理**：模型（如 `BAAI/bge-small-zh-v1.5`）读取一段文字，将其映射为一个固定长度的浮点数数组（例如 512 个数字）。
    *   **语义空间**：在这个高维空间中，**语义相似的句子，其向量距离也越近**。
        *   例：“苹果很好吃”和“香蕉很美味”的向量距离会很近。
        *   例：“苹果很好吃”和“苹果公司发布了新手机”的距离会较远。
    *   **归一化 (`normalize_embeddings=True`)**：将向量长度统一缩放为 1。这使得我们可以直接使用**余弦相似度 (Cosine Similarity)** 来计算角度差异，无需考虑向量本身的模长，计算更快且更稳定。


2. **数据存储：Collection Schema (结构化仓库)**
**代码对应**：`_init_collection` 中的 `FieldSchema` 定义。

Milvus 不像 MySQL 那样只存表，它定义了一个包含**向量字段**的特殊结构。我们设计了四个字段：

| 字段名 | 类型 | 作用 | 类比 |
| :--- | :--- | :--- | :--- |
| **`id`** | `VARCHAR` (字符串) | **主键**。唯一标识一条知识。 | 图书馆的**索书号** |
| **`vector`** | `FLOAT_VECTOR` (浮点向量) | **核心字段**。存储由 Embedding 模型生成的 512 维数组。 | 书的**内容指纹** (用于快速比对) |
| **`text`** | `VARCHAR` (长文本) | **原始内容**。存储用户能看懂的原文。 | 书的**正文内容** |
| **`metadata`** | `JSON` | **扩展信息**。存储分类、来源、时间等标签。 | 书的**元数据卡片** (作者/出版社) |

> **💡 设计亮点**：
> 这种设计实现了**“向量检索 + 标量过滤”**的混合能力。你可以先搜索“语义最接近的”，再过滤“分类为 HR 政策的”数据。


3. **加速引擎：HNSW 索引 (超级导航图)**
**代码对应**：`create_index` 中的 `HNSW` 参数配置。

这是 Milvus 最快的原因所在。如果不建索引，搜索时需要把新问题和库里的几百万条数据逐一比对（暴力搜索），速度极慢。**HNSW** 算法解决了这个问题。

*   **算法名称**：**HNSW (Hierarchical Navigable Small World，分层可导航小世界)**
*   **通俗解释**：**“高速公路 + 本地街道”的多层地图策略**
    *   **分层结构**：
        *   **顶层 (高速公路)**：节点很少，但连接距离很远的节点。让你能从“北京”瞬间跳到“上海”大区。
        *   **中层 (国道)**：节点增多，连接范围缩小，帮你定位到“上海市区”。
        *   **底层 (本地街道)**：节点最密，连接最近的邻居，帮你精准找到“具体的某条路”。
    *   **搜索过程**：
        1.  从顶层入口进入，快速粗略定位到大致区域。
        2.  逐层向下，搜索范围越来越精细。
        3.  在底层找到最近的 K 个邻居。
    *   **优势**：将搜索复杂度从 $O(N)$ (线性扫描) 降低到 $O(\log N)$ (对数级)。即使库里有 1000 万条数据，也只需几十次跳跃就能找到结果。

*   **关键参数策略**：
    *   **`M` (最大连接数)**：每个节点在每层最多连几条线。
        *   *策略*：设为 8。越大搜索越准，但内存占用越高，建索引越慢。
    *   **`efConstruction` (建图时的探索深度)**：建索引时，每插入一个点，要考察多少个邻居来决定怎么连线。
        *   *策略*：设为 200。越大建出来的图质量越好（路修得越直），但建库时间越长。
    *   **`ef` (搜索时的探索深度)**：搜索时，临时保留多少个候选节点进行比对。
        *   *策略*：设为 64。越大搜索越准（不容易错过小路），但耗时稍增。通常 `ef` ≥ `top_k`。


4. **检索流程：ANN 搜索与阈值过滤**
**代码对应**：`search` 方法。

这是用户提问时的实际执行路径：

1.  **向量化 (Embedding)**：
    *   调用 `embed_text(query)`，将用户的问题“最好的产品是什么？”转化为一个 512 维的向量 `V_query`。
2.  **近似最近邻搜索 (ANN Search)**：
    *   利用建好的 **HNSW 索引图**，从 `V_query` 出发，在图中快速跳跃。
    *   计算 `V_query` 与库中向量的**余弦相似度**（夹角越小，分数越高，最高为 1）。
    *   找出分数最高的 `top_k` (例如 3 个) 候选项。
3.  **阈值过滤 (Threshold Filtering)**：
    *   **策略**：`if hit.score >= settings.RAG_SCORE_THRESHOLD`
    *   **目的**：防止“强行作答”。如果最高相似度只有 0.3（说明完全不相关），直接丢弃，返回“未找到”，避免 LLM 产生幻觉。
    *   **默认值**：通常设为 0.5 ~ 0.6，根据实际业务调整。


#### 🔄 完整数据流向总结

```mermaid
graph LR
    User[用户提问] --> Embed[Embedding 模型]
    Embed --> V_Query[查询向量]
    
    V_Query --> HNSW{HNSW 索引导航}
    HNSW --> TopK[Top-K 候选集]
    
    TopK --> Filter{分数 >= 阈值？}
    Filter -- 是 --> Result[返回原文 + 分数]
    Filter -- 否 --> Empty[返回空/未找到]
    
    subgraph 知识库
    DB[(Milvus Collection)]
    DB -- 存储 --> Vector[向量字段]
    DB -- 存储 --> Text[原文字段]
    end
    
    HNSW -.-> DB
```


### 3. 更新工具调用 

新建 `src/mini_agent/tools/rag_tools.py`

```python
# src/mini_agent/tools/rag_tools.py
from langchain_core.tools import tool
from src.core.milvus_client import get_milvus_client
from src.core.config import settings # 如果需要访问阈值等配置
import uuid
import logging

logger = logging.getLogger(__name__)

# 获取单例客户端
milvus = get_milvus_client()

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
        milvus.insert_data(
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
        results = milvus.search(query, top_k=top_k)
        if not results:
            return "ℹ️ 未找到相关知识。"
        
        relevant_texts = []
        for r in results:
            # 这里不再硬编码 0.5，而是直接使用过滤后的结果
            # 因为 milvus.search() 内部已经根据 settings.RAG_SCORE_THRESHOLD 过滤了
            relevant_texts.append(f"- {r['text']} (置信度：{r['score']:.2f})")
        
        if not relevant_texts:
            return "ℹ️ 找到了匹配项，但相关性较低 (低于阈值)。"
            
        return "📚 相关知识:\n" + "\n".join(relevant_texts)
    except Exception as e:
        logger.error(f"搜索知识失败：{e}")
        return f"❌ 搜索失败：{str(e)}"

rag_tools = [add_knowledge, search_knowledge]
```



### 4. 注册导出工具

在 `src/mini_agent/tools/__init__.py` 中注册导出工具：
```python
from .base_tools import base_tools
from .memory_tools import memory_tools
from .rag_tools import rag_tools

# 统一导出所有工具
ALL_TOOLS = base_tools + memory_tools + rag_tools
```

### 5. 实战测试

创建一个测试脚本 `src/test/test_rag.py`

```python
# src/test/test_rag.py

from src.mini_agent.tools.rag_tools import add_knowledge, search_knowledge

if __name__ == "__main__":
    print("🚀 开始 RAG 测试...")
    
    # 1. 添加知识
    print("\n--- 添加知识 ---")
    res1 = add_knowledge.invoke({
        "text": "公司的年假政策是：入职满 1 年有 5 天年假，满 3 年有 10 天年假。",
        "category": "hr_policy"
    })
    print(res1)
    
    res2 = add_knowledge.invoke({
        "text": "我们的旗舰产品是 Qwen-Agent，它支持多模态输入和长上下文记忆。",
        "category": "product"
    })
    print(res2)
    
    # 2. 搜索知识
    print("\n--- 搜索知识：我有多少天年假？ ---")
    query = "我入职 3 年了，有多少天年假？"
    res3 = search_knowledge.invoke({"query": query, "top_k": 2})
    print(res3)
    
    print("\n--- 搜索知识：旗舰产品是什么？ ---")
    query2 = "你们最好的产品叫什么？"
    res4 = search_knowledge.invoke({"query": query2, "top_k": 2})
    print(res4)
    
    print("\n✅ 测试完成！请打开 Attu (http://localhost:3000) 查看数据。")
```

运行测试脚本：
```
🚀 开始 RAG 测试...

--- 搜索知识：我有多少天年假？ ---
📚 相关知识:
- 公司的年假政策是：入职满 1 年有 5 天年假，满 3 年有 10 天年假。 (置信度：0.78)

--- 搜索知识：旗舰产品是什么？ ---
ℹ️ 未找到相关知识。
```

一旦测试通过，你的 Agent 就拥有了外脑！
你可以尝试问 Agent：
> “根据知识库，入职 3 年有多少天年假？”
Agent 会自动调用 search_knowledge 工具，获取上下文，然后结合 Qwen 的推理能力给出完美回答。

```py
# python -m src.main
if __name__ == "__main__":
    # 示例运行
    # run_agent("我叫pd，记住我。", thread_id="session_004")
    # run_agent("我是谁？", thread_id="session_004")
    run_agent("“根据知识库，入职 3 年有多少天年假？”", thread_id="session_005")
```

没写入知识库前：
```text
👤 用户 (Thread: session_005): “根据知识库，入职 3 年有多少天年假？”
🧠 思考：准备调用工具 ['search_knowledge']
📝 工具结果：ℹ️ 未找到相关知识。
🤖 Agent: 根据中国《职工带薪年休假条例》，连续工作满12个月的职工享受5天年假，每多工作1年增加1天（最多15天）。但具体天数可能因公司政策调整，建议您查阅所在公司的《员工手册》或人力资源部门的规定以确认准确信息。
```

**写入知识库后**(注意在写入知识库前运行了`main.py`，产生了上面的结果，写入知识库后，想让他输出知识库的内容，需要另起一个session_id, 否则会使用记忆，因为之前执行过知识库搜索的工具，没获取到东西，下一轮对话默认不会再重复执行，可以通过redis证实):
```text
👤 用户 (Thread: session_006): “根据知识库，入职 3 年有多少天年假？”
🧠 思考：准备调用工具 ['search_knowledge']
📝 工具结果：📚 相关知识:
- 公司的年假政策是：入职满 1 年有 5 天年假，满 3 年有 10 天年假。 (置信度：0.80)
🤖 Agent: 根据公司政策，入职满3年的员工享有10天年假。
```

![alt text](./img/redis证实.png)

### 仍然存在的问题

运行测试时 用户问到 query2 = "你们最好的产品叫什么？"
+ 模型返回
> ℹ️ 未找到相关知识。

这是一个非常经典且关键的 RAG 问题！🎯

这个问题在业界被称为 **“语义鸿沟” (Semantic Gap)** 或 **“查询 - 文档不匹配” (Query-Document Mismatch)。**

🔍 深度解析：为什么查不到？
你存入的数据是：
> “我们的旗舰产品是 Qwen-Agent...”

你查询的问题是：
> “你们最好的产品叫什么？”

虽然人类知道“旗舰产品” ≈ “最好的产品”，但在向量空间里，这两个短语的向量距离可能比你想象的要远。原因如下：
1. 词汇不匹配：
   + Embedding 模型（即使是 BGE）也是基于统计训练的。如果训练数据中“旗舰产品”和“最好的产品”经常出现在不同语境下，它们的向量重心就会偏移。
   + “旗舰”强调“代表性、最高端”，“最好”强调“质量、排名”。语义有重叠，但不完全重合。
2. 陈述句 vs 疑问句：
   + 存入的是陈述句（事实描述）。
   + 查询的是疑问句（寻求信息）。
   + 某些模型对句式的敏感度较高，导致向量分布不同。
3. 粒度问题：
   + 如果文档太长，关键信息被淹没，向量会代表整段话的“平均语义”，而不是那个关键词的语义。
4. 阈值过高：
   + 我们在 config.py 里设置了 RAG_SCORE_THRESHOLD = 0.5。
   + 可能其实召回了结果，但分数是 0.48，直接被代码过滤掉了，返回了“未找到”。



# RAG 技术的三大范式演进：从原生到模块化  
> 聚焦“知识如何入库”与“问题如何作答”

在大模型时代，RAG（Retrieval-Augmented Generation）已成为连接静态知识与动态生成的核心桥梁。但很多人只关注“怎么回答”，却忽略了“知识是怎么准备的”。事实上，RAG 的演进不仅体现在问答流程的优化，更体现在**知识构建方式**与**推理架构设计**的双重升级。

本文章将从两个维度解析 RAG 的三大范式：
- **检索前（Offline）**：原始文档 → 知识库
- **检索后（Online）**：用户提问 → 检索 → 回复

---

## 1. Native RAG（原生 RAG）

### 检索前：知识库构建流程
这是最朴素的知识注入方式：
- 原始文档（如 PDF、网页、TXT）被切分为固定长度的文本块（Chunking）。
- 每个 chunk 通过一个**静态嵌入模型**（如 sentence-transformers）转换为向量。
- 向量与原文本一起存入向量数据库（如 FAISS、Chroma）。

> 无元数据、无重排、无清洗——“所见即所得”。

```mermaid
flowchart LR
    A[原始文档] --> B[文本分块\n固定长度]
    B --> C[嵌入模型编码]
    C --> D[向量 + 文本存入\n向量数据库]
```

### 检索后：问答推理流程
- 用户输入问题。
- 问题经同一嵌入模型编码，在向量库中做近似最近邻搜索（ANN），返回 Top-K chunks。
- 将问题与检索结果拼接成 Prompt，送入 LLM 生成答案。

```mermaid
flowchart LR
    Q[用户提问] --> E[问题嵌入]
    E --> F[向量库检索\nTop-K chunks]
    F --> G[构建 Prompt:\nQuery + Context]
    G --> H[LLM 生成答案]
    H --> I[输出回复]
```
特点
- **构建简单，但知识粒度粗糙**
- **问答链路直接，但容错性差**


## 2. Advanced RAG（高级 RAG）

### 检索前：知识库构建流程
为了提升知识质量，Advanced RAG 在入库阶段引入更多智能处理：
- 文档先经过**清洗与结构化**（如去除页眉页脚、提取表格）。
- 分块策略更智能：**语义分块**（Semantic Chunking）或**滑动窗口**，避免切断句子。
- 可加入**元数据**（来源、时间、实体标签）。
- 部分系统还会对 chunk 进行**摘要**或**关键词提取**，辅助后续检索。

```mermaid
flowchart LR
    A[原始文档] --> B[清洗与结构化]
    B --> C[智能分块\n语义/滑动窗口]
    C --> D[提取元数据\n时间/实体/章节]
    D --> E[嵌入编码]
    E --> F[向量 + 元数据存入\n增强型向量库]
```

### 检索后：问答推理流程
- 用户提问后，可能先进行**查询重写**或**意图识别**。
- 初检使用向量检索，再用**Cross-Encoder 重排序**提升相关性。
- 支持**多跳检索**：第一轮结果用于生成新查询，继续检索。
- 最终上下文经过筛选后送入 LLM。

```mermaid
flowchart LR
    Q[用户提问] --> R[查询重写/扩展]
    R --> S[向量初检]
    S --> T[重排序\nCross-Encoder]
    T --> U{是否多跳?}
    U -- 是 --> V[基于上下文生成新查询]
    V --> W[第二轮检索]
    W --> X[合并上下文]
    U -- 否 --> X
    X --> Y[构建增强 Prompt]
    Y --> Z[LLM 生成答案]
    Z --> AA[输出回复]
```

特点
- **知识更结构化，检索更精准**
- **支持复杂推理，但延迟较高**


## 3. Modular RAG（模块化 RAG）

### 检索前：知识库构建流程
Modular RAG 将知识构建视为**可配置的数据管道**：
- 支持**多源异构数据**：文本、表格、图像、API 数据等。
- 引入**知识图谱**或**向量+标量混合索引**（如 Elasticsearch + FAISS）。
- 可选**增量更新机制**：只更新变更部分，而非全量重建。
- 支持**版本控制**与**审计日志**，满足企业合规需求。

```mermaid
flowchart LR
    A1[文本] --> M[统一摄入层]
    A2[表格/CSV] --> M
    A3[数据库/API] --> M
    A4[图像/多模态] --> M
    M --> N[智能解析与标准化]
    N --> O[多策略分块 + 元数据标注]
    O --> P[多模态嵌入 / 图谱三元组]
    P --> Q[混合索引存储\n向量+关键词+图]
```

### 检索后：问答推理流程
- 用户提问首先进入**路由模块**，判断任务类型（事实问答？计算？工具调用？）。
- 根据路由结果，动态选择检索路径：
  - 调用向量库
  - 查询知识图谱
  - 调用外部工具（如计算器、数据库）
- LLM 不仅生成答案，还可能参与**反思（Self-Critique）** 或 **迭代检索**。
- 输出前可经过**安全过滤**、**格式化**等后处理模块。

```mermaid
flowchart LR
    Q[用户提问] --> R[智能路由]
    R -->|事实类| S[向量/图谱检索]
    R -->|计算类| T[调用工具/代码解释器]
    R -->|多模态| U[图像/音频检索]
    S --> V[上下文验证与融合]
    T --> V
    U --> V
    V --> W[LLM 生成 + 自我反思]
    W --> X[后处理\n安全/格式/引用]
    X --> Y[输出最终回复]
```

### 特点
- **构建灵活，支持多模态与实时更新**
- **推理可编程，适配复杂业务场景**

---

## 总结：三大范式的本质差异

| 维度 | Native RAG | Advanced RAG | Modular RAG |
|------|-----------|--------------|-------------|
| **检索前（知识构建）** | 固定分块 + 单一嵌入 | 智能分块 + 元数据 + 清洗 | 多源摄入 + 混合索引 + 增量更新 |
| **检索后（问答流程）** | 直接检索 → 生成 | 查询优化 + 重排序 + 多跳 | 动态路由 + 工具调用 + 自省机制 |
| **核心目标** | 快速原型 | 提升准确率 | 支持生产级复杂系统 |
| **适用阶段** | 实验/POC | 专业领域应用 | 企业级 Agent 架构 |

> **演进逻辑**：  
> **知识从“堆进去” → “理清楚” → “活起来”**；  
> **问答从“问了就答” → “想清楚再答” → “会思考、会调工具地答”**。

未来，随着 RAG 与 Agent、Memory、Tool Use 的深度融合，**Modular RAG 将成为智能体的“外脑操作系统”**——而这一切，都始于你对“文档如何变成知识”的认真对待。

接下来，让我们回到之前检索的问题，并设计一个更智能的检索方案。

#  方案落地：基于 Advanced RAG 范式的方案设计

📍 当前定位诊断
+ 之前状态 (Native RAG)：
   + 检索前：简单分块 + 静态 Embedding。
   + 检索后：用户提问 → 直接向量检索 → LLM 回答。
   + 痛点：词汇不匹配（“最好的产品”vs“旗舰产品”），导致召回失败。
+ 目标状态 (Advanced RAG)：
   + 检索后优化：引入 查询重写 (Query Rewriting) 模块，在检索前对用户意图进行对齐和增强。

在Advanced RAG的架构图中，这一步对应：
> 用户提问 → [新增] 查询重写/扩展 → 向量初检 → ...


我们将创建一个独立的 Retrieval Pipeline (检索管道) 模块，将“重写”和“搜索”封装在一起，而不是散落在 Tool 函数里。这符合 Modular RAG 的设计思想，为未来扩展（如重排序、多跳）留出接口。

## 目录结构更新

```text
src/
├── core/
│   ├── config.py              # 配置中心
│   └── milvus_client.py       # 底层向量库操作 (保持纯净，只负责存/搜)
├── rag/                       # 👈 新增：RAG 专用模块 (Advanced RAG 核心)
│   ├── __init__.py
│   ├── pipeline.py            # 👈 新增：检索管道 (重写 + 搜索 + 重排序预留)
│   └── rewriter.py            # 👈 新增：查询重写器
├── mini_agent/
│   ├── tools/
│   │   └── rag_tools.py       # 工具层 (调用 Pipeline)
│   └── graph.py               # Agent 流程
└── test/
    └── test_rag_advanced.py   # 测试脚本
```

## 代码实现

A. 关键词增强式重写 (`src/rag/rewriter.py`)

```py
# src/rag/rewriter.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)

class QueryRewriter:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            temperature=0.3  # 稍微增加创造性以扩展同义词
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
            
            logger.info(f"🔄 [Rewriter] 原始：{original_query}")
            logger.info(f"✨ [Rewriter] 重写：{rewritten_query}")
            
            return rewritten_query
        except Exception as e:
            logger.error(f"❌ [Rewriter] 失败，降级使用原查询：{e}")
            return original_query

# 单例
rewriter_instance = QueryRewriter()
```

B. 检索管道 (`src/rag/pipeline.py`)

对应架构：`查询重写` → `向量初检` → `重排序 (预留)` 的全流程


```py
# src/rag/pipeline.py
from src.core.milvus_client import get_milvus_client
from src.rag.rewriter import rewriter_instance
from src.core.config import settings
from src.utils.xml_parser import remove_think_and_n
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class RetrievalPipeline:
    def __init__(self):
        self.milvus = get_milvus_client()
        self.rewriter = rewriter_instance
        self.use_rewrite = True  # 可通过配置开关
        self.enable_fallback = True # 是否启用降级回退

    def run(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        执行完整的检索流程 (Advanced RAG Online Flow)
        Flow: Query -> Rewrite -> Search -> (Fallback) -> Result
        """
        final_query = query
        
        # Step 1: 查询重写 (Advanced RAG 特性)
        if self.use_rewrite:
            final_query = self.rewriter.rewrite(query)
        
        # 去除think和\n的内容
        final_query = remove_think_and_n(final_query)
        # Step 2: 向量初检
        logger.info(f"🔍 [Pipeline] 正在检索：{final_query}")
        results = self.milvus.search(final_query, top_k=top_k)
        
        # Step 3: 降级策略 (Fallback) - 如果重写后没结果，尝试原查询
        if not results and self.enable_fallback and final_query != query:
            logger.warning("⚠️ [Pipeline] 重写查询无结果，触发降级策略，尝试原查询...")
            results = self.milvus.search(query, top_k=top_k)
        
        # Step 4: (预留) 重排序 (Re-ranking)
        # 未来可在此处接入 Cross-Encoder 对 results 进行二次排序
        
        # Step 5: 阈值过滤
        filtered_results = []
        for r in results:
            if r["score"] >= settings.RAG_SCORE_THRESHOLD:
                filtered_results.append(r)
            else:
                logger.debug(f"🚫 [Pipeline] 分数 {r['score']:.2f} 低于阈值 {settings.RAG_SCORE_THRESHOLD}, 过滤")
        
        return filtered_results

# 单例
pipeline_instance = RetrievalPipeline()
```

C. 工具层适配 (`src/mini_agent/tools/rag_tools.py`)

工具层现在变得非常轻薄，它只负责调用 `Pipeline`，不再关心具体的检索逻辑。这符合 Modular RAG 的解耦思想。

```python
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
            # 因为 milvus.search() 内部已经根据 settings.RAG_SCORE_THRESHOLD 过滤了
            relevant_texts.append(f"- {r['text']} (置信度：{r['score']:.2f})")
        
        if not relevant_texts:
            return "ℹ️ 找到了匹配项，但相关性较低 (低于阈值)。"
            
        return "📚 相关知识:\n" + "\n".join(relevant_texts)
    except Exception as e:
        logger.error(f"搜索知识失败：{e}")
        return f"❌ 搜索失败：{str(e)}"

rag_tools = [add_knowledge, search_knowledge]
```

在 `test_rag.py`前面设置日志级别

```py
import logging

# 设置根日志记录器的级别为 INFO，并配置输出格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

再次尝试运行测试脚本
```bash
python -m src.test.test_rag
```

输出：
```text
--- 搜索知识：旗舰产品是什么？ ---
2026-02-28 15:40:49,655 - httpx - INFO - HTTP Request: POST http://localhost:7575/v1/chat/completions "HTTP/1.1 200 OK"
2026-02-28 15:40:49,656 - src.rag.rewriter - INFO - 🔄 [Rewriter] 原始：你们最好的产品叫什么？
2026-02-28 15:40:49,657 - src.rag.rewriter - INFO - ✨ [Rewriter] 重写：<think>
好的，用户问的是“你们最好的产品叫什么？”，我需要按照RAG系统的查询优化策略来改写这个问题。首先，根据策略1，要将口语化的词汇转换为专业术语。“最好的”可以换成“旗舰、核心、最佳”这样的词。然后，策略2要求将疑问句转为陈述句，因为向量数据库通常存储的是事实陈述，而不是问题。所以原
句是疑问句，需要改成陈述句。策略3是去噪，去除礼貌用语，这里“你们”可能可以换成更正式的“该公司的”或者“本公司的”。策略4是扩展，如果原句太短，可能需要补充主语或背景，但这里已经比较清晰了，可能不需要扩展。

现在，原句是“你们最好的产品叫什么？”，改写后的句子应该是“该公司的旗舰产品的名称。”或者“本公司的核心产品的名称。”但根据示例，用户给出的例子是“该公司的旗舰产品的名称。”所以可能需要保持一致性。另外，用户的问题可能隐含的是询问产品名称，所以可能需要更明确地指出是产品名称。因 
此，正确的改写应该是“该公司的旗舰产品的名称。”这样符合所有策略的要求，没有添加额外内容，保持了专业术语，转换为陈述句，并去除了口语化的“你们”和“叫什么”。
</think>

该公司的旗舰产品的名称。
2026-02-28 15:40:49,657 - src.rag.pipeline - INFO - 🔍 [Pipeline] 正在检索：该公司的旗舰产品的名称。
📚 相关知识:
- 我们的旗舰产品是 Qwen-Agent，它支持多模态输入和长上下文记忆。 (置信度：0.60)
```

## 🎓 总结：我们在架构中做了什么？

1. 检索前 (Offline)：整段Embedding
2. 检索后 (Online)：
   + 插入模块：在 Query 和 Vector Search 之间，插入了 QueryRewriter。
   + 流程编排：创建了 RetrievalPipeline，将重写、搜索、降级策略编排成一个标准流程。
   + 解耦设计：Agent 工具层不再感知复杂的检索逻辑，只调用 pipeline.run()，为未来升级为 Modular RAG（如加入重排序、多跳）打下坚实基础。

现在的系统状态：
你已经成功构建了一个 初级的 Advanced RAG 系统！它具备了工业级 RAG 的雏形：能理解用户意图，能自动修正查询偏差，并有容错机制。

### 下一步：

+ 检索前优化 (Offline Optimization) —— 让知识“颗粒度”更精细
+ 检索后优化 (Online Optimization) —— 让结果“更相关”
+ 架构模块化 (Modular Architecture) —— 打造可插拔组件
+ 端到端验收 (End-to-End Evaluation)