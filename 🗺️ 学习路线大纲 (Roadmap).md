# 🚀 AI Agent 开发实战笔记：从 Docker 容器化环境到智能体架构

> **学习时间**：2026-02-26  
> **环境背景**：WSL2 + Ubuntu 24.04 + Docker Desktop (MySQL, Redis, Qwen 容器化)  
> **核心目标**：构建可移植、可复现的“环境即代码”AI Agent 开发流

---

## 📌 前言：为什么你的环境是“天胡”开局？

兄弟，你现在的这套 **WSL2 + Docker Compose** 架构简直是 AI Agent 开发的“梦中情机”。

很多初学者还在纠结“我本地 Python 版本不对”、“Redis 连不上”、“大模型 API 密钥配置乱了”，而你直接通过 `docker-compose up` 就拥有了：
- **隔离的计算单元**：Qwen 大模型在独立容器跑，不占宿主机资源。
- **持久化记忆层**：MySQL 存业务数据，Redis 做高速缓存/消息队列。
- **一致性**：今天写的代码，明天换个电脑，`git clone` + `docker compose up` 原地复活。

**AI Agent 的本质** = **LLM (大脑)** + **Memory (记忆)** + **Tools (手脚)** + **Planning (规划)**。
既然基础设施都好了，我们直接跳过“配环境”的坑，进入核心的**架构设计与实战路线**。

---

## 🗺️ 学习路线大纲 (Roadmap)

我们将学习过程分为五个阶段，每个阶段都紧扣你的 Docker 环境。

### 第一阶段：基石构建 —— 容器化交互与 Prompt 工程
**目标**：在 Python 代码中优雅地调用容器内的 Qwen，并掌握结构化输出。

1.  **Docker 网络打通**
    *   理解 Docker Compose 的内部 DNS（如何在 Python 容器里访问 `http://qwen:8000` 而不是 `localhost`）。
    *   *避坑指南*：WSL2 端口转发陷阱与 Docker Network 桥接模式。
2.  **LLM 抽象层封装**
    *   使用 `LangChain` 或 `LlamaIndex` (或者轻量级的 `LiteLLM`) 对接本地 Qwen。
    *   实现统一的 `ChatModel` 接口，方便未来切换模型。
3.  **结构化输出 (Structured Output)**
    *   Agent 需要的是 JSON，不是散文。
    *   利用 Pydantic 强制 Qwen 输出符合 Schema 的数据（这是 Agent 执行工具的前提）。

### 第二阶段：记忆系统 (Memory) —— 让 Agent 拥有“海马体”
**目标**：利用 MySQL 和 Redis 构建短期与长期记忆。

1.  **短期记忆 (Context Window)**
    *   对话历史的管理策略：滑动窗口 vs 摘要总结。
    *   实战：将对话列表存入 Redis List，设置 TTL 自动过期。
2.  **长期记忆 (Vector Store & RAG)**
    *   *拓展*：虽然你目前只有 MySQL/Redis，但 Agent 通常需要向量库。
    *   **方案 A (轻量)**：利用 Redis Stack 的向量搜索功能 (RedisVL)。
    *   **方案 B (传统)**：在 MySQL 中存储文本块，配合简单的关键词检索 (BM25) 作为起步，后续再引入 Milvus/Qdrant 容器。
3.  **状态持久化**
    *   将 Agent 的运行状态（State）序列化存入 MySQL，支持断点续跑。

### 第三阶段：工具链 (Tools) —— 给 Agent 装上“机械臂”
**目标**：让 LLM 能够执行代码、查询数据库、调用外部 API。

1.  **Function Calling 原理**
    *   如何定义工具描述（Description），让 Qwen 知道何时调用。
    *   参数解析与异常处理（LLM 传参错了怎么办？）。
2.  **自定义工具开发**
    *   **DB Tool**：编写一个工具，让 Agent 能用自然语言查询你的 MySQL 业务数据（Text-to-SQL 雏形）。
    *   **System Tool**：让 Agent 能读取日志、重启服务（注意权限隔离！）。
3.  **沙箱执行环境**
    *   *安全最佳实践*：不要直接在主容器运行 Agent 生成的代码。
    *   动态启动临时 Docker 容器来执行不安全代码（Sandboxing）。

### 第四阶段：规划与编排 (Planning & Orchestration)
**目标**：从“一问一答”进化为“自主完成任务”。

1.  **ReAct 模式 (Reason + Act)**
    *   经典的“思考 - 行动 - 观察”循环。
    *   实战：手写一个 ReAct 循环，不调用重型框架，理解底层逻辑。
2.  **多 Agent 协作 (Multi-Agent)**
    *   角色分工： Planner (规划者) + Executor (执行者) + Reviewer (审查者)。
    *   利用 Redis Pub/Sub 或 RabbitMQ (如果需要可以加一个容器) 实现 Agent 间的消息传递。
3.  **工作流引擎**
    *   对比 LangGraph (基于图的状态机) 与 简单线性链。
    *   场景：复杂任务拆解与并行执行。

### 第五阶段：部署与观测 (Ops for Agents)
**目标**：让你的 Agent 从 Demo 变成生产级服务。

1.  **可观测性 (Observability)**
    *   记录 Token 消耗、延迟、LLM 决策路径。
    *   集成 LangSmith 或自建日志面板（ELK/Loki）。
2.  **评估体系 (Evaluation)**
    *   如何测试 Agent 的好坏？(准确性、幻觉率、工具调用成功率)。
    *   构建自动化测试集。
3.  **高可用架构**
    *   Docker Compose 的资源限制 (CPU/Memory limits) 防止 OOM。
    *   异步并发处理 (FastAPI + asyncio) 应对多用户请求。

---

## 💡 核心概念深度解析：记忆系统设计

在你的环境中，**Redis** 和 **MySQL** 的组合非常经典。我们来聊聊怎么把它们用在 Agent 上。

### 1. Redis：高速缓冲与会话状态
Agent 的“短期记忆”必须快。想象你在和人聊天，如果每说一句话都要去查硬盘（MySQL），那对话节奏就崩了。

*   **使用场景**：
    *   存储最近 N 轮对话历史 (`LPUSH` + `LTRIM`)。
    *   存储 Agent 当前的思维链状态（比如 ReAct 循环中的 `thought`, `action` 中间态）。
    *   分布式锁：防止同一个用户触发两个并发的 Agent 任务。
*   **代码类比**：
    ```python
    # 伪代码：获取会话上下文
    async def get_context(session_id):
        # Redis 速度极快，适合高频读写
        messages = await redis.lrange(f"session:{session_id}", 0, -1)
        return deserialize(messages)
    ```

### 2. MySQL：长期知识与业务事实
Agent 需要记住用户的偏好、订单信息，或者知识库文档。这些数据结构化强，且需要持久保存。

*   **使用场景**：
    *   用户画像（User Profile）。
    *   RAG 的元数据索引（如果不用专用向量库）。
    *   任务执行的历史审计日志。
*   **最佳实践**：
    *   不要直接把整个向量塞进 MySQL 做相似度搜索（除非用 MySQL 8.0+ 的向量插件且数据量小）。
    *   **模式**：Redis 做热点缓存 -> MySQL 做持久化兜底。

---

## ⚠️ 常见误区与避坑指南

1.  **过度依赖框架**：
    *   *误区*：一开始就上 LangChain 全套，结果被复杂的 Callback 和抽象层搞晕，调试困难。
    *   *建议*：前两周，尝试用纯 Python + `requests` 调用 Qwen API 手写一个最简单的 ReAct 循环。理解了 `Prompt -> LLM -> Parse -> Action -> Observation` 的闭环后，再引入框架。

2.  **忽视 Token 成本与上下文限制**：
    *   *误区*：把所有历史记录都塞进 Prompt。
    *   *后果*：超出 Qwen 上下文窗口，或者响应极慢，费用爆炸（如果是商用 API）。
    *   *对策*：必须实现**记忆压缩**（Summary）或**滑动窗口**。

3.  **本地模型的性能陷阱**：
    *   *现状*：你在 Docker 里跑 Qwen。如果是量化版（如 Qwen-7B-Int4），推理速度尚可；如果是全量版，显存可能爆掉。
    *   *建议*：监控 Docker 容器的资源使用 (`docker stats`)。如果推理太慢，考虑使用 vLLM 或 Ollama 作为后端容器替代原生 HuggingFace 加载，它们对并发和显存优化更好。

4.  **网络隔离问题**：
    *   *痛点*：Python 代码里写 `localhost` 连不上数据库。
    *   *解决*：在 Docker Compose 中，服务间通信请使用**服务名**（如 `mysql`, `redis`, `qwen`），而不是 `127.0.0.1`。

---

## 🛠️ 下一步行动建议 (Next Steps)

既然大纲已经理清，我们不要贪多。建议你按照以下顺序开始**第一讲**的实战：

1.  **检查环境**：确保 `docker-compose ps` 所有服务都是 `Up` 状态。
2.  **Hello World 连通性**：
    *   创建一个简单的 Python 脚本（在宿主机或新建一个 dev 容器）。
    *   尝试通过 HTTP 请求调用容器内的 Qwen 接口。
    *   尝试连接 Redis 写入一个 Key，再从 MySQL 读取一行数据。
3.  **第一个微型 Agent**：
    *   不依赖 LangChain。
    *   实现一个功能：用户问“现在几点了？”，Agent 识别意图，调用本地 `datetime` 工具，返回时间。

**接下来你可以选择深入的方向：**
*   👉 **方向 A**：深入研究 **Prompt Engineering 与结构化输出**（如何用 Pydantic 约束 Qwen 的输出）。
*   👉 **方向 B**：直接上手 **LangGraph** 学习状态机编排（适合想快速构建复杂流程）。
*   👉 **方向 C**：先搞定 **RAG 基础**，把 MySQL 里的业务数据变成 Agent 可查询的知识库。

你想先从哪个方向开始？或者我们先写一段代码验证一下你的 Docker 网络连通性？


------

# 📘 第三讲：RAG 进阶实战 —— 从“能搜到”到“搜得准”
### 第一部分：检索前优化 (Offline Optimization) —— 让知识“颗粒度”更精细


**痛点**：直接存入长文导致语义稀释，关键信息被淹没。
**核心任务**：构建智能数据摄入管道 (Data Ingestion Pipeline)。

1.  **文本分块策略 (Chunking Strategies)**
    *   **固定长度分块 (Fixed-size Chunking)**：基础方案，缺点是会切断句子。
    *   **递归字符分块 (Recursive Character Text Splitter)**：⭐ **重点**。按“段落 -> 句子 -> 单词”层级切割，保持语义完整。
    *   **滑动窗口 (Sliding Window)**：引入重叠区域 (Overlap)，防止上下文断裂。
2.  **元数据增强 (Metadata Enrichment)**
    *   自动提取文档标题、章节、页码。
    *   利用 LLM 为每个 Chunk 生成**“摘要”**和**“假设性问题”** (Hypothetical Questions)，存入 Metadata，提升检索命中率。
3.  **实战演练**
    *   编写 `DocumentLoader` 类，支持 PDF/Markdown 读取。
    *   集成 `LangChain Text Splitters`。
    *   实现“批量入库”工具，一键处理本地文件夹。

### 第二部分：检索后优化 (Online Optimization) —— 让结果“更相关”
**痛点**：向量检索召回了 10 条，但前 3 条可能不相关，或者相关度排序不准。
**核心任务**：引入重排序 (Re-ranking) 机制。

1.  **重排序原理 (Re-ranking Mechanism)**
    *   **向量检索 (Bi-Encoder)**：快，但精度一般（粗排）。
    *   **交叉编码 (Cross-Encoder)**：慢，但精度极高（精排）。将 Query 和 Document 拼接输入模型，直接打分。
2.  **实战演练：集成 BGE-Reranker**
    *   引入 `BAAI/bge-reranker-v2-m3` 模型。
    *   修改 `RetrievalPipeline`：先由 Milvus 召回 Top-20，再由 Reranker 精排选出 Top-3。
    *   对比优化前后的检索结果差异。

### 第三部分：架构模块化 (Modular Architecture) —— 打造可插拔组件
**痛点**：代码耦合，想换分块策略或重排模型需要大改代码。
**核心任务**：重构 `rag/` 模块，实现策略模式。

1.  **工厂模式设计**
    *   `ChunkerFactory`：动态切换分块策略（配置驱动）。
    *   `RerankerFactory`：动态切换重排模型（或开关）。
2.  **配置驱动开发**
    *   在 `config.py` 中新增 `CHUNK_SIZE`, `CHUNK_OVERLAP`, `ENABLE_RERANK` 等开关。
    *   实现“零代码”调整检索策略。

### 第四部分：端到端验收 (End-to-End Evaluation)
**任务**：构建一个完整的测试集，量化评估优化效果。

1.  **构建测试集**
    *   准备 5 个典型问题（包含语义模糊、多跳推理等难点）。
2.  **A/B 测试**
    *   **Baseline**：无分块 + 无重排。
    *   **Optimized**：递归分块 + 滑动窗口 + 重排序。
3.  **指标观察**
    *   命中率 (Hit Rate)。
    *   响应质量 (LLM 回答的准确性)。

---

## 🛠️ 技术栈预告
*   **LangChain Text Splitters**: 递归分块核心库。
*   **PyMuPDF / Unstructured**: 文档解析工具。
*   **FlagEmbedding / BGE-Reranker**: 高性能重排序模型。
*   **Config Pattern**: 策略模式在 Python 中的落地。

---

## 💡 预期成果
课程结束后，你将拥有：
1.  一个**支持上传本地文件**（PDF/MD）并自动分块入库的工具。
2.  一个**具备“粗排 + 精排”双重过滤**的高精度检索管道。
3.  一套**可配置、可插拔**的 RAG 模块化代码架构。

---

**👉 请评估：**
1.  这个大纲是否符合你当前的预期？
2.  是否有特定想优先深入的技术点（比如特别关注 PDF 解析，或者特别关注重排序模型的选择）？
3.  是否需要调整节奏（例如将“分块”和“重排”拆分为两讲详细讲解）？

请给出你的反馈，确认后我们立即开始 **第一部分：智能分块与数据摄入** 的实战！


# 📘 第四讲：RAG 深度优化实战 —— 从“可用”到“卓越”

> **核心目标**：填补现有架构中的“未竟之地”，挖掘 Metadata 与 Summary 的深层价值，引入多路召回机制，解决复杂场景下的检索失效问题。
> **演进路线**：**Advanced RAG** $\rightarrow$ **High-Performance / Modular RAG**。
> **关键指标**：召回率 (Recall) 提升 30%+，复杂问答准确率显著提升。

---

## 📅 课程大纲预览

### 第一部分：元数据与摘要的深度利用 (Metadata & Summary Deep Dive)
**痛点**：目前的 Metadata（摘要、假设性问题）仅存储在数据库中，检索时未被充分利用，导致“存而不用”。
**核心任务**：实现**混合检索 (Hybrid Search)** 与 **预过滤/后过滤** 策略。

1.  **摘要增强检索 (Summary-Augmented Retrieval)**
    *   **策略 A**：将 `summary` 字段也进行向量化，与 `text` 字段共同构建索引（双字段检索）。
    *   **策略 B**：在 Prompt 中动态注入 Summary，让 LLM 基于摘要快速判断相关性。
2.  **假设性问题匹配 (Hypothetical Question Matching)**
    *   利用存储的 `questions` 字段进行**关键词匹配 (BM25)** 或 **专用向量检索**。
    *   解决“用户提问”直接匹配“文档陈述句”的语义鸿沟。
3.  **元数据过滤 (Metadata Filtering)**
    *   实现基于 `category`, `source`, `page` 的**预过滤 (Pre-filtering)**。
    *   场景：用户问“HR 政策里的年假规定”，自动过滤掉“产品手册”中的内容。

### 第二部分：多路召回与融合 (Multi-Retrieval & Fusion)
**痛点**：单一向量检索无法覆盖所有场景（如精确匹配专有名词、代码片段、特定术语）。
**核心任务**：构建**多路召回 (Multi-Query / Hybrid Retrieval)** 管道。

1.  **关键词检索 (Sparse Retrieval / BM25)**
    *   引入 **BM25 算法**（Milvus 支持或集成 Elasticsearch/Whoosh）。
    *   优势：精确匹配专有名词、型号、代码错误码。
2.  **多查询生成 (Multi-Query Retrieval)**
    *   让 LLM 从不同角度生成 3-5 个变体查询（如：正面、反面、侧面描述）。
    *   并行执行多次向量检索，扩大召回覆盖面。
3.  **结果融合策略 (Fusion Strategies)**
    *   **倒数排名融合 (RRF, Reciprocal Rank Fusion)**：工业界标准融合算法，无需调参，自动平衡各路结果。
    *   **加权评分融合**：根据业务需求调整向量分与关键词分的权重。

### 第三部分：父子索引与上下文窗口优化 (Parent-Child Indexing)
**痛点**：小 Chunk 检索准但上下文缺失，大 Chunk 上下文全但检索噪声大。
**核心任务**：实现 **父子文档索引 (Parent-Child Retrieval)**。

1.  **大小块双索引策略**
    *   **子块 (Child)**：小粒度分块（200字），用于高精度向量检索。
    *   **父块 (Parent)**：大粒度分块（1000字）或原文档，用于提供给 LLM 完整上下文。
2.  **检索与替换逻辑**
    *   检索命中“子块”后，自动通过 ID 映射找回对应的“父块”。
    *   将“父块”送入 LLM，既保证了检索精度，又提供了充足上下文。
3.  **实战演练**
    *   修改 `ingestion.py` 建立父子映射关系。
    *   修改 `milvus_client.py` 支持 ID 反向查找。

### 第四部分：端到端评估与自动化测试 (Evaluation & Testing)
**痛点**：优化效果靠“体感”，缺乏量化数据支撑。
**核心任务**：构建 **RAG 评估框架**。

1.  **构建金标准数据集 (Golden Dataset)**
    *   准备 20-50 个典型问答对 (Query, Ground Truth Answer, Relevant Chunks)。
2.  **核心指标计算**
    *   **Hit Rate @ K**：正确答案是否在 Top-K 中。
    *   **MRR (Mean Reciprocal Rank)**：正确答案的平均排名倒数。
    *   **Faithfulness**：LLM 回答是否忠实于检索内容（防幻觉）。
3.  **A/B 测试框架**
    *   对比配置：`Base RAG` vs `Hybrid Search` vs `Parent-Child`。
    *   输出可视化报告，量化每一步优化的收益。

---

## 🛠️ 技术栈预告
*   **Milvus Scalar Filtering**: 元数据预过滤。
*   **Rank-BM25 / Milvus Text Embedding**: 稀疏向量/关键词检索。
*   **LangChain RRF**: 倒数排名融合算法。
*   **Ragas / Arize Phoenix** (可选): 自动化评估框架（或自建简易版）。
*   **ID Mapping Logic**: 父子索引核心逻辑。

---

## 💡 预期成果
课程结束后，你将拥有：
1.  **混合检索引擎**：同时支持向量语义匹配 + 关键词精确匹配 + 元数据过滤。
2.  **上下文增强机制**：通过父子索引，彻底解决“检索准但回答缺上下文”的矛盾。
3.  **量化评估体系**：一套可复用的测试脚本，用数据证明优化的价值。
4.  **真正的企业级 RAG**：能够处理复杂、模糊、专业术语多的真实业务场景。

---

## 🗓️ 建议学习节奏
*   **第 1 步**：元数据深度利用 (最快见效，改动最小)。
*   **第 2 步**：多路召回与融合 (核心难点，提升最大)。
*   **第 3 步**：父子索引 (架构升级，解决长上下文问题)。
*   **第 4 步**：自动化评估 (收尾验证，数据驱动)。

---

**👉 请评估：**
1.  这个大纲是否覆盖了你关心的“坑”（Summary, Metadata, 多路召回）？
2.  **父子索引 (Parent-Child)** 是否是你当前迫切需要的？（如果不是，我们可以替换为其他优化点，如“查询路由 Query Routing”）。
3.  是否需要调整顺序？例如先做“多路召回”还是先做“元数据过滤”？

请给出你的反馈，确认后我们将立即开始 **第一部分：元数据与摘要的深度利用**！