# src/test/recall_chunk_strategy.py
# 用法: python -m src.test.recall_chunk_strategy
import asyncio
import logging
import statistics
from typing import List, Dict, Any
from src.rag.pipeline import pipeline_instance
from src.core.config import settings
from src.rag.strategies.base import SearchResult

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ChunkStrategyEval")

# ==========================================
# 1. 测试数据集 (与主测试集保持一致，确保可比性)
# ==========================================

TEST_CASES = [
    {
        "category": "精确关键词匹配 (Easy)",
        "questions": [
            "根据《奇葩星球有限公司员工差旅与报销标准手册》，乘坐高铁的票价超过多少元将被视为“奢侈出行”？",
            "在《小模型训练奇谭》中，推荐的“摸鱼Transformer”模型有多少参数？",
            "奇葩星球公司禁止报销哪种咖啡？",
        ],
        "keywords": [
            ["高铁", "奢侈出行", "10元"],
            ["摸鱼Transformer", "参数", "42000"],
            ["禁止报销", "咖啡", "星巴克"]
        ]
    },
    {
        "category": "语义理解与推理 (Medium)",
        "questions": [
            "如果一名员工想通过步行出差获得最高荣誉，他每走一公里能拿到多少补贴？",
            "在预算极其有限的情况下，《小模型训练奇谭》认为真正的智能取决于什么？",
            "《小模型训练奇谭》推荐哪些非传统的数据来源用于训练小模型？",
        ],
        "keywords": [
            ["步行", "补贴", "50元"],
            ["智能", "脑洞密度", "参数量"],
            ["数据来源", "微信群", "分手短信", "流浪猫"]
        ]
    },
    {
        "category": "多跳推理 (Hard)",
        "questions": [
            "一名员工因工作压力在出差途中哭泣，根据公司规定，他如何能将情绪转化为实际的经济补偿？",
            "《小模型训练奇谭》中提到的“穷人版分布式训练”是如何利用社交工具实现的？",
        ],
        "keywords": [
            ["哭泣", "眼泪", "情感天平", "补贴"],
            ["穷人版分布式训练", "微信群", "梯度", "红包"]
        ]
    },
    {
        "category": "区分相似概念 (Confusion)",
        "questions": [
            "“泡面”在《奇葩星球有限公司员工差旅与报销标准手册》和《小模型训练奇谭》中分别扮演什么角色？",
            "两份文件都提到了“老板”，它们各自的态度或处理方式有何核心区别？",
        ],
        "keywords": [
            ["泡面", "觉醒补贴", "泡面RNN", "电热水壶"],
            ["老板", "黑暗料理", "画饼", "老板探测器"]
        ]
    }
]

# ==========================================
# 2. 评估逻辑
# ==========================================

def check_hit(results: List[SearchResult], keywords: List[str]) -> Dict[str, Any]:
    """检查检索结果是否命中"""
    for rank, res in enumerate(results):
        text = res.text.lower()
        is_hit = False
        for kw in keywords:
            if len(kw) > 2 and kw.lower() in text:
                is_hit = True
                break
        
        if is_hit:
            return {
                "hit": True,
                "rank": rank + 1,
                "source_plugin": res.source_field,
                "snippet": text[:100] + "...",
                "score": res.score,
                "text_length": len(res.text), # 👈 新增：记录返回文本长度
                "is_parent_mode": res.source_field.endswith("_parent") or res.metadata.get('_retrieval_mode') == 'child_to_parent' # 👈 新增：检测是否触发父子模式
            }
    
    return {
        "hit": False,
        "rank": -1,
        "source_plugin": "None",
        "snippet": "No relevant chunk found.",
        "score": 0,
        "text_length": 0,
        "is_parent_mode": False
    }

async def run_single_test(question: str, keywords: List[str], top_k: int = 5) -> Dict[str, Any]:
    """运行单个问题的测试"""
    logger.info(f"\n❓ 测试问题：{question}")
    
    try:
        results = await pipeline_instance.run(query=question, top_k=top_k)
        
        if not results:
            logger.warning("⚠️ 无检索结果")
            return {"question": question, "hit": False, "rank": -1, "total_results": 0, "text_lengths": [], "parent_mode_count": 0}
        
        eval_result = check_hit(results, keywords)
        
        # 统计本次查询的文本长度分布和父子模式触发情况
        text_lengths = [len(r.text) for r in results]
        parent_mode_count = sum(1 for r in results if r.source_field.endswith("_parent") or r.metadata.get('_retrieval_mode') == 'child_to_parent')
        
        logger.info(f"✅ 评估结果：Hit={eval_result['hit']}, Rank={eval_result['rank']}, Source={eval_result['source_plugin']}")
        logger.debug(f"   片段：{eval_result['snippet']} | 平均长度：{statistics.mean(text_lengths):.0f}")
        
        return {
            "question": question,
            "hit": eval_result["hit"],
            "rank": eval_result["rank"],
            "source_plugin": eval_result["source_plugin"],
            "total_results": len(results),
            "top_score": results[0].score if results else 0,
            "text_lengths": text_lengths,       # 👈 新增
            "parent_mode_count": parent_mode_count # 👈 新增
        }
        
    except Exception as e:
        logger.error(f"❌ 执行出错：{e}")
        import traceback
        traceback.print_exc()
        return {"question": question, "hit": False, "rank": -1, "error": str(e), "text_lengths": [], "parent_mode_count": 0}

async def main():
    print("="*80)
    print("🧪 RAG 分块策略对比测试 (单路召回)")
    print("="*80)
    
    # 打印当前关键配置
    strategy = settings.rag_offline.chunk_strategy
    use_parent = strategy == "parent_child"
    hybrid = settings.search.enable_hybrid_search
    
    print(f"⚙️ 当前配置:")
    print(f"   - 分块策略 (CHUNK_STRATEGY): {strategy}")
    print(f"   - 启用父子上下文 (USE_PARENT_CONTEXT): {use_parent}")
    print(f"   - 混合检索 (ENABLE_HYBRID_SEARCH): {hybrid} (建议为 False 以测试单路)")
    print("="*80)
    
    if hybrid:
        logger.warning("⚠️ 检测到混合检索已开启。为了纯粹对比分块策略，建议在 .env 中设置 SEARCH_ENABLE_HYBRID_SEARCH=False")
        exit(1)

    all_results = []
    category_stats = {}
    
    # 用于统计全局的文本长度和父子模式使用情况
    all_text_lengths = []
    total_parent_mode_hits = 0

    for case in TEST_CASES:
        cat_name = case["category"]
        logger.info(f"\n--- 类别：{cat_name} ---")
        category_stats[cat_name] = {"total": 0, "hit": 0, "ranks": []}
        
        for q, kws in zip(case["questions"], case["keywords"]):
            res = await run_single_test(q, kws, top_k=2)
            res["category"] = cat_name
            all_results.append(res)
            
            category_stats[cat_name]["total"] += 1
            if res.get("hit"):
                category_stats[cat_name]["hit"] += 1
                category_stats[cat_name]["ranks"].append(res["rank"])
            
            # 累积统计
            if res.get("text_lengths"):
                all_text_lengths.extend(res["text_lengths"])
            if res.get("parent_mode_count"):
                total_parent_mode_hits += res["parent_mode_count"]

    # ==========================================
    # 3. 生成报告
    # ==========================================
    print("\n" + "="*80)
    print("📊 评估报告总结")
    print("="*80)
    
    total_q = len(all_results)
    total_hit = sum(1 for r in all_results if r.get("hit"))
    overall_hit_rate = total_hit / total_q if total_q > 0 else 0
    
    # 计算 MRR
    reciprocal_ranks = [1.0/r["rank"] for r in all_results if r.get("hit") and r["rank"] > 0]
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    # 计算文本长度统计
    avg_text_len = statistics.mean(all_text_lengths) if all_text_lengths else 0
    median_text_len = statistics.median(all_text_lengths) if all_text_lengths else 0
    
    print(f"\n🏆 总体表现:")
    print(f"   - 总问题数：{total_q}")
    print(f"   - 命中数 (Hit@5): {total_hit}")
    print(f"   - 命中率 (Hit Rate): {overall_hit_rate:.2%}")
    print(f"   - 平均倒数排名 (MRR): {mrr:.4f}")
    
    print(f"\n📏 上下文窗口分析:")
    print(f"   - 返回片段平均长度：{avg_text_len:.0f} 字")
    print(f"   - 返回片段中位长度：{median_text_len:.0f} 字")
    if strategy == "parent_child" and use_parent:
        print(f"   - 触发'查子返父'次数：{total_parent_mode_hits} 次 (共 {total_q * 5} 个结果)")
        print(f"   - 父子模式覆盖率：{(total_parent_mode_hits / (total_q * 5)) * 100:.1f}%")
    else:
        print(f"   - 父子模式：未启用或非 Parent-Child 策略")

    print(f"\n📈 分类别表现:")
    print(f"{'类别':<30} | {'命中率':<10} | {'MRR':<10} | {'平均排名':<10}")
    print("-" * 70)
    
    for cat, stats in category_stats.items():
        hit_rate = stats["hit"] / stats["total"] if stats["total"] > 0 else 0
        ranks = stats["ranks"]
        cat_mrr = sum([1.0/r for r in ranks]) / len(ranks) if ranks else 0
        avg_rank = sum(ranks) / len(ranks) if ranks else 0
        
        print(f"{cat:<30} | {hit_rate:>6.2%}     | {cat_mrr:>6.4f}     | {avg_rank:>6.2f}")
    
    print("\n💡 策略对比指南:")
    print("   请保存此报告，然后修改 .env 文件：")
    print("   1. 将 CHUNK_STRATEGY 从 'recursive' 改为 'parent_child' (或反之)")
    print("   2. 再次运行本测试脚本")
    print("   3. 对比两份报告的 [命中率] 和 [上下文窗口分析]")
    print("   👉 预期：Parent-Child 策略在保持命中率的同时，应显著增加 [平均长度]，从而提升 LLM 回答质量。")
    
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())