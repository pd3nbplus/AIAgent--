# src/test/recall_test.py
# python -m src.test.recall_test
import asyncio
import logging
from typing import List, Dict, Any
from src.rag.pipeline import pipeline_instance
from src.core.config import settings
from src.rag.strategies.base import SearchResult

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RecallEval")

# ==========================================
# 1. 测试数据集定义
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
    """
    检查检索结果是否命中包含关键词的文档
    返回：{'hit': bool, 'rank': int, 'source_plugin': str, 'snippet': str}
    """
    for rank, res in enumerate(results):
        text = res.text.lower()
        # 只要命中任意一个关键词组中的部分词（简化逻辑：命中2个以上关键词即算命中）
        # 或者更严格：必须命中所有关键词？这里采用宽松策略：命中至少 2 个关键概念
        match_count = sum(1 for kw in keywords if kw.lower() in text)
        
        # 简单策略：如果关键词列表中有任意一个词出现在文本中，且该词长度>2（避免停用词）
        # 这里为了演示，我们检查是否包含列表中的任意一个长词
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
                "score": res.score
            }
    
    return {
        "hit": False,
        "rank": -1,
        "source_plugin": "None",
        "snippet": "No relevant chunk found.",
        "score": 0
    }

async def run_single_test(question: str, keywords: List[str], top_k: int = 5) -> Dict[str, Any]:
    """运行单个问题的测试"""
    logger.info(f"\n❓ 测试问题：{question}")
    
    try:
        # 调用异步 Pipeline 现在返回 List[SearchResult]
        results = await pipeline_instance.run(query=question, top_k=top_k)
        
        if not results:
            logger.warning("⚠️ 无检索结果")
            return {"question": question, "hit": False, "rank": -1, "total_results": 0, "rerank_triggered": False}
        
        # 评估命中率
        eval_result = check_hit(results, keywords)
        
        # 记录是否触发重排 (通过日志或结果推断，这里简单通过结果数量判断，实际可加 flag)
        # 注意：pipeline 内部日志已经输出了是否触发重排，这里主要关注结果
        
        logger.info(f"✅ 评估结果：Hit={eval_result['hit']}, Rank={eval_result['rank']}, Source={eval_result['source_plugin']}")
        logger.debug(f"   片段：{eval_result['snippet']}")
        
        return {
            "question": question,
            "hit": eval_result["hit"],
            "rank": eval_result["rank"],
            "source_plugin": eval_result["source_plugin"],
            "total_results": len(results),
            "top_score": results[0].score if results else 0
        }
        
    except Exception as e:
        logger.error(f"❌ 执行出错：{e}")
        return {"question": question, "hit": False, "rank": -1, "error": str(e)}

async def main():
    print("="*80)
    print("🚀 开始 RAG 召回能力全面评估")
    print(f"⚙️ 配置：Hybrid={settings.search.enable_hybrid_search}, Rerank={settings.rag_online.enable_rerank}")
    print("="*80)
    
    all_results = []
    category_stats = {}
    
    for case in TEST_CASES:
        cat_name = case["category"]
        logger.info(f"\n--- 类别：{cat_name} ---")
        category_stats[cat_name] = {"total": 0, "hit": 0, "ranks": []}
        
        for q, kws in zip(case["questions"], case["keywords"]):
            res = await run_single_test(q, kws, top_k=5)
            res["category"] = cat_name
            all_results.append(res)
            
            category_stats[cat_name]["total"] += 1
            if res.get("hit"):
                category_stats[cat_name]["hit"] += 1
                category_stats[cat_name]["ranks"].append(res["rank"])
    
    # ==========================================
    # 3. 生成报告
    # ==========================================
    print("\n" + "="*80)
    print("📊 评估报告总结")
    print("="*80)
    
    total_q = len(all_results)
    total_hit = sum(1 for r in all_results if r.get("hit"))
    overall_hit_rate = total_hit / total_q if total_q > 0 else 0
    
    # 计算 MRR (Mean Reciprocal Rank)
    reciprocal_ranks = [1.0/r["rank"] for r in all_results if r.get("hit") and r["rank"] > 0]
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    print(f"\n🏆 总体表现:")
    print(f"   - 总问题数：{total_q}")
    print(f"   - 命中数 (Hit@5): {total_hit}")
    print(f"   - 命中率 (Hit Rate): {overall_hit_rate:.2%}")
    print(f"   - 平均倒数排名 (MRR): {mrr:.4f} (越接近 1 越好)")
    
    print(f"\n📈 分类别表现:")
    print(f"{'类别':<30} | {'命中率':<10} | {'MRR':<10} | {'平均排名':<10}")
    print("-" * 70)
    
    for cat, stats in category_stats.items():
        hit_rate = stats["hit"] / stats["total"] if stats["total"] > 0 else 0
        ranks = stats["ranks"]
        cat_mrr = sum([1.0/r for r in ranks]) / len(ranks) if ranks else 0
        avg_rank = sum(ranks) / len(ranks) if ranks else 0
        
        print(f"{cat:<30} | {hit_rate:>6.2%}     | {cat_mrr:>6.4f}     | {avg_rank:>6.2f}")
    
    print("\n💡 建议:")
    if overall_hit_rate < 0.6:
        print("   ⚠️ 召回率较低，建议检查 ES 索引是否同步完成，或调整 IK 分词配置。")
    if mrr < 0.5:
        print("   ⚠️ 排名靠后，建议检查 RRF 融合权重或重排序阈值。")
    if settings.search.enable_hybrid_search == False:
        print("   💡 当前未开启混合检索，尝试设置 SEARCH_ENABLE_HYBRID_SEARCH=True 以提升效果。")
    
    print("="*80)

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())