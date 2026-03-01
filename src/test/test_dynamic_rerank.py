# src/test/test_dynamic_rerank.py
from src.rag.pipeline import pipeline_instance
import logging

# 设置根日志记录器的级别为 INFO，并配置输出格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# 测试用例 1: 明确查询 (期望：分数差距大，跳过重排)
QUERY_EASY = "公司的年假政策是什么？" 
# 测试用例 2: 模糊/易混淆查询 (期望：分数差距小，触发重排)
QUERY_HARD = "报销规则是怎样的？" 

def test_case(query, label):
    print(f"\n--- {label}: {query} ---")
    results = pipeline_instance.run(query, top_k=3)
    if not results:
        print("❌ 无结果")
        return
    
    print(f"✅ 最终结果 ({len(results)} 条):")
    for i, r in enumerate(results):
        # 显示来源分数和重排分数 (如果有)
        score_info = f"Vec:{r['score']:.4f}"
        if 'rerank_score' in r:
            score_info += f" | Rank:{r['rerank_score']:.4f}"
        print(f"{i+1}. [{score_info}] {r['text'][:40]}...")

if __name__ == "__main__":
    print("🚀 开始动态重排策略测试...")
    test_case(QUERY_EASY, "场景 A (明确查询)")
    test_case(QUERY_HARD, "场景 B (模糊查询)")
    
    print("\n💡 请观察日志中的 '触发重排序' 或 '跳过重排序' 提示。")