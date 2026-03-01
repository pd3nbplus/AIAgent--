# src/test/test_rag.py

from src.Mini_Agent.tools.rag_tools import add_knowledge, search_knowledge
import logging

# 设置根日志记录器的级别为 INFO，并配置输出格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    print("🚀 开始 RAG 测试...")
    
    # # 1. 添加知识
    # print("\n--- 添加知识 ---")
    # res1 = add_knowledge.invoke({
    #     "text": "公司的年假政策是：入职满 1 年有 5 天年假，满 3 年有 10 天年假。",
    #     "category": "hr_policy"
    # })
    # print(res1)
    
    # res2 = add_knowledge.invoke({
    #     "text": "我们的旗舰产品是 Qwen-Agent，它支持多模态输入和长上下文记忆。",
    #     "category": "product"
    # })
    # print(res2)
    
    # 2. 搜索知识
    # print("\n--- 搜索知识：我有多少天年假？ ---")
    # query = "我入职 3 年了，有多少天年假？"
    # res3 = search_knowledge.invoke({"query": query, "top_k": 2})
    # print(res3)
    
    # print("\n--- 搜索知识：旗舰产品是什么？ ---")
    # query2 = "你们最好的产品叫什么？"
    # res4 = search_knowledge.invoke({"query": query2, "top_k": 2})
    # print(res4)

    print("\n--- 搜索知识：员工报销相关规定是什么？ ---")
    query5 = "为什么不给我报销高铁票？"
    res5 = search_knowledge.invoke({"query": query5, "top_k": 2})
    print(res5)
    
    # print("\n✅ 测试完成！请打开 Attu (http://localhost:3000) 查看数据。")