# src/test/test_ingestion.py
# python -m src.test.test_ingestion
import os
from src.rag.ingestion import ingestion_pipeline
import logging

# 设置根日志记录器的级别为 INFO，并配置输出格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    # 获取项目根目录（即 src 的上一级目录）
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # 配置你的文档目录，直接基于项目根目录拼接
    docs_folder = os.path.join(project_root, "data", "docs")
    
    if not os.path.exists(docs_folder):
        print(f"❌ 目录不存在：{docs_folder}, 请先创建并放入测试文件")
        exit(1)
    
    # 开始批量处理
    # category 用于后续过滤，比如 "hr", "product"
    ingestion_pipeline.process_directory(docs_folder, category="company_knowledge")
    
    print("\n💡 提示：请登录 Attu (http://localhost:3000) 与 ES数据库 查看新增的数据和元数据！")