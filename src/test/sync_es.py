from src.core.es_client import es_client_instance

if __name__ == "__main__":
    if not es_client_instance.is_available():
        print("❌ ES 不可用，请检查配置")
        exit(1)
    
    print("🚀 开始存量同步...")
    es_client_instance.sync_from_milvus(batch_size=500)
    print("✅ 同步结束")