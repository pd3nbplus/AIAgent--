import redis
from langgraph.checkpoint.redis import RedisSaver
from src.core.config import settings

# 单例模式
class RedisClient:
    _instance = None
    def __new__(cls, *args, **kwargs):
        # 如果实例不存在，则创建新实例
        if not cls._instance:
            cls._instance = super().__new__(cls)
            # print("创建新的单例实例")
        else:
            print("返回已存在的单例实例")
        return cls._instance
    def __init__(self):
        self.host = settings.db.redis_host
        self.port = settings.db.redis_port
        self.db = 0
        # TTL 配置，包含：
        # default_ttl：检查点存活时间（分钟），默认 60 分钟。
        # refresh_on_read：读取时是否刷新 TTL，默认 True。
        self.ttl_config = {"default_ttl": 120, "refresh_on_read": False}
        self.redis = redis.Redis(host=self.host, port=self.port, db=self.db)
        self.redis_saver = RedisSaver(redis_client=self.redis,ttl=self.ttl_config)