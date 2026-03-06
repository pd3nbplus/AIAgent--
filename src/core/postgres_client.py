import logging

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import settings

logger = logging.getLogger(__name__)


class PostgresClient:
    def __init__(self, database_url: str | None = None) -> None:
        # 统一走 core 配置：像 MySQL 一样用一行 URL 管理 PostgreSQL 连接
        self.url = database_url or settings.db.postgres_database_url
        # 连接池参数复用全局 db 配置，避免在多处重复维护。
        self.engine: Engine = create_engine(
            self.url,
            pool_pre_ping=settings.db.pool_pre_ping,
            pool_size=settings.db.pool_size,
            max_overflow=settings.db.max_overflow,
            echo=settings.db.echo,
            future=True,
        )
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)

    def get_session(self) -> Session:
        # 仅提供会话，不绑定具体业务逻辑；具体建表/写入由上层组件实现。
        return self.SessionLocal()


_postgres_client_instance: PostgresClient | None = None


def get_postgres_client(database_url: str | None = None) -> PostgresClient:
    """
    获取 PostgreSQL 客户端：
    - 未传 database_url: 返回全局单例（默认用于主流程）
    - 传入 database_url: 创建独立实例（用于特定功能隔离）
    """
    global _postgres_client_instance
    if database_url:
        return PostgresClient(database_url=database_url)
    if _postgres_client_instance is None:
        _postgres_client_instance = PostgresClient()
    return _postgres_client_instance
