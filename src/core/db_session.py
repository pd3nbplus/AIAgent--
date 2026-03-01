# src/core/db_session.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from src.core.models import Base
from src.core.config import settings

# 创建引擎
engine = create_engine(
    settings.db.database_url, 
    pool_pre_ping=settings.db.pool_pre_ping,        # 自动检测断连并重连
    pool_size=settings.db.pool_size,                # 连接池大小
    max_overflow=settings.db.max_overflow,          # 最大溢出连接数
    echo=settings.db.echo                           # 生产环境关闭 SQL 日志，调试时设为 True
)

# 创建 Session 工厂
# scoped_session 确保在多线程/异步环境下每个线程有独立的 Session
SessionLocal = scoped_session(sessionmaker(bind=engine))

def init_db():
    """初始化数据库：如果表不存在则自动创建"""
    Base.metadata.create_all(bind=engine)
    print("✅ 数据库表结构已同步 (ORM 模式)")

def get_db_session():
    """获取数据库会话"""
    return SessionLocal()