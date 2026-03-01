# src/core/models.py
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, UniqueConstraint, Index
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class UserProfile(Base):
    __tablename__ = 'user_profiles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(String(255), nullable=False, index=True)  # 会话ID
    user_key = Column(String(100), nullable=False)               # 键：name, preference
    user_value = Column(Text, nullable=False)                    # 值：IronMan, Pizza
    confidence_score = Column(Float, default=1.0)                # 置信度
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 定义唯一约束：同一个 session 下，key 不能重复
    __table_args__ = (
        UniqueConstraint('thread_id', 'user_key', name='uk_thread_key'),
        Index('idx_thread', 'thread_id'),
    )

    def __repr__(self):
        return f"<UserProfile(thread_id={self.thread_id}, key={self.user_key}, value={self.user_value})>"