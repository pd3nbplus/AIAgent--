# src\Mini_Agent\tools\memory_tools.py
from langchain_core.tools import tool
from src.core.db_session import get_db_session
from src.core.models import UserProfile
from sqlalchemy import select
from typing import List

@tool
def save_user_memory(key: str, value: str, thread_id: str) -> str:
    """
    保存用户的关键信息到长期记忆。
    参数：
        key: 信息类别 (如 'name', 'job')
        value: 具体内容
        thread_id: 当前会话 ID
    """
    session = get_db_session()
    try:
        # 1. 尝试查询是否已存在
        stmt = select(UserProfile).where(
            UserProfile.thread_id == thread_id,
            UserProfile.user_key == key
        )
        existing_profile = session.execute(stmt).scalars().first()

        if existing_profile:
            # 2. 如果存在，更新值
            existing_profile.user_value = value
            existing_profile.confidence_score = 1.0
            # updated_at 会自动由 onupdate 触发
        else:
            # 3. 如果不存在，插入新记录
            new_profile = UserProfile(
                thread_id=thread_id,
                user_key=key,
                user_value=value,
                confidence_score=1.0
            )
            session.add(new_profile)

        session.commit()
        return f"✅ 成功记住：{key} = {value}"
    
    except Exception as e:
        session.rollback()
        return f"❌ 保存记忆失败：{str(e)}"
    finally:
        session.close()

@tool
def get_user_memory(keys: List[str], thread_id: str) -> str:
    """
    查询用户的长期记忆。
    参数：
        keys: 想要查询的键列表 (如 ['name', 'job'])。为空则查所有。
        thread_id: 当前会话 ID
    """
    session = get_db_session()
    try:
        query = select(UserProfile).where(UserProfile.thread_id == thread_id)
        
        if keys:
            query = query.where(UserProfile.user_key.in_(keys))
        
        # 按创建时间排序，确保拿到最新的（虽然 unique 约束保证只有一个）
        query = query.order_by(UserProfile.created_at.desc())
        
        results = session.execute(query).scalars().all()
        
        if not results:
            return "ℹ️ 未找到相关记忆。"
        
        # 格式化为易读字符串
        memory_list = [f"{p.user_key}: {p.user_value}" for p in results]
        return "🧠 长期记忆加载：" + "; ".join(memory_list)

    except Exception as e:
        return f"❌ 查询记忆失败：{str(e)}"
    finally:
        session.close()

memory_tools = [save_user_memory, get_user_memory]