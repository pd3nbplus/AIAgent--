-- Active: 1772039059386@@127.0.0.1@3307@agent_memory
-- 创建数据库（如果还没创建）
CREATE DATABASE IF NOT EXISTS agent_memory CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE agent_memory;

-- 创建用户画像表
CREATE TABLE IF NOT EXISTS user_profiles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    thread_id VARCHAR(255) NOT NULL, -- 关联会话 ID (可选，也可以按 user_id)
    user_key VARCHAR(100) NOT NULL, -- 记忆的键，如 "name", "favorite_food"
    user_value TEXT NOT NULL, -- 记忆的值，如 "IronMan", "Pizza"
    confidence_score FLOAT DEFAULT 1.0, -- 置信度 (0.0-1.0)，Agent 不确定时可以存低分
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    -- 索引优化：快速查找某个会话的所有记忆，或特定键
    UNIQUE KEY uk_thread_key (thread_id, user_key),
    INDEX idx_thread (thread_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;