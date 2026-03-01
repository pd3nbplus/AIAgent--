import redis
import json

# 连接 Redis
r = redis.Redis(host="127.0.0.1", port=6379, db=0, decode_responses=True)

# 查找所有包含 session_001 的 Key
pattern = "*session_001*"
keys = r.keys(pattern)

print(f"🔍 找到 {len(keys)} 个相关 Key:")
for key in keys:
    print(f"\n📂 Key: {key}")
    
    # 获取类型
    k_type = r.type(key)
    print(f"   类型：{k_type}")
    
    # 获取 TTL
    ttl = r.ttl(key)
    print(f"   剩余寿命：{ttl} 秒")
    
    # 获取内容
    if k_type == 'string':
        value = r.get(key)
        # 尝试格式化 JSON
        try:
            data = json.loads(value)
            # 简单打印消息内容
            if 'channel_values' in data and 'messages' in data['channel_values']:
                msgs = data['channel_values']['messages']
                print(f"   📝 内容预览 (最后一条): {msgs[-1]['content'][:50]}...")
            else:
                print(f"   📝 内容预览：{value[:100]}...")
        except:
            print(f"   📝 内容：{value[:100]}...")