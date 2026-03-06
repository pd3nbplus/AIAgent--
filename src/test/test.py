from openai import OpenAI

client = OpenAI(
    api_key="ms-02d60235-2ddc-4431-b158-b8ea872682ec", # 请替换成您的ModelScope Access Token
    base_url="https://api-inference.modelscope.cn/v1/"
)


response = client.chat.completions.create(
    model="MiniMax/MiniMax-M2.5", # ModelScope Model-Id
    messages=[
        {
            'role': 'system',
            'content': 'You are a helpful assistant.'
        },
        {
            'role': 'user',
            'content': '用python写一下快排'
        }
    ],
    stream=False
)

print(response.choices[0].message.content)