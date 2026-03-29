import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv(override=True)

print("Testing DeepSeek API connection...")
print(f"API Key: {os.environ.get('OPENAI_API_KEY')}")
print(f"Base URL: {os.environ.get('OPENAI_BASE_URL')}")

try:
    # 创建OpenAI客户端
    llm = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL")
    )
    
    # 测试简单的生成
    response = llm.invoke("Hello, how are you?")
    print("\nAPI Test Success!")
    print(f"Response: {response.content}")
    
except Exception as e:
    print(f"\nAPI Test Failed: {e}")
    import traceback
    traceback.print_exc()
