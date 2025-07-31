from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

llm=init_chat_model(
    model=os.getenv("QWEN3_MODEL"),
    model_provider="openai",  # 指定为openai兼容的API
    api_key=os.getenv("BAILIAN_API_KEY"),
    base_url=os.getenv("BAILIAN_API_BASE_URL"),
    extra_body={"enable_thinking": False}
)
def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent=create_react_agent(
    model=llm,
    tools=[get_weather],
    prompt="You are a helpful assistant on get weather.",
)

response =agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in Zhuhai"}]}
)
print(response)