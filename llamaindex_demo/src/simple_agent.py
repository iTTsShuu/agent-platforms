import os
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from llama_index.llms.dashscope import DashScope
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
import asyncio

# 加载环境变量
load_dotenv()

# 获取配置
api_key = os.getenv("BAILIAN_API_KEY")
model_name = os.getenv("QWEN3_MODEL")
base_url = os.getenv("BAILIAN_API_BASE_URL")

def get_weather() -> str:
    return f"天气信息:晴朗"

weather_tool = FunctionTool.from_defaults(
    fn=get_weather,
    name="get_weather",
    description="获取天气信息"
)

# 使用 OpenAI 兼容的 LLM 类，避免 Pydantic 冲突
llm = DashScope(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
    extra_body={"enable_thinking": False}
)

# 创建 FunctionAgent
agent = FunctionAgent(
    tools=[weather_tool], 
    prompt="你是一个天气助手，负责回答用户关于天气的问题。如果用户询问天气，请调用 get_weather 工具。",
    llm=llm,
    verbose=True
)

async def test_llm_directly():
    """直接测试 LLM，不通过 FunctionAgent"""
    try:
        print("🧪 直接测试 LLM...")
        response = await llm.acomplete("你好，请简单介绍一下你自己")
        print(f"🤖 LLM 响应: {response}")
        return True
    except Exception as e:
        print(f"❌ LLM 测试失败: {e}")
        return False

async def main():
    try:
        print("🚀 启动天气查询 Agent...")
        print(f"📋 使用模型: {model_name}")
        print(f"🔗 API 地址: {base_url}")
        
        # 检查环境变量
        if not api_key or not base_url:
            print("❌ 请设置 BAILIAN_API_KEY 和 BAILIAN_API_BASE_URL 环境变量")
            return
        
        # 首先测试 LLM 是否正常工作
        llm_works = await test_llm_directly()
        if not llm_works:
            print("❌ LLM 测试失败，跳过 FunctionAgent 测试")
            return
        
        print("\n💬 开始 FunctionAgent 测试...")
        response = await agent.run("今天天气怎么样？")
        print(f"🤖 助手: {response}")
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        print(f"🔍 错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())