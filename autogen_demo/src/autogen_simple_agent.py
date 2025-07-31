import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily, ModelInfo
# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.
import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 从环境变量中获取apikey、modelname和baseurl
api_key = os.getenv("BAILIAN_API_KEY")
model_name = os.getenv("QWEN3_MODEL")
base_url = os.getenv("BAILIAN_API_BASE_URL")

# 创建ModelInfo对象，包含必需的vision字段
model_info = ModelInfo(
    name=model_name,
    vision=False,  # 设置vision字段为False，表示不支持视觉功能
    function_calling=True,  # 支持函数调用
    json_output=True,  # 支持json输出
    family=ModelFamily.ANY,  # 设置模型家族为ANY
    structured_output=True,  # 支持结构化输出
    multiple_system_messages=True,  # 支持多个非连续的系统消息
)

model_client = OpenAIChatCompletionClient(
    model=model_name,
    api_key=api_key,
    model_info=model_info,
    base_url=base_url,
    enable_thinking=False,
)

# Define a simple function tool that the agent can use.
# For this example, we use a fake weather tool for demonstration purposes.
async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is 73 degrees and Sunny."

# Define an AssistantAgent with the model, tool, system message, and reflection enabled.
# The system message instructs the agent via natural language.
# Create the primary agent.
primary_agent = AssistantAgent(
    "primary",
    model_client=model_client,
    system_message="You are a helpful AI assistant.",
    # reflect_on_tool_use=True,
    model_client_stream=True,
)

# Create the critic agent.
critic_agent = AssistantAgent(
    "critic",
    model_client=model_client,
    system_message="Provide constructive feedback. Respond with 'APPROVE' to when your feedbacks are addressed.",
    #表示在使用工具后，智能体会对工具调用的结果进行反思和自我评估，从而提升回复的准确性和可靠性。
    # reflect_on_tool_use=True, 
    # 启用后，模型客户端会以流式方式返回生成的内容，实现边生成边输出。
    model_client_stream=True,
)

# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("APPROVE")

# Create a team with the primary and critic agents.
team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=text_termination)

async def main():
    await Console(team.run_stream(task="Write a short poem about the fall season."))

asyncio.run(main())
