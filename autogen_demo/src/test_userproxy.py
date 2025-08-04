import asyncio
from autogen_core import CancellationToken
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.messages import TextMessage


async def simple_user_agent():
    agent = UserProxyAgent("user_proxy")
    response = await asyncio.create_task(
        agent.on_messages(
            [TextMessage(content="What is your name? ", source="user")],
            cancellation_token=CancellationToken(),
        )
    )
    assert isinstance(response.chat_message, TextMessage)
    print(f"Your name is {response.chat_message.content}")

