from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
import os
from dotenv import load_dotenv
load_dotenv()
def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."

tool_node = ToolNode([get_weather])

model = init_chat_model(
    model=os.getenv("QWEN3_MODEL"),
    model_provider="openai",  # 指定为openai兼容的API
    api_key=os.getenv("BAILIAN_API_KEY"),
    base_url=os.getenv("BAILIAN_API_BASE_URL"),
    extra_body={"enable_thinking": False}
)
model_with_tools = model.bind_tools([get_weather])

def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

builder = StateGraph(MessagesState)

# Define the two nodes we will cycle between
builder.add_node("call_model", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", should_continue, ["tools", END])
builder.add_edge("tools", "call_model")

graph = builder.compile()

graph.invoke({"messages": [{"role": "user", "content": "what's the weather in sf?"}]})