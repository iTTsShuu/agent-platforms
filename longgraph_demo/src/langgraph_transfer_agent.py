from typing import Dict, Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START,MessagesState
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from pydantic import BaseModel
import uuid
import os
import json
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 模拟数据库 - 使用用户名称作为键
accounts_db = {
    "张三": {"balance": 10000.0},
    "李四": {"balance": 5000.0},
    "王五": {"balance": 2000.0},
    "赵六": {"balance": 50000.0},
    "我": {"balance": 10000.0}
}

# 工具函数1: 查询余额
def get_balance(user_name: str) -> str:
    """
    查询指定用户的余额
    Args:
        user_name: 用户名称，例如 '张三', '李四', '我'
    Returns:
        用户余额信息
    """
    print(f"--tool called--查询余额: {user_name}")
    if user_name in accounts_db:
        account_info = accounts_db[user_name]
        return f"用户 {user_name} 的余额为: {account_info['balance']} 元"
    else:
        return f"用户 {user_name} 不存在"

# 工具函数2: 查询账户
def get_account(user_name: str) -> str:
    """
    查询指定用户
    Args:
        user_name: 用户名称，例如 '张三', '李四', '我'
    Returns:
        用户是否存在
    """
    print(f"--tool called--查询账户: {user_name}")
    if user_name in accounts_db:
        return f"用户 {user_name} 存在"
    else:
        return f"用户 {user_name} 不存在"

# 工具函数3: 执行转账
def execute_transfer(to_user: str, amount: float) -> str:
    """
    执行转账操作，从我的账户转出资金
    Args:
        to_user: 目标用户名称
        amount: 转账金额
    Returns:
        转账结果信息
    """
    print(f"--tool called--执行转账: {to_user} {amount}")
    # 检查我的余额是否足够
    if accounts_db["我"]["balance"] < amount:
        return f"转账失败：余额不足。当前余额: {accounts_db['我']['balance']} 元，需要: {amount} 元"
    
    # 执行转账
    accounts_db["我"]["balance"] -= amount
    accounts_db[to_user]["balance"] += amount
    return f"转账成功！ 从我的账户转出 {amount} 元到 {to_user}"

# 工具函数4: 向用户提问
def reply_to_user(content: str) -> str:
    """
    向用户提问
    Args:
        content: 提问内容
        state: 转账状态
    Returns:
        回答内容
    """
    print(f"--tool called--提问用户: {content}")
    # 获取用户输入
    user_input = input("reply:")
    return user_input

# 初始化 LLM
model = init_chat_model(
    model=os.getenv("QWEN3_MODEL"),
    model_provider="openai",  # 指定为openai兼容的API
    api_key=os.getenv("BAILIAN_API_KEY"),
    base_url=os.getenv("BAILIAN_API_BASE_URL"),
    extra_body={"enable_thinking": False}
)

# 构建系统提示
transfer_prompt = """
你是一个专业的银行转账助手，负责处理用户的转账请求。

你的任务：
1. 分析用户的转账需求，并整理转账信息
2. 收集必要的转账信息（目标账户、金额）
3. 验证信息的有效性，包含目标账户是否存在，金额是否足够
4. 当信息不足时，主动通过工具函数`reply_to_user`询问用户，直到信息收集完整。
5. 当信息收集完整后，先总结信息向用户确认，再通过工具函数`execute_transfer`执行转账，并返回转账结果

注意事项：
- 所有转账都从我的账户转出
- 确认转账信息中金额，并确保余额足够`get_balance`
- 目标账户必须存在`get_account`
- 请与用户进行自然对话，逐步收集信息，确保转账安全准确
- 当转账成功后，请发送"DONE"
"""
# 关键的状态类
class TransferState(MessagesState):
    """
    转账状态，包含整个转账流程中需要追踪和传递的关键信息。
    字段说明：
    - to_account: 转入账户的标识
    - amount: 转账金额
    """
    to_account: str
    amount: float
#toolnode
tools = ToolNode(tools=[get_balance,get_account,reply_to_user,execute_transfer])
# 绑定工具
model_with_tools = model.bind_tools([get_balance,get_account,reply_to_user,execute_transfer])

def call_model_transfer(state: TransferState):
    """调用模型转账"""
    # 构建完整消息列表，包含系统提示
    full_messages = [SystemMessage(content=transfer_prompt)] + state["messages"]
    response = model_with_tools.invoke(full_messages)
    print("转账助手：" + response.content)
    return {
        **state,
        "messages": state["messages"] + [response]
    } 

def user_input(state: TransferState):
    user_input_text = input("你：")
    return {
        **state,
        "messages": state["messages"] + [HumanMessage(content=user_input_text)]
    }

def is_tool_call(state: TransferState):
    last_message = state["messages"][-1]
    # 检查是否有工具调用
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    if "DONE" in state["messages"][-1].content:
        return END
    return "user_input"

# 创建 LangGraph
def create_transfer_graph():
    """创建支持多轮聊天的转账流程图"""
    workflow = StateGraph(TransferState)
    # 添加节点
    workflow.add_node("user_input", user_input)
    workflow.add_node("call_model_transfer", call_model_transfer)
    workflow.add_node("tools", tools)
    # 设置入口点
    workflow.add_edge(START, "user_input")
    workflow.add_edge("user_input", "call_model_transfer")
    # 工具执行后回到模型调用
    workflow.add_edge("tools", "call_model_transfer")
    # 添加条件边 - 检查是否有工具调用
    workflow.add_conditional_edges(
        "call_model_transfer", 
        is_tool_call,
        {
            "tools": "tools",
            "user_input": "user_input",
            END: END,
        }
    )
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)
    # 添加条件边 - 完成转账后退出
    # workflow.add_conditional_edges(
    #     "call_model_transfer", 
    #     success_transfer,
    #     {
    #         END: END,
    #         "user_input": "user_input"
    #     }
    # )


transfer_graph = create_transfer_graph()
transfer_graph.get_graph().draw_mermaid_png()
result = transfer_graph.invoke(
    {"messages": []},
    {"configurable": {"thread_id": "1"}}
)
# initial_state = {
#     "to_account": "",
#     "amount": 0.0,
#     "info_collected": False,
#     "success": False
# }