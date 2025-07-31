from typing import Dict, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel
import uuid
import os
import json
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 我的账户信息
MY_ACCOUNT = "my_account"
MY_BALANCE = 10000.0

# 模拟数据库
accounts_db = {
    "acc_001": {"balance": 10000.0, "owner": "张三"},
    "acc_002": {"balance": 5000.0, "owner": "李四"},
    "acc_003": {"balance": 2000.0, "owner": "王五"},
    "acc_004": {"balance": 50000.0, "owner": "赵六"},
    MY_ACCOUNT: {"balance": MY_BALANCE, "owner": "我"}
}

# 定义状态类型 - 优化后的字段
class TransferState(TypedDict):
    """
    转账状态，包含整个转账流程中需要追踪和传递的关键信息。
    
    字段说明：
    - user_request: 用户的原始转账请求文本
    - to_account: 转入账户的标识
    - amount: 转账金额
    - success: 转账是否成功的标志
    - message: 当前流程节点的提示信息或处理结果
    - transfer_id: 本次转账的唯一标识
    """
    user_request: str
    to_account: str
    amount: float
    success: bool
    message: str
    transfer_id: str

# 初始化 LLM
llm = init_chat_model(
    model=os.getenv("QWEN3_MODEL"),
    model_provider="openai",  # 指定为openai兼容的API
    api_key=os.getenv("BAILIAN_API_KEY"),
    base_url=os.getenv("BAILIAN_API_BASE_URL"),
    extra_body={"enable_thinking": False}
)

# 工具函数1: 查询余额
def query_balance(account_id: str) -> str:
    """
    查询指定账户的余额
    Args:
        account_id: 账户ID，例如 'acc_001', 'my_account'
    Returns:
        账户余额信息
    """
    if account_id in accounts_db:
        account_info = accounts_db[account_id]
        return f"账户 {account_id} ({account_info['owner']}) 的余额为: {account_info['balance']} 元"
    else:
        return f"账户 {account_id} 不存在"

# 工具函数2: 查询账户信息
def query_account(account_id: str) -> str:
    """
    查询指定账户的详细信息
    Args:
        account_id: 账户ID，例如 'acc_001', 'my_account'
    Returns:
        账户详细信息
    """
    if account_id in accounts_db:
        account_info = accounts_db[account_id]
        return f"账户ID: {account_id}, 户主: {account_info['owner']}, 余额: {account_info['balance']} 元"
    else:
        return f"账户 {account_id} 不存在"

# 工具函数3: 执行转账
def execute_transfer(to_account: str, amount: float) -> str:
    """
    执行转账操作，从我的账户转出资金
    Args:
        to_account: 目标账户ID
        amount: 转账金额
    Returns:
        转账结果信息
    """
    # 检查目标账户是否存在
    if to_account not in accounts_db:
        return f"转账失败：目标账户 {to_account} 不存在"
    
    # 检查我的余额是否足够
    if accounts_db[MY_ACCOUNT]["balance"] < amount:
        return f"转账失败：余额不足。当前余额: {accounts_db[MY_ACCOUNT]['balance']} 元，需要: {amount} 元"
    
    # 执行转账
    accounts_db[MY_ACCOUNT]["balance"] -= amount
    accounts_db[to_account]["balance"] += amount
    
    # 生成转账ID
    transfer_id = str(uuid.uuid4())[:8]
    
    return f"转账成功！转账ID: {transfer_id}, 从我的账户转出 {amount} 元到 {to_account} ({accounts_db[to_account]['owner']})"

def information_collector_agent(state: TransferState) -> TransferState:
    """第一个Agent：负责总结用户需求为固定格式的一句话"""
    user_request = state["user_request"]
    
    system_prompt = """
    你是一个银行转账需求总结助手。你的任务是：
    1. 分析用户的转账请求
    2. 将用户需求总结为固定格式的一句话
    
    总结格式：转[金额]元给[账户持有人姓名]
    
    可用账户信息:
    - acc_001: 张三
    - acc_002: 李四  
    - acc_003: 王五
    - acc_004: 赵六
    
    如果用户没有明确指定金额，请合理推断一个金额。
    如果用户没有明确指定目标账户，请选择最合适的账户。
    
    请直接返回总结后的固定格式句子，不要包含其他内容。
    """
    
    user_message = f"用户请求: {user_request}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    response = llm.invoke(messages)
    
    # 将总结的需求存储到message字段中
    return {
        **state,
        "message": response.content.strip()
    }

def transfer_executor_agent(state: TransferState) -> TransferState:
    """第二个Agent：负责解析固定格式句子并执行转账"""
    summary_message = state["message"]
    
    system_prompt = """
    你是一个银行转账执行助手。你的任务是：
    1. 解析固定格式的转账需求句子
    2. 提取目标账户和转账金额
    3. 执行转账操作
    4. 返回转账结果
    
    固定格式：转[金额]元给[账户持有人姓名]
    
    账户映射关系：
    - 张三 -> acc_001
    - 李四 -> acc_002
    - 王五 -> acc_003
    - 赵六 -> acc_004
    
    请解析句子并提取信息，然后以JSON格式返回：
    {
        "to_account": "账户ID",
        "amount": 金额
    }
    
    注意：请确保返回的是有效的JSON格式，不要包含其他文字。
    """
    
    # 构建解析消息
    parse_message = f"请解析这个转账需求：{summary_message}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=parse_message)
    ]
    
    response = llm.invoke(messages)
    
    try:
        # 解析响应中的JSON
        response_text = response.content.strip()
        print(f"DEBUG - 第二个Agent收到的消息: {summary_message}")
        print(f"DEBUG - 第二个Agent的解析结果: {response_text}")
        
        if "{" in response_text and "}" in response_text:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_str = response_text[start:end]
            extracted_data = json.loads(json_str)
            
            to_account = extracted_data.get("to_account", "acc_001")
            amount = float(extracted_data.get("amount", 1000))
            
            # 执行转账
            transfer_result = execute_transfer.invoke({
                "to_account": to_account,
                "amount": amount
            })
            
            # 检查转账是否成功
            success = "转账成功" in transfer_result
            
            # 提取转账ID（如果成功）
            transfer_id = ""
            if success and "转账ID:" in transfer_result:
                transfer_id = transfer_result.split("转账ID: ")[1].split(",")[0]
            
            return {
                **state,
                "to_account": to_account,
                "amount": amount,
                "success": success,
                "message": transfer_result,
                "transfer_id": transfer_id
            }
        else:
            return {
                **state,
                "to_account": "acc_001",
                "amount": 1000.0,
                "success": False,
                "message": "解析失败，使用默认值",
                "transfer_id": ""
            }
    except Exception as e:
        print(f"DEBUG - 解析异常: {e}")
        return {
            **state,
            "to_account": "acc_001",
            "amount": 1000.0,
            "success": False,
            "message": f"解析错误，使用默认值",
            "transfer_id": ""
        }

# 创建 LangGraph
def create_llm_transfer_graph():
    """创建使用两个协作Agent的转账流程图"""
    workflow = StateGraph(TransferState)
    
    # 添加节点
    workflow.add_node("collect_info", information_collector_agent)
    workflow.add_node("execute_transfer", transfer_executor_agent)
    
    # 设置入口点
    workflow.set_entry_point("collect_info")
    
    # 添加边
    workflow.add_edge("collect_info", "execute_transfer")
    workflow.add_edge("execute_transfer", END)
    
    return workflow.compile()

# 主函数
def main():
    """主函数 - 演示使用两个协作Agent的转账流程"""
    # 创建转账图
    transfer_graph = create_llm_transfer_graph()
    
    print("=== 使用两个协作Agent的转账系统演示 ===\n")
    
    # 显示初始账户状态
    print("初始账户状态:")
    for acc_id, acc_info in accounts_db.items():
        print(f"  {acc_id} ({acc_info['owner']}): {acc_info['balance']} 元")
    print()
    
    # 测试用例1: 正常转账
    print("测试1: 正常转账")
    initial_state = {
        "user_request": "我想转1000元给张三",
        "to_account": "",
        "amount": 0.0,
        "success": False,
        "message": "",
        "transfer_id": ""
    }
    
    result = transfer_graph.invoke(initial_state)
    print(f"用户请求: {result['user_request']}")
    print(f"目标账户: {result['to_account']}")
    print(f"转账金额: {result['amount']} 元")
    print(f"转账结果: {result['message']}")
    print()
    
    # 测试用例2: 大额转账
    print("测试2: 大额转账")
    initial_state2 = {
        "user_request": "转5000元给李四",
        "to_account": "",
        "amount": 0.0,
        "success": False,
        "message": "",
        "transfer_id": ""
    }
    
    result2 = transfer_graph.invoke(initial_state2)
    print(f"用户请求: {result2['user_request']}")
    print(f"目标账户: {result2['to_account']}")
    print(f"转账金额: {result2['amount']} 元")
    print(f"转账结果: {result2['message']}")
    print()
    
    # 测试用例3: 模糊请求
    print("测试3: 模糊请求")
    initial_state3 = {
        "user_request": "帮我转点钱给王五",
        "to_account": "",
        "amount": 0.0,
        "success": False,
        "message": "",
        "transfer_id": ""
    }
    
    result3 = transfer_graph.invoke(initial_state3)
    print(f"用户请求: {result3['user_request']}")
    print(f"目标账户: {result3['to_account']}")
    print(f"转账金额: {result3['amount']} 元")
    print(f"转账结果: {result3['message']}")
    print()
    
    # 显示最终账户状态
    print("最终账户状态:")
    for acc_id, acc_info in accounts_db.items():
        print(f"  {acc_id} ({acc_info['owner']}): {acc_info['balance']} 元")

if __name__ == "__main__":
    main() 