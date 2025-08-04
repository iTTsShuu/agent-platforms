import string
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModelSettings
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("BAILIAN_API_KEY")
base_url = os.getenv("BAILIAN_API_BASE_URL")
model_name = os.getenv("QWEN3_MODEL")

# 模拟数据库 - 使用用户名称作为键
accounts_db = {
    "张三": {"balance": 10000.0},
    "李四": {"balance": 5000.0},
    "王五": {"balance": 2000.0},
    "赵六": {"balance": 50000.0},
    "我": {"balance": 10000.0}
}

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

model=OpenAIModel(
    model_name,
    provider=OpenAIProvider(
        api_key=api_key,
        base_url=base_url,
    ),
    settings=OpenAIModelSettings(
        extra_body={"enable_thinking": False}
    )
)

transfer_agent = Agent(  
    model,
    # 这项配置指定了依赖项的类型为int，通常用于agent在处理工具函数参数时进行类型校验或推断，确保传递给工具的参数类型正确。
    # deps_type=int,
    # output_type=bool,
    system_prompt=transfer_prompt,
)

# 工具函数1: 查询余额
@transfer_agent.tool
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
@transfer_agent.tool
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
@transfer_agent.tool
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
@transfer_agent.tool
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

def chat_with_transfer_agent():
    print("输入 'quit' 退出对话")
    print("-" * 50)
    # 创建上下文
    
    while True:
        try:
            # 获取用户输入
            user_input = input("用户: ").strip()
            
            # 检查退出条件
            if user_input.lower() in ['exit', 'quit', '退出', 'q']:
                print("再见！")
                break
            if not user_input:
                continue
            # 运行 agent
            response =  transfer_agent.run_sync(user_input)
            print(response)
            print(f"助手: {response.output}")
            
        except Exception as e:
            print(f"发生错误: {e}")
            print("请重试或输入 'quit' 退出")
            continue

def main():
    """
    主函数
    """
    # 运行对话
    chat_with_transfer_agent()

if __name__ == "__main__":
    main()
