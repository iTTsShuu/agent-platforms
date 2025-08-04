from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 模拟数据库
# 模拟数据库 - 使用用户名称作为键
accounts_db = {
    "张三": {"balance": 10000.0},
    "李四": {"balance": 5000.0},
    "王五": {"balance": 2000.0},
    "赵六": {"balance": 50000.0},
    "我": {"balance": 10000.0}
}

# 工具类定义
class GetAccountTool(BaseTool):
    name: str = "get_account"
    description: str = "获取指定用户是否存在"
    
    def _run(self, user_name: str) -> str:
        print(f"--toolcall--查询用户: {user_name}")
        if user_name in accounts_db:
            info = accounts_db[user_name]
            return f"用户 {user_name} 存在"
        else:
            return f"用户 {user_name} 不存在"

class GetBalanceTool(BaseTool):
    name: str = "get_balance"
    description: str = "获取指定用户余额"
    
    def _run(self, user_name: str) -> str:
        print(f"--toolcall--查询余额: {user_name}")
        if user_name in accounts_db:
            return f"用户 {user_name} 的余额：{accounts_db[user_name]['balance']}"
        else:
            return f"用户 {user_name} 不存在"

class ExecuteTransferTool(BaseTool):
    name: str = "execute_transfer"
    description: str = "执行转账"
    
    def _run(self, to_user: str, amount: float) -> str:
        print(f"--toolcall--执行转账: {to_user} {amount}")
        if to_user in accounts_db:
            accounts_db[to_user]['balance'] += amount
            return f"转账成功，转账金额：{amount}，转账后余额：{accounts_db[to_user]['balance']}"
        else:
            return f"转账失败，目标账户 {to_user} 不存在"

class ReplyToUserTool(BaseTool):
    name: str = "reply_to_user"
    description: str = "向用户提问获取信息"
    
    def _run(self, question: str) -> str:
        print(f"--toolcall--向用户提问: {question}")
        return f"用户即将回答"

def main():
    """主函数 - 创建并运行聊天 Agent"""
    #百炼模型
    llm = LLM(
        model="openai/"+os.getenv("QWEN3_MODEL"),
        api_key=os.getenv("BAILIAN_API_KEY"),
        base_url=os.getenv("BAILIAN_API_BASE_URL"),
        enable_thinking=False
    )
    #ollama模型
    # llm = LLM(
    #     model=os.getenv("OLLAMA_MODEL"),
    #     base_url=os.getenv("OLLAMA_API_BASE_URL"),
    # )
    # 直接创建 Agent（所有配置都在这里）
    
    transfer_agent = Agent(
        name="transfer_agent",
        role="银行转账助手",
        goal="""
        1. 分析用户的转账需求，并整理转账信息
        2. 收集必要的转账信息（目标账户、金额）
        3. 验证信息的有效性，包含目标账户是否存在，金额是否足够
        4. 当信息不足时，主动通过工具函数`reply_to_user`询问用户，直到信息收集完整。
        5. 当信息收集完整后，先总结信息向用户确认，再通过工具函数`execute_transfer`执行转账，并返回转账结果
        """,
        backstory="""
        你是一个专业的银行转账助手，负责处理用户的转账请求。
        注意事项：
        - 所有转账都从我的账户转出
        - 确认转账信息中金额，并确保余额足够`get_balance`
        - 目标账户必须存在`get_account`
        - 请与用户进行自然对话，逐步收集信息，确保转账安全准确
        - 确认信息后通过`execute_transfer`执行转账
        - 当转账成功后，请发送"DONE"
        """,
        verbose=True,
        # 是否允许 Agent 将任务委托给其他 Agent。设置为 False 表示不允许委托，所有任务都由当前 Agent 处理。
        allow_delegation=False,
        tools=[GetAccountTool(), GetBalanceTool(), ExecuteTransferTool(), ReplyToUserTool()],
        llm=llm
    )
    
    print("智能助手已启动！输入 'quit' 退出对话")
    print("=" * 50)
    
    while True:
        # 获取用户输入
        user_input = input("\n你: ").strip()
        if user_input.lower() in ['quit', 'exit', '退出']:
            break
        if not user_input:
            continue
        try:
            task = Task(
                description=f"""处理用户请求：{user_input}
                注意事项：
                - 所有转账都从我的账户转出
                - 确认转账信息中金额，并确保余额足够`get_balance`
                - 目标账户必须存在`get_account`
                - 请与用户进行自然对话，逐步收集信息，确保转账安全准确
                - 确认信息后通过`execute_transfer`执行转账
                - 当转账成功后，请发送"DONE"
                """,
                expected_output="对用户请求的友好回复",
                agent=transfer_agent
            )
            # 创建 Crew 并执行
            crew = Crew(
                agents=[transfer_agent],
                tasks=[task],
                process=Process.sequential,
                memory=True,
                verbose=True
            )
            result = crew.kickoff()
            print(f"\n助手: {result}")
            
        except Exception as e:
            print(f"错误: {str(e)}")

if __name__ == "__main__":
    main() 