from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import uuid
import os
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

# 工具1: 查询余额
class QueryBalanceInput(BaseModel):
    """查询余额工具的输入参数"""
    account_id: str = Field(..., description="账户ID，例如 'acc_001', 'my_account'")

class GetBalanceTool(BaseTool):
    name: str = "get_balance"
    description: str = "查询指定账户的余额信息"
    args_schema: Type[BaseModel] = QueryBalanceInput

    def _run(self, account_id: str) -> str:
        if account_id in accounts_db:
            account_info = accounts_db[account_id]
            return f"账户 {account_id} ({account_info['owner']}) 的余额为: {account_info['balance']} 元"
        else:
            return f"账户 {account_id} 不存在"

# 工具2: 查询账户信息
class QueryAccountInput(BaseModel):
    """查询账户信息的输入参数"""
    account_id: str = Field(..., description="账户ID，例如 'acc_001', 'my_account'")

class GetAccountInfoTool(BaseTool):
    name: str = "get_account_info"
    description: str = "查询指定账户的详细信息"
    args_schema: Type[BaseModel] = QueryAccountInput

    def _run(self, account_id: str) -> str:
        if account_id in accounts_db:
            account_info = accounts_db[account_id]
            return f"账户ID: {account_id}, 户主: {account_info['owner']}, 余额: {account_info['balance']} 元"
        else:
            return f"账户 {account_id} 不存在"

# 工具3: 执行转账
class ExecuteTransferInput(BaseModel):
    """执行转账的输入参数"""
    to_account: str = Field(..., description="目标账户ID")
    amount: float = Field(..., description="转账金额")

class ExecuteTransferTool(BaseTool):
    name: str = "execute_transfer"
    description: str = "执行转账操作，从我的账户转出资金"
    args_schema: Type[BaseModel] = ExecuteTransferInput

    def _run(self, to_account: str, amount: float) -> str:
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

# 工具4: 获取所有账户信息
class GetAllAccountsInput(BaseModel):
    """获取所有账户信息的输入参数"""
    dummy: str = Field(default="", description="占位参数")

class GetAllAccountsTool(BaseTool):
    name: str = "get_all_accounts"
    description: str = "获取所有可用账户的信息"
    args_schema: Type[BaseModel] = GetAllAccountsInput

    def _run(self, dummy: str = "") -> str:
        accounts_info = []
        for acc_id, acc_info in accounts_db.items():
            accounts_info.append(f"{acc_id} ({acc_info['owner']}): {acc_info['balance']} 元")
        return "可用账户信息:\n" + "\n".join(accounts_info)

# 工具5: 向用户提问
class AskUserInput(BaseModel):
    """向用户提问的输入参数"""
    question: str = Field(..., description="要向用户提出的问题")
    context: str = Field(default="", description="问题的上下文信息")

class AskUserTool(BaseTool):
    name: str = "ask_user"
    description: str = "当转账信息不足时，向用户提问获取更多信息"
    args_schema: Type[BaseModel] = AskUserInput

    def _run(self, question: str, context: str = "") -> str:
        """
        向用户提问并获取回答
        Args:
            question: 要向用户提出的问题
            context: 问题的上下文信息
        Returns:
            用户的回答
        """
        # 显示问题和上下文
        if context:
            print(f"\n[上下文] {context}")
        print(f"\n[问题] {question}")
        
        # 获取用户输入
        try:
            user_answer = input("请回答: ").strip()
            if not user_answer:
                # 如果用户没有输入，使用默认回答
                return self._get_default_answer(question)
            return user_answer
        except (KeyboardInterrupt, EOFError):
            # 如果用户中断输入，使用默认回答
            print("\n使用默认回答...")
            return self._get_default_answer(question)
    
    def _get_default_answer(self, question: str) -> str:
        """获取默认回答"""
        question_lower = question.lower()
        
        # 根据问题类型返回默认回答
        if "账户" in question_lower or "转给谁" in question_lower:
            if "张三" in question_lower or "李四" in question_lower or "王五" in question_lower or "赵六" in question_lower:
                return "好的，我确认转账给这个账户"
            else:
                return "我想转给张三"
        
        elif "金额" in question_lower or "多少钱" in question_lower or "转多少" in question_lower:
            if "1000" in question_lower or "1000元" in question_lower:
                return "1000元"
            elif "5000" in question_lower or "5000元" in question_lower:
                return "5000元"
            else:
                return "1000元"
        
        elif "余额" in question_lower or "不够" in question_lower:
            return "那我转少一点，500元吧"
        
        else:
            return "我确认这个转账请求" 