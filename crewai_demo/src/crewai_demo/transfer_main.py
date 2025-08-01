#!/usr/bin/env python
import warnings
from datetime import datetime
from crewai_demo.transfer_crew import TransferCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run_transfer(user_request: str):
    """
    运行转账功能
    Args:
        user_request: 用户的转账请求，例如 "我想转1000元给张三"
    """
    inputs = {
        'user_request': user_request,
        'current_time': str(datetime.now())
    }
    
    try:
        print(f"=== 开始处理转账请求 ===")
        print(f"用户请求: {user_request}")
        print()
        
        # 运行转账Crew
        result = TransferCrew().crew().kickoff(inputs=inputs)
        
        print(f"=== 转账处理完成 ===")
        print(f"处理结果: {result}")
        
        return result
        
    except Exception as e:
        print(f"转账处理过程中出现错误: {e}")
        raise Exception(f"转账处理失败: {e}")

def demo_transfer():
    """
    演示转账功能
    """
    print("=== CrewAI 转账系统演示 ===\n")
    
    # 测试用例1: 正常转账
    print("测试1: 正常转账")
    result1 = run_transfer("我想转1000元给张三")
    print(f"结果1: {result1}\n")
    
    # 测试用例2: 大额转账
    print("测试2: 大额转账")
    result2 = run_transfer("转5000元给李四")
    print(f"结果2: {result2}\n")
    
    # 测试用例3: 模糊请求
    print("测试3: 模糊请求")
    result3 = run_transfer("帮我转点钱给王五")
    print(f"结果3: {result3}\n")
    
    # 测试用例4: 余额不足
    print("测试4: 余额不足")
    result4 = run_transfer("转20000元给赵六")
    print(f"结果4: {result4}\n")

if __name__ == "__main__":
    # 运行演示
    demo_transfer() 