from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai import LLM
import os
from dotenv import load_dotenv

# 导入转账工具
from crewai_demo.tools.transfer_tools import (
    GetBalanceTool, 
    GetAccountInfoTool, 
    ExecuteTransferTool, 
    GetAllAccountsTool,
    AskUserTool
)

# 加载环境变量
load_dotenv()

# 配置百炼模型
model_name = os.getenv("QWEN3_MODEL")
api_key = os.getenv("BAILIAN_API_KEY")
base_url = os.getenv("BAILIAN_API_BASE_URL")

llm = LLM(
    model="openai/"+model_name,
    api_key=api_key,
    base_url=base_url,
    enable_thinking=False
)

@CrewBase
class TransferCrew():
    """转账Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def transfer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['transfer_agent'],
            verbose=True,
            llm=llm,
            tools=[
                GetBalanceTool(),
                GetAccountInfoTool(),
                ExecuteTransferTool(),
                GetAllAccountsTool(),
                AskUserTool()
            ]
        )

    @task
    def transfer_task(self) -> Task:
        return Task(
            config=self.tasks_config['transfer_task'],
            output_file='crewai_demo/output/transfer_report.md'
        )

    @crew
    def crew(self) -> Crew:
        """创建转账Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        ) 