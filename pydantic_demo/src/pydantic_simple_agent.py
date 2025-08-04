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

roulette_agent = Agent(  
    # 'dashscope:qwen3-30b-a3b',
    model,
    deps_type=int,
    output_type=bool,
    system_prompt=(
        'Use the `roulette_wheel` function to see if the '
        'customer has won based on the number they provide.'
    ),
)

@roulette_agent.tool
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:  
    """check if the square is a winner"""
    return 'winner' if square == ctx.deps else 'loser'

# Run the agent
success_number = 18  
result = roulette_agent.run_sync('Put my money on square eighteen', deps=success_number)
print(result.output)  
#> True

# 解释：run_sync方法是Agent类中的一个同步运行方法，用于以阻塞方式执行智能体的推理流程。
# 它会根据传入的输入（如用户的自然语言指令）和依赖参数（如deps=success_number），
# 调用模型进行推理，并返回一个包含推理结果的对象（如result.output）。
# 适用于需要同步获取结果的场景。
result = roulette_agent.run_sync('I bet five is the winner', deps=success_number)
print(result.output)
#> False