from __future__ import annotations
from dataclasses import dataclass
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

# Graph简单示例
# 定义一个节点，判断数字是否能被5整除
@dataclass
class DivisibleBy5(BaseNode[None, None, int]):  
    foo: int
    async def run(
        self,
        ctx: GraphRunContext,
    ) -> Increment | End[int]:
        # 如果foo能被5整除，则结束流程并返回结果
        if self.foo % 5 == 0:
            return End(self.foo)
        else:
            # 否则进入Increment节点，foo加1
            return Increment(self.foo)

# 定义一个节点，对数字加1
@dataclass
class Increment(BaseNode):  
    foo: int
    async def run(self, ctx: GraphRunContext) -> DivisibleBy5:
        # 返回DivisibleBy5节点，foo加1
        return DivisibleBy5(self.foo + 1)

# 构建图，包含DivisibleBy5和Increment两个节点
fives_graph = Graph(nodes=[DivisibleBy5, Increment])  
# 从DivisibleBy5(4)开始运行图
result = fives_graph.run_sync(DivisibleBy5(9))  
# 输出最终结果
print(result.output)