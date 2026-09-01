from ..core.agent import BaseAgent
from ..core.message import Message


class PlanAndSolveAgent(BaseAgent):
    def run(self, query: str) -> str:
        # 第一步生成任务规划
        plan_msg = [Message.human(f"请把下面任务拆解成分步执行计划，只输出计划：{query}")]
        plan = self.llm.invoke(plan_msg).content

        # 第二步执行任务
        final_prompt = f"""任务:{query}
执行计划:
{plan}
按照计划完成任务，输出最终答案。
"""
        solve_msg = [Message.human(final_prompt)]
        result = self.llm.invoke(solve_msg).content
        return result