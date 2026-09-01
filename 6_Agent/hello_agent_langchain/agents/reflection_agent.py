from ..core.agent import BaseAgent
from ..core.message import Message


class ReflectionAgent(BaseAgent):
    def run(self, query: str) -> str:
        # 第一轮回答
        msg1 = [Message.human(query)]
        ans1 = self.llm.invoke(msg1).content

        # 反思优化
        reflect_prompt = f"""原始问题:{query}
初次回答:{ans1}
请检查回答有没有错误，遗漏，重新给出一份更完善的答案。
"""
        msg2 = [Message.human(reflect_prompt)]
        final_ans = self.llm.invoke(msg2).content
        return final_ans