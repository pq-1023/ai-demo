from ..core.agent import BaseAgent
from ..core.message import Message


class SimpleAgent(BaseAgent):
    def run(self, query: str) -> str:
        messages = [
            Message.system("你是一个乐于助人的AI助手"),
            Message.human(query)
        ]
        resp = self.llm.invoke(messages)
        return resp.content