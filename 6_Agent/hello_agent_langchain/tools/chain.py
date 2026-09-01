from typing import List
from langchain_core.tools import BaseTool


class ToolChain:
    def __init__(self, tools: List[BaseTool]):
        self.tools = tools

    def execute_sync(self, input_str: str) -> List[str]:
        results = []
        for tool in self.tools:
            res = tool.invoke(input_str)
            results.append(str(res))
        return results