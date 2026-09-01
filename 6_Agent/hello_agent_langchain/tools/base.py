from langchain_core.tools import BaseTool
from pydantic import Field


class HelloBaseTool(BaseTool):
    name: str = Field(description="工具名称")
    description: str = Field(description="工具描述，提供给大模型选择工具")

    def _run(self, query: str) -> str:
        raise NotImplementedError

    async def _arun(self, query: str) -> str:
        raise NotImplementedError