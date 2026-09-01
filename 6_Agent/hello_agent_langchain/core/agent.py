from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.tools import BaseTool
from .llm import HelloAgentsLLM


class BaseAgent(ABC):
    def __init__(self, llm: Optional[HelloAgentsLLM] = None, tools: Optional[List[BaseTool]] = None):
        self.llm_wrapper = llm or HelloAgentsLLM()
        self.llm = self.llm_wrapper.get_llm()
        self.tools = tools or []

    @abstractmethod
    def run(self, query: str) -> str:
        """同步运行agent"""
        pass

    async def arun(self, query: str) -> str:
        """异步运行agent"""
        raise NotImplementedError("异步运行未实现")