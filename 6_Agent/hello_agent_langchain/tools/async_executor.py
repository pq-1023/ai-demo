import asyncio
from typing import List
from langchain_core.tools import BaseTool


async def async_execute_tools(tools: List[BaseTool], query: str):
    tasks = [tool.ainvoke(query) for tool in tools]
    return await asyncio.gather(*tasks)