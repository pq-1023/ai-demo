from typing import Dict, Type, List
from .base import HelloBaseTool
from ..core.exceptions import ToolNotFoundError


class ToolRegistry:
    _tools: Dict[str, Type[HelloBaseTool]] = {}

    @classmethod
    def register(cls, tool_cls: Type[HelloBaseTool]):
        cls._tools[tool_cls.name] = tool_cls
        return tool_cls

    @classmethod
    def get_tool(cls, name: str) -> Type[HelloBaseTool]:
        if name not in cls._tools:
            raise ToolNotFoundError(f"工具 {name} 不存在")
        return cls._tools[name]

    @classmethod
    def get_all_tools(cls) -> List[HelloBaseTool]:
        instances = []
        for t_cls in cls._tools.values():
            instances.append(t_cls())
        return instances