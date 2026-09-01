from langchain_community.tools import CalculatorTool
from ..registry import ToolRegistry

CalculatorToolInstance = CalculatorTool()
ToolRegistry.register(CalculatorToolInstance.__class__)