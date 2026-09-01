from langchain_community.tools.tavily_search import TavilySearchResults
from ..registry import ToolRegistry

SearchTool = TavilySearchResults(max_results=3)
ToolRegistry.register(SearchTool.__class__)