from dotenv import load_dotenv
# 加载 .env 文件中的环境变量
load_dotenv()

import os
import re
from tavily import TavilyClient
from typing import Dict, Any

def clean_text(text: str, max_len: int = 450) -> str:
    """清洗搜索结果文本，去除多余换行、表格标记、截断超长内容"""
    # 移除连续换行
    text = re.sub(r'\n+', '\n', text)
    # 剔除markdown表格分隔线
    text = re.sub(r'(\|.*\|\n)+', '', text)
    # 截断
    if len(text) > max_len:
        text = text[:max_len] + " ...(内容已截断)"
    return text


def search(query: str) -> str:
    """
    一个基于Tavily的实战网页搜索引擎工具。
    Tavily专为AI Agent优化，返回干净、适合大模型阅读的搜索结果。
    """
    print(f"🔍 正在执行 [Tavily] 网页搜索: {query}")
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "错误：TAVILY_API_KEY 未在 .env 文件中配置。"

        tavily_client = TavilyClient(api_key=api_key)

        results = tavily_client.search(
            query=query,
            max_results=3,
            search_depth="basic",
            include_answer=True   # 开启 Tavily AI 预生成答案！
        )

        # 如果 Tavily 已经给出总结答案，优先返回（最适合Agent）
        if results.get("answer"):
            return f"搜索总结：{results['answer']}"

        result_list = results.get("results", [])
        if not result_list:
            return f"对不起，没有找到关于 '{query}' 的信息。"

        snippets = []
        for i, res in enumerate(result_list):
            # 过滤相关性很低的结果
            score = res.get("score", 0.0)
            if score < 0.3:
                continue
            title = res.get('title', '')
            raw_content = res.get('content', '')
            clean_content = clean_text(raw_content)
            snippets.append(f"[{i+1}] {title}\n{clean_content}")

        if not snippets:
            return f"对不起，没有找到关于 '{query}' 的高相关信息。"

        return "\n\n".join(snippets)

    except Exception as e:
        return f"搜索时发生错误: {e}"


class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            print(f"警告：工具 '{name}' 已存在，将被覆盖。")

        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> callable:
        """
        根据名称获取一个工具的执行函数。
        """
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])


# --- 工具初始化与使用示例 ---
if __name__ == '__main__':
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册我们的实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)

    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 4. 智能体的Action调用
    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误：未找到名为 '{tool_name}' 的工具。")