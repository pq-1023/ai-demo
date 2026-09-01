from .core.llm import HelloAgentsLLM
from .agents.react_agent import ReActAgent
from .tools.registry import ToolRegistry

if __name__ == "__main__":
    llm = HelloAgentsLLM()
    tools = ToolRegistry.get_all_tools()
    agent = ReActAgent(llm=llm, tools=tools)
    output = agent.run("计算 256 * 789 的结果")
    print("\n====最终结果====")
    print(output)