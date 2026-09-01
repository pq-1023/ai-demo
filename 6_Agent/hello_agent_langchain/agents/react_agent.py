from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from ..core.agent import BaseAgent


REACT_PROMPT_TPL = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""


class ReActAgent(BaseAgent):
    def run(self, query: str) -> str:
        prompt = PromptTemplate.from_template(REACT_PROMPT_TPL)
        react_agent = create_react_agent(self.llm, self.tools, prompt)
        agent_executor = AgentExecutor(agent=react_agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
        result = agent_executor.invoke({"input": query})
        return result["output"]