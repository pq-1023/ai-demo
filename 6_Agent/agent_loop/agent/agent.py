from agent.state import AgentState
from agent.decision import make_decision
from agent.exceptions import AgentMaxIterationsError
from tools.product import search_product
from tools.calculator import calculator
from tools.search import search_knowledge

MAX_ITERATIONS = 5

TOOL_MAP = {
    "search_product": search_product,
    "calculator": calculator,
    "search_knowledge": search_knowledge
}

class CampusAgent:
    def call_tool(self, tool_name: str, tool_input):
        tool_func = TOOL_MAP[tool_name]
        return tool_func(tool_input)

    def run(self, user_question: str):
        state = AgentState(
            question=user_question,
            messages=[],
            current_step=0,
            observations=[],
            final_answer=None
        )

        for _ in range(MAX_ITERATIONS):
            state.current_step += 1
            print(f"\n===== Step {state.current_step} =====")

            decision = make_decision(state)
            print(f"Decision: {decision}")

            if decision["type"] == "tool":
                tool_name = decision["tool"]
                tool_input = decision["input"]
                try:
                    obs = self.call_tool(tool_name, tool_input)
                    print(f"Observation: {obs}")
                    state.observations.append(obs)
                except Exception as e:
                    print(f"Tool Error: {e}")
                    state.observations.append(f"工具执行失败：{str(e)}")
                continue

            elif decision["type"] == "final":
                state.final_answer = decision["content"]
                return state.final_answer

        raise AgentMaxIterationsError("Agent达到最大迭代次数，终止任务")