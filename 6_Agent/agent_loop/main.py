from agent.agent import CampusAgent
from agent.exceptions import AgentMaxIterationsError

if __name__ == "__main__":
    agent = CampusAgent()
    user_query = "帮我看看校园里有没有500元以内的二手显示器，如果有，告诉我平均价格。"
    print("User:", user_query)
    try:
        ans = agent.run(user_query)
        print("\n✅ Final Answer：", ans)
    except AgentMaxIterationsError as e:
        print("\n❌", e)