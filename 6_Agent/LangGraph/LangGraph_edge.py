from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# ========== Step1 定义State ==========
class State(TypedDict):
    question: str
    intent: str
    documents: list[str]
    answer: str


# ========== Step2 定义节点 Node ==========
# 意图节点
def intent_node(state: State):
    question = state["question"]
    if "学校" in question or "图书馆" in question or "食堂" in question or "体育馆" in question:
        intent = "rag"
    else:
        intent = "chat"
    print(f"【intent_node】判断意图: {intent}")
    return {"intent": intent}


# 伪知识库
knowledge_base = [
    "学校图书馆每天晚上10点闭馆",
    "学校食堂早餐开放时间为7点到9点",
    "学校体育馆晚上9点关闭"
]

# RAG检索节点
def rag_node(state: State):
    question = state["question"]
    documents = []
    # 修复原来的bug：提取关键词循环匹配
    keywords = ["图书馆", "食堂", "体育馆"]
    for doc in knowledge_base:
        for kw in keywords:
            if kw in question and kw in doc:
                documents.append(doc)
                break
    print(f"【rag_node】检索得到文档: {documents}")
    return {"documents": documents}


# 答案生成节点
def answer_node(state: State):
    documents = state["documents"]
    question = state["question"]
    if documents:
        answer = documents[0]
    else:
        answer = "暂时没有找到相关信息"
    print(f"【answer_node】生成回答：{answer}")
    return {"answer": answer}


# ========== Step3 条件路由函数 ==========
def route_intent(state: State):
    if state["intent"] == "rag":
        return "rag"
    return "answer"


# ========== Step4 组装Graph ==========
builder = StateGraph(State)

# 注册节点
builder.add_node("intent", intent_node)
builder.add_node("rag", rag_node)
builder.add_node("answer", answer_node)

# 连线
builder.add_edge(START, "intent")
builder.add_conditional_edges(
    "intent",
    route_intent,
    {
        "rag": "rag",
        "answer": "answer"
    }
)
builder.add_edge("rag", "answer")
builder.add_edge("answer", END)

# 编译图
graph = builder.compile()


# ========== Step5 测试函数，方便批量测试 ==========
def run_graph(question: str):
    print("-" * 50)
    print(f"用户提问：{question}")
    result = graph.invoke({
        "question": question,
        "intent": "",
        "documents": [],
        "answer": ""
    })
    print(f"✅ 最终完整State：{result}\n")
    return result


# 批量测试3种场景
if __name__ == "__main__":
    # 场景1：知识库问题，走RAG分支
    run_graph("学校图书馆几点关门？")
    # 场景2：闲聊，直接到answer，跳过RAG
    run_graph("你好呀！")
    # 场景3：知识库问题（食堂）
    run_graph("学校食堂早餐几点开始？")