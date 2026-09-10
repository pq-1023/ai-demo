from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. 定义State
class State(TypedDict):
    question: str
    intent: str
    documents: list[str]
    answer: str


# 2. Node：意图节点
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

# Node：伪RAG检索
def rag_node(state: State):
    question = state["question"]
    documents = []
    keywords = ["图书馆", "食堂", "体育馆"]
    for doc in knowledge_base:
        for kw in keywords:
            if kw in question and kw in doc:
                documents.append(doc)
                break
    print(f"【rag_node】检索得到文档: {documents}")
    return {"documents": documents}


# Node：生成答案
def answer_node(state: State):
    documents = state["documents"]
    if documents:
        answer = documents[0]
    else:
        answer = "暂时没有找到相关信息"
    print(f"【answer_node】生成回答：{answer}")
    return {"answer": answer}


# 3. 路由函数（条件边用）
def route_intent(state: State):
    if state["intent"] == "rag":
        return "rag"
    return "answer"


# ========== 组装Graph【重点：这里是改动的地方】 ==========
builder = StateGraph(State)

# 注册节点
builder.add_node("intent", intent_node)
builder.add_node("rag", rag_node)
builder.add_node("answer", answer_node)

# 起点固定走到intent
builder.add_edge(START, "intent")

# ===== 核心：intent节点执行完，走条件分支 =====
builder.add_conditional_edges(
    source="intent",
    path=route_intent,
    path_map={
        "rag": "rag",
        "answer": "answer"
    }
)

# RAG跑完之后，固定走向 answer
builder.add_edge("rag", "answer")

# answer执行完成，结束流程
builder.add_edge("answer", END)

# 编译图
graph = builder.compile()

# 测试入口
def run_graph(question: str):
    print("-" * 60)
    print(f"用户提问：{question}")
    result = graph.invoke({
        "question": question,
        "intent": "",
        "documents": [],
        "answer": ""
    })
    print(f"✅ 最终State：{result}\n")
    return result


if __name__ == "__main__":
    run_graph("学校图书馆几点关门？")  # 走分支：intent -> rag -> answer
    run_graph("你好呀")               # 走分支：intent -> answer，跳过rag
    run_graph("学校体育馆几点关门？")  # 走分支：intent -> rag -> answer