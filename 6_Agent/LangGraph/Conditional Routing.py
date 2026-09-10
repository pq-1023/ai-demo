from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# ========== 1. 定义State ==========
class State(TypedDict):
    question: str
    route_type: str
    docs: list[str]
    search_result: str
    answer: str


# ========== 2. 定义所有Node ==========

# 路由节点（Router）：根据用户问题分类，决定走哪个分支
def router_node(state: State):
    q = state["question"]
    if any(k in q for k in ["图书馆", "食堂", "体育馆"]):
        route_type = "rag"
    elif any(k in q for k in ["天气", "新闻", "股票"]):
        route_type = "search"
    else:
        route_type = "chat"
    print(f"【路由节点】问题: {q} -> 判定路由: {route_type}")
    return {"route_type": route_type}


local_kb = [
    "学校图书馆每天晚上10点闭馆",
    "学校食堂早餐7点-9点开放",
    "学校体育馆晚上9点关闭"
]
def rag_node(state: State):
    q = state["question"]
    res = []
    for doc in local_kb:
        if any(k in q and k in doc for k in ["图书馆", "食堂", "体育馆"]):
            res.append(doc)
    print(f"【RAG节点】读取本地知识库: {res}")
    return {"docs": res}


def search_node(state: State):
    q = state["question"]
    fake_search = f"[联网搜索结果] 关于`{q}`的最新外网信息"
    print(f"【Search节点】执行联网搜索: {fake_search}")
    return {"search_result": fake_search}


def chat_node(state: State):
    print(f"【Chat节点】闲聊模式，无需工具调用")
    return {}


def answer_node(state: State):
    rt = state["route_type"]
    q = state["question"]
    if rt == "rag":
        docs = state["docs"]
        ans = f"关于「{q}」，本地知识库查到：{docs[0] if docs else '暂无相关信息'}"
    elif rt == "search":
        info = state["search_result"]
        ans = f"关于「{q}」，{info}"
    else:
        ans = f"闲聊回复：你问的「{q}」，很高兴和你聊天~"
    print(f"【Answer节点】汇总生成最终回答: {ans}")
    return {"answer": ans}


# ========== 3. 条件路由函数 ==========
def route_selector(state: State):
    if state["route_type"] == "rag":
        return "rag"
    elif state["route_type"] == "search":
        return "search"
    else:
        return "chat"


# ========== 4. 组装Graph ==========
builder = StateGraph(State)

builder.add_node("router", router_node)
builder.add_node("rag", rag_node)
builder.add_node("search", search_node)
builder.add_node("chat", chat_node)
builder.add_node("answer", answer_node)

builder.add_edge(START, "router")

builder.add_conditional_edges(
    source="router",
    path=route_selector,
    path_map={
        "rag": "rag",
        "search": "search",
        "chat": "chat"
    }
)

builder.add_edge("rag", "answer")
builder.add_edge("search", "answer")
builder.add_edge("chat", "answer")
builder.add_edge("answer", END)

graph = builder.compile()


# ========== 测试入口 ==========
def run(question: str):
    print("-" * 70)
    print(f"用户提问：{question}")
    result = graph.invoke({
        "question": question,
        "route_type": "",
        "docs": [],
        "search_result": "",
        "answer": ""
    })
    print(f"最终State：{result}\n")
    return result


if __name__ == "__main__":
    run("学校图书馆几点关门？")
    run("今天成都天气怎么样？")
    run("哈喽，你在吗？")