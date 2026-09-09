import time
import os
from typing import List, Optional
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.agents import create_agent
from pydantic import BaseModel, Field

# ============================================================
# 一、复用阶段2：Retriever 知识检索（非结构化文档）
# ============================================================

knowledge_texts = [
    Document(page_content="ThinkPad T480，内存16GB，固态硬盘512GB SSD，适配Java开发，价格区间2000~3000元"),
    Document(page_content="ThinkPad T480 适合后端开发、代码编译，扩展性良好，可更换内存与硬盘，键盘手感优秀，商务本稳定性高"),
    Document(page_content="Dell Latitude 5420，16GB内存，512GB SSD，适合后端开发通用场景，轻薄便携，续航较好"),
    Document(page_content="Java开发推荐16GB以上内存，因为IDE（IDEA）+ 虚拟机 + 浏览器同时占用内存较大")
]

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
embedding = HuggingFaceEmbeddings(
    model_name=os.path.join(_BASE_DIR, "model", "all-MiniLM-L6-v2")
)
vector_db = Chroma.from_documents(
    documents=knowledge_texts,
    embedding=embedding,
    collection_name="laptop_kb_v2"
)

class LaptopRetriever(BaseRetriever):
    vector_store: Chroma
    top_k: int = 2

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.vector_store.similarity_search(query, k=self.top_k)

retriever = LaptopRetriever(vector_store=vector_db, top_k=2)

# 将 Retriever 包装成 Tool，Agent 才能调用它
@tool
def search_knowledge(query: str) -> str:
    """
    笔记本知识库检索工具。
    用于查询设备适配场景、性能优势、开发环境建议、专业知识等说明性内容。
    :param query: 知识查询问题
    :return: 检索到的相关知识库文本
    """
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])


# ============================================================
# 二、复用阶段3：search_product Tool（结构化业务数据）
# ============================================================

@tool
def search_product(keyword: str, max_price: int) -> List[dict]:
    """
    二手笔记本商品查询工具。
    用于查询具体商品型号、价格、配置参数等结构化业务数据。
    :param keyword: 商品关键词/使用场景，例如 Java开发、办公
    :param max_price: 最大预算价格
    :return: 符合条件的商品列表（结构化字典数组）
    """
    products = [
        {"name": "ThinkPad T480", "price": 2800, "ram": "16GB", "ssd": "512GB", "usage": "Java开发适配"},
        {"name": "Dell Latitude 5420", "price": 2900, "ram": "16GB", "ssd": "512GB", "usage": "后端开发通用"}
    ]
    return [p for p in products if p["price"] <= max_price and keyword in p["usage"]]


# ============================================================
# 三、定义结构化输出（Pydantic 模型）
# ============================================================

class LaptopAnswer(BaseModel):
    """笔记本咨询的结构化回答"""
    intent: str = Field(description="用户意图分类：商品查询 / 知识咨询 / 综合咨询")
    recommendation: str = Field(description="给用户的核心推荐结论")
    product_list: Optional[List[dict]] = Field(default=None, description="推荐的商品列表，无则为null")
    knowledge_summary: Optional[str] = Field(default=None, description="检索到的知识要点总结，无则为null")
    reason: str = Field(description="做出该推荐的理由")


# ============================================================
# 四、组装 Agent：自主决策 + 双工具
# ============================================================

# Agent 系统提示词：明确告诉模型什么时候用哪个工具
SYSTEM_PROMPT = """你是二手笔记本智能导购助手。你拥有两个工具：
1. search_product：查询具体商品型号、价格、配置参数
2. search_knowledge：查询设备适配场景、性能优势、开发建议等知识说明

严格遵守以下规则：
- 最多只调用工具1~2次，拿到结果后必须立即停止，直接整理最终回答
- 绝对不要在拿到工具结果后再次调用同一个工具
- 如果已经有足够信息，直接输出答案，不要调用任何工具
- 最终输出符合 LaptopAnswer 结构的 JSON"""

llm = ChatOllama(model="qwen2.5:3b-instruct-q4_K_M", temperature=0)

tools = [search_product, search_knowledge]

agent = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)
agent = agent.with_config({"recursion_limit": 15})


def run_agent_demo(user_query: str):
    print(f"\n{'='*60}")
    print(f"用户问题：{user_query}")
    print(f"{'='*60}")

    result = agent.invoke({"messages": [("user", user_query)]})
    raw_answer = result["messages"][-1].content

    structured_llm = llm.with_structured_output(LaptopAnswer)
    final_json = structured_llm.invoke(
        f"基于以下回答，输出结构化JSON：\n{raw_answer}\n\n用户原问题：{user_query}"
    )

    print("\n===== 最终结构化 JSON 结果 =====")
    print(final_json.model_dump_json(indent=2, ensure_ascii=False))
    return final_json


# ========== 测试样例：验证 Agent 自主决策 ==========
if __name__ == "__main__":
    # 测试1：纯商品查询 → Agent 应自主调用 search_product
    run_agent_demo("3000以内适合Java开发的笔记本有哪些？多少钱？")
    time.sleep(2)
    # 测试2：纯知识咨询 → Agent 应自主调用 search_knowledge
    run_agent_demo("Java开发为什么需要16GB内存？ThinkPad T480有什么优势？")
    time.sleep(2)
    # 测试3：综合咨询 → Agent 应两个工具都调用
    run_agent_demo("我预算3000做Java开发，推荐哪款？说说它为什么适合。")