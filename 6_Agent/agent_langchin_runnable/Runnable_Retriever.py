from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List

# ===================== 1. 准备固定知识库素材 =====================
knowledge_texts = [
    Document(page_content="ThinkPad T480，内存16GB，固态硬盘512GB SSD，适配Java开发，价格区间2000~3000元"),
    Document(page_content="ThinkPad T480 适合后端开发、代码编译，扩展性良好，可更换内存与硬盘"),
    Document(page_content="2000-3000价位二手笔记本，T480是Java开发常用机型")
]

# ===================== 2. 初始化向量数据库(Chroma，仅存储向量) =====================
# Embedding模型：文本转向量
embedding = HuggingFaceEmbeddings(
    model_name="./model/all-MiniLM-L6-v2"
)
# 向量库：只负责向量存储、相似度匹配，不是Retriever
vector_db = Chroma.from_documents(
    documents=knowledge_texts,
    embedding=embedding,
    collection_name="laptop_kb"
)

# ===================== 3. 自定义Retriever【检索抽象层，本阶段核心】 =====================
# 继承BaseRetriever，实现统一检索接口，解耦向量库
class LaptopRetriever(BaseRetriever):
    vector_store: Chroma
    top_k: int = 2

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Retriever职责：调用向量库、控制返回条数、过滤文档，封装检索逻辑
        docs = self.vector_store.similarity_search(query, k=self.top_k)
        # 可在这里增加过滤逻辑（过滤低分、过滤无效文档），这就是Retriever的价值
        return docs

# 实例化检索器，传入向量库
retriever = LaptopRetriever(vector_store=vector_db, top_k=2)

# ===================== 4. 构建Prompt模板：拼接用户问题 + 检索到的上下文 =====================
prompt_template = ChatPromptTemplate.from_template("""
基于下面检索到的知识库内容回答用户问题，不要编造信息。
知识库内容：
{context}

用户问题：{question}
""")

# ===================== 5. 初始化LLM，组装完整RAG链路 =====================
llm = ChatOllama(
    model="qwen2.5:3b-instruct-q4_K_M",
    temperature=0
)
rag_chain = prompt_template | llm | StrOutputParser()

# ===================== 6. 主执行函数：完整RAG链路 =====================
def run_rag_demo(user_query: str):
    # Step1：Retriever检索相关文档（Retriever层，调用向量库）
    retrieved_docs = retriever.invoke(user_query)
    # 把检索结果拼接成上下文字符串
    context_text = "\n".join([doc.page_content for doc in retrieved_docs])
    # Step2：传入prompt，调用大模型生成答案
    answer = rag_chain.invoke({
        "context": context_text,
        "question": user_query
    })
    print("===== 检索到的知识库片段 =====")
    for idx, doc in enumerate(retrieved_docs):
        print(f"{idx+1}. {doc.page_content}")
    print("\n===== LLM回答 =====")
    print(answer)
    return answer

# ========== 测试样例 ==========
if __name__ == "__main__":
    # 测试问题1
    run_rag_demo("适合Java开发的二手笔记本，2千到3千有什么推荐？")
    print("-" * 60)
    # 测试问题2
    run_rag_demo("ThinkPad T480配置和价格是多少？")