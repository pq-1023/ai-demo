from sentence_transformers import SentenceTransformer
import chromadb
import requests


# ====================== 简易 Ollama客户端 ======================
class ModelClient:
    """调用本地Ollama，不需要api‑key"""

    def chat(self, prompt: str):
        url = "http://127.0.0.1:11434/api/generate"
        payload = {
            "model": "qwen2.5:3b-instruct-q4_K_M",
            "prompt": prompt,
            "stream": False
        }
        resp = requests.post(url, json=payload)
        return resp.json()["response"]


# ====================== 1、初始化工具 ======================
#直接读取本地model文件夹
emb_model = SentenceTransformer("./model/all-MiniLM-L6-v2")
# 本地持久化向量数据库
chroma_client = chromadb.PersistentClient(path="./rag_db")
collection = chroma_client.get_or_create_collection(name="knowledge_base")
# 初始化大模型
llm = ModelClient()

# ====================== 2、测试知识库入库 ======================
docs = [
    "RAG全称检索增强生成，可以减少大模型幻觉。",
    "RAG基础流程：问题向量化、向量检索、拼接上下文、大模型生成。",
    "幻觉是大模型编造不存在事实的现象。"
]
ids = [f"doc_{i}" for i in range(len(docs))]
# 文档转为向量数字
embeddings = emb_model.encode(docs).tolist()
collection.add(documents=docs, ids=ids, embeddings=embeddings)


# ======================3、向量检索函数 ======================
def retrieve(query, top_k=2):
    # 用户问题转向量
    query_vector = emb_model.encode(query).tolist()
    # 相似度搜索
    search_result = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )
    # 把查到的几段文档拼成一个字符串
    context = "\n".join(search_result["documents"][0])
    return context


# ======================4、组装RAG专用提示词 ======================
def build_rag_prompt(context: str, question: str):
    prompt = f"""你是知识库问答助手，请严格根据下面【参考资料】里的内容回答用户问题。
如果参考资料中没有相关信息，不要编造答案，直接回复："知识库中暂未查询到相关信息"。
禁止使用知识库以外你自己的知识作答。

【参考资料】
{context}

【用户问题】
{question}
"""
    return prompt


# ======================5、完整RAG主流程 ======================
def rag_chat(question):
    context_data = retrieve(question)
    final_prompt = build_rag_prompt(context_data, question)
    answer = llm.chat(final_prompt)
    return answer


# ======================6、对比测试：直接提问 VS RAG问答 ======================
if __name__ == "__main__":
    user_q = "什么是RAG，可以解决什么问题？"
    print("====不带知识库 直接问大模型====")
    print(llm.chat(user_q))
    print("\n====开启RAG知识库问答====")
    print(rag_chat(user_q))

    # 测试知识库没有的问题，验证兜底话术
    print("\n====测试知识库不存在的问题：宇宙多大====")
    print(rag_chat("宇宙多大"))
