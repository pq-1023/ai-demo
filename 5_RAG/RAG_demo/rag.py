from sentence_transformers import SentenceTransformer
import chromadb
import requests
from FlagEmbedding import FlagReranker  # 新增重排库


# ====================== 简易 Ollama客户端 ======================
class ModelClient:
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
# 初始化重排序模型 bge-reranker-base
reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
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


# ======================3、【原版检索-保留不动】纯向量检索 ======================
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


# ======================3‑NEW 带Rerank重排序检索函数 ======================
def retrieve_with_rerank(query, top_k=5, final_top=3, enable_rerank=True):
    """
    :param query: 用户问题
    :param top_k: 向量粗召回数量（先捞5条候选文档）
    :param final_top: 最终送给LLM的文档条数
    :param enable_rerank: 是否开启重排序开关
    :return:拼接好的上下文字符串
    """
    # 第一步：向量粗召回
    query_vector = emb_model.encode(query).tolist()
    search_result = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )
    doc_list = search_result["documents"][0]

    # 如果关闭重排序，直接截取前final_top条返回
    if not enable_rerank:
        return "\n---\n".join(doc_list[:final_top])

    # 开启Rerank：问题和每一段文档打分
    score_doc_pairs = []
    for doc in doc_list:
        score = reranker.compute_score([query, doc])
        score_doc_pairs.append((score, doc))

    # 分数降序，相关性越高分数越大
    score_doc_pairs.sort(reverse=True, key=lambda x: x[0])
    best_docs = [item[1] for item in score_doc_pairs[:final_top]]
    return "\n---\n".join(best_docs)


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


# ======================5、新版RAG对话函数（支持rerank开关） ======================
def rag_chat_v2(question, enable_rerank=True):
    context_data = retrieve_with_rerank(question, enable_rerank=enable_rerank)
    final_prompt = build_rag_prompt(context_data, question)
    answer = llm.chat(final_prompt)
    return answer


# ======================6、测试入口 ======================
if __name__ == "__main__":
    user_q = "什么是幻觉？"
    print("====不带知识库 直接问大模型====")
    print(llm.chat(user_q))

    print("\n====关闭Rerank重排序====")
    print(rag_chat_v2(user_q, enable_rerank=False))

    print("\n====开启Rerank重排序====")
    print(rag_chat_v2(user_q, enable_rerank=True))

    # 测试知识库没有的问题，验证兜底话术
    print("\n====测试知识库不存在的问题：宇宙多大====")
    print(rag_chat_v2("宇宙多大"))