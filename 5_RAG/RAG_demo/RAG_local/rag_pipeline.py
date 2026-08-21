from dotenv import load_dotenv
import os
import logging
from config import ENABLE_QUERY_REWRITE

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

TOP_K = int(os.getenv("TOP_K"))
SIMILAR_THRESHOLD = float(os.getenv("SIMILAR_THRESHOLD"))
ENABLE_RERANK = os.getenv("ENABLE_RERANK") == "True"
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "20"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None and ENABLE_RERANK:
        try:
            from rerank_provider import RerankProvider
            model_path = os.getenv("RERANK_MODEL_PATH", "../model/bge-reranker-base")
            hf_model_name = os.getenv("RERANK_HF_MODEL_NAME", "BAAI/bge-reranker-base")
            _reranker = RerankProvider(model_path=model_path, hf_model_name=hf_model_name)
            if _reranker.is_available():
                logging.info(f"Rerank 模型加载成功")
            else:
                logging.warning("Rerank 模型不可用，将跳过重排序")
                _reranker = None
        except Exception as e:
            logging.warning(f"Rerank 初始化失败: {e}")
            _reranker = None
    return _reranker


def retrieval_pipeline(db, query: str, history=None) -> list[str]:
    search_query = query
    if history and ENABLE_QUERY_REWRITE:
        search_query = _rewrite_query_with_history(query, history)
        if search_query != query:
            logging.info(f"问题改写: [{query}] → [{search_query}]")

    fetch_k = RERANK_TOP_K if ENABLE_RERANK and get_reranker() else TOP_K
    raw_result = db.search(search_query, fetch_k)

    logging.info("========检索片段========")
    convert_result = []
    for idx, (doc, sim_score) in enumerate(raw_result):
        convert_result.append((doc, sim_score))
        logging.info(f"片段{idx + 1} 相似度:{round(sim_score, 3)} \n{doc}\n")

    if ENABLE_RERANK:
        reranker = get_reranker()
        if reranker and reranker.is_available():
            logging.info(f"Rerank 重排序: {len(convert_result)} 候选 → Top {RERANK_TOP_N}")
            docs_only = [doc for doc, _ in convert_result]
            reranked = reranker.rerank(search_query, docs_only, top_n=RERANK_TOP_N)
            convert_result = reranked
            for idx, (doc, score) in enumerate(convert_result):
                logging.info(f"  Rerank片段{idx + 1} 精排分:{round(score, 3)} \n{doc[:200]}...\n")

    filter_res = [(doc, score) for doc, score in convert_result if score >= SIMILAR_THRESHOLD]

    if ENABLE_RERANK and len(filter_res) > 0:
        filter_res.sort(key=lambda x: x[1], reverse=True)

    return [item[0] for item in filter_res]


def _rewrite_query_with_history(query, history):
    history_text = ""
    recent = history[-6:] if len(history) > 6 else history
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            history_text += f"用户: {content}\n"
        elif role == "assistant":
            history_text += f"助手: {content}\n"
        elif role == "system":
            history_text += f"【历史摘要】{content}\n"
    if not history_text.strip():
        return query
    rewrite_prompt = (
        f"根据以下对话历史，将用户最后一个问题改写为独立的、完整的、适合搜索的中文问题。\n"
        f"如果最后一个问题已经是完整的独立问题，直接返回原问题。\n\n"
        f"对话历史:\n{history_text}\n"
        f"用户最后问题: {query}\n\n"
        f"改写后的问题:"
    )
    try:
        from llm_provider import llm_chat_sync
        result = llm_chat_sync(rewrite_prompt)
        return result.strip() if result.strip() else query
    except Exception as e:
        logging.warning(f"问题改写失败，使用原问题: {e}")
        return query


def build_rag_prompt(context_list, question, history=None):
    history_text = ""
    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                history_text += f"用户: {content}\n"
            elif role == "assistant":
                history_text += f"助手: {content}\n"
            elif role == "system":
                history_text += f"【历史摘要】{content}\n"
    if not context_list:
        context = "无匹配知识库内容"
    else:
        context = "\n\n====文档片段====\n".join(context_list)

    if history_text.strip():
        prompt = f"""你是知识库问答助手，请严格根据下面【参考资料】中的内容回答用户问题。
仔细阅读参考资料，如果里面**完全没有可以回答该问题的信息**，不要自行编造，直接回复"知识库中暂未查询到相关信息"。
禁止使用知识库以外的知识作答，答案尽量简明。

【对话历史】
{history_text}
【参考资料】
{context}
【用户新问题】
{question}
"""
    else:
        prompt = f"""你是知识库问答助手，请严格根据下面【参考资料】中的内容回答用户问题。
仔细阅读参考资料，如果里面**完全没有可以回答该问题的信息**，不要自行编造，直接回复"知识库中暂未查询到相关信息"。
禁止使用知识库以外的知识作答，答案尽量简明。

【参考资料】
{context}
【用户问题】
{question}
"""
    return prompt