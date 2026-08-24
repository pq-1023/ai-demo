import logging
from vector_db import load_vector_db
from config import TOP_K, ENABLE_RERANK

logger = logging.getLogger(__name__)


def get_retriever(filter_meta: dict = None):
    db = load_vector_db()
    search_kwargs = {"k": TOP_K}
    if filter_meta:
        search_kwargs["filter"] = filter_meta

    base_retriever = db.as_retriever(search_kwargs=search_kwargs)

    if not ENABLE_RERANK:
        logger.info("使用普通检索")
        return base_retriever

    try:
        from rerank_module import build_rerank_retriever
        retriever = build_rerank_retriever(base_retriever)
        logger.info("使用 Rerank 重排序检索")
        return retriever
    except Exception as e:
        logger.warning(f"Rerank 初始化失败，降级为普通检索: {e}")
        return base_retriever