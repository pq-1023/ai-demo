import os
import logging
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from config import RERANK_MODEL, RERANK_TOP_N

logger = logging.getLogger(__name__)


def build_rerank_retriever(base_retriever):
    if not os.path.exists(RERANK_MODEL):
        raise FileNotFoundError(f"Rerank 模型路径不存在: {RERANK_MODEL}")

    logger.info(f"加载 Rerank 模型: {RERANK_MODEL}")
    try:
        model = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)
        compressor = CrossEncoderReranker(model=model, top_n=RERANK_TOP_N)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        logger.info("Rerank 重排序器构建完成")
        return retriever
    except Exception as e:
        logger.error(f"Rerank 模型加载失败: {e}")
        raise