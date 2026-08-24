import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBED_MODEL

logger = logging.getLogger(__name__)

_embedding_model = None


def get_embedding():
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"加载 Embedding 模型: {EMBED_MODEL}")
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    return _embedding_model