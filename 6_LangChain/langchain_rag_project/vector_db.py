import os
import logging
from langchain_chroma import Chroma
from langchain_core.documents import Document
from embedding_provider import get_embedding
from config import CHROMA_PATH, COLLECTION_NAME

logger = logging.getLogger(__name__)


def create_vector_db(documents):
    embedding = get_embedding()
    os.makedirs(CHROMA_PATH, exist_ok=True)
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME
    )
    logger.info(f"向量库已创建: {CHROMA_PATH}, 共 {len(documents)} 条")
    return db


def append_to_vector_db(documents):
    if not os.path.exists(CHROMA_PATH):
        logger.error("向量库不存在，无法追加，请先初始化")
        return 0
    embedding = get_embedding()
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding,
        collection_name=COLLECTION_NAME
    )
    db.add_documents(documents)
    count = db._collection.count()
    logger.info(f"已追加 {len(documents)} 条, 向量库现有 {count} 条")
    return count


def load_vector_db():
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(f"向量库路径不存在: {CHROMA_PATH}，请先运行 ingest.py")
    embedding = get_embedding()
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding,
        collection_name=COLLECTION_NAME
    )
    count = db._collection.count()
    logger.info(f"向量库已加载: {CHROMA_PATH}, 共 {count} 条")
    return db