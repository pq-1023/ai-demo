import sys
import os
import logging
from langchain_core.documents import Document
from document_loader import load_document
from text_splitter import split_documents
from vector_db import create_vector_db, append_to_vector_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

for _logger in ["httpx", "http", "langchain_ollama", "urllib3", "requests"]:
    logging.getLogger(_logger).setLevel(logging.WARNING)


def ingest_file(file_path: str, clear: bool = False):
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return

    logger.info(f"读取文档: {file_path}")
    docs = load_document(file_path)

    logger.info("文本分片...")
    chunks = split_documents(docs)

    if clear:
        logger.info("清空已有向量库...")
        import shutil
        shutil.rmtree("./chroma_db", ignore_errors=True)

    logger.info("写入向量库...")
    create_vector_db(chunks)
    logger.info("文档入库完成")


def ingest_text(text: str, source: str = "手动输入"):
    if not text.strip():
        logger.warning("输入文本为空")
        return 0

    doc = Document(page_content=text.strip(), metadata={"source": source})
    logger.info(f"文本分片... (来源: {source})")
    chunks = split_documents([doc])

    if os.path.exists("./chroma_db"):
        count = append_to_vector_db(chunks)
    else:
        logger.info("写入向量库...")
        create_vector_db(chunks)
        count = len(chunks)

    logger.info(f"手动入库完成, 本次 {len(chunks)} 块, 共 {count} 条")
    return count


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python ingest.py <文件路径> [--clear]")
        print("示例: python ingest.py test.pdf --clear")
        sys.exit(1)

    file_path = sys.argv[1]
    clear = "--clear" in sys.argv
    ingest_file(file_path, clear)