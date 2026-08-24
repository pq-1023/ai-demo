import os
import re
import logging
from pypdf import PdfReader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    text = text.replace("\t", " ")
    text = re.sub(r"[^\S\n]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    result_lines = []
    empty_count = 0
    for line in lines:
        if line == "":
            empty_count += 1
            if empty_count <= 1:
                result_lines.append("")
        else:
            empty_count = 0
            result_lines.append(line)
    return "\n".join(result_lines).strip()


def _load_pdf(file_path: str):
    reader = PdfReader(file_path)
    docs = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        doc = Document(
            page_content=text,
            metadata={"source": file_path, "page": page_num}
        )
        docs.append(doc)
    return docs


def _load_text(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [Document(page_content=text, metadata={"source": file_path})]


def load_document(file_path: str):
    suffix = os.path.splitext(file_path)[1].lower()
    logger.info(f"加载文档: {file_path}, 类型: {suffix}")

    if suffix == ".pdf":
        docs = _load_pdf(file_path)
    elif suffix in (".txt", ".md"):
        docs = _load_text(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {suffix}，仅支持 PDF、TXT、MD")

    for doc in docs:
        doc.page_content = _clean_text(doc.page_content)

    logger.info(f"文档加载完成, 共 {len(docs)} 段")
    return docs