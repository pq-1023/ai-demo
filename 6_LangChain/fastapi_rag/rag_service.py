import os
import uuid
import pdfplumber
import chromadb
from config import settings
from embedding_provider import embedding_client
from logger import logger


#文本清洗
def clean_text(text: str) -> str:
    lines = text.split("\n")
    new_lines = [line.strip() for line in lines if line.strip()]
    return "\n".join(new_lines)


#固定长度切块
def split_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + settings.CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start += (settings.CHUNK_SIZE - settings.CHUNK_OVERLAP)
    return chunks


#解析PDF字节流
def extract_pdf_bytes(pdf_bytes: bytes) -> str:
    temp_file = f"temp_{uuid.uuid4()}.pdf"
    with open(temp_file, "wb") as f:
        f.write(pdf_bytes)
    full_text = ""
    with pdfplumber.open(temp_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    os.remove(temp_file)
    return full_text


#获取持久化向量库
def get_chroma_collection():
    client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
    coll = client.get_or_create_collection(settings.COLLECTION_NAME)
    return coll


#PDF入库
def pdf_ingest(pdf_bytes: bytes):
    logger.info("开始解析PDF文件")
    raw_text = extract_pdf_bytes(pdf_bytes)
    if not raw_text.strip():
        raise Exception("PDF没有识别到任何文字")
    clean_txt = clean_text(raw_text)
    chunk_list = split_text(clean_txt)
    logger.info(f"文档切块完成，一共 {len(chunk_list)} 块")
    coll = get_chroma_collection()
    embeddings = embedding_client.encode(chunk_list)
    ids = [str(uuid.uuid4()) for _ in chunk_list]
    coll.add(documents=chunk_list, embeddings=embeddings, ids=ids)
    logger.info("PDF成功存入向量库")
    return len(chunk_list)


#知识库检索
def knowledge_query(question: str, top_k: int):
    coll = get_chroma_collection()
    q_emb = embedding_client.encode([question])[0]
    result = coll.query(query_embeddings=[q_emb], n_results=top_k)
    docs = result["documents"][0]
    distances = result["distances"][0]

    valid_chunk = []
    for doc, dist in zip(docs, distances):
        sim = 1 - dist
        if sim >= settings.SIM_THRESHOLD:
            valid_chunk.append(doc)

    if len(valid_chunk) == 0:
        return {
            "answer": "知识库中没有找到相关信息",
            "source": []
        }
    context = "\n".join(valid_chunk)
    answer = f"【知识库检索结果】\n{context}"
    return {
        "answer": answer,
        "source": valid_chunk
    }