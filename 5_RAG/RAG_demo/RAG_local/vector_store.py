import chromadb
from dotenv import load_dotenv
import os
from embedding_provider import get_embedding_client

load_dotenv()
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(_BASE_DIR, os.getenv("CHROMA_PATH"))

class ChromaVectorStore:
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(path=CHROMA_PATH)
            self.collection = self.client.get_or_create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
            self.embed_client = get_embedding_client()
        except Exception as e:
            raise RuntimeError(f"向量数据库初始化失败:{e}")

    @property
    def is_empty(self) -> bool:
        return self.collection.count() == 0

    def clear_collection(self):
        self.collection.delete(ids=self.collection.get()["ids"])

    def upsert_documents(self, chunks: list[str], id_offset: int = 0):
        try:
            embeddings = self.embed_client.encode(chunks)
            ids = [f"chunk_{id_offset + i}" for i in range(len(chunks))]
            self.collection.upsert(
                documents=chunks,
                embeddings=embeddings,
                ids=ids
            )
        except Exception as e:
            raise RuntimeError(f"文档入库失败:{e}")

    def search(self, query: str, top_k: int):
        try:
            query_vec = self.embed_client.encode([query])[0]
            res = self.collection.query(
                query_embeddings=[query_vec],
                n_results=top_k
            )
            docs = res["documents"][0]
            distances = res["distances"][0]
            sim_scores = [1 - d for d in distances]
            return list(zip(docs, sim_scores))
        except Exception as e:
            raise RuntimeError(f"向量检索失败:{e}")