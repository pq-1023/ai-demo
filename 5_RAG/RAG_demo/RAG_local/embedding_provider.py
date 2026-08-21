import os
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import requests
from dotenv import load_dotenv

load_dotenv()

class BaseEmbedding(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        pass

class LocalEmbedding(BaseEmbedding):
    def __init__(self):
        try:
            self.model_path = os.getenv("LOCAL_EMBED_MODEL_PATH", "all-MiniLM-L6-v2")
            self.model = SentenceTransformer(
                self.model_path,
                local_files_only=True
            )
        except Exception as e:
            raise RuntimeError(f"本地Embedding模型加载失败:{e}")

    def encode(self, texts: list[str]) -> list[list[float]]:
        try:
            emb = self.model.encode(texts, normalize_embeddings=True)
            return emb.tolist()
        except Exception as e:
            raise RuntimeError(f"本地向量编码失败:{e}")


class CloudEmbedding(BaseEmbedding):
    def __init__(self):
        self.api_key = os.getenv("DOUBAO_API_KEY")
        self.url = os.getenv("DOUBAO_EMB_URL")
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def encode(self, texts: list[str]) -> list[list[float]]:
        try:
            payload = {"model": "text-embedding", "input": texts}
            resp = requests.post(self.url, json=payload, headers=self.headers,timeout=120)
            resp.raise_for_status()
            data = resp.json()
            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings
        except Exception as e:
            raise RuntimeError(f"云端Embedding调用失败:{e}")


def get_embedding_client() -> BaseEmbedding:
    mode = os.getenv("EMBED_MODE", "local")
    if mode == "cloud":
        return CloudEmbedding()
    return LocalEmbedding()