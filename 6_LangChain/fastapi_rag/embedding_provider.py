from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from config import settings
from logger import logger


class BaseEmbedding(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        pass


class LocalEmbedding(BaseEmbedding):
    def __init__(self):
        logger.info("正在加载本地Embedding模型")
        self.model = SentenceTransformer(
            settings.LOCAL_EMBED_MODEL_PATH,
            local_files_only=True
        )
        logger.info("Embedding模型加载完成")

    def encode(self, texts: list[str]) -> list[list[float]]:
        emb = self.model.encode(texts, normalize_embeddings=True)
        return emb.tolist()


embedding_client = LocalEmbedding()