import os
import logging

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class RerankProvider:
    def __init__(self, model_path=None, hf_model_name=None):
        self.model = None
        self.model_path = model_path
        self.hf_model_name = hf_model_name
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder

            if self.model_path and os.path.exists(self.model_path):
                self.model = CrossEncoder(self.model_path)
                logger.info(f"Rerank 模型加载成功(本地): {self.model_path}")
                return

            if self.hf_model_name:
                logger.info(f"本地无模型，从 HuggingFace 镜像下载: {self.hf_model_name}")
                logger.info("如果下载失败，请手动从 ModelScope 下载模型")
                self.model = CrossEncoder(self.hf_model_name)
                if self.model_path:
                    os.makedirs(self.model_path, exist_ok=True)
                    self.model.save(self.model_path)
                    logger.info(f"Rerank 模型已保存到: {self.model_path}")
                logger.info(f"Rerank 模型加载成功(在线): {self.hf_model_name}")
                return

            logger.warning(f"Rerank 模型路径不存在且未配置 HuggingFace 模型名")
            self.model = None
        except ImportError:
            logger.error("缺少 sentence_transformers 依赖，请运行: pip install sentence-transformers")
            self.model = None
        except Exception as e:
            logger.error(f"Rerank 模型加载失败: {e}")
            logger.info("请手动下载模型: https://www.modelscope.cn/models/BAAI/bge-reranker-base")
            logger.info(f"然后放到: {self.model_path}")
            self.model = None

    def rerank(self, query, documents, top_n=3):
        if not self.model or not documents:
            return [(doc, 0.0) for doc in documents[:top_n]]
        try:
            pairs = [[query, doc] for doc in documents]
            scores = self.model.predict(pairs)
            if not isinstance(scores, list):
                scores = [float(scores)]
            ranked = sorted(
                zip(documents, scores),
                key=lambda x: x[1],
                reverse=True
            )
            logger.info(f"Rerank 完成: {len(documents)} 候选 → Top {top_n}")
            return ranked[:top_n]
        except Exception as e:
            logger.error(f"Rerank 执行失败: {e}")
            return [(doc, 0.0) for doc in documents[:top_n]]

    def is_available(self):
        return self.model is not None

    def close(self):
        if self.model:
            del self.model
            self.model = None