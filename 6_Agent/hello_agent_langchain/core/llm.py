import requests
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from .config import settings
from .exceptions import LLMInvokeError


def check_ollama_model_exists(base_url: str, model_name: str) -> bool:
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        model_list = [item["name"] for item in resp.json()["models"]]
        return model_name in model_list
    except Exception:
        return False


class HelloAgentsLLM:
    def __init__(self):
        self.provider = settings.llm_provider.lower().strip()
        self._llm = self._build_llm()

    def _build_llm(self):
        if self.provider == "ollama":
            if not check_ollama_model_exists(settings.ollama_base_url, settings.ollama_model):
                raise LLMInvokeError(
                    f"本地Ollama未找到模型: {settings.ollama_model}\n"
                    f"请执行: ollama pull {settings.ollama_model}"
                )
            return ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=settings.ollama_temperature
            )
        elif self.provider == "deepseek":
            if not settings.deepseek_api_key:
                raise LLMInvokeError("DeepSeek API Key不能为空，请在.env配置")
            return ChatOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                model=settings.deepseek_model,
                temperature=settings.deepseek_temperature
            )
        else:
            raise LLMInvokeError(f"不支持的LLM提供商: {self.provider},仅支持 ollama / deepseek")

    def get_llm(self):
        return self._llm