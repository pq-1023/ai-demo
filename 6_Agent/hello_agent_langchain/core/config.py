from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()

class AgentConfig(BaseSettings):
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")

    # ollama
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct-q4_K_M")
    ollama_temperature: float = float(os.getenv("OLLAMA_TEMP", "0.0"))

    # deepseek api
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    deepseek_temperature: float = float(os.getenv("DEEPSEEK_TEMP", "0.0"))

settings = AgentConfig()