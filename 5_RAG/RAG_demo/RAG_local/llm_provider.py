import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

LLM_MODE = os.getenv("LLM_MODE")

# Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# 豆包云端大模型
DOUBAO_LLM_API_KEY = os.getenv("DOUBAO_LLM_API_KEY")
DOUBAO_LLM_URL = os.getenv("DOUBAO_LLM_URL")
DOUBAO_LLM_MODEL = os.getenv("DOUBAO_LLM_MODEL")


def ollama_stream_chat(prompt: str):
    """本地Ollama流式调用，使用 /api/chat 接口（兼容新版Ollama）"""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": True
        }
        resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                if "message" in data:
                    yield data["message"].get("content", "")
                else:
                    yield data.get("response", "")
    except requests.exceptions.ConnectionError:
        yield "\n【错误】无法连接Ollama，请检查Ollama服务是否启动"
    except requests.exceptions.Timeout:
        yield "\n【错误】Ollama请求超时"
    except Exception as e:
        yield f"\n【Ollama调用异常】{str(e)}"


def doubao_stream_chat(prompt: str):
    """豆包云端大模型 流式返回"""
    headers = {
        "Authorization": f"Bearer {DOUBAO_LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": DOUBAO_LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": True
    }
    try:
        resp = requests.post(DOUBAO_LLM_URL, headers=headers, json=body, stream=True, timeout=120)
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            if line.startswith("data: "):
                chunk_str = line[6:]
                if chunk_str == "[DONE]":
                    break
                chunk_data = json.loads(chunk_str)
                delta = chunk_data["choices"][0]["delta"].get("content", "")
                yield delta
    except requests.exceptions.ConnectionError:
        yield "\n【错误】网络异常，无法访问豆包API"
    except requests.exceptions.Timeout:
        yield "\n【错误】豆包API请求超时"
    except Exception as e:
        yield f"\n【豆包调用异常】{str(e)}"


def llm_stream_chat(prompt: str):
    """统一入口，自动选择本地Ollama /云端豆包"""
    if LLM_MODE == "ollama":
        return ollama_stream_chat(prompt)
    elif LLM_MODE == "doubao":
        return doubao_stream_chat(prompt)
    else:
        def err_gen():
            yield f"\n【配置错误】不支持的LLM_MODE:{LLM_MODE}"
        return err_gen()


def ollama_chat_sync(prompt: str) -> str:
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if "message" in data:
            return data["message"].get("content", "")
        return data.get("response", "")
    except requests.exceptions.ConnectionError:
        return "[错误]无法连接Ollama"
    except requests.exceptions.Timeout:
        return "[错误]Ollama请求超时"
    except Exception as e:
        return f"[Ollama调用异常]{str(e)}"


def doubao_chat_sync(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {DOUBAO_LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": DOUBAO_LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    try:
        resp = requests.post(DOUBAO_LLM_URL, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        return "[错误]网络异常"
    except requests.exceptions.Timeout:
        return "[错误]请求超时"
    except Exception as e:
        return f"[豆包调用异常]{str(e)}"


def llm_chat_sync(prompt: str) -> str:
    if LLM_MODE == "ollama":
        return ollama_chat_sync(prompt)
    elif LLM_MODE == "doubao":
        return doubao_chat_sync(prompt)
    return f"[配置错误]不支持的LLM_MODE:{LLM_MODE}"