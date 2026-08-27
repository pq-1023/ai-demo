import requests
import json
from typing import List, Dict, Any, Optional


class ToolFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class ToolCall:
    def __init__(self, id: str, function: ToolFunction):
        self.id = id
        self.function = function


class ChatMessage:
    def __init__(self, content: Optional[str], tool_calls: Optional[List[ToolCall]] = None):
        self.content = content
        self.tool_calls = tool_calls


OLLAMA_URL = "http://127.0.0.1:11434/v1/chat/completions"


def chat_with_tools(
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: str = "qwen2.5:3b-instruct-q4_K_M"
) -> ChatMessage:
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    msg = data["choices"][0]["message"]
    content = msg.get("content")

    tool_calls = None
    if msg.get("tool_calls"):
        tool_calls = [
            ToolCall(
                id=tc["id"],
                function=ToolFunction(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"]
                )
            )
            for tc in msg["tool_calls"]
        ]

    return ChatMessage(content=content, tool_calls=tool_calls)