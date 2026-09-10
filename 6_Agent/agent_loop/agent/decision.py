import json
import os
import re
import requests
from dotenv import load_dotenv
from .state import AgentState

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")

SYSTEM_PROMPT = """你是一个智能Agent，使用工具来回答用户问题。

## 可用工具
1. search_product(keyword: str) - 搜索商品，返回 [{"name": str, "price": int}, ...]
2. calculator(expr: str) - 计算数学表达式，例如 "399 + 450"，返回数值
3. search_knowledge(query: str) - 校园知识库查询

## 决策规则
1. 第一步：用户问商品相关的，先用 search_product 搜索关键词
2. 拿到商品列表后，从列表中筛选符合条件(如价格<=500)的商品，把它们的价格用 " + " 拼接，调用 calculator 求和
3. 拿到求和结果后，自己算平均(总和/数量)，然后给出最终答案
4. 不要重复调用同一个工具！如果已经有搜索结果了，就不要再搜了
5. 当你有足够信息回答问题时，直接输出 final

## 输出格式
严格输出 JSON，不要其他内容：
- 调工具：{"type":"tool","tool":"工具名","input":"参数"}
- 最终答：{"type":"final","content":"回答内容"}"""


def _build_messages(state: AgentState) -> list:
    user_parts = [f"用户问题：{state.question}"]
    user_parts.append(f"当前是第 {state.current_step} 步（最多5步）")

    if state.observations:
        user_parts.append("历史观测结果：")
        for i, obs in enumerate(state.observations):
            user_parts.append(f"  第{i+1}步: {json.dumps(obs, ensure_ascii=False)}")
        user_parts.append(f"注意：你已经调用过 {len(state.observations)} 次工具了，不要重复！")
    else:
        user_parts.append("历史观测结果：暂无（第一步）")

    user_parts.append("现在请决策下一步，只输出JSON。")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)}
    ]


def _call_deepseek(messages: list) -> str:
    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL_ID,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.3
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _parse_decision(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group())
        raise ValueError(f"无法解析LLM输出为JSON: {raw}")


def make_decision(state: AgentState) -> dict:
    messages = _build_messages(state)
    print(f"  [LLM User] {messages[-1]['content'][:200]}...")

    raw_response = _call_deepseek(messages)
    print(f"  [LLM Raw] {raw_response}")

    decision = _parse_decision(raw_response)

    if decision["type"] == "tool":
        valid_tools = {"search_product", "calculator", "search_knowledge"}
        if decision["tool"] not in valid_tools:
            raise ValueError(f"LLM返回了无效的工具名: {decision['tool']}")
    elif decision["type"] == "final":
        pass
    else:
        raise ValueError(f"未知决策类型: {decision.get('type')}")

    return decision