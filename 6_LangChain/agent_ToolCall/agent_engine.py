import json
from llm_client import chat_with_tools, ChatMessage
from tools_registry import TOOL_LIST, TOOL_EXEC_MAP


def _msg_to_dict(msg: ChatMessage) -> dict:
    d = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            }
            for tc in msg.tool_calls
        ]
    return d


def run_agent(user_query: str):
    # ① 初始化对话上下文（系统提示 + 用户问题）
    messages = [
        {
            "role": "system",
            "content": "你是工具调用助手。用户的问题如果需要查询订单或者数学计算，你必须调用对应的工具函数，不要自己计算、不要编造答案。调用工具后等待工具返回结果，再回答用户。"
        },
        {"role": "user", "content": user_query}
    ]

    tool_called = False # 记录是否调用过工具
    max_loop = 3 # 最多循环 3 次（防止死循环）
    # ② Agent 循环：LLM 推理 → 工具调用 → 再次推理
    for _ in range(max_loop):
        msg = chat_with_tools(messages, TOOL_LIST)# 调用 LLM 推理，获取模型回复
        # ③ LLM 决定：不调用工具 → 直接返回答案
        if msg.tool_calls is None or len(msg.tool_calls) == 0:
            return {
                "success": True,
                "answer": msg.content,
                "tool_called": tool_called
            }
        # ④ LLM 决定：调用工具 → 执行并塞回上下文
        tool_called = True
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name# 找到业务函数
            try:
                args = json.loads(tool_call.function.arguments)# 解析参数
            except json.JSONDecodeError:
                return {"success": False, "error": "模型生成的工具参数JSON格式错误"}

            func = TOOL_EXEC_MAP.get(func_name)# 找到业务函数
            if not func:
                return {"success": False, "error": f"不存在工具：{func_name}"}

            tool_result = func(**args)# 执行工具函数
            # 把 assistant 消息和 tool 结果都追加到 messages
            messages.append(_msg_to_dict(msg)) # LLM 的 tool_call 请求
            messages.append({  # 工具执行结果
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result, ensure_ascii=False)
            })
    # ⑤ 兜底：循环耗尽后让 LLM 生成最终答案
    final_msg = chat_with_tools(messages, TOOL_LIST)
    return {
        "success": True,
        "answer": final_msg.content,
        "tool_called": True
    }