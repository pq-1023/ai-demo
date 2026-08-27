from typing import Dict, Any

# 工具描述，发给LLM
TOOL_LIST = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "根据订单编号查询订单状态",
            "parameters": {
                "type": "object",
                "properties": {
                    "orderId": {
                        "type": "string",
                        "description": "订单编号"
                    }
                },
                "required": ["orderId"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator_add",
            "description": "两个数字相加",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个数字"},
                    "b": {"type": "number", "description": "第二个数字"}
                },
                "required": ["a", "b"]
            }
        }
    }
]

# 实际执行的业务函数
def query_order(orderId: str) -> Dict[str, Any]:
    mock_db = {
        "2026001": {"orderId": "2026001", "status": "已发货", "delivery_time": "2026‑08‑28"},
        "2026002": {"orderId": "2026002", "status": "待付款"}
    }
    return mock_db.get(orderId, {"msg": "未找到该订单"})


def calculator_add(a: float, b: float) -> float:
    return a + b


TOOL_EXEC_MAP = {
    "query_order": query_order,
    "calculator_add": calculator_add
}