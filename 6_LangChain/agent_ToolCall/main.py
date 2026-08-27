from fastapi import FastAPI
from pydantic import BaseModel
from agent_engine import run_agent

# ① 创建应用实例（相当于 Flask 的 app）
app = FastAPI(title="Ollama‑Agent‑ToolCall")
# ② 定义请求体的数据模型（自动校验 JSON 格式）
class AgentRequest(BaseModel):
    query: str # 客户端必须传一个 query 字符串

# ③ 注册路由：把 URL /api/agent/chat 绑定到处理函数
@app.post("/api/agent/chat")
def agent_chat(req: AgentRequest):
    result = run_agent(req.query)# 调用业务层
    return result# 自动序列化为 JSON 返回

# ④ 启动 ASGI 服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)