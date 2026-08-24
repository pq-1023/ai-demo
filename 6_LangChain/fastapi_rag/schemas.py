from pydantic import BaseModel, Field


#问答请求体
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="用户问题")
    top_k: int = Field(default=3, ge=1, le=10, description="召回片段数量")


#问答返回体
class ChatResponse(BaseModel):
    answer: str
    source: list[str]


#通用返回封装
class CommonResp(BaseModel):
    code: int
    msg: str
    data: dict | None = None