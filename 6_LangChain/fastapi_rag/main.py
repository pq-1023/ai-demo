from fastapi import FastAPI, UploadFile, File
from schemas import ChatRequest, ChatResponse, CommonResp
from exception_handler import global_exception_handler
from rag_service import pdf_ingest, knowledge_query
from logger import logger
from config import settings

app = FastAPI(title="RAG知识库后端服务")
#注册全局异常
app.add_exception_handler(Exception, global_exception_handler)


#接口1：PDF上传入库
@app.post("/api/rag/upload", response_model=CommonResp)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return CommonResp(code=400, msg="只支持PDF文件", data=None)
    bytes_data = await file.read()
    chunk_count = pdf_ingest(bytes_data)
    return CommonResp(
        code=200,
        msg="文档入库成功",
        data={"chunk_count": chunk_count}
    )


#接口2：知识库问答
@app.post("/api/rag/chat", response_model=ChatResponse)
async def rag_chat(req: ChatRequest):
    res = knowledge_query(req.question, req.top_k)
    return ChatResponse(**res)


if __name__ == "__main__":
    import uvicorn
    logger.info(f"服务启动，端口:{settings.SERVER_PORT}")
    uvicorn.run("main:app", host="0.0.0.0", port=settings.SERVER_PORT, reload=False)
