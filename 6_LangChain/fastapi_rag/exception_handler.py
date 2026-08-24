from fastapi import Request
from fastapi.responses import JSONResponse
from logger import logger


async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"全局捕获异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "msg": f"服务器内部错误：{str(exc)}",
            "data": None
        }
    )