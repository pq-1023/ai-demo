import time
import random
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from agent.exceptions import (
    ToolTimeoutError, ToolNetworkError,
    ToolParamError, ToolLogicError, PermanentToolError, TransientToolError
)

MAX_RETRY = 3
TOOL_TIMEOUT = 2


def _validate_query(query):
    if query is None:
        raise ToolParamError("search_knowledge: query 不能为 None")
    if not isinstance(query, str):
        raise ToolParamError(f"search_knowledge: query 必须是 str，收到 {type(query).__name__}")
    if len(query.strip()) == 0:
        raise ToolParamError("search_knowledge: query 不能为空字符串")


def _do_search(query):
    time.sleep(random.uniform(0.1, 0.4))

    if random.random() < 0.1:
        raise ToolNetworkError("模拟网络抖动")

    mock_lib = {
        "图书馆几点关门": "图书馆22:00关门，周末21:30关门"
    }
    return mock_lib.get(query, "知识库暂无该信息")


def search_knowledge(query: str):
    _validate_query(query)

    retry = 0
    while retry < MAX_RETRY:
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_do_search, query)
                return future.result(timeout=TOOL_TIMEOUT)

        except FutureTimeout:
            retry += 1
            print(f"    [search_knowledge] 超时，retry {retry}/{MAX_RETRY}")
            if retry >= MAX_RETRY:
                raise ToolTimeoutError("search_knowledge 超时（已重试3次）")

        except TransientToolError as e:
            retry += 1
            print(f"    [search_knowledge] {e}，retry {retry}/{MAX_RETRY}")
            if retry >= MAX_RETRY:
                raise

        except PermanentToolError:
            raise

        except Exception as e:
            raise ToolLogicError(f"search_knowledge 未知错误: {e}")