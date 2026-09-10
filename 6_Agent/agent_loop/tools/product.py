import time
import random
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from agent.exceptions import (
    ToolTimeoutError, ToolNetworkError,
    ToolParamError, ToolLogicError, PermanentToolError, TransientToolError
)

MAX_RETRY = 3
TOOL_TIMEOUT = 2


def _validate_keyword(keyword):
    if keyword is None:
        raise ToolParamError("search_product: keyword 不能为 None")
    if not isinstance(keyword, str):
        raise ToolParamError(f"search_product: keyword 必须是 str，收到 {type(keyword).__name__}")
    if len(keyword.strip()) == 0:
        raise ToolParamError("search_product: keyword 不能为空字符串")


def _do_search(keyword):
    time.sleep(random.uniform(0.1, 0.5))

    if random.random() < 0.15:
        raise ToolNetworkError("模拟网络波动，连接超时")

    return [
        {"name": "A显示器", "price": 399},
        {"name": "B显示器", "price": 450},
        {"name": "C显示器", "price": 699}
    ]


def search_product(keyword: str):
    _validate_keyword(keyword)

    retry = 0
    while retry < MAX_RETRY:
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_do_search, keyword)
                return future.result(timeout=TOOL_TIMEOUT)

        except FutureTimeout:
            retry += 1
            print(f"    [search_product] 超时，retry {retry}/{MAX_RETRY}")
            if retry >= MAX_RETRY:
                raise ToolTimeoutError("search_product 超时（已重试3次）")

        except TransientToolError as e:
            retry += 1
            print(f"    [search_product] {e}，retry {retry}/{MAX_RETRY}")
            if retry >= MAX_RETRY:
                raise

        except PermanentToolError:
            raise

        except Exception as e:
            raise ToolLogicError(f"search_product 未知错误: {e}")