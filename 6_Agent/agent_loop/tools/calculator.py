import time
import random
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from agent.exceptions import (
    ToolTimeoutError, ToolNetworkError,
    ToolParamError, ToolLogicError, PermanentToolError, TransientToolError
)

MAX_RETRY = 3
TOOL_TIMEOUT = 2


def _validate_expr(expr):
    if expr is None:
        raise ToolParamError("calculator: expr 不能为 None")
    if not isinstance(expr, str):
        raise ToolParamError(f"calculator: expr 必须是 str，收到 {type(expr).__name__}")
    if len(expr.strip()) == 0:
        raise ToolParamError("calculator: expr 不能为空字符串")


def _do_calc(expr):
    time.sleep(random.uniform(0.05, 0.3))

    if random.random() < 0.05:
        raise ToolNetworkError("模拟网络抖动")

    result = eval(expr, {"__builtins__": {}}, {})
    return float(result)


def calculator(expr: str) -> float:
    _validate_expr(expr)

    retry = 0
    while retry < MAX_RETRY:
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_do_calc, expr)
                return future.result(timeout=TOOL_TIMEOUT)

        except FutureTimeout:
            retry += 1
            print(f"    [calculator] 超时，retry {retry}/{MAX_RETRY}")
            if retry >= MAX_RETRY:
                raise ToolTimeoutError("calculator 超时（已重试3次）")

        except TransientToolError as e:
            retry += 1
            print(f"    [calculator] {e}，retry {retry}/{MAX_RETRY}")
            if retry >= MAX_RETRY:
                raise

        except ZeroDivisionError as e:
            raise ToolLogicError(f"calculator 除零错误: {e}")

        except SyntaxError as e:
            raise ToolLogicError(f"calculator 表达式语法错误: {e}")

        except NameError as e:
            raise ToolLogicError(f"calculator 表达式含非法变量: {e}")

        except PermanentToolError:
            raise

        except Exception as e:
            raise ToolLogicError(f"calculator 计算失败: {e}")