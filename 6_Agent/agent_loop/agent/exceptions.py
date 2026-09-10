class AgentMaxIterationsError(Exception):
    """Agent 达到最大迭代次数"""
    pass


class ToolError(Exception):
    """工具错误基类"""
    pass


class TransientToolError(ToolError):
    """
    暂时性错误 —— 重试有意义
    例：网络超时、服务暂时不可用、并发限流
    """
    pass


class PermanentToolError(ToolError):
    """
    永久性错误 —— 重试没用
    例：参数错误(None)、SQL语法错、除零、文件不存在
    """
    pass


class ToolTimeoutError(TransientToolError):
    """工具调用超时（可重试）"""
    pass


class ToolNetworkError(TransientToolError):
    """网络波动（可重试）"""
    pass


class ToolParamError(PermanentToolError):
    """参数校验失败（不可重试）"""
    pass


class ToolLogicError(PermanentToolError):
    """业务逻辑错误，如 eval 除零、索引越界（不可重试）"""
    pass