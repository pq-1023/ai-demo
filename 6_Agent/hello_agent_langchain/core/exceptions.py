class HelloAgentError(Exception):
    """Agent基础异常"""
    pass

class LLMInvokeError(HelloAgentError):
    """大模型调用失败"""
    pass

class ToolNotFoundError(HelloAgentError):
    """工具未找到"""
    pass

class AgentRunError(HelloAgentError):
    """Agent运行异常"""
    pass