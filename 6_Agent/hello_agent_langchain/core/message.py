from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

class Message:
    @staticmethod
    def human(content: str) -> HumanMessage:
        return HumanMessage(content=content)

    @staticmethod
    def ai(content: str) -> AIMessage:
        return AIMessage(content=content)

    @staticmethod
    def system(content: str) -> SystemMessage:
        return SystemMessage(content=content)