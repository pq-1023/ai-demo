import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
from rag_retriever import get_retriever
from config import OLLAMA_MODEL, OLLAMA_URL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是知识库问答助手，只能依靠给出的上下文回答问题。
如果上下文里面完全没有相关信息，直接输出：【知识库暂未查询到此内容】，严禁自己编造、脑补答案。

参考资料：
{context}"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])


def build_chain(filter_meta: dict = None):
    retriever = get_retriever(filter_meta)
    llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
    doc_chain = create_stuff_documents_chain(llm, PROMPT)
    chain = create_retrieval_chain(retriever, doc_chain)
    logger.info(f"RAG Chain 构建完成, model={OLLAMA_MODEL}")
    return chain


def format_history(history: list) -> list:
    messages = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def stream_answer(chain, question: str, history: list = None):
    input_data = {"input": question}
    if history:
        input_data["history"] = format_history(history)

    for chunk in chain.stream(input_data):
        if "answer" in chunk:
            yield chunk["answer"]