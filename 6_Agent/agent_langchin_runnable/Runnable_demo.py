from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

# ===================== 1. 定义输出数据结构 =====================
class DevInfo(BaseModel):
    budget: int = Field(description="项目预算")
    usage: str = Field(description="用途描述")

# ===================== 2. 初始化组件 =====================
parser = JsonOutputParser(pydantic_object=DevInfo)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你需要按照指定JSON格式返回结果。{format_instructions}"),
    ("human", "请生成一条开发项目信息")
])

llm = ChatOllama(
    model="qwen2.5:3b-instruct-q4_K_M",
    temperature=0
)

# ===================== 3. 构建Runnable链路：Prompt | Model | Parser =====================
chain = prompt | llm | parser

# ===================== 4. 执行调用 =====================
if __name__ == "__main__":
    result = chain.invoke({"format_instructions": parser.get_format_instructions()})
    print(type(result))
    print(result)