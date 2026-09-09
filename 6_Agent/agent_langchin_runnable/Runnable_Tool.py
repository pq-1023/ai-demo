from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# ===================== 1. 使用 @tool 装饰器，封装商品查询工具 =====================
@tool
def search_product(keyword: str, max_price: int):
    """
    二手笔记本商品查询工具
    :param keyword: 商品关键词/使用场景，例如Java开发、办公
    :param max_price: 最大预算价格
    :return: 符合条件的商品列表，结构化字典数组
    """
    # 模拟二手笔记本数据库
    products = [
        {
            "name": "ThinkPad T480",
            "price": 2800,
            "ram": "16GB",
            "usage": "Java开发适配"
        },
        {
            "name": "Dell Latitude 5420",
            "price": 2900,
            "ram": "16GB",
            "usage": "后端开发通用"
        }
    ]
    # 业务筛选：价格上限匹配 + 使用场景关键词匹配
    res = [p for p in products if p["price"] <= max_price and keyword in p["usage"]]
    return res

# ===================== 2. Prompt模板：把工具返回的结构化数据交给LLM整理话术 =====================
prompt_template = ChatPromptTemplate.from_template("""
你是二手笔记本导购助手，基于下面工具查询到的商品数据，整理成通顺自然的回答。
商品查询结果：
{product_data}
用户问题：{question}
""")

# ===================== 3. 初始化LLM，组装链路 =====================
llm = ChatOllama(
    model="qwen2.5:3b-instruct-q4_K_M",
    temperature=0
)
chain = prompt_template | llm | StrOutputParser()

# ===================== 4. 主执行函数：手动触发Tool调用（本阶段无Agent自动选工具） =====================
def run_tool_demo(user_query: str, search_keyword: str, max_price: int):
    # Step1：手动调用Tool，查询结构化商品数据
    product_result = search_product.invoke({
        "keyword": search_keyword,
        "max_price": max_price
    })
    print("===== Tool返回结构化商品数据 =====")
    print(product_result)

    # Step2：将tool结果塞入prompt，交给LLM生成自然语言回答
    answer = chain.invoke({
        "product_data": product_result,
        "question": user_query
    })
    print("\n===== LLM整理后的回答 =====")
    print(answer)
    return answer

# ========== 测试样例 ==========
if __name__ == "__main__":
    run_tool_demo(
        user_query="2000-3000预算，适合Java开发的笔记本有哪些？",
        search_keyword="Java开发",
        max_price=3000
    )