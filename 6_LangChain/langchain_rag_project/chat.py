import sys
import os
import logging
from rag_chain import build_chain, stream_answer
from ingest import ingest_file
from config import MAX_HISTORY_ROUNDS

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

for _logger in ["httpx", "http", "langchain_ollama", "urllib3", "requests"]:
    logging.getLogger(_logger).setLevel(logging.WARNING)


class Conversation:
    def __init__(self, max_rounds=MAX_HISTORY_ROUNDS):
        self.history = []
        self.max_rounds = max_rounds

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_rounds * 2:
            self.history = self.history[-(self.max_rounds * 2):]

    def get(self) -> list:
        return self.history.copy()

    def clear(self):
        self.history = []


def file_ingest():
    print("\n📂 文件入库")
    print("-" * 50)
    file_path = input("请输入文件路径: ").strip().strip('"')
    if not file_path:
        print("⚠️ 未输入路径，已取消")
        return
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    mode = input("入库方式 [1] 追加  [2] 清空重建 (回车默认追加): ").strip()
    clear = mode == "2"

    print()
    try:
        ingest_file(file_path, clear=clear)
        print("✅ 文件入库完成")
    except Exception as e:
        print(f"❌ 入库失败: {e}")


def chat_loop(chain):
    conv = Conversation()

    print("\n✅ 问答就绪 | 命令:")
    print("   exit    - 退出")
    print("   clear   - 清空对话")
    print("   history - 查看历史")
    print("   file    - 读取文件入库")
    print("-" * 50)

    while True:
        try:
            question = input("\n你: ").strip()
            if not question:
                continue

            cmd = question.lower()
            if cmd == "exit":
                print("再见！")
                break
            if cmd == "clear":
                conv.clear()
                print("✅ 对话历史已清空")
                continue
            if cmd == "history":
                ctx = conv.get()
                if not ctx:
                    print("暂无对话历史")
                else:
                    for msg in ctx:
                        label = "用户" if msg["role"] == "user" else "助手"
                        print(f"  {label}: {msg['content'][:100]}")
                continue
            if cmd == "file":
                file_ingest()
                continue

            conv.add("user", question)

            print("\n助手: ", end="", flush=True)
            full_answer = ""

            try:
                for token in stream_answer(chain, question, conv.get()):
                    print(token, end="", flush=True)
                    full_answer += token
            except Exception as e:
                print(f"\n❌ 回答生成失败: {e}")
                conv.history.pop()
                continue

            print()
            if full_answer.strip():
                conv.add("assistant", full_answer.strip())

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except EOFError:
            print("\n\n再见！")
            break


def main():
    print("=" * 50)
    print("  知识库问答系统 (LangChain)")
    print("=" * 50)

    try:
        chain = build_chain()
        logging.info("RAG Chain 就绪")
    except Exception as e:
        print(f"\n❌ 系统初始化失败: {e}")
        print("请检查: 向量库是否已初始化")
        sys.exit(1)

    while True:
        print("\n请选择操作:")
        print("  [1] 问答对话")
        print("  [2] 读取文件入库")
        print("  [0] 退出")

        choice = input("\n请输入选项 (0-2): ").strip()

        if choice == "0":
            print("再见！")
            break
        elif choice == "1":
            chat_loop(chain)
            break
        elif choice == "2":
            file_ingest()
            while True:
                again = input("\n继续入库? (y/n): ").strip().lower()
                if again == "y":
                    file_ingest()
                else:
                    break
        else:
            print("⚠️ 无效选项，请输入 0-2")


if __name__ == "__main__":
    main()