from document_loader import load_document, split_text
from vector_store import ChromaVectorStore
from rag_pipeline import retrieval_pipeline, build_rag_prompt
from llm_provider import llm_stream_chat
from conversation import ConversationManager
from config import DEFAULT_USER
from dotenv import load_dotenv
import os

load_dotenv()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))

try:
    vector_db = ChromaVectorStore()
except Exception as e:
    print(f"向量库初始化失败: {e}")
    vector_db = None

try:
    conv_manager = ConversationManager()
except Exception as e:
    print(f"会话管理器初始化失败（对话历史功能不可用）: {e}")
    conv_manager = None


def ingest_file(file_path):
    if vector_db is None:
        print("⚠️向量库未初始化，无法导入文档")
        return
    try:
        if not os.path.exists(file_path):
            print("⚠️文件不存在！")
            return
        print("正在读取文档...")
        raw_text = load_document(file_path)
        print("正在文本分片...")
        chunks = split_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"分片总数 {len(chunks)} ,开始分批入库")
        if not vector_db.is_empty:
            print("检测到已有知识库，将清空旧数据后重新导入...")
            vector_db.clear_collection()
        for batch_idx in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[batch_idx:batch_idx+BATCH_SIZE]
            vector_db.upsert_documents(batch, id_offset=batch_idx)
            print(f"已入库 {min(batch_idx+BATCH_SIZE,len(chunks))}/{len(chunks)}")
        print(f"✅文档导入成功，分片总数:{len(chunks)}")
    except Exception as e:
        print(f"\n❌文档导入失败：{e}")


def chat(question, session_id=None):
    try:
        history = None
        if conv_manager and session_id:
            history = conv_manager.get_history(session_id)
        chunk_list = retrieval_pipeline(vector_db, question, history)
        prompt = build_rag_prompt(chunk_list, question, history)
        gen = llm_stream_chat(prompt)
        answer = ""
        for token in gen:
            print(token, end="", flush=True)
            answer += token
        if conv_manager and session_id and answer.strip():
            conv_manager.add_message(session_id, "user", question)
            conv_manager.add_message(session_id, "assistant", answer)
        print()
    except Exception as e:
        print(f"\n❌问答异常:{e}")


def show_sessions(user_id):
    if not conv_manager:
        print("⚠️会话管理器不可用")
        return
    sessions = conv_manager.list_sessions(user_id)
    if not sessions:
        print("暂无历史会话")
        return
    print("\n====历史会话列表====")
    for idx, s in enumerate(sessions, 1):
        status = "活跃" if s.get("status") == "active" else "已归档"
        title = s.get("title", "新对话")
        sid = s.get("id", "")
        print(f"  {idx}. [{status}] {title} (ID: {sid[:12]}...)")
    print("====================\n")


def chat_mode():
    if vector_db and vector_db.is_empty:
        print("⚠️知识库为空，请先导入文档（选择功能1）")
        return
    if conv_manager is None:
        print("⚠️会话管理器不可用，将使用单轮对话模式")
        print("问答模式，输入exit退出")
        while True:
            user_q = input("\n你的提问：")
            if user_q.strip() == "exit":
                break
            chat(user_q)
        return

    user_id = DEFAULT_USER
    current_session = conv_manager.create_session(user_id)
    print(f"✅已创建新会话 (ID: {current_session[:12]}...)")
    print("问答模式，输入以下命令操作：")
    print("  exit         退出程序")
    print("  clear        清空当前会话历史")
    print("  new          创建新会话")
    print("  sessions     查看历史会话列表")
    print("  switch <id>  切换会话 (输入会话ID前12位即可)")
    print("  delete <id>  删除会话")
    print("  archive <id> 归档会话")

    while True:
        try:
            user_q = input("\n你的提问：").strip()
            if not user_q:
                continue

            if user_q == "exit":
                conv_manager.close()
                print("程序已退出")
                break

            if user_q == "clear":
                confirm = input("确认清空当前会话历史？(y/n): ").strip().lower()
                if confirm == "y":
                    conv_manager.delete_session(current_session, user_id)
                    current_session = conv_manager.create_session(user_id)
                    print(f"✅已清空并创建新会话 (ID: {current_session[:12]}...)")
                continue

            if user_q == "new":
                conv_manager.flush_all()
                current_session = conv_manager.create_session(user_id)
                print(f"✅已创建新会话 (ID: {current_session[:12]}...)")
                continue

            if user_q == "sessions":
                show_sessions(user_id)
                continue

            if user_q.startswith("switch "):
                sid_prefix = user_q[7:].strip()
                sessions = conv_manager.list_sessions(user_id)
                found = None
                for s in sessions:
                    if s["id"].startswith(sid_prefix):
                        found = s
                        break
                if found:
                    conv_manager.flush_all()
                    conv_manager.switch_session(found["id"])
                    current_session = found["id"]
                    print(f"✅已切换到会话: {found.get('title','新对话')} (ID: {current_session[:12]}...)")
                else:
                    print(f"⚠️未找到匹配的会话 (前缀: {sid_prefix})")
                continue

            if user_q.startswith("delete "):
                sid_prefix = user_q[7:].strip()
                sessions = conv_manager.list_sessions(user_id)
                found = None
                for s in sessions:
                    if s["id"].startswith(sid_prefix):
                        found = s
                        break
                if found:
                    if found["id"] == current_session:
                        conv_manager.delete_session(current_session, user_id)
                        current_session = conv_manager.create_session(user_id)
                        print(f"✅已删除当前会话并创建新会话 (ID: {current_session[:12]}...)")
                    else:
                        conv_manager.delete_session(found["id"], user_id)
                        print(f"✅已删除会话: {found.get('title')}")
                else:
                    print(f"⚠️未找到匹配的会话 (前缀: {sid_prefix})")
                continue

            if user_q.startswith("archive "):
                sid_prefix = user_q[8:].strip()
                sessions = conv_manager.list_sessions(user_id)
                found = None
                for s in sessions:
                    if s["id"].startswith(sid_prefix):
                        found = s
                        break
                if found:
                    conv_manager.archive_session(found["id"], user_id)
                    print(f"✅已归档会话: {found.get('title')}")
                else:
                    print(f"⚠️未找到匹配的会话 (前缀: {sid_prefix})")
                continue

            chat(user_q, current_session)

        except KeyboardInterrupt:
            conv_manager.close()
            print("\n程序已退出")
            break


if __name__ == "__main__":
    print("====私有化企业知识库问答系统====")
    print("1、导入文档  2、知识库问答")
    try:
        op = input("请输入功能序号：")
        if op == "1":
            path = input("输入文件路径(例 ./docs/test.pdf):")
            ingest_file(path)
        elif op == "2":
            chat_mode()
        else:
            print("⚠️无效的功能序号")
    except KeyboardInterrupt:
        print("\n程序已退出")