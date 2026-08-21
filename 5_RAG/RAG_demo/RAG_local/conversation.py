import threading
import time
from config import (
    MAX_HISTORY_TOKENS, MAX_RECENT_ROUNDS,
    ASYNC_SAVE_INTERVAL
)
import mysql_client
import redis_client
import logging

logger = logging.getLogger(__name__)


class ConversationManager:
    def __init__(self):
        self._write_queue = []
        self._queue_lock = threading.Lock()
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name="mysql-writer"
        )
        self._flush_thread.start()
        self._ensure_db_init()

    def _ensure_db_init(self):
        try:
            mysql_client.init_database()
        except Exception as e:
            logger.warning(f"MySQL 初始化失败（可忽略，功能将受限）: {e}")

    def _flush_loop(self):
        while True:
            time.sleep(ASYNC_SAVE_INTERVAL)
            self._flush_to_mysql()

    def _flush_to_mysql(self):
        with self._queue_lock:
            if not self._write_queue:
                return
            batch = list(self._write_queue)
            self._write_queue.clear()
        try:
            mysql_client.batch_save_messages(batch)
            logger.debug(f"异步写入 MySQL: {len(batch)} 条消息")
        except Exception as e:
            logger.error(f"异步写入 MySQL 失败: {e}")

    def create_session(self, user_id, title="新对话"):
        session_id = mysql_client.create_session(user_id, title)
        redis_client.save_session(session_id, user_id, title)
        logger.info(f"创建会话: {session_id}")
        return session_id

    def switch_session(self, session_id):
        session = redis_client.get_session(session_id)
        if not session:
            session = mysql_client.get_session(session_id)
            if session:
                redis_client.save_session(
                    session["id"],
                    session["user_id"],
                    session.get("title", "新对话")
                )
                history = mysql_client.get_history(session_id)
                for msg in history:
                    redis_client.add_message(
                        session_id, msg["role"], msg["content"]
                    )
                logger.info(f"从 MySQL 回源会话: {session_id}")
        return session

    def add_message(self, session_id, role, content):
        token_count = self._estimate_tokens(content)
        redis_client.add_message(session_id, role, content)
        redis_client.update_session_time(session_id)
        self._enqueue_mysql(session_id, role, content, token_count)
        self._check_compress(session_id)

    def _enqueue_mysql(self, session_id, role, content, token_count):
        with self._queue_lock:
            self._write_queue.append((session_id, role, content, token_count))

    def get_history(self, session_id, recent_n=None):
        return redis_client.get_history(session_id, recent_n)

    def get_full_history(self, session_id):
        mysql_history = mysql_client.get_history(session_id)
        if mysql_history:
            return [
                {"role": m["role"], "content": m["content"], "time": str(m["created_at"])}
                for m in mysql_history
            ]
        return redis_client.get_all_history(session_id)

    def compress_history(self, session_id):
        history = redis_client.get_all_history(session_id)
        if not history:
            return
        total_tokens = sum(
            self._estimate_tokens(m["content"]) for m in history
        )
        if total_tokens <= MAX_HISTORY_TOKENS:
            return
        keep_n = MAX_RECENT_ROUNDS
        keep_msgs = history[-keep_n * 2:]
        old_msgs = history[:-keep_n * 2]
        old_text = ""
        for msg in old_msgs:
            prefix = "用户" if msg["role"] == "user" else "助手"
            old_text += f"{prefix}: {msg['content']}\n"
        summary = f"历史摘要：对话早期内容涉及以下要点：{self._simple_summarize(old_text)}"
        redis_client.clear_history(session_id)
        redis_client.add_message(session_id, "system", summary)
        for msg in keep_msgs:
            redis_client.add_message(session_id, msg["role"], msg["content"])
        logger.info(f"会话 {session_id} 历史压缩完成")

    def _simple_summarize(self, text):
        sentences = [s.strip() for s in text.replace("\n", "。").split("。") if s.strip()]
        if len(sentences) <= 3:
            return "；".join(sentences)
        return "；".join(sentences[:3])

    def _check_compress(self, session_id):
        history = redis_client.get_all_history(session_id)
        total_tokens = sum(
            self._estimate_tokens(m["content"]) for m in history
        )
        if total_tokens > MAX_HISTORY_TOKENS * 1.5:
            self.compress_history(session_id)

    def list_sessions(self, user_id):
        mysql_sessions = mysql_client.get_sessions(user_id)
        redis_sessions = redis_client.get_user_sessions(user_id)
        redis_ids = {s["id"] for s in redis_sessions}
        all_sessions = redis_sessions[:]
        for s in mysql_sessions:
            if s["id"] not in redis_ids:
                all_sessions.append({
                    "id": s["id"],
                    "title": s["title"],
                    "status": s["status"],
                    "updated_at": str(s["updated_at"])
                })
        return all_sessions

    def archive_session(self, session_id, user_id):
        redis_client.archive_session(session_id, user_id)
        mysql_client.archive_session(session_id)
        logger.info(f"归档会话: {session_id}")

    def delete_session(self, session_id, user_id):
        redis_client.delete_session(session_id, user_id)
        mysql_client.delete_session(session_id)
        logger.info(f"删除会话: {session_id}")

    def get_session_stats(self, user_id):
        return mysql_client.get_session_stats(user_id)

    def flush_all(self):
        self._flush_to_mysql()

    def _estimate_tokens(self, text):
        return max(1, len(text) // 2)

    def close(self):
        self.flush_all()