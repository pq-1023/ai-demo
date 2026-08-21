import pymysql
import uuid
import json
import time
import threading
from config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD,
    MYSQL_DATABASE, MYSQL_CHARSET
)
import logging

logger = logging.getLogger(__name__)

_connection_pool = []
_pool_lock = threading.Lock()
MAX_POOL_SIZE = 5


def _create_connection():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        charset=MYSQL_CHARSET,
        database=MYSQL_DATABASE,
        autocommit=True
    )


def get_connection():
    with _pool_lock:
        if _connection_pool:
            return _connection_pool.pop()
    return _create_connection()


def release_connection(conn):
    with _pool_lock:
        if len(_connection_pool) < MAX_POOL_SIZE:
            _connection_pool.append(conn)
        else:
            conn.close()


def init_database():
    conn = pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT,
        user=MYSQL_USER, password=MYSQL_PASSWORD,
        charset=MYSQL_CHARSET
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` "
                f"DEFAULT CHARACTER SET {MYSQL_CHARSET}"
            )
        conn.commit()
        logger.info(f"数据库 {MYSQL_DATABASE} 初始化成功")
    finally:
        conn.close()

    conn = _create_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation (
                    id VARCHAR(32) PRIMARY KEY,
                    user_id VARCHAR(32) NOT NULL,
                    title VARCHAR(100),
                    status ENUM('active','archived') DEFAULT 'active',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_user_id (user_id),
                    INDEX idx_status (status)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS message (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(32) NOT NULL,
                    role ENUM('user','assistant','system') NOT NULL,
                    content TEXT NOT NULL,
                    token_count INT DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_session_id (session_id),
                    INDEX idx_created_at (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
        conn.commit()
        logger.info("数据表初始化成功")
    finally:
        release_connection(conn)


def create_session(user_id, title="新对话"):
    conn = get_connection()
    try:
        session_id = uuid.uuid4().hex[:16]
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO conversation (id, user_id, title) VALUES (%s, %s, %s)",
                (session_id, user_id, title)
            )
        conn.commit()
        return session_id
    finally:
        release_connection(conn)


def update_session(session_id, title=None, status=None):
    conn = get_connection()
    try:
        fields = []
        params = []
        if title:
            fields.append("title=%s")
            params.append(title)
        if status:
            fields.append("status=%s")
            params.append(status)
        if not fields:
            return
        params.append(session_id)
        with conn.cursor() as cursor:
            cursor.execute(
                f"UPDATE conversation SET {','.join(fields)} WHERE id=%s",
                params
            )
        conn.commit()
    finally:
        release_connection(conn)


def save_message(session_id, role, content, token_count=0):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO message (session_id, role, content, token_count) "
                "VALUES (%s, %s, %s, %s)",
                (session_id, role, content, token_count)
            )
        conn.commit()
    finally:
        release_connection(conn)


def batch_save_messages(messages):
    if not messages:
        return
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO message (session_id, role, content, token_count) "
                "VALUES (%s, %s, %s, %s)",
                messages
            )
        conn.commit()
    finally:
        release_connection(conn)


def get_history(session_id, limit=None):
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            if limit:
                cursor.execute(
                    "SELECT role, content, token_count, created_at "
                    "FROM message WHERE session_id=%s ORDER BY id DESC LIMIT %s",
                    (session_id, limit)
                )
            else:
                cursor.execute(
                    "SELECT role, content, token_count, created_at "
                    "FROM message WHERE session_id=%s ORDER BY id ASC",
                    (session_id,)
                )
            return cursor.fetchall()
    finally:
        release_connection(conn)


def get_sessions(user_id, status=None):
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            if status:
                cursor.execute(
                    "SELECT id, title, status, created_at, updated_at "
                    "FROM conversation WHERE user_id=%s AND status=%s "
                    "ORDER BY updated_at DESC",
                    (user_id, status)
                )
            else:
                cursor.execute(
                    "SELECT id, title, status, created_at, updated_at "
                    "FROM conversation WHERE user_id=%s "
                    "ORDER BY updated_at DESC",
                    (user_id,)
                )
            return cursor.fetchall()
    finally:
        release_connection(conn)


def get_session(session_id):
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(
                "SELECT id, user_id, title, status, created_at, updated_at "
                "FROM conversation WHERE id=%s",
                (session_id,)
            )
            return cursor.fetchone()
    finally:
        release_connection(conn)


def archive_session(session_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE conversation SET status='archived' WHERE id=%s",
                (session_id,)
            )
        conn.commit()
    finally:
        release_connection(conn)


def delete_session(session_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM message WHERE session_id=%s", (session_id,))
            cursor.execute("DELETE FROM conversation WHERE id=%s", (session_id,))
        conn.commit()
    finally:
        release_connection(conn)


def get_session_stats(user_id):
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(
                "SELECT COUNT(*) as total, "
                "SUM(CASE WHEN status='active' THEN 1 ELSE 0 END) as active_count "
                "FROM conversation WHERE user_id=%s",
                (user_id,)
            )
            return cursor.fetchone()
    finally:
        release_connection(conn)