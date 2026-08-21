import redis
import json
import time
from config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB, MAX_RECENT_ROUNDS, SESSION_TIMEOUT_HOURS
import logging

logger = logging.getLogger(__name__)

_redis_client = None


def get_redis_client():
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                db=REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=5
            )
            _redis_client.ping()
            logger.info("Redis 连接成功")
        except Exception as e:
            raise RuntimeError(f"Redis 连接失败: {e}")
    return _redis_client


def _session_key(session_id):
    return f"session:{session_id}"


def _history_key(session_id):
    return f"history:{session_id}"


def _user_sessions_key(user_id):
    return f"user_sessions:{user_id}"


def save_session(session_id, user_id, title="新对话"):
    r = get_redis_client()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    r.hset(_session_key(session_id), "user_id", user_id)
    r.hset(_session_key(session_id), "title", title)
    r.hset(_session_key(session_id), "status", "active")
    r.hset(_session_key(session_id), "created_at", now)
    r.hset(_session_key(session_id), "updated_at", now)
    _set_ttl(session_id)
    r.sadd(_user_sessions_key(user_id), session_id)
    return session_id


def get_session(session_id):
    r = get_redis_client()
    data = r.hgetall(_session_key(session_id))
    return data if data else None


def update_session_time(session_id):
    r = get_redis_client()
    r.hset(_session_key(session_id), "updated_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    _set_ttl(session_id)


def _set_ttl(session_id):
    r = get_redis_client()
    ttl_seconds = SESSION_TIMEOUT_HOURS * 3600
    r.expire(_session_key(session_id), ttl_seconds)
    r.expire(_history_key(session_id), ttl_seconds)


def add_message(session_id, role, content):
    r = get_redis_client()
    msg = json.dumps({
        "role": role,
        "content": content,
        "time": time.strftime("%Y-%m-%d %H:%M:%S")
    }, ensure_ascii=False)
    r.rpush(_history_key(session_id), msg)
    recent_rounds = MAX_RECENT_ROUNDS * 2
    if r.llen(_history_key(session_id)) > recent_rounds:
        r.ltrim(_history_key(session_id), -recent_rounds, -1)
    _set_ttl(session_id)


def get_history(session_id, recent_n=None):
    r = get_redis_client()
    if recent_n:
        raw_list = r.lrange(_history_key(session_id), -recent_n * 2, -1)
    else:
        raw_list = r.lrange(_history_key(session_id), 0, -1)
    result = []
    for item in raw_list:
        result.append(json.loads(item))
    return result


def get_all_history(session_id):
    r = get_redis_client()
    raw_list = r.lrange(_history_key(session_id), 0, -1)
    result = []
    for item in raw_list:
        result.append(json.loads(item))
    return result


def get_user_sessions(user_id):
    r = get_redis_client()
    session_ids = r.smembers(_user_sessions_key(user_id))
    sessions = []
    for sid in session_ids:
        session = get_session(sid)
        if session and session.get("status") == "active":
            sessions.append({"id": sid, **session})
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return sessions


def set_session_title(session_id, title):
    r = get_redis_client()
    r.hset(_session_key(session_id), "title", title)


def archive_session(session_id, user_id):
    r = get_redis_client()
    r.hset(_session_key(session_id), "status", "archived")
    r.srem(_user_sessions_key(user_id), session_id)


def delete_session(session_id, user_id):
    r = get_redis_client()
    r.delete(_session_key(session_id))
    r.delete(_history_key(session_id))
    r.srem(_user_sessions_key(user_id), session_id)


def clear_history(session_id):
    r = get_redis_client()
    r.delete(_history_key(session_id))


def is_session_active(session_id):
    r = get_redis_client()
    data = r.hgetall(_session_key(session_id))
    return data and data.get("status") == "active"