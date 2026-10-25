"""
Persistence layer for conversation history - SQLite, local file.
One connection per call: fine at this scale, no concurrent-writer
problem to solve for a single local dev tool.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
import os


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chat_history.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # access columns by name, e.g. row["title"]
    return conn


def create_conversation(title=None):
    conversation_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    conn.execute(
        "INSERT INTO conversations (conversation_id, title, created_at) VALUES (?, ?, ?)",
        (conversation_id, title, created_at),
    )
    conn.commit()
    conn.close()
    return conversation_id


def save_message(conversation_id, role, text, sources=None):
    message_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    sources_json = json.dumps(sources) if sources is not None else None

    conn = get_connection()
    conn.execute(
        "INSERT INTO messages (message_id, conversation_id, role, text, sources, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (message_id, conversation_id, role, text, sources_json, created_at),
    )
    conn.commit()
    conn.close()
    return message_id


def list_conversations():
    conn = get_connection()
    rows = conn.execute(
        "SELECT conversation_id, title, created_at FROM conversations ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_conversation_messages(conversation_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT role, text, sources, created_at FROM messages "
        "WHERE conversation_id = ? ORDER BY created_at ASC",
        (conversation_id,),
    ).fetchall()
    conn.close()

    messages = []
    for row in rows:
        message = dict(row)
        message["sources"] = json.loads(message["sources"]) if message["sources"] else None
        messages.append(message)
    return messages