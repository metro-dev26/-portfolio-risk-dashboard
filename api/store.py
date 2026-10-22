"""SQLite persistence for saved portfolios. Ids are random tokens, so one person's link
can't be guessed from another's, and there is no list-everything query. stdlib only. A
fresh connection per call — FastAPI runs sync handlers in a threadpool and a sqlite3
connection cannot cross threads. Path resolves at call time so tests can inject a temp DB."""
import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone

_DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "portfolios.db")


def _resolve(path):
    return path or os.environ.get("PORTFOLIO_DB", _DEFAULT_DB)


_CREATE_TABLE = (
    "CREATE TABLE IF NOT EXISTS saved_portfolios ("
    "id TEXT PRIMARY KEY, name TEXT NOT NULL, holdings TEXT NOT NULL, created_at TEXT NOT NULL)"
)


def save_portfolio(name, holdings, path=None):
    path = _resolve(path)
    pid = secrets.token_urlsafe(16)            # 22 characters, 128 bits
    with sqlite3.connect(path) as conn:
        conn.execute(_CREATE_TABLE)
        conn.execute(
            "INSERT INTO saved_portfolios (id, name, holdings, created_at) VALUES (?, ?, ?, ?)",
            (pid, name, json.dumps(holdings), datetime.now(timezone.utc).isoformat()),
        )
    return pid


def get_portfolio(pid, path=None):
    path = _resolve(path)
    with sqlite3.connect(path) as conn:
        conn.execute(_CREATE_TABLE)
        row = conn.execute(
            "SELECT id, name, holdings, created_at FROM saved_portfolios WHERE id = ?", (pid,)
        ).fetchone()
    if row is None:
        return None
    return {"id": row[0], "name": row[1], "holdings": json.loads(row[2]), "created_at": row[3]}
