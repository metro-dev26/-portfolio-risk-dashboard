"""SQLite persistence for saved portfolios. stdlib only. A fresh connection per
call — FastAPI runs sync handlers in a threadpool and a sqlite3 connection cannot
cross threads. Path resolves at call time so tests can inject a temp DB."""
import json
import os
import sqlite3
from datetime import datetime, timezone

_DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "portfolios.db")


def _resolve(path):
    return path or os.environ.get("PORTFOLIO_DB", _DEFAULT_DB)


def _init(path):
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS portfolios ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "name TEXT NOT NULL, holdings TEXT NOT NULL, created_at TEXT NOT NULL)"
        )


def save_portfolio(name, holdings, path=None):
    path = _resolve(path)
    _init(path)
    with sqlite3.connect(path) as conn:
        cur = conn.execute(
            "INSERT INTO portfolios (name, holdings, created_at) VALUES (?, ?, ?)",
            (name, json.dumps(holdings), datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def get_portfolio(pid, path=None):
    path = _resolve(path)
    _init(path)
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT id, name, holdings, created_at FROM portfolios WHERE id = ?", (pid,)
        ).fetchone()
    if row is None:
        return None
    return {"id": row[0], "name": row[1], "holdings": json.loads(row[2]), "created_at": row[3]}


def list_portfolios(path=None):
    path = _resolve(path)
    _init(path)
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "SELECT id, name, holdings, created_at FROM portfolios ORDER BY id"
        ).fetchall()
    return [
        {"id": r[0], "name": r[1], "holdings": json.loads(r[2]), "created_at": r[3]}
        for r in rows
    ]
