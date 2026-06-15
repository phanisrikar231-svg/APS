from __future__ import annotations

import sqlite3
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "fedmsme.db"


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                company_name TEXT NOT NULL,
                phone TEXT,
                role TEXT NOT NULL DEFAULT 'Admin',
                verified INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS otp_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                code_hash TEXT NOT NULL,
                purpose TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                used INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS msmes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                industry TEXT NOT NULL,
                city TEXT,
                state TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS machines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                msme_id INTEGER,
                name TEXT NOT NULL,
                machine_type TEXT NOT NULL,
                sensor_schema TEXT NOT NULL,
                workflow_notes TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY(msme_id) REFERENCES msmes(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS sensor_batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                raw_csv TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                FOREIGN KEY(machine_id) REFERENCES machines(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id INTEGER NOT NULL,
                risk REAL NOT NULL,
                rul REAL NOT NULL,
                health REAL NOT NULL,
                status TEXT NOT NULL,
                reasons_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(machine_id) REFERENCES machines(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                rounds INTEGER NOT NULL,
                clients INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                loss REAL NOT NULL,
                rmse REAL NOT NULL,
                history_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_store (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                model_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.commit()


def scalar(conn: sqlite3.Connection, query: str, params: tuple = ()) -> object:
    row = conn.execute(query, params).fetchone()
    if row is None:
        return None
    return row[0]

