from __future__ import annotations

import json
from pathlib import Path

import sqlite3

from . import ml
from .database import ROOT_DIR
from .security import hash_password, iso, utcnow


DEMO_EMAIL = "admin@demo.msme"
DEMO_PASSWORD = "demo1234"


def ensure_model(conn: sqlite3.Connection) -> dict:
    row = conn.execute("SELECT model_json FROM model_store WHERE id = 1").fetchone()
    if row:
        return ml.model_from_json(row["model_json"])

    result = ml.train_strategy("FedProx", rounds=8, local_epochs=3)
    now = iso(utcnow())
    conn.execute(
        """
        INSERT INTO training_runs(strategy, rounds, clients, accuracy, loss, rmse, history_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.strategy,
            result.rounds,
            result.clients,
            result.accuracy,
            result.loss,
            result.rmse,
            json.dumps(result.history),
            now,
        ),
    )
    conn.execute(
        "INSERT INTO model_store(id, model_json, updated_at) VALUES (1, ?, ?)",
        (ml.model_to_json(result.model), now),
    )
    conn.commit()
    return result.model


def seed_demo_data(conn: sqlite3.Connection) -> None:
    now = iso(utcnow())
    user = conn.execute("SELECT * FROM users WHERE lower(email) = lower(?)", (DEMO_EMAIL,)).fetchone()
    if user is None:
        digest, salt = hash_password(DEMO_PASSWORD)
        conn.execute(
            """
            INSERT INTO users(name, email, password_hash, salt, company_name, phone, role, verified, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'Admin', 1, ?)
            """,
            ("Demo Admin", DEMO_EMAIL, digest, salt, "FedMSME Demo Consortium", "+91-90000-00000", now),
        )
        conn.commit()
        user = conn.execute("SELECT * FROM users WHERE lower(email) = lower(?)", (DEMO_EMAIL,)).fetchone()

    msme = conn.execute("SELECT * FROM msmes WHERE user_id = ? LIMIT 1", (user["id"],)).fetchone()
    if msme is None:
        conn.execute(
            """
            INSERT INTO msmes(user_id, name, industry, city, state, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                "FedMSME Demo Consortium",
                "Mixed light manufacturing",
                "Bengaluru",
                "Karnataka",
                now,
            ),
        )
        conn.commit()
        msme = conn.execute("SELECT * FROM msmes WHERE user_id = ? LIMIT 1", (user["id"],)).fetchone()

    existing_count = conn.execute("SELECT COUNT(*) FROM machines WHERE user_id = ?", (user["id"],)).fetchone()[0]
    if existing_count:
        model = ensure_model(conn)
        ensure_demo_predictions(conn, user["id"], model)
        return

    machines = [
        ("CNC-07", "CNC Milling", "vibration, temperature, current, rpm, load", "Batch machining; two 8-hour shifts"),
        ("TXT-MOTOR-12", "Textile Motor", "vibration, temperature, current, rpm", "Continuous loom motor line"),
        ("PUMP-03", "Pump Unit", "pressure, vibration, current, temperature, load", "Intermittent fluid transfer"),
        ("COMP-02", "Air Compressor", "pressure, acoustic, temperature, current", "Compressed-air utility cell"),
    ]
    sample_dir = ROOT_DIR / "sample_data"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for idx, (name, machine_type, sensors, notes) in enumerate(machines, start=1):
        cur = conn.execute(
            """
            INSERT INTO machines(user_id, msme_id, name, machine_type, sensor_schema, workflow_notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user["id"], msme["id"], name, machine_type, sensors, notes, now),
        )
        machine_id = cur.lastrowid
        degradation = 0.20 + 0.14 * idx
        csv_text = ml.timeseries_to_csv(machine_type, seed=80 + idx, degradation=degradation)
        filename = f"{name.lower().replace('-', '_')}_sample.csv"
        (sample_dir / filename).write_text(csv_text, encoding="utf-8")
        conn.execute(
            """
            INSERT INTO sensor_batches(machine_id, name, raw_csv, uploaded_at)
            VALUES (?, ?, ?, ?)
            """,
            (machine_id, f"{name} seeded sensor sample", csv_text, now),
        )

    conn.commit()
    model = ensure_model(conn)
    ensure_demo_predictions(conn, user["id"], model)


def ensure_demo_predictions(conn: sqlite3.Connection, user_id: int, model: dict) -> None:
    now = iso(utcnow())
    machines = conn.execute("SELECT * FROM machines WHERE user_id = ? ORDER BY id", (user_id,)).fetchall()
    for machine in machines:
        batch = conn.execute(
            "SELECT * FROM sensor_batches WHERE machine_id = ? ORDER BY id DESC LIMIT 1", (machine["id"],)
        ).fetchone()
        if not batch:
            continue
        latest = conn.execute(
            "SELECT * FROM predictions WHERE machine_id = ? ORDER BY id DESC LIMIT 1", (machine["id"],)
        ).fetchone()
        if latest:
            try:
                latest_reasons = json.loads(latest["reasons_json"])
                if latest_reasons and "zone" in latest_reasons[0]:
                    continue
            except Exception:
                pass
        prediction = ml.predict_from_csv(batch["raw_csv"], model)
        conn.execute(
            """
            INSERT INTO predictions(machine_id, risk, rul, health, status, reasons_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                machine["id"],
                prediction["risk"],
                prediction["rul"],
                prediction["health"],
                prediction["status"],
                json.dumps(prediction["reasons"]),
                now,
            ),
        )
    conn.commit()
