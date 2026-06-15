from __future__ import annotations

import json
import re

from . import ml


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _safe_json(value: str, fallback):
    try:
        return json.loads(value)
    except Exception:
        return fallback


def machine_context(conn, machine) -> dict:
    prediction = conn.execute(
        "SELECT * FROM predictions WHERE machine_id = ? ORDER BY id DESC LIMIT 1", (machine["id"],)
    ).fetchone()
    batch = conn.execute(
        "SELECT * FROM sensor_batches WHERE machine_id = ? ORDER BY id DESC LIMIT 1", (machine["id"],)
    ).fetchone()
    features = {}
    if batch:
        try:
            features = ml.extract_features(ml.coerce_uploaded_csv(batch["raw_csv"]))
        except Exception:
            features = {}
    reasons = _safe_json(prediction["reasons_json"], []) if prediction else []
    return {
        "machine": machine,
        "prediction": prediction,
        "batch": batch,
        "features": features,
        "reasons": reasons,
    }


def answer_question(context: dict, question: str) -> str:
    machine = context["machine"]
    prediction = context["prediction"]
    batch = context["batch"]
    features = context["features"]
    reasons = context["reasons"]
    tokens = _tokens(question)

    if not prediction:
        return (
            f"{machine['name']} has no prediction yet. Upload sensor CSV data or generate demo data, "
            "then run prediction so I can discuss risk, RUL, and critical zones."
        )

    risk_pct = prediction["risk"] * 100
    base = (
        f"{machine['name']} is a {machine['machine_type']} with status {prediction['status']}. "
        f"Failure risk is {risk_pct:.1f}%, health score is {prediction['health']:.1f}/100, "
        f"and estimated RUL is {prediction['rul']:.1f} operating hours."
    )
    primary = reasons[0] if reasons else {}
    zone = primary.get("zone", "general machine-health zone")
    action = primary.get("action", "Review the latest sensor batch and maintenance history.")

    if tokens & {"why", "reason", "reasons", "critical", "problem", "issue", "fault", "where"}:
        detail = []
        for reason in reasons[:3]:
            detail.append(
                f"{reason.get('label')} ({reason.get('zone', 'machine-health zone')}) "
                f"{reason.get('impact', 'affected')} risk with value {reason.get('value')}."
            )
        joined = " ".join(detail) if detail else "No detailed reason was recorded."
        return f"{base} The main concern is the {zone}. {joined} Recommended action: {action}"

    if tokens & {"rul", "life", "remaining", "hours", "long"}:
        return (
            f"{machine['name']} has an estimated RUL of {prediction['rul']:.1f} operating hours. "
            f"Because the main risk zone is {zone}, plan inspection around that area before the RUL window closes."
        )

    if tokens & {"sensor", "sensors", "data", "csv", "reading", "readings"}:
        available = [
            label
            for label, flag in [
                ("vibration", features.get("has_vibration")),
                ("temperature", features.get("has_temperature")),
                ("current", features.get("has_current")),
                ("rpm", features.get("has_rpm")),
                ("pressure", features.get("has_pressure")),
                ("acoustic", features.get("has_acoustic")),
                ("load", features.get("has_load")),
            ]
            if flag
        ]
        batch_name = batch["name"] if batch else "no batch"
        return (
            f"The latest batch is '{batch_name}'. Available sensor channels are: "
            f"{', '.join(available) if available else 'not detected'}. The app converts these into common "
            "health features so different MSME machines can still be compared."
        )

    if tokens & {"fix", "maintenance", "repair", "action", "technician", "inspect"}:
        return (
            f"Recommended maintenance for {machine['name']}: {action} "
            f"Also check the top risk zone: {zone}. If status is Critical, inspect before the next production cycle."
        )

    if tokens & {"federated", "privacy", "training", "model", "learn"}:
        return (
            "This app uses a federated-learning simulation. Each MSME keeps raw CSV data locally, converts it "
            "into health features, trains locally, and only model updates are aggregated. FedProx is used because "
            "different machines create non-IID data."
        )

    return (
        f"{base} The most important zone is {zone}. Ask me things like: "
        "'why is it critical?', 'what should I inspect?', 'what sensors are used?', or 'what is the RUL?'"
    )

