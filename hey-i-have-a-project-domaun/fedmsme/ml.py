from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from typing import Iterable

import numpy as np
import pandas as pd


UTC = timezone.utc

SENSOR_COLUMNS = ["vibration", "temperature", "current", "rpm", "pressure", "acoustic", "load"]

FEATURE_COLUMNS = [
    "vibration_rms",
    "vibration_std",
    "temperature_mean",
    "temperature_slope",
    "current_mean",
    "current_std",
    "rpm_mean",
    "pressure_mean",
    "acoustic_rms",
    "load_mean",
    "operating_hours",
    "has_vibration",
    "has_temperature",
    "has_current",
    "has_rpm",
    "has_pressure",
    "has_acoustic",
    "has_load",
]

FEATURE_LABELS = {
    "vibration_rms": "vibration intensity",
    "vibration_std": "vibration instability",
    "temperature_mean": "average temperature",
    "temperature_slope": "temperature rising trend",
    "current_mean": "average electrical current",
    "current_std": "current fluctuation",
    "rpm_mean": "rotation speed",
    "pressure_mean": "pressure level",
    "acoustic_rms": "acoustic noise intensity",
    "load_mean": "machine load",
    "operating_hours": "operating hours",
    "has_vibration": "vibration sensor availability",
    "has_temperature": "temperature sensor availability",
    "has_current": "current sensor availability",
    "has_rpm": "RPM sensor availability",
    "has_pressure": "pressure sensor availability",
    "has_acoustic": "acoustic sensor availability",
    "has_load": "load sensor availability",
}

MACHINE_PROFILES = {
    "CNC Milling": ["vibration", "temperature", "current", "rpm", "load"],
    "Lathe Machine": ["vibration", "temperature", "current", "rpm", "load"],
    "Drilling Machine": ["vibration", "temperature", "current", "rpm", "load"],
    "Textile Motor": ["vibration", "temperature", "current", "rpm"],
    "Injection Molding Machine": ["temperature", "pressure", "current", "load"],
    "Hydraulic Press": ["pressure", "temperature", "current", "load"],
    "Pump Unit": ["pressure", "vibration", "current", "temperature", "load"],
    "Air Compressor": ["pressure", "acoustic", "temperature", "current"],
    "Conveyor Belt Motor": ["vibration", "temperature", "current", "rpm", "load"],
    "Packaging Machine": ["vibration", "temperature", "current", "rpm", "load"],
    "Cooling Fan": ["vibration", "temperature", "current", "rpm"],
    "Industrial Oven": ["temperature", "current", "load"],
    "Welding Transformer": ["temperature", "current", "load"],
    "Boiler Feed Pump": ["pressure", "vibration", "temperature", "current", "load"],
    "Gearbox Assembly": ["vibration", "acoustic", "temperature", "rpm", "load"],
    "General Motor": ["vibration", "temperature", "current", "rpm"],
    "Custom / Other": ["vibration", "temperature", "current", "rpm", "pressure", "acoustic", "load"],
}

DIAGNOSIS_RULES = {
    "vibration_rms": ("Bearing / shaft vibration zone", "Inspect bearings, shaft alignment, foundation looseness, and lubrication."),
    "vibration_std": ("Vibration stability zone", "Check imbalance, looseness, bearing wear, coupling misalignment, and resonance."),
    "temperature_mean": ("Thermal / cooling zone", "Inspect cooling fan, lubrication, friction points, blocked vents, and overload heating."),
    "temperature_slope": ("Thermal rise trend", "Temperature is rising over time; check cooling, lubrication, and recent load changes."),
    "current_mean": ("Electrical load zone", "Check motor overcurrent, supply imbalance, winding stress, and abnormal mechanical load."),
    "current_std": ("Electrical fluctuation zone", "Look for unstable load, loose wiring, voltage dips, or intermittent motor stress."),
    "rpm_mean": ("Speed / drive zone", "Check belt slip, drive control, gearbox, spindle speed, or motor speed instability."),
    "pressure_mean": ("Hydraulic / pneumatic pressure zone", "Inspect pump/compressor pressure, valve blockage, leakage, and line restrictions."),
    "acoustic_rms": ("Acoustic noise zone", "Check abnormal noise from bearings, gears, compressor valves, cavitation, or mechanical rubbing."),
    "load_mean": ("Mechanical load zone", "Review workload, jamming, friction, tool wear, and process overload conditions."),
    "operating_hours": ("Usage age zone", "Schedule inspection based on operating hours and recent degradation trend."),
}


@dataclass
class TrainingResult:
    strategy: str
    rounds: int
    clients: int
    accuracy: float
    loss: float
    rmse: float
    history: list[dict]
    model: dict


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -45, 45)
    return 1.0 / (1.0 + np.exp(-values))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    aliases = {
        "vib": "vibration",
        "vibration_rms": "vibration",
        "temp": "temperature",
        "motor_temp": "temperature",
        "amps": "current",
        "ampere": "current",
        "motor_current": "current",
        "speed": "rpm",
        "rotational_speed": "rpm",
        "press": "pressure",
        "sound": "acoustic",
        "noise": "acoustic",
        "machine_load": "load",
    }
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        renamed[col] = aliases.get(key, key)
    return df.rename(columns=renamed)


def coerce_uploaded_csv(raw_csv: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(raw_csv.strip()))
    df = normalize_columns(df)

    numeric_cols = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= max(3, len(converted) // 4):
            df[col] = converted.interpolate(limit_direction="both").fillna(converted.median())
            numeric_cols.append(col)

    recognized = [col for col in SENSOR_COLUMNS if col in numeric_cols]
    if not recognized:
        # Practical demo fallback: map arbitrary numeric sensor columns by order.
        mapping_order = ["vibration", "temperature", "current", "rpm", "load", "pressure", "acoustic"]
        for source, target in zip(numeric_cols, mapping_order):
            df[target] = df[source]
    return df


def rms(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def slope(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2:
        return 0.0
    x = np.arange(arr.size, dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def extract_features(df: pd.DataFrame) -> dict[str, float]:
    df = normalize_columns(df.copy())
    features = {name: 0.0 for name in FEATURE_COLUMNS}
    row_count = max(len(df), 1)

    if "vibration" in df:
        values = pd.to_numeric(df["vibration"], errors="coerce").dropna()
        features["vibration_rms"] = rms(values)
        features["vibration_std"] = float(values.std(ddof=0) if len(values) else 0.0)
        features["has_vibration"] = 1.0
    if "temperature" in df:
        values = pd.to_numeric(df["temperature"], errors="coerce").dropna()
        features["temperature_mean"] = float(values.mean() if len(values) else 0.0)
        features["temperature_slope"] = slope(values.tail(min(len(values), 30)))
        features["has_temperature"] = 1.0
    if "current" in df:
        values = pd.to_numeric(df["current"], errors="coerce").dropna()
        features["current_mean"] = float(values.mean() if len(values) else 0.0)
        features["current_std"] = float(values.std(ddof=0) if len(values) else 0.0)
        features["has_current"] = 1.0
    if "rpm" in df:
        values = pd.to_numeric(df["rpm"], errors="coerce").dropna()
        features["rpm_mean"] = float(values.mean() if len(values) else 0.0)
        features["has_rpm"] = 1.0
    if "pressure" in df:
        values = pd.to_numeric(df["pressure"], errors="coerce").dropna()
        features["pressure_mean"] = float(values.mean() if len(values) else 0.0)
        features["has_pressure"] = 1.0
    if "acoustic" in df:
        values = pd.to_numeric(df["acoustic"], errors="coerce").dropna()
        features["acoustic_rms"] = rms(values)
        features["has_acoustic"] = 1.0
    if "load" in df:
        values = pd.to_numeric(df["load"], errors="coerce").dropna()
        features["load_mean"] = float(values.mean() if len(values) else 0.0)
        features["has_load"] = 1.0

    features["operating_hours"] = float(row_count / 60.0)
    return features


def feature_vector(features: dict[str, float]) -> np.ndarray:
    return np.asarray([float(features.get(name, 0.0)) for name in FEATURE_COLUMNS], dtype=float)


def make_raw_timeseries(machine_type: str, rows: int = 180, seed: int = 1, degradation: float = 0.35) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sensors = MACHINE_PROFILES.get(machine_type, MACHINE_PROFILES["General Motor"])
    t = np.arange(rows)
    d = np.clip(degradation + (t / max(rows - 1, 1)) * 0.22, 0.0, 1.0)
    data = {"timestamp": pd.date_range("2026-01-01", periods=rows, freq="min")}

    if "vibration" in sensors:
        data["vibration"] = np.abs(rng.normal(0.18 + 2.8 * d**2, 0.04 + 0.30 * d, rows))
    if "temperature" in sensors:
        data["temperature"] = rng.normal(39 + 34 * d + 3.5 * np.sin(t / 18), 1.4 + d, rows)
    if "current" in sensors:
        data["current"] = np.abs(rng.normal(6.8 + 5.2 * d, 0.45 + 0.7 * d, rows))
    if "rpm" in sensors:
        data["rpm"] = rng.normal(1480 - 260 * d + 18 * np.sin(t / 15), 16 + 10 * d, rows)
    if "pressure" in sensors:
        data["pressure"] = np.abs(rng.normal(3.4 + 2.7 * d + 0.4 * np.sin(t / 13), 0.18 + 0.45 * d, rows))
    if "acoustic" in sensors:
        data["acoustic"] = np.abs(rng.normal(0.22 + 2.4 * d**2, 0.08 + 0.25 * d, rows))
    if "load" in sensors:
        data["load"] = np.clip(rng.normal(0.48 + 0.34 * d, 0.05 + 0.04 * d, rows), 0, 1.25)

    return pd.DataFrame(data)


def timeseries_to_csv(machine_type: str, seed: int = 1, degradation: float = 0.35) -> str:
    return make_raw_timeseries(machine_type, seed=seed, degradation=degradation).to_csv(index=False)


def demo_csv_for_machine(machine_type: str, condition: str = "warning", seed: int = 99) -> str:
    condition_map = {
        "safe": 0.18,
        "warning": 0.48,
        "critical": 0.78,
    }
    degradation = condition_map.get(condition.lower(), 0.48)
    return timeseries_to_csv(machine_type, seed=seed, degradation=degradation)


def synthetic_client(machine_type: str, client_name: str, samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    xs = []
    ys = []
    ruls = []
    for i in range(samples):
        degradation = float(np.clip(rng.beta(2.0, 2.3), 0.02, 0.98))
        raw = make_raw_timeseries(
            machine_type,
            rows=int(rng.integers(72, 220)),
            seed=int(seed * 10_000 + i),
            degradation=degradation,
        )
        feats = extract_features(raw)
        x = feature_vector(feats)
        sensor_penalty = 0.04 * (7 - sum(feats[f"has_{name}"] for name in SENSOR_COLUMNS))
        risk = np.clip(
            0.14
            + 0.56 * degradation
            + 0.10 * feats["vibration_std"]
            + 0.006 * max(feats["temperature_mean"] - 45, 0)
            + sensor_penalty
            + rng.normal(0, 0.035),
            0,
            1,
        )
        label = 1 if risk >= 0.58 else 0
        rul = max(4.0, 145.0 * (1.0 - risk) + rng.normal(0, 7.5))
        xs.append(x)
        ys.append(label)
        ruls.append(rul)
    return {
        "name": client_name,
        "machine_type": machine_type,
        "X": np.vstack(xs),
        "y": np.asarray(ys, dtype=float),
        "rul": np.asarray(ruls, dtype=float),
    }


def make_federated_clients(samples_per_client: int = 90) -> list[dict]:
    return [
        synthetic_client("CNC Milling", "Rajkot CNC Unit", samples_per_client, 11),
        synthetic_client("Textile Motor", "Surat Textile Motor Line", samples_per_client, 17),
        synthetic_client("Pump Unit", "Coimbatore Pump Workshop", samples_per_client, 23),
        synthetic_client("Air Compressor", "Pune Compressor Cell", samples_per_client, 31),
        synthetic_client("General Motor", "Ludhiana General Motor Shop", samples_per_client, 43),
    ]


def scale_clients(clients: list[dict]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    all_x = np.vstack([client["X"] for client in clients])
    mean = all_x.mean(axis=0)
    std = all_x.std(axis=0)
    std[std < 1e-6] = 1.0
    scaled = []
    for client in clients:
        clone = dict(client)
        clone["X"] = (client["X"] - mean) / std
        scaled.append(clone)
    return scaled, mean, std


def bce_loss(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    p = sigmoid(x @ w + b)
    eps = 1e-8
    return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


def local_update(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    epochs: int,
    lr: float,
    global_w: np.ndarray,
    global_b: float,
    mu: float,
) -> tuple[np.ndarray, float]:
    w_local = w.copy()
    b_local = float(b)
    for _ in range(epochs):
        p = sigmoid(x @ w_local + b_local)
        grad_w = (x.T @ (p - y)) / len(y)
        grad_b = float(np.mean(p - y))
        if mu:
            grad_w += mu * (w_local - global_w)
            grad_b += mu * (b_local - global_b)
        w_local -= lr * grad_w
        b_local -= lr * grad_b
    return w_local, b_local


def evaluate(x: np.ndarray, y: np.ndarray, rul: np.ndarray, w: np.ndarray, b: float) -> dict:
    p = sigmoid(x @ w + b)
    pred = (p >= 0.5).astype(float)
    accuracy = float((pred == y).mean())
    loss = bce_loss(x, y, w, b)
    rul_pred = np.maximum(3.0, 145.0 * (1.0 - p) + 5.0)
    rmse = float(math.sqrt(np.mean((rul_pred - rul) ** 2)))
    return {"accuracy": accuracy, "loss": loss, "rmse": rmse}


def train_strategy(strategy: str = "FedProx", rounds: int = 10, local_epochs: int = 3) -> TrainingResult:
    clients, mean, std = scale_clients(make_federated_clients())
    all_x = np.vstack([client["X"] for client in clients])
    all_y = np.concatenate([client["y"] for client in clients])
    all_rul = np.concatenate([client["rul"] for client in clients])

    w = np.zeros(len(FEATURE_COLUMNS), dtype=float)
    b = 0.0
    lr = 0.09
    mu = 0.14 if strategy.lower() == "fedprox" else 0.0
    history = []

    for rnd in range(1, rounds + 1):
        updates = []
        total = 0
        for client in clients:
            wc, bc = local_update(
                client["X"],
                client["y"],
                w,
                b,
                epochs=local_epochs,
                lr=lr,
                global_w=w,
                global_b=b,
                mu=mu,
            )
            size = len(client["y"])
            updates.append((wc, bc, size, client["name"]))
            total += size
        w = sum((wc * size for wc, _, size, _ in updates), np.zeros_like(w)) / total
        b = float(sum((bc * size for _, bc, size, _ in updates)) / total)
        metrics = evaluate(all_x, all_y, all_rul, w, b)
        history.append(
            {
                "round": rnd,
                "loss": round(metrics["loss"], 4),
                "accuracy": round(metrics["accuracy"], 4),
                "rmse": round(metrics["rmse"], 3),
            }
        )

    metrics = evaluate(all_x, all_y, all_rul, w, b)
    model = {
        "strategy": strategy,
        "feature_columns": FEATURE_COLUMNS,
        "feature_labels": FEATURE_LABELS,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": w.tolist(),
        "bias": b,
        "trained_at": now_iso(),
        "client_profiles": [
            {"name": client["name"], "machine_type": client["machine_type"], "samples": len(client["y"])}
            for client in clients
        ],
    }
    return TrainingResult(
        strategy=strategy,
        rounds=rounds,
        clients=len(clients),
        accuracy=float(metrics["accuracy"]),
        loss=float(metrics["loss"]),
        rmse=float(metrics["rmse"]),
        history=history,
        model=model,
    )


def train_comparison() -> tuple[TrainingResult, TrainingResult]:
    fedavg = train_strategy("FedAvg", rounds=8, local_epochs=3)
    fedprox = train_strategy("FedProx", rounds=8, local_epochs=3)
    return fedavg, fedprox


def model_to_json(model: dict) -> str:
    return json.dumps(model, separators=(",", ":"))


def model_from_json(value: str) -> dict:
    return json.loads(value)


def predict_from_csv(raw_csv: str, model: dict) -> dict:
    df = coerce_uploaded_csv(raw_csv)
    features = extract_features(df)
    x = feature_vector(features)
    mean = np.asarray(model["mean"], dtype=float)
    std = np.asarray(model["std"], dtype=float)
    w = np.asarray(model["weights"], dtype=float)
    b = float(model["bias"])
    xs = (x - mean) / std
    risk = float(sigmoid(np.asarray([xs @ w + b]))[0])
    health = float(np.clip(100.0 * (1.0 - risk), 0.0, 100.0))
    rul = float(max(3.0, 145.0 * (1.0 - risk) + 5.0))
    status = "Safe" if risk < 0.38 else "Warning" if risk < 0.66 else "Critical"

    contributions = xs * w
    ranked = np.argsort(np.abs(contributions))[::-1][:7]
    reasons = []
    diagnosis = []
    for idx in ranked:
        feature = FEATURE_COLUMNS[int(idx)]
        if feature.startswith("has_"):
            continue
        value = float(features.get(feature, 0.0))
        direction = "increased" if contributions[idx] > 0 else "reduced"
        zone, action = DIAGNOSIS_RULES.get(
            feature,
            ("General machine-health zone", "Review the latest sensor batch and maintenance history."),
        )
        reasons.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "value": round(value, 3),
                "impact": direction,
                "score": round(float(contributions[idx]), 4),
                "zone": zone,
                "action": action,
            }
        )
        if direction == "increased":
            diagnosis.append(
                {
                    "zone": zone,
                    "feature": FEATURE_LABELS.get(feature, feature),
                    "severity": "Critical" if risk >= 0.66 and len(diagnosis) == 0 else "Warning",
                    "action": action,
                }
            )
        if len(reasons) == 4:
            break

    primary_zone = diagnosis[0]["zone"] if diagnosis else "General machine-health zone"
    status_detail = (
        f"Critical at {primary_zone}"
        if status == "Critical"
        else f"Warning at {primary_zone}"
        if status == "Warning"
        else f"Safe; monitor {primary_zone}"
    )

    return {
        "risk": round(risk, 4),
        "health": round(health, 2),
        "rul": round(rul, 1),
        "status": status,
        "status_detail": status_detail,
        "critical_zone": primary_zone,
        "features": {key: round(float(value), 4) for key, value in features.items()},
        "reasons": reasons,
        "diagnosis": diagnosis[:3],
    }
