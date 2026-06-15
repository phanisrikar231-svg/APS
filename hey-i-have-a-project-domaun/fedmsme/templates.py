from __future__ import annotations

import html
import json
from typing import Iterable


def h(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def layout(title: str, body: str, user=None, message: str | None = None) -> str:
    if user:
        nav = f"""
        <nav class="nav">
            <a href="/dashboard">Dashboard</a>
            <a href="/machines">Machines</a>
            <a href="/training">Federated Training</a>
            <a href="/design">Non-IID Design</a>
            <a href="/datasets">Datasets</a>
            <a href="/logout">Logout</a>
        </nav>
        """
        identity = f"""
        <div class="identity">
            <span>{h(user['company_name'])}</span>
            <small>{h(user['email'])}</small>
        </div>
        """
    else:
        nav = """
        <nav class="nav">
            <a href="/login">Login</a>
            <a href="/signup">Signup</a>
        </nav>
        """
        identity = ""

    flash = f'<div class="flash">{h(message)}</div>' if message else ""
    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{h(title)} | FedMSME-PdM</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <header class="topbar">
        <a class="brand" href="/"><span class="brand-mark">FM</span><span>FedMSME-PdM</span></a>
        {nav}
        {identity}
    </header>
    <main class="shell">
        {flash}
        {body}
    </main>
</body>
</html>"""


def public_hero(kind: str, inner: str) -> str:
    return f"""
    <section class="auth-shell">
        <div class="auth-copy">
            <p class="eyebrow">Python-only software demo</p>
            <h1>Federated predictive maintenance for MSMEs</h1>
            <p>
                Simulate different factories, train without sharing raw sensor data,
                and show machine health, RUL, failure risk, and explainable alerts.
            </p>
            <div class="hero-preview">
                <div>
                    <span>Fleet health</span>
                    <strong>87.4%</strong>
                </div>
                <div>
                    <span>Critical alerts</span>
                    <strong>02</strong>
                </div>
                <div>
                    <span>FedProx accuracy</span>
                    <strong>92%</strong>
                </div>
            </div>
            <div class="mini-grid">
                <span>OTP verification</span>
                <span>FedAvg/FedProx</span>
                <span>CSV sensor data</span>
                <span>PDF reports</span>
            </div>
        </div>
        <div class="auth-panel {h(kind)}">{inner}</div>
    </section>
    """


def metric_card(label: str, value: str, detail: str = "", tone: str = "") -> str:
    return f"""
    <article class="metric {h(tone)}">
        <span>{h(label)}</span>
        <strong>{h(value)}</strong>
        <small>{h(detail)}</small>
    </article>
    """


def status_badge(status: str) -> str:
    tone = status.lower()
    return f'<span class="badge {h(tone)}">{h(status)}</span>'


def simple_table(headers: Iterable[str], rows: Iterable[Iterable[object]]) -> str:
    head = "".join(f"<th>{h(col)}</th>" for col in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def sparkline(values: list[float], width: int = 260, height: int = 72) -> str:
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    span = hi - lo if hi != lo else 1.0
    points = []
    for idx, value in enumerate(values):
        x = idx * (width / max(len(values) - 1, 1))
        y = height - ((value - lo) / span) * (height - 8) - 4
        points.append(f"{x:.1f},{y:.1f}")
    return f"""
    <svg class="sparkline" viewBox="0 0 {width} {height}" role="img" aria-label="training trend">
        <polyline points="{' '.join(points)}"></polyline>
    </svg>
    """


def reasons_list(reasons_json: str) -> str:
    try:
        reasons = json.loads(reasons_json)
    except Exception:
        reasons = []
    if not reasons:
        return "<p>No explanations available yet.</p>"
    items = []
    for reason in reasons:
        items.append(
            f"""
            <li>
                <strong>{h(reason.get('label'))}</strong>
                <span>{h(reason.get('impact'))} risk; value {h(reason.get('value'))}</span>
                <em>{h(reason.get('zone', 'Machine-health zone'))}</em>
                <small>{h(reason.get('action', 'Review latest sensor trend and maintenance history.'))}</small>
            </li>
            """
        )
    return f'<ul class="reasons">{"".join(items)}</ul>'


def prediction_detail(status: str, reasons_json: str) -> tuple[str, str]:
    try:
        reasons = json.loads(reasons_json)
    except Exception:
        reasons = []
    zone = reasons[0].get("zone", "general machine-health zone") if reasons else "general machine-health zone"
    if status == "Critical":
        return f"Critical at {zone}", zone
    if status == "Warning":
        return f"Warning at {zone}", zone
    return f"Safe; monitor {zone}", zone
