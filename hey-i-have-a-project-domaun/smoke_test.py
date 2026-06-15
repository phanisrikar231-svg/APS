from __future__ import annotations

import threading
import time
import re
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from http.server import ThreadingHTTPServer

from fedmsme.app import FedMSMEHandler
from fedmsme.database import get_connection, init_db
from fedmsme.seed import DEMO_EMAIL, DEMO_PASSWORD, seed_demo_data


def fetch(opener, url: str, data: dict | None = None):
    encoded = None
    headers = {}
    if data is not None:
        encoded = urllib.parse.urlencode(data).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=encoded, headers=headers)
    return opener.open(req, timeout=20)


def main() -> None:
    init_db()
    with get_connection() as conn:
        seed_demo_data(conn)

    server = ThreadingHTTPServer(("127.0.0.1", 8765), FedMSMEHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.8)

    jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    base = "http://127.0.0.1:8765"

    assert fetch(opener, f"{base}/login").status == 200
    assert fetch(opener, f"{base}/static/style.css").status == 200
    otp_email = f"otp{int(time.time())}@example.com"
    signup = fetch(
        opener,
        f"{base}/signup",
        {
            "name": "OTP Test User",
            "email": otp_email,
            "company_name": "OTP Test MSME",
            "phone": "",
            "password": "secret123",
        },
    ).read().decode("utf-8")
    match = re.search(r"<strong>(\d{6})</strong>", signup)
    assert match, "OTP was not shown in local demo mode"
    verified = fetch(opener, f"{base}/verify", {"email": otp_email, "otp": match.group(1)}).read().decode("utf-8")
    assert "OTP verified" in verified
    login = fetch(opener, f"{base}/login", {"email": DEMO_EMAIL, "password": DEMO_PASSWORD})
    html = login.read().decode("utf-8")
    assert "Predictive maintenance dashboard" in html
    assert fetch(opener, f"{base}/machines").status == 200
    machine = fetch(opener, f"{base}/machine?id=1").read().decode("utf-8")
    assert "Explainable alert" in machine
    assert "Generate Critical CSV" in machine
    demo = fetch(opener, f"{base}/demo-csv", {"machine_id": "1", "condition": "critical"}).read().decode("utf-8")
    assert "Critical" in demo or "Warning" in demo
    datasets = fetch(opener, f"{base}/datasets").read().decode("utf-8")
    assert "Where to get machine CSV data" in datasets
    chat = fetch(opener, f"{base}/chat?machine_id=1&q=why+is+it+critical").read().decode("utf-8")
    assert "FedMSME Tiny Context Bot" in chat
    training = fetch(opener, f"{base}/training").read().decode("utf-8")
    assert "FedAvg vs FedProx" in training
    report = fetch(opener, f"{base}/report?machine_id=1")
    assert report.headers.get_content_type() == "application/pdf"
    assert len(report.read()) > 1000
    server.shutdown()
    print("Smoke test passed: OTP signup, login, dashboard, machines, training page, and PDF report are working.")


if __name__ == "__main__":
    main()
