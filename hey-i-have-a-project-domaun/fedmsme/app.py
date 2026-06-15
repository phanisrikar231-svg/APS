from __future__ import annotations

import cgi
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse

from .chatbot import answer_question, machine_context
from . import ml
from .database import ROOT_DIR, get_connection, init_db
from .reports import prediction_pdf
from .security import (
    create_otp,
    create_session,
    current_user,
    expired_session_cookie,
    hash_password,
    iso,
    send_otp_email,
    session_cookie,
    utcnow,
    verify_otp,
    verify_password,
)
from .seed import DEMO_EMAIL, DEMO_PASSWORD, seed_demo_data
from .templates import (
    h,
    layout,
    metric_card,
    prediction_detail,
    public_hero,
    reasons_list,
    simple_table,
    sparkline,
    status_badge,
)


def qmsg(params: dict[str, list[str]]) -> str | None:
    values = params.get("msg")
    return values[0] if values else None


class FedMSMEHandler(BaseHTTPRequestHandler):
    server_version = "FedMSME-PdM/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[FedMSME] {self.address_string()} - {fmt % args}")

    def send_html(self, html: str, status: int = 200, cookie: str | None = None) -> None:
        data = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(data)

    def send_bytes(self, data: bytes, content_type: str, filename: str | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(data)

    def redirect(self, location: str, cookie: str | None = None) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()

    def not_found(self, user=None) -> None:
        self.send_html(layout("Not found", "<section class='panel'><h1>Page not found</h1></section>", user), 404)

    def get_user(self, conn):
        return current_user(conn, self.headers.get("Cookie"))

    def require_user(self, conn):
        user = self.get_user(conn)
        if not user:
            self.redirect("/login?msg=Please+login+to+continue")
            return None
        return user

    def parse_post(self) -> dict[str, str]:
        ctype = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", "0") or 0)
        if "multipart/form-data" in ctype:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": ctype,
                    "CONTENT_LENGTH": str(length),
                },
            )
            values: dict[str, str] = {}
            for key in form.keys():
                item = form[key]
                if isinstance(item, list):
                    item = item[0]
                if getattr(item, "filename", None):
                    raw = item.file.read()
                    values[key] = raw.decode("utf-8", errors="replace")
                else:
                    values[key] = item.value
            return values

        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        parsed = parse_qs(raw)
        return {key: values[0] for key, values in parsed.items()}

    def do_GET(self) -> None:
        init_db()
        with get_connection() as conn:
            seed_demo_data(conn)
            parsed = urlparse(self.path)
            path = parsed.path
            params = parse_qs(parsed.query)
            user = self.get_user(conn)

            if path == "/static/style.css":
                css_path = ROOT_DIR / "static" / "style.css"
                self.send_bytes(css_path.read_bytes(), "text/css; charset=utf-8")
                return
            if path.startswith("/sample_data/"):
                self.serve_sample_file(path)
                return
            if path == "/":
                self.redirect("/dashboard" if user else "/login")
                return
            if path == "/login":
                self.send_html(self.login_page(qmsg(params)))
                return
            if path == "/signup":
                self.send_html(self.signup_page(qmsg(params)))
                return
            if path == "/verify":
                self.send_html(self.verify_page(params, qmsg(params)))
                return
            if path == "/logout":
                token = self.headers.get("Cookie")
                if token:
                    from .security import token_from_cookie

                    session_token = token_from_cookie(token)
                    if session_token:
                        conn.execute("DELETE FROM sessions WHERE token = ?", (session_token,))
                        conn.commit()
                self.redirect("/login?msg=You+are+logged+out", expired_session_cookie())
                return

            user = self.require_user(conn)
            if not user:
                return
            if path == "/dashboard":
                self.send_html(self.dashboard_page(conn, user, qmsg(params)))
            elif path == "/machines":
                self.send_html(self.machines_page(conn, user, qmsg(params)))
            elif path == "/machines/new":
                self.send_html(self.new_machine_page(conn, user, qmsg(params)))
            elif path == "/machine":
                self.send_html(self.machine_page(conn, user, params, qmsg(params)))
            elif path == "/upload":
                self.send_html(self.upload_page(conn, user, params, qmsg(params)))
            elif path == "/training":
                self.send_html(self.training_page(conn, user, qmsg(params)))
            elif path == "/design":
                self.send_html(self.design_page(user, qmsg(params)))
            elif path == "/datasets":
                self.send_html(self.datasets_page(user, qmsg(params)))
            elif path == "/chat":
                self.send_html(self.chat_page(conn, user, params, qmsg(params)))
            elif path == "/report":
                self.report_response(conn, user, params)
            else:
                self.not_found(user)

    def do_POST(self) -> None:
        init_db()
        with get_connection() as conn:
            seed_demo_data(conn)
            parsed = urlparse(self.path)
            path = parsed.path
            data = self.parse_post()

            if path == "/signup":
                self.handle_signup(conn, data)
                return
            if path == "/verify":
                self.handle_verify(conn, data)
                return
            if path == "/resend-otp":
                self.handle_resend_otp(conn, data)
                return
            if path == "/login":
                self.handle_login(conn, data)
                return

            user = self.require_user(conn)
            if not user:
                return
            if path == "/machines/new":
                self.handle_new_machine(conn, user, data)
            elif path == "/upload":
                self.handle_upload(conn, user, data)
            elif path == "/predict":
                self.handle_predict(conn, user, data)
            elif path == "/demo-csv":
                self.handle_demo_csv(conn, user, data)
            elif path == "/chat":
                self.handle_chat(conn, user, data)
            elif path == "/training/run":
                self.handle_training(conn, user)
            else:
                self.not_found(user)

    def login_page(self, message: str | None = None) -> str:
        panel = f"""
        <h2>Login</h2>
        <form method="post" action="/login" class="form">
            <label>Email <input name="email" type="email" value="{h(DEMO_EMAIL)}" required></label>
            <label>Password <input name="password" type="password" value="{h(DEMO_PASSWORD)}" required></label>
            <button class="primary" type="submit">Login</button>
        </form>
        <p class="hint">Demo account: {h(DEMO_EMAIL)} / {h(DEMO_PASSWORD)}</p>
        <p class="hint">New users must verify a six-digit OTP before login.</p>
        """
        return layout("Login", public_hero("login", panel), None, message)

    def signup_page(self, message: str | None = None) -> str:
        panel = """
        <h2>Create MSME account</h2>
        <form method="post" action="/signup" class="form">
            <label>Name <input name="name" required></label>
            <label>Email <input name="email" type="email" required></label>
            <label>Company name <input name="company_name" required></label>
            <label>Phone <input name="phone"></label>
            <label>Password <input name="password" type="password" minlength="6" required></label>
            <button class="primary" type="submit">Send OTP</button>
        </form>
        <p class="hint">Without SMTP settings, this demo displays the OTP on screen for evaluation.</p>
        """
        return layout("Signup", public_hero("signup", panel), None, message)

    def verify_page(self, params: dict[str, list[str]], message: str | None = None) -> str:
        email = params.get("email", [""])[0]
        demo_otp = params.get("demo_otp", [""])[0]
        demo_box = (
            f"<div class='otp-demo'><span>Demo OTP</span><strong>{h(demo_otp)}</strong></div>" if demo_otp else ""
        )
        panel = f"""
        <h2>Verify OTP</h2>
        {demo_box}
        <form method="post" action="/verify" class="form">
            <label>Email <input name="email" type="email" value="{h(email)}" required></label>
            <label>Six-digit OTP <input name="otp" pattern="[0-9]{{6}}" inputmode="numeric" required></label>
            <button class="primary" type="submit">Verify account</button>
        </form>
        <form method="post" action="/resend-otp" class="inline-form">
            <input name="email" type="hidden" value="{h(email)}">
            <button type="submit" class="ghost">Resend OTP</button>
        </form>
        """
        return layout("Verify OTP", public_hero("verify", panel), None, message)

    def dashboard_page(self, conn, user, message: str | None = None) -> str:
        machine_count = conn.execute("SELECT COUNT(*) FROM machines WHERE user_id = ?", (user["id"],)).fetchone()[0]
        batch_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM sensor_batches b
            JOIN machines m ON m.id = b.machine_id
            WHERE m.user_id = ?
            """,
            (user["id"],),
        ).fetchone()[0]
        latest_run = conn.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT 1").fetchone()
        critical_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM predictions p
            JOIN machines m ON m.id = p.machine_id
            WHERE m.user_id = ? AND p.id IN (
                SELECT MAX(id) FROM predictions GROUP BY machine_id
            ) AND p.status = 'Critical'
            """,
            (user["id"],),
        ).fetchone()[0]
        metrics = "".join(
            [
                metric_card("Machines", str(machine_count), "registered assets"),
                metric_card("Sensor batches", str(batch_count), "CSV/demo datasets"),
                metric_card("Critical alerts", str(critical_count), "latest machine state", "danger" if critical_count else ""),
                metric_card(
                    "Global model",
                    f"{latest_run['accuracy'] * 100:.1f}%" if latest_run else "Not trained",
                    "latest federated accuracy" if latest_run else "run simulation",
                ),
            ]
        )
        machines = conn.execute(
            """
            SELECT m.*,
                   p.status, p.risk, p.health, p.rul, p.created_at AS predicted_at
            FROM machines m
            LEFT JOIN predictions p ON p.id = (
                SELECT id FROM predictions WHERE machine_id = m.id ORDER BY id DESC LIMIT 1
            )
            WHERE m.user_id = ?
            ORDER BY m.id
            """,
            (user["id"],),
        ).fetchall()
        rows = []
        for machine in machines:
            rows.append(
                [
                    f"<a href='/machine?id={machine['id']}'>{h(machine['name'])}</a>",
                    h(machine["machine_type"]),
                    status_badge(machine["status"] or "No prediction"),
                    f"{machine['risk'] * 100:.1f}%" if machine["risk"] is not None else "-",
                    f"{machine['rul']:.1f}h" if machine["rul"] is not None else "-",
                ]
            )
        table = simple_table(["Machine", "Type", "Status", "Risk", "RUL"], rows) if rows else "<p>No machines yet.</p>"
        body = f"""
        <section class="page-head">
            <div>
                <p class="eyebrow">MSME command center</p>
                <h1>Predictive maintenance dashboard</h1>
                <p>Software demo using simulated industrial sensor streams and federated model training.</p>
            </div>
            <a class="button primary" href="/machines/new">Add Machine</a>
        </section>
        <section class="metrics">{metrics}</section>
        <section class="panel">
            <div class="section-title">
                <h2>Machine health overview</h2>
                <a href="/training">View training</a>
            </div>
            {table}
        </section>
        """
        return layout("Dashboard", body, user, message)

    def machines_page(self, conn, user, message: str | None = None) -> str:
        machines = conn.execute("SELECT * FROM machines WHERE user_id = ? ORDER BY id", (user["id"],)).fetchall()
        rows = []
        for machine in machines:
            rows.append(
                [
                    f"<a href='/machine?id={machine['id']}'>{h(machine['name'])}</a>",
                    h(machine["machine_type"]),
                    h(machine["sensor_schema"]),
                    h(machine["workflow_notes"] or "-"),
                    f"<a class='mini-button' href='/upload?machine_id={machine['id']}'>Upload</a>",
                ]
            )
        table = simple_table(["Machine", "Type", "Sensors", "Workflow", "Action"], rows)
        body = f"""
        <section class="page-head">
            <div>
                <p class="eyebrow">Assets</p>
                <h1>Machines</h1>
            </div>
            <a class="button primary" href="/machines/new">Register machine</a>
        </section>
        <section class="panel">{table}</section>
        """
        return layout("Machines", body, user, message)

    def new_machine_page(self, conn, user, message: str | None = None) -> str:
        msmes = conn.execute("SELECT * FROM msmes WHERE user_id = ? ORDER BY id", (user["id"],)).fetchall()
        options = "".join(f"<option value='{m['id']}'>{h(m['name'])}</option>" for m in msmes)
        types = "".join(f"<option>{h(name)}</option>" for name in ml.MACHINE_PROFILES)
        body = f"""
        <section class="panel narrow">
            <h1>Register machine</h1>
            <form method="post" action="/machines/new" class="form">
                <label>MSME unit <select name="msme_id">{options}</select></label>
                <label>Machine name <input name="name" placeholder="CNC-07" required></label>
                <label>Machine type <select name="machine_type">{types}</select></label>
                <label>Custom machine type <input name="custom_machine_type" placeholder="Use only if Machine type is Custom / Other"></label>
                <label>Available sensors <input name="sensor_schema" placeholder="vibration, temperature, current" required></label>
                <label>Workflow notes <textarea name="workflow_notes" rows="4" placeholder="Shift pattern, batch process, maintenance cycle"></textarea></label>
                <button class="primary" type="submit">Save machine</button>
            </form>
            <p class="hint">Choose Custom / Other for machines not listed, then add the sensor names you actually have.</p>
        </section>
        """
        return layout("Register Machine", body, user, message)

    def machine_page(self, conn, user, params: dict[str, list[str]], message: str | None = None) -> str:
        machine = self.machine_for_user(conn, user, params)
        if not machine:
            return layout("Machine", "<section class='panel'><h1>Machine not found</h1></section>", user, message)
        prediction = conn.execute(
            "SELECT * FROM predictions WHERE machine_id = ? ORDER BY id DESC LIMIT 1", (machine["id"],)
        ).fetchone()
        batch = conn.execute(
            "SELECT * FROM sensor_batches WHERE machine_id = ? ORDER BY id DESC LIMIT 1", (machine["id"],)
        ).fetchone()
        if prediction:
            detail, zone = prediction_detail(prediction["status"], prediction["reasons_json"])
            risk_card = "".join(
                [
                    metric_card("Status", detail, "latest prediction", prediction["status"].lower()),
                    metric_card("Failure risk", f"{prediction['risk'] * 100:.1f}%", "probability"),
                    metric_card("Health score", f"{prediction['health']:.1f}/100", "higher is better"),
                    metric_card("RUL", f"{prediction['rul']:.1f}h", "estimated operating hours"),
                ]
            )
            explanations = reasons_list(prediction["reasons_json"])
            report_link = f"<a class='button' href='/report?machine_id={machine['id']}'>Download PDF report</a>"
            chat_link = f"<a class='button' href='/chat?machine_id={machine['id']}'>Ask machine chatbot</a>"
        else:
            risk_card = "<p>No prediction yet. Upload or run prediction using the seeded sample batch.</p>"
            explanations = ""
            report_link = ""
            chat_link = f"<a class='button' href='/chat?machine_id={machine['id']}'>Ask machine chatbot</a>"
        predict_form = f"""
        <form method="post" action="/predict" class="inline-form">
            <input type="hidden" name="machine_id" value="{machine['id']}">
            <button class="primary" type="submit">Run prediction</button>
        </form>
        """
        body = f"""
        <section class="page-head">
            <div>
                <p class="eyebrow">{h(machine['machine_type'])}</p>
                <h1>{h(machine['name'])}</h1>
                <p>{h(machine['workflow_notes'] or 'No workflow notes added.')}</p>
            </div>
            <div class="actions">
                <a class="button" href="/upload?machine_id={machine['id']}">Upload CSV</a>
                {chat_link}
                {report_link}
            </div>
        </section>
        <section class="metrics">{risk_card}</section>
        <section class="split">
            <article class="panel">
                <div class="section-title">
                    <h2>Explainable alert</h2>
                    {predict_form}
                </div>
                {explanations or '<p>Prediction explanations will appear here.</p>'}
            </article>
            <article class="panel">
                <h2>Data profile</h2>
                <dl class="details">
                    <dt>Sensors</dt><dd>{h(machine['sensor_schema'])}</dd>
                    <dt>Latest batch</dt><dd>{h(batch['name']) if batch else 'No batch uploaded'}</dd>
                    <dt>Last uploaded</dt><dd>{h(batch['uploaded_at']) if batch else '-'}</dd>
                </dl>
                <form method="post" action="/demo-csv" class="demo-actions">
                    <input type="hidden" name="machine_id" value="{machine['id']}">
                    <button name="condition" value="safe" type="submit">Generate Safe CSV</button>
                    <button name="condition" value="warning" type="submit">Generate Warning CSV</button>
                    <button name="condition" value="critical" type="submit" class="primary">Generate Critical CSV</button>
                </form>
            </article>
        </section>
        """
        return layout(machine["name"], body, user, message)

    def upload_page(self, conn, user, params: dict[str, list[str]], message: str | None = None) -> str:
        machine = self.machine_for_user(conn, user, params)
        if not machine:
            return layout("Upload", "<section class='panel'><h1>Machine not found</h1></section>", user, message)
        body = f"""
        <section class="panel narrow">
            <p class="eyebrow">{h(machine['name'])}</p>
            <h1>Upload sensor CSV</h1>
            <p class="muted">Accepted columns include vibration, temperature, current, rpm, pressure, acoustic, and load. Unknown numeric columns are mapped automatically for demo use.</p>
            <form method="post" action="/upload" enctype="multipart/form-data" class="form">
                <input type="hidden" name="machine_id" value="{machine['id']}">
                <label>Batch name <input name="name" value="{h(machine['name'])} new sensor batch" required></label>
                <label>CSV file <input name="csv_file" type="file" accept=".csv,text/csv"></label>
                <label>Or paste CSV text <textarea name="csv_text" rows="10" placeholder="timestamp,vibration,temperature,current"></textarea></label>
                <button class="primary" type="submit">Upload and predict</button>
            </form>
            <p class="hint">Sample files are available in the project sample_data folder.</p>
        </section>
        """
        return layout("Upload Sensor CSV", body, user, message)

    def training_page(self, conn, user, message: str | None = None) -> str:
        runs = conn.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT 8").fetchall()
        rows = []
        for run in runs:
            rows.append(
                [
                    h(run["created_at"]),
                    h(run["strategy"]),
                    str(run["rounds"]),
                    str(run["clients"]),
                    f"{run['accuracy'] * 100:.1f}%",
                    f"{run['loss']:.4f}",
                    f"{run['rmse']:.2f}",
                ]
            )
        table = simple_table(["Created", "Strategy", "Rounds", "Clients", "Accuracy", "Loss", "RUL RMSE"], rows)
        latest = runs[0] if runs else None
        trend = ""
        if latest:
            history = json.loads(latest["history_json"])
            trend = sparkline([float(item["loss"]) for item in reversed(history)])
        model_row = conn.execute("SELECT model_json, updated_at FROM model_store WHERE id = 1").fetchone()
        clients = []
        if model_row:
            model = ml.model_from_json(model_row["model_json"])
            for profile in model.get("client_profiles", []):
                clients.append(
                    f"<li><strong>{h(profile['name'])}</strong><span>{h(profile['machine_type'])}, {h(profile['samples'])} windows</span></li>"
                )
        body = f"""
        <section class="page-head">
            <div>
                <p class="eyebrow">Federated simulation</p>
                <h1>FedAvg vs FedProx training</h1>
                <p>Simulates five MSMEs with different machine types and sensor schemas. Raw data stays inside each client.</p>
            </div>
            <form method="post" action="/training/run">
                <button class="primary" type="submit">Run training simulation</button>
            </form>
        </section>
        <section class="split">
            <article class="panel">
                <h2>Latest convergence</h2>
                {trend or '<p>No training trend yet.</p>'}
                <p class="muted">Lower loss over federated rounds means the shared model is learning across heterogeneous clients.</p>
            </article>
            <article class="panel">
                <h2>Federated clients</h2>
                <ul class="client-list">{''.join(clients)}</ul>
            </article>
        </section>
        <section class="panel">
            <h2>Training runs</h2>
            {table}
        </section>
        """
        return layout("Federated Training", body, user, message)

    def design_page(self, user, message: str | None = None) -> str:
        body = """
        <section class="page-head">
            <div>
                <p class="eyebrow">Research design</p>
                <h1>Handling different MSME machines and workflows</h1>
                <p>This project treats different factory data as a non-IID federated learning problem.</p>
            </div>
        </section>
        <section class="panel prose">
            <h2>Why one raw model is not enough</h2>
            <p>CNC machines, pumps, motors, and compressors do not expose the same sensors. One may have vibration and RPM, another may have pressure and acoustic data. Directly forcing all raw columns into one model would make the training unstable.</p>
            <h2>What this demo does</h2>
            <p>Each client converts its raw stream into a common health-feature vector: vibration intensity, temperature trend, current fluctuation, pressure level, load, operating hours, and sensor availability flags. This gives the global model a shared language while preserving each factory's local raw data.</p>
            <h2>Why FedProx is included</h2>
            <p>FedAvg is a good baseline, but FedProx is better suited for non-IID clients because it discourages each local model from drifting too far away from the shared global model.</p>
            <h2>How to present it</h2>
            <p>The hardware layer is simulated through benchmark-style sensor CSVs. The software layer demonstrates the real research contribution: privacy-preserving collaborative learning, machine health prediction, RUL estimation, and explainable maintenance alerts.</p>
        </section>
        """
        return layout("Non-IID Design", body, user, message)

    def datasets_page(self, user, message: str | None = None) -> str:
        rows = [
            [
                "<a href='https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/' target='_blank'>NASA PCoE Data Repository</a>",
                "Bearing, milling, turbofan, battery, and other prognostics datasets.",
                "Best for academic predictive-maintenance/RUL experiments.",
            ],
            [
                "<a href='https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset' target='_blank'>UCI AI4I 2020 Predictive Maintenance</a>",
                "Synthetic industrial machine dataset with temperature, speed, torque, tool wear, and failure labels.",
                "Best starter CSV because it is small and easy to understand.",
            ],
            [
                "<a href='https://engineering.case.edu/bearingdatacenter' target='_blank'>CWRU Bearing Data Center</a>",
                "Bearing fault signals for motor/bearing condition diagnosis.",
                "Best when you want vibration-heavy fault diagnosis.",
            ],
        ]
        table = simple_table(["Source", "What it gives", "How to use it"], rows)
        body = f"""
        <section class="page-head">
            <div>
                <p class="eyebrow">Data sources</p>
                <h1>Where to get machine CSV data</h1>
                <p>You do not need to type CSV data manually. Use the built-in demo generator, sample_data files, or public predictive-maintenance datasets.</p>
            </div>
        </section>
        <section class="panel prose">
            <h2>Fastest demo path</h2>
            <p>Open any machine page and click Generate Safe CSV, Generate Warning CSV, or Generate Critical CSV. The app creates a realistic sensor batch and immediately runs prediction.</p>
            <h2>Public dataset links</h2>
            {table}
            <h2>CSV columns this app accepts</h2>
            <p>Use columns such as timestamp, vibration, temperature, current, rpm, pressure, acoustic, and load. If a dataset has different numeric column names, the app maps common aliases automatically.</p>
        </section>
        """
        return layout("Datasets", body, user, message)

    def chat_page(self, conn, user, params: dict[str, list[str]], message: str | None = None) -> str:
        machine = self.machine_for_user(conn, user, params)
        if not machine:
            return layout("Machine Chatbot", "<section class='panel'><h1>Machine not found</h1></section>", user, message)
        question = params.get("q", [""])[0]
        answer = ""
        if question:
            answer = answer_question(machine_context(conn, machine), question)
        starter_questions = [
            "Why is this machine critical?",
            "What should the technician inspect?",
            "What sensors are used?",
            "What is the remaining useful life?",
            "How does federated learning help privacy?",
        ]
        starter_links = "".join(
            f"<a class='question-chip' href='/chat?machine_id={machine['id']}&q={quote(q)}'>{h(q)}</a>"
            for q in starter_questions
        )
        chat_answer = (
            f"""
            <div class="chat-turn user-turn"><strong>You</strong><p>{h(question)}</p></div>
            <div class="chat-turn bot-turn"><strong>FedMSME Tiny Context Bot</strong><p>{h(answer)}</p></div>
            """
            if answer
            else "<p class='muted'>Ask about the machine risk, critical zone, sensor data, RUL, or maintenance action.</p>"
        )
        body = f"""
        <section class="page-head">
            <div>
                <p class="eyebrow">Local lightweight chatbot</p>
                <h1>Ask about {h(machine['name'])}</h1>
                <p>This chatbot answers only from the machine profile, latest sensor batch, prediction, and explanation data already inside the app.</p>
            </div>
            <a class="button" href="/machine?id={machine['id']}">Back to machine</a>
        </section>
        <section class="chat-layout">
            <article class="panel chat-panel">
                {chat_answer}
                <form method="get" action="/chat" class="chat-form">
                    <input type="hidden" name="machine_id" value="{machine['id']}">
                    <input name="q" placeholder="Ask: why is it critical, what should I inspect, what is the RUL..." required>
                    <button class="primary" type="submit">Ask</button>
                </form>
            </article>
            <aside class="panel">
                <h2>Try asking</h2>
                <div class="question-grid">{starter_links}</div>
                <p class="hint">The assistant is intentionally local and lightweight. It does not call an external API.</p>
            </aside>
        </section>
        """
        return layout("Machine Chatbot", body, user, message)

    def serve_sample_file(self, path: str) -> None:
        rel = unquote(path.replace("/sample_data/", "", 1))
        target = (ROOT_DIR / "sample_data" / rel).resolve()
        root = (ROOT_DIR / "sample_data").resolve()
        if not str(target).startswith(str(root)) or not target.exists():
            self.send_response(404)
            self.end_headers()
            return
        content_type = mimetypes.guess_type(str(target))[0] or "text/plain"
        self.send_bytes(target.read_bytes(), content_type, target.name)

    def machine_for_user(self, conn, user, params: dict[str, list[str]]):
        raw = params.get("machine_id", params.get("id", [""]))[0]
        try:
            machine_id = int(raw)
        except ValueError:
            return None
        return conn.execute("SELECT * FROM machines WHERE id = ? AND user_id = ?", (machine_id, user["id"])).fetchone()

    def latest_batch(self, conn, machine_id: int):
        return conn.execute(
            "SELECT * FROM sensor_batches WHERE machine_id = ? ORDER BY id DESC LIMIT 1", (machine_id,)
        ).fetchone()

    def current_model(self, conn) -> dict:
        row = conn.execute("SELECT model_json FROM model_store WHERE id = 1").fetchone()
        if not row:
            from .seed import ensure_model

            return ensure_model(conn)
        return ml.model_from_json(row["model_json"])

    def store_prediction(self, conn, machine_id: int, result: dict):
        now = iso(utcnow())
        conn.execute(
            """
            INSERT INTO predictions(machine_id, risk, rul, health, status, reasons_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                machine_id,
                result["risk"],
                result["rul"],
                result["health"],
                result["status"],
                json.dumps(result["reasons"]),
                now,
            ),
        )
        conn.commit()

    def handle_signup(self, conn, data: dict[str, str]) -> None:
        email = data.get("email", "").strip().lower()
        if not email or not data.get("password"):
            self.redirect("/signup?msg=Email+and+password+are+required")
            return
        existing = conn.execute("SELECT * FROM users WHERE lower(email) = lower(?)", (email,)).fetchone()
        if existing and existing["verified"]:
            self.redirect("/login?msg=Account+already+exists.+Please+login")
            return
        if existing:
            user_id = existing["id"]
        else:
            digest, salt = hash_password(data["password"])
            now = iso(utcnow())
            cur = conn.execute(
                """
                INSERT INTO users(name, email, password_hash, salt, company_name, phone, role, verified, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'Admin', 0, ?)
                """,
                (
                    data.get("name", "").strip() or "MSME Admin",
                    email,
                    digest,
                    salt,
                    data.get("company_name", "").strip() or "New MSME",
                    data.get("phone", "").strip(),
                    now,
                ),
            )
            user_id = cur.lastrowid
            conn.execute(
                "INSERT INTO msmes(user_id, name, industry, city, state, created_at) VALUES (?, ?, ?, '', '', ?)",
                (user_id, data.get("company_name", "").strip() or "New MSME", "Manufacturing", now),
            )
            conn.commit()

        code = create_otp(conn, user_id)
        sent = send_otp_email(email, code)
        suffix = "" if sent else f"&demo_otp={quote(code)}"
        self.redirect(f"/verify?email={quote(email)}{suffix}&msg=OTP+sent")

    def handle_verify(self, conn, data: dict[str, str]) -> None:
        ok, message = verify_otp(conn, data.get("email", ""), data.get("otp", ""))
        if ok:
            self.redirect("/login?msg=OTP+verified.+Please+login")
        else:
            self.redirect(f"/verify?email={quote(data.get('email', ''))}&msg={quote(message)}")

    def handle_resend_otp(self, conn, data: dict[str, str]) -> None:
        email = data.get("email", "").strip().lower()
        user = conn.execute("SELECT * FROM users WHERE lower(email) = lower(?)", (email,)).fetchone()
        if not user:
            self.redirect("/verify?msg=No+account+found")
            return
        code = create_otp(conn, user["id"])
        sent = send_otp_email(email, code)
        suffix = "" if sent else f"&demo_otp={quote(code)}"
        self.redirect(f"/verify?email={quote(email)}{suffix}&msg=New+OTP+sent")

    def handle_login(self, conn, data: dict[str, str]) -> None:
        email = data.get("email", "").strip().lower()
        user = conn.execute("SELECT * FROM users WHERE lower(email) = lower(?)", (email,)).fetchone()
        if not user or not verify_password(data.get("password", ""), user["password_hash"], user["salt"]):
            self.redirect("/login?msg=Invalid+email+or+password")
            return
        if not user["verified"]:
            code = create_otp(conn, user["id"])
            sent = send_otp_email(email, code)
            suffix = "" if sent else f"&demo_otp={quote(code)}"
            self.redirect(f"/verify?email={quote(email)}{suffix}&msg=Verify+OTP+before+login")
            return
        token = create_session(conn, user["id"])
        self.redirect("/dashboard?msg=Welcome+back", session_cookie(token))

    def handle_new_machine(self, conn, user, data: dict[str, str]) -> None:
        now = iso(utcnow())
        machine_type = data.get("machine_type", "General Motor")
        custom_type = data.get("custom_machine_type", "").strip()
        if machine_type == "Custom / Other" and custom_type:
            machine_type = custom_type
        conn.execute(
            """
            INSERT INTO machines(user_id, msme_id, name, machine_type, sensor_schema, workflow_notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                int(data.get("msme_id") or 0) or None,
                data.get("name", "").strip() or "New Machine",
                machine_type,
                data.get("sensor_schema", "").strip() or "temperature, current",
                data.get("workflow_notes", "").strip(),
                now,
            ),
        )
        conn.commit()
        self.redirect("/machines?msg=Machine+registered")

    def handle_upload(self, conn, user, data: dict[str, str]) -> None:
        try:
            machine_id = int(data.get("machine_id", ""))
        except ValueError:
            self.redirect("/machines?msg=Invalid+machine")
            return
        machine = conn.execute("SELECT * FROM machines WHERE id = ? AND user_id = ?", (machine_id, user["id"])).fetchone()
        if not machine:
            self.redirect("/machines?msg=Machine+not+found")
            return
        raw_csv = data.get("csv_file", "").strip() or data.get("csv_text", "").strip()
        if not raw_csv:
            self.redirect(f"/upload?machine_id={machine_id}&msg=Please+upload+or+paste+CSV+data")
            return
        try:
            ml.coerce_uploaded_csv(raw_csv)
        except Exception as exc:
            self.redirect(f"/upload?machine_id={machine_id}&msg={quote('CSV error: ' + str(exc)[:90])}")
            return
        now = iso(utcnow())
        conn.execute(
            "INSERT INTO sensor_batches(machine_id, name, raw_csv, uploaded_at) VALUES (?, ?, ?, ?)",
            (machine_id, data.get("name", "").strip() or "Uploaded sensor batch", raw_csv, now),
        )
        model = self.current_model(conn)
        prediction = ml.predict_from_csv(raw_csv, model)
        self.store_prediction(conn, machine_id, prediction)
        self.redirect(f"/machine?id={machine_id}&msg=Sensor+data+uploaded+and+prediction+refreshed")

    def handle_predict(self, conn, user, data: dict[str, str]) -> None:
        try:
            machine_id = int(data.get("machine_id", ""))
        except ValueError:
            self.redirect("/machines?msg=Invalid+machine")
            return
        machine = conn.execute("SELECT * FROM machines WHERE id = ? AND user_id = ?", (machine_id, user["id"])).fetchone()
        if not machine:
            self.redirect("/machines?msg=Machine+not+found")
            return
        batch = self.latest_batch(conn, machine_id)
        if not batch:
            self.redirect(f"/machine?id={machine_id}&msg=No+sensor+batch+available")
            return
        model = self.current_model(conn)
        prediction = ml.predict_from_csv(batch["raw_csv"], model)
        self.store_prediction(conn, machine_id, prediction)
        self.redirect(f"/machine?id={machine_id}&msg=Prediction+completed")

    def handle_demo_csv(self, conn, user, data: dict[str, str]) -> None:
        try:
            machine_id = int(data.get("machine_id", ""))
        except ValueError:
            self.redirect("/machines?msg=Invalid+machine")
            return
        machine = conn.execute("SELECT * FROM machines WHERE id = ? AND user_id = ?", (machine_id, user["id"])).fetchone()
        if not machine:
            self.redirect("/machines?msg=Machine+not+found")
            return
        condition = data.get("condition", "warning").lower()
        csv_text = ml.demo_csv_for_machine(machine["machine_type"], condition=condition, seed=machine_id * 101 + len(condition))
        now = iso(utcnow())
        conn.execute(
            "INSERT INTO sensor_batches(machine_id, name, raw_csv, uploaded_at) VALUES (?, ?, ?, ?)",
            (machine_id, f"{machine['name']} generated {condition} demo CSV", csv_text, now),
        )
        model = self.current_model(conn)
        prediction = ml.predict_from_csv(csv_text, model)
        self.store_prediction(conn, machine_id, prediction)
        self.redirect(f"/machine?id={machine_id}&msg={quote('Generated ' + condition + ' demo CSV and refreshed prediction')}")

    def handle_chat(self, conn, user, data: dict[str, str]) -> None:
        try:
            machine_id = int(data.get("machine_id", ""))
        except ValueError:
            self.redirect("/machines?msg=Invalid+machine")
            return
        question = data.get("q", "").strip()
        self.redirect(f"/chat?machine_id={machine_id}&q={quote(question)}")

    def handle_training(self, conn, user) -> None:
        fedavg, fedprox = ml.train_comparison()
        now = iso(utcnow())
        for result in (fedavg, fedprox):
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
            """
            INSERT INTO model_store(id, model_json, updated_at)
            VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET model_json = excluded.model_json, updated_at = excluded.updated_at
            """,
            (ml.model_to_json(fedprox.model), now),
        )
        conn.commit()
        self.redirect("/training?msg=Federated+simulation+completed")

    def report_response(self, conn, user, params: dict[str, list[str]]) -> None:
        machine = self.machine_for_user(conn, user, params)
        if not machine:
            self.not_found(user)
            return
        prediction = conn.execute(
            "SELECT * FROM predictions WHERE machine_id = ? ORDER BY id DESC LIMIT 1", (machine["id"],)
        ).fetchone()
        if not prediction:
            self.redirect(f"/machine?id={machine['id']}&msg=Run+a+prediction+before+downloading+the+report")
            return
        pdf = prediction_pdf(machine, prediction)
        safe_name = machine["name"].lower().replace(" ", "_").replace("-", "_")
        self.send_bytes(pdf, "application/pdf", f"{safe_name}_health_report.pdf")


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    init_db()
    with get_connection() as conn:
        seed_demo_data(conn)
    server = ThreadingHTTPServer((host, port), FedMSMEHandler)
    print(f"FedMSME-PdM running at http://{host}:{port}")
    print(f"Demo login: {DEMO_EMAIL} / {DEMO_PASSWORD}")
    server.serve_forever()
