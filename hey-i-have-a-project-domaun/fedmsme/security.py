from __future__ import annotations

import hashlib
import os
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from http.cookies import SimpleCookie
from typing import Optional

import sqlite3


SESSION_COOKIE = "fedmsme_session"
UTC = timezone.utc


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000
    ).hex()
    return digest, salt


def verify_password(password: str, digest: str, salt: str) -> bool:
    candidate, _ = hash_password(password, salt)
    return secrets.compare_digest(candidate, digest)


def hash_otp(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def generate_otp() -> str:
    return f"{secrets.randbelow(900000) + 100000}"


def create_session(conn: sqlite3.Connection, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = utcnow()
    conn.execute(
        "INSERT INTO sessions(token, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
        (token, user_id, iso(now + timedelta(hours=10)), iso(now)),
    )
    conn.commit()
    return token


def session_cookie(token: str) -> str:
    cookie = SimpleCookie()
    cookie[SESSION_COOKIE] = token
    cookie[SESSION_COOKIE]["path"] = "/"
    cookie[SESSION_COOKIE]["httponly"] = True
    cookie[SESSION_COOKIE]["samesite"] = "Lax"
    return cookie.output(header="").strip()


def expired_session_cookie() -> str:
    cookie = SimpleCookie()
    cookie[SESSION_COOKIE] = ""
    cookie[SESSION_COOKIE]["path"] = "/"
    cookie[SESSION_COOKIE]["max-age"] = 0
    cookie[SESSION_COOKIE]["httponly"] = True
    cookie[SESSION_COOKIE]["samesite"] = "Lax"
    return cookie.output(header="").strip()


def token_from_cookie(header: str | None) -> Optional[str]:
    if not header:
        return None
    cookie = SimpleCookie()
    cookie.load(header)
    morsel = cookie.get(SESSION_COOKIE)
    return morsel.value if morsel else None


def current_user(conn: sqlite3.Connection, cookie_header: str | None):
    token = token_from_cookie(cookie_header)
    if not token:
        return None
    row = conn.execute(
        """
        SELECT u.*
        FROM sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.token = ?
        """,
        (token,),
    ).fetchone()
    if row is None:
        return None
    session = conn.execute("SELECT expires_at FROM sessions WHERE token = ?", (token,)).fetchone()
    if session and from_iso(session["expires_at"]) < utcnow():
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        return None
    return row


def create_otp(conn: sqlite3.Connection, user_id: int, purpose: str = "signup") -> str:
    code = generate_otp()
    now = utcnow()
    conn.execute(
        """
        INSERT INTO otp_codes(user_id, code_hash, purpose, expires_at, attempts, used, created_at)
        VALUES (?, ?, ?, ?, 0, 0, ?)
        """,
        (user_id, hash_otp(code), purpose, iso(now + timedelta(minutes=10)), iso(now)),
    )
    conn.commit()
    return code


def verify_otp(conn: sqlite3.Connection, email: str, code: str, purpose: str = "signup") -> tuple[bool, str]:
    user = conn.execute("SELECT * FROM users WHERE lower(email) = lower(?)", (email,)).fetchone()
    if user is None:
        return False, "No account was found for that email."

    otp = conn.execute(
        """
        SELECT *
        FROM otp_codes
        WHERE user_id = ? AND purpose = ? AND used = 0
        ORDER BY id DESC
        LIMIT 1
        """,
        (user["id"], purpose),
    ).fetchone()
    if otp is None:
        return False, "No active OTP was found. Please request a new OTP."
    if from_iso(otp["expires_at"]) < utcnow():
        return False, "That OTP expired. Please request a new one."
    if otp["attempts"] >= 5:
        return False, "Too many wrong attempts. Please request a new OTP."

    conn.execute("UPDATE otp_codes SET attempts = attempts + 1 WHERE id = ?", (otp["id"],))
    if not secrets.compare_digest(hash_otp(code.strip()), otp["code_hash"]):
        conn.commit()
        return False, "Invalid OTP. Please try again."

    conn.execute("UPDATE otp_codes SET used = 1 WHERE id = ?", (otp["id"],))
    if purpose == "signup":
        conn.execute("UPDATE users SET verified = 1 WHERE id = ?", (user["id"],))
    conn.commit()
    return True, "OTP verified successfully."


def send_otp_email(to_email: str, code: str) -> bool:
    host = os.getenv("FEDMSME_SMTP_HOST")
    sender = os.getenv("FEDMSME_SMTP_FROM")
    username = os.getenv("FEDMSME_SMTP_USER")
    password = os.getenv("FEDMSME_SMTP_PASSWORD")
    port = int(os.getenv("FEDMSME_SMTP_PORT", "587"))
    if not host or not sender:
        return False

    msg = EmailMessage()
    msg["Subject"] = "Your FedMSME-PdM verification OTP"
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(
        f"Your FedMSME-PdM OTP is {code}. It expires in 10 minutes. "
        "If you did not request this, ignore this email."
    )
    with smtplib.SMTP(host, port, timeout=10) as smtp:
        smtp.starttls()
        if username and password:
            smtp.login(username, password)
        smtp.send_message(msg)
    return True

