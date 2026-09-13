"""Notifications: email wrapper + OTP make / send / verify.

The OTP is stored in a tiny SQLite table keyed by ``thread_id``. This is what
makes the step-up idempotent: LangGraph re-runs a node from the top when a run
resumes after ``interrupt()``, so the "send OTP" side effect must not fire twice.
``send_otp`` only sends when no unexpired code already exists for the thread.
"""

from __future__ import annotations

import html as _html
import re
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from fraudgraph import config, messenger

_TS_FMT = "%Y-%m-%d %H:%M:%S"


def send_email(subject: str, text_body: str, html_body: str | None = None,
               to: str | None = None) -> None:
    """Thin wrapper over the messenger module's send_email."""
    messenger.send_email(subject, text_body, html_body, to=to)


# --------------------------------------------------------------------------
# OTP store
# --------------------------------------------------------------------------
def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or config.OTP_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS otp_codes ("
        "thread_id TEXT PRIMARY KEY, code TEXT NOT NULL, "
        "expires_at TEXT NOT NULL, sent_at TEXT NOT NULL)"
    )
    conn.row_factory = sqlite3.Row
    return conn


def make_otp(length: int | None = None) -> str:
    """Generate a numeric OTP of the configured length."""
    length = length or config.OTP_LENGTH
    return "".join(secrets.choice("0123456789") for _ in range(length))


def _active_code(conn, thread_id: str, now: datetime):
    row = conn.execute(
        "SELECT code, expires_at FROM otp_codes WHERE thread_id = ?", (thread_id,)
    ).fetchone()
    if row and datetime.strptime(row["expires_at"], _TS_FMT) > now:
        return row["code"]
    return None


def send_otp(thread_id: str, email: str, db_path: Path | None = None) -> str | None:
    """Send an OTP for this thread, IDEMPOTENTLY.

    If an unexpired code already exists for ``thread_id`` (i.e. the step-up node
    is re-running on resume), no new email is sent and ``None`` is returned.
    Otherwise a fresh code is generated, emailed, stored, and returned.
    """
    now = datetime.now()
    conn = _connect(db_path)
    try:
        if _active_code(conn, thread_id, now) is not None:
            return None  # already sent for this thread — do not resend

        code = make_otp()
        expires = now + timedelta(seconds=config.OTP_TTL_SEC)
        conn.execute(
            "INSERT OR REPLACE INTO otp_codes (thread_id, code, expires_at, sent_at) "
            "VALUES (?, ?, ?, ?)",
            (thread_id, code, expires.strftime(_TS_FMT), now.strftime(_TS_FMT)),
        )
        conn.commit()
    finally:
        conn.close()

    minutes = max(1, config.OTP_TTL_SEC // 60)
    send_email(
        subject="Your verification code",
        text_body=f"Your one-time verification code is {code}. It expires in {minutes} minutes.",
        html_body=f"<p>Your one-time verification code is <b>{code}</b>.</p>"
                  f"<p>It expires in {minutes} minutes.</p>",
        to=email,
    )
    return code


def get_otp(thread_id: str, db_path: Path | None = None) -> str | None:
    """Return the active (unexpired) code for a thread, or None. (For tests/demo.)"""
    conn = _connect(db_path)
    try:
        return _active_code(conn, thread_id, datetime.now())
    finally:
        conn.close()


def verify_otp(thread_id: str, code: str, db_path: Path | None = None) -> bool:
    """True iff ``code`` matches the thread's stored, unexpired OTP."""
    conn = _connect(db_path)
    try:
        active = _active_code(conn, thread_id, datetime.now())
        return active is not None and str(code).strip() == active
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Escalation email (formatted)
# --------------------------------------------------------------------------
def _md_to_html(src: str) -> str:
    """Minimal markdown -> HTML (headings, bold, code, lists) for emails."""
    def inline(t: str) -> str:
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
        return t

    out, in_list = [], False
    for raw in _html.escape(src or "").split("\n"):
        line = raw.strip()
        if re.match(r"^#{2,4}\s+", line):
            if in_list:
                out.append("</ul>"); in_list = False
            out.append(f"<h4 style='margin:12px 0 4px;color:#10233f'>{inline(re.sub(r'^#+\s+', '', line))}</h4>")
        elif re.match(r"^([-*]|\d+\.)\s+", line):
            if not in_list:
                out.append("<ul style='margin:6px 0 6px 18px;padding:0'>"); in_list = True
            out.append(f"<li style='margin:2px 0'>{inline(re.sub(r'^([-*]|\d+\.)\s+', '', line))}</li>")
        elif line == "":
            if in_list:
                out.append("</ul>"); in_list = False
        else:
            if in_list:
                out.append("</ul>"); in_list = False
            out.append(f"<p style='margin:6px 0'>{inline(line)}</p>")
    if in_list:
        out.append("</ul>")
    return "".join(out)


def send_escalation(*, transaction: dict, risk_band=None, rule_score=None,
                    reasons=None, analyst_summary="", sar_draft="", to: str | None = None) -> None:
    """Send a formatted escalation email for a case sent to senior review."""
    txn_id = transaction.get("txn_id", "unknown")
    reasons_str = ", ".join(reasons or []) or "none"
    recipient = to or config.DEMO_USER_EMAIL or config.EMAIL_ADDRESS

    text = (
        f"FRAUD CASE ESCALATED for senior review.\n\n"
        f"Transaction : {txn_id}\n"
        f"Amount      : {transaction.get('amount')}\n"
        f"Merchant    : {transaction.get('merchant_id')}\n"
        f"Geo         : {transaction.get('geo')}\n"
        f"Risk band   : {risk_band}   Rule score: {rule_score}\n"
        f"Reason codes: {reasons_str}\n\n"
        f"ANALYST SUMMARY\n{analyst_summary or '(none)'}\n\n"
        f"SAR DRAFT\n{sar_draft or '(none)'}\n"
    )

    def row(label, value):
        return (f"<tr><td style='padding:3px 12px 3px 0;color:#5a6472'>{label}</td>"
                f"<td style='padding:3px 0;font-weight:600'>{_html.escape(str(value))}</td></tr>")

    html = f"""
    <div style="font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif;color:#1a1a1a;max-width:640px;margin:0 auto">
      <div style="background:#10233f;color:#fff;padding:14px 20px;border-radius:8px 8px 0 0">
        <h2 style="margin:0;font-size:18px">🚨 Fraud case escalated</h2>
        <p style="margin:4px 0 0;color:#b9c6db;font-size:13px">Sent to senior review by an analyst</p>
      </div>
      <div style="border:1px solid #e0e3e8;border-top:none;padding:16px 20px;border-radius:0 0 8px 8px">
        <table style="border-collapse:collapse;font-size:13px">
          {row("Transaction", txn_id)}
          {row("Amount", transaction.get("amount"))}
          {row("Merchant", transaction.get("merchant_id"))}
          {row("Geo", transaction.get("geo"))}
          {row("Risk band", risk_band)}
          {row("Rule score", rule_score)}
          {row("Reason codes", reasons_str)}
        </table>
        <h3 style="margin:16px 0 4px;font-size:14px;color:#5a6472;text-transform:uppercase;letter-spacing:.04em">Analyst summary</h3>
        <div style="background:#f7f8fa;border:1px solid #e6e9ee;border-radius:6px;padding:4px 12px;font-size:13px">
          {_md_to_html(analyst_summary) or "<p>(none)</p>"}
        </div>
        <h3 style="margin:16px 0 4px;font-size:14px;color:#5a6472;text-transform:uppercase;letter-spacing:.04em">SAR draft</h3>
        <div style="background:#f7f8fa;border:1px solid #e6e9ee;border-radius:6px;padding:4px 12px;font-size:13px">
          {_md_to_html(sar_draft) or "<p>(none)</p>"}
        </div>
        <p style="color:#8a93a0;font-size:12px;margin-top:16px">FraudGraph · the decision is set by a human or the deterministic engine, never the LLM.</p>
      </div>
    </div>
    """

    send_email(subject=f"[FraudGraph] Case escalated — {txn_id}",
               text_body=text, html_body=html, to=recipient)
