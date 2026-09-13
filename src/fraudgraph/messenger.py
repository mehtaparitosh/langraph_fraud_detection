"""Email transport over SMTP (e.g. Gmail app password).

Interface matches the course-lab module: ``send_email(subject, text_body,
html_body)`` and ``push(text)``. Reads EMAIL_SMTP_SERVER / EMAIL_ADDRESS /
EMAIL_APP_PASSWORD from config (.env). Swap this file for your own if you prefer.
"""

from __future__ import annotations

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from fraudgraph import config

SMTP_SSL_PORT = 465


def send_email(subject: str, text_body: str, html_body: str | None = None,
               to: str | None = None) -> None:
    """Send an email via SMTP over SSL. Raises if credentials are missing."""
    sender = config.EMAIL_ADDRESS
    password = config.EMAIL_APP_PASSWORD
    server = config.EMAIL_SMTP_SERVER or "smtp.gmail.com"
    recipient = to or config.DEMO_USER_EMAIL or sender

    if not (sender and password):
        raise RuntimeError("EMAIL_ADDRESS / EMAIL_APP_PASSWORD not set in .env")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(text_body, "plain"))
    if html_body:
        msg.attach(MIMEText(html_body, "html"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(server, SMTP_SSL_PORT, context=context) as smtp:
        smtp.login(sender, password)
        smtp.sendmail(sender, [recipient], msg.as_string())


def push(text: str) -> None:
    """Fallback notification channel (no push provider wired up — just prints)."""
    print(f"[push] {text}")
