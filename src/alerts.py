"""
Social-worker (사회복지사) alert channels.

Privacy first: by default an alert is written ONLY to a local file log on the
device. Conversation content never leaves the machine unless the operator
explicitly opts in to an external channel (Slack webhook or SMTP email) by
setting the relevant environment variables - which is only appropriate when
an authorised social-worker channel exists for the deployment.

Channels
  - LocalFileSink : always on. Appends JSONL + a human-readable .txt log.
  - SlackSink     : on only if ALERT_SLACK_WEBHOOK is set.
  - EmailSink     : on only if ALERT_SMTP_* env vars are set.
"""
from __future__ import annotations
import json
import os
import smtplib
import urllib.request
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path
from typing import List

from . import config


class LocalFileSink:
    """Always-on local log. Safe default - nothing leaves the device."""

    def __init__(self, log_dir: Path = config.ALERT_LOG_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.log_dir / "alert_log.jsonl"
        self.txt = self.log_dir / "alert_log.txt"

    @property
    def name(self) -> str:
        return f"local-file:{self.jsonl}"

    def send(self, user_name: str, urgency: str, alert_text: str) -> None:
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "user": user_name,
            "urgency": urgency,
            "alert": alert_text,
        }
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with self.txt.open("a", encoding="utf-8") as f:
            f.write(alert_text + "\n" + ("-" * 60) + "\n")


class SlackSink:
    """Opt-in: posts the alert summary to a Slack incoming webhook."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    @property
    def name(self) -> str:
        return "slack-webhook"

    def send(self, user_name: str, urgency: str, alert_text: str) -> None:
        data = json.dumps({"text": alert_text}).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url, data=data,
            headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception as exc:
            print(f"[alerts] Slack send failed: {exc.__class__.__name__}")


class EmailSink:
    """Opt-in: emails the alert summary to the social worker via SMTP."""

    def __init__(self):
        self.host = os.environ["ALERT_SMTP_HOST"]
        self.port = int(os.environ.get("ALERT_SMTP_PORT", "587"))
        self.user = os.environ["ALERT_SMTP_USER"]
        self.password = os.environ["ALERT_SMTP_PASSWORD"]
        self.to_addr = os.environ["ALERT_EMAIL_TO"]

    @property
    def name(self) -> str:
        return f"email:{self.to_addr}"

    def send(self, user_name: str, urgency: str, alert_text: str) -> None:
        msg = MIMEText(alert_text, _charset="utf-8")
        msg["Subject"] = f"[{urgency}] 독거노인 이상신호 알림 - {user_name}"
        msg["From"] = self.user
        msg["To"] = self.to_addr
        try:
            with smtplib.SMTP(self.host, self.port, timeout=15) as s:
                s.starttls()
                s.login(self.user, self.password)
                s.send_message(msg)
        except Exception as exc:
            print(f"[alerts] Email send failed: {exc.__class__.__name__}")


class AlertDispatcher:
    """Fans an alert out to every configured channel."""

    def __init__(self):
        self.sinks: List = [LocalFileSink()]           # always on
        webhook = os.environ.get("ALERT_SLACK_WEBHOOK")
        if webhook:
            self.sinks.append(SlackSink(webhook))
        if os.environ.get("ALERT_SMTP_HOST"):
            try:
                self.sinks.append(EmailSink())
            except KeyError as exc:
                print(f"[alerts] email disabled - missing env var {exc}")

    @property
    def channels(self) -> List[str]:
        return [s.name for s in self.sinks]

    def dispatch(self, user_name: str, urgency: str, alert_text: str) -> None:
        for sink in self.sinks:
            sink.send(user_name, urgency, alert_text)


if __name__ == "__main__":
    d = AlertDispatcher()
    print("active channels:", d.channels)
    d.dispatch("테스트 어르신", "주의", "[주의] 테스트 알림입니다.")
    print("wrote alert to local log:", config.ALERT_LOG_DIR)
