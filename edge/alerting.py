from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .utils import Event


@dataclass
class AlertConfig:
    enabled: bool = False
    provider: str = "twilio"
    twilio_from: str = ""
    twilio_to: str = ""


def send_alert(event: Event, cfg: AlertConfig, extra: dict[str, Any] | None = None) -> None:
    """Send an alert when a dangerous event is detected."""
    if not cfg.enabled:
        return

    payload = {
        "kind": event.kind,
        "confidence": event.confidence,
        "timestamp_ms": event.timestamp_ms,
        "details": event.details,
        "extra": extra or {},
        "provider": cfg.provider,
    }

    if cfg.provider.lower() == "twilio":
        _send_twilio_sms(payload, cfg)
        return

    # Fallback: log to console
    print(f"[ALERT] {payload}")


def _send_twilio_sms(payload: dict[str, Any], cfg: AlertConfig) -> None:
    try:
        from twilio.rest import Client
    except Exception as e:
        print(f"[ALERT] Twilio not available ({e}). Payload={payload}")
        return

    sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    if not sid or not token:
        print("[ALERT] Missing TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN env vars.")
        print(f"[ALERT] Payload={payload}")
        return

    if not cfg.twilio_from or not cfg.twilio_to:
        print("[ALERT] Missing alerts.twilio_from or alerts.twilio_to in config.")
        print(f"[ALERT] Payload={payload}")
        return

    text = (
        f"ElderWatch alert: {payload['kind']} "
        f"(conf={payload['confidence']:.2f}) "
        f"t={payload['timestamp_ms']}"
    )
    client = Client(sid, token)
    msg = client.messages.create(from_=cfg.twilio_from, to=cfg.twilio_to, body=text)
    print(f"[ALERT] Twilio SMS sent. sid={msg.sid}")

