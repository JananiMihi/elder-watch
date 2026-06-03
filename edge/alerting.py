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
    except ImportError as e:
        print(f"[ALERT] FATAL: Twilio SDK not installed: {e}")
        return

    # Get credentials from environment
    sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    
    print(f"[ALERT] *** ATTEMPT TO SEND SMS ***")
    print(f"[ALERT] Credentials: SID={sid[:10]}... (len={len(sid)}), Token=****** (len={len(token)})")
    
    # Validate credentials
    if not sid:
        print(f"[ALERT] FATAL: TWILIO_ACCOUNT_SID not set in environment")
        return
    if not token:
        print(f"[ALERT] FATAL: TWILIO_AUTH_TOKEN not set in environment")
        return

    # Validate config
    if not cfg.twilio_from:
        print(f"[ALERT] FATAL: twilio_from not set in config")
        return
    if not cfg.twilio_to:
        print(f"[ALERT] FATAL: twilio_to not set in config")
        return

    print(f"[ALERT] From: {cfg.twilio_from} -> To: {cfg.twilio_to}")
    
    try:
        text = (
            f"ElderWatch alert: {payload['kind']} "
            f"(conf={payload['confidence']:.2f}) "
            f"t={payload['timestamp_ms']}"
        )
        print(f"[ALERT] Message: {repr(text)}")
        
        print(f"[ALERT] Initializing Twilio Client...")
        client = Client(sid, token)
        
        print(f"[ALERT] Sending SMS...")
        msg = client.messages.create(from_=cfg.twilio_from, to=cfg.twilio_to, body=text)
        
        print(f"[ALERT] *** SUCCESS: SMS sent! SID={msg.sid} ***")
    except Exception as e:
        print(f"[ALERT] *** FAILED TO SEND SMS ***")
        print(f"[ALERT] Error type: {type(e).__name__}")
        print(f"[ALERT] Error message: {str(e)}")
        import traceback
        traceback.print_exc()


