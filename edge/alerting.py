from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
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
        print("[ALERT] Alerts disabled in config")
        return

    payload = {
        "kind": event.kind,
        "confidence": event.confidence,
        "timestamp_ms": event.timestamp_ms,
        "details": event.details,
        "extra": extra or {},
        "provider": cfg.provider,
    }

    provider = cfg.provider.lower()
    if provider == "twilio":
        _send_twilio_message(payload, cfg, channel="sms")
        return
    if provider in {"twilio_whatsapp", "whatsapp"}:
        _send_twilio_message(payload, cfg, channel="whatsapp")
        return

    # Fallback: log to console
    print(f"[ALERT] {payload}")


def _send_twilio_message(payload: dict[str, Any], cfg: AlertConfig, channel: str) -> None:
    try:
        from twilio.rest import Client
        from twilio.base.exceptions import TwilioRestException
    except ImportError as e:
        print(f"[ALERT] FATAL: Twilio SDK not installed in this Python environment: {e}")
        print("[ALERT] Install it with: .\\.venv\\Scripts\\python -m pip install twilio")
        return

    _load_local_env()

    # Get credentials from environment
    sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    
    channel_name = "WhatsApp" if channel == "whatsapp" else "SMS"
    print(f"[ALERT] *** ATTEMPT TO SEND {channel_name.upper()} ***")
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

    if channel == "sms" and cfg.twilio_from == "+14155238886":
        print("[ALERT] FATAL: +14155238886 is not your account SMS number. Use your Twilio SMS-capable phone number.")
        return

    from_address = _format_twilio_address(cfg.twilio_from, channel)
    to_address = _format_twilio_address(cfg.twilio_to, channel)

    print(f"[ALERT] From: {from_address} -> To: {to_address}")
    
    try:
        text = (
            f"ElderWatch alert: {payload['kind']} "
            f"(conf={payload['confidence']:.2f}) "
            f"t={payload['timestamp_ms']}"
        )
        print(f"[ALERT] Message: {repr(text)}")
        
        print(f"[ALERT] Initializing Twilio Client...")
        client = Client(sid, token)
        
        print(f"[ALERT] Sending {channel_name}...")
        msg = client.messages.create(from_=from_address, to=to_address, body=text)
        
        print(f"[ALERT] *** SUCCESS: Twilio accepted {channel_name} request! SID={msg.sid}, status={msg.status} ***")
        _poll_twilio_delivery_status(client, msg.sid)
    except TwilioRestException as e:
        print(f"[ALERT] *** FAILED TO SEND SMS ***")
        print(f"[ALERT] Twilio error: status={e.status}, code={e.code}, message={e.msg}")
        if e.code == 21660:
            print(
                "[ALERT] Fix: for SMS, use a Twilio SMS-capable number owned by your account. "
                "For WhatsApp sandbox, set provider to twilio_whatsapp."
            )
    except Exception as e:
        print(f"[ALERT] *** FAILED TO SEND SMS ***")
        print(f"[ALERT] Error type: {type(e).__name__}")
        print(f"[ALERT] Error message: {str(e)}")


def _format_twilio_address(number: str, channel: str) -> str:
    number = number.strip()
    if channel == "whatsapp" and not number.startswith("whatsapp:"):
        return f"whatsapp:{number}"
    return number


def _poll_twilio_delivery_status(client: Any, message_sid: str) -> None:
    """Poll briefly so console runs show carrier delivery failures when Twilio has them."""
    terminal_statuses = {"delivered", "undelivered", "failed"}
    for delay_s in (5, 10, 20):
        print(f"[ALERT] Checking delivery status in {delay_s} seconds...")
        time.sleep(delay_s)
        msg = client.messages(message_sid).fetch()
        print(
            "[ALERT] Delivery check: "
            f"status={msg.status}, error_code={msg.error_code}, error_message={msg.error_message}"
        )
        if msg.status in terminal_statuses:
            return

    print(
        "[ALERT] Twilio still has no final delivery result. "
        "Check Twilio Console message logs for this SID and verify SMS Geo Permissions/trial recipient settings."
    )


def _load_local_env() -> None:
    """Load Twilio credentials from .env when python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    root_env = Path(__file__).resolve().parent.parent / ".env"
    edge_env = Path(__file__).resolve().parent / ".env"
    load_dotenv(root_env)
    load_dotenv(edge_env, override=True)


