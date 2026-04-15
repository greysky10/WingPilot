from __future__ import annotations

import json
import sys
import urllib.request
from typing import Any


def _post_discord_content(webhook_url: str, content: str) -> bool:
    webhook = str(webhook_url or "").strip()
    message = str(content or "").strip()
    if not webhook or not message:
        return False

    try:
        body = json.dumps({"content": message}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            webhook,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "DaySpy/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5):
            return True
    except Exception as exc:
        print(f"Discord alert failed: {exc}", file=sys.stderr)
        return False


def send_discord_text_alert(webhook_url: str, message: str) -> bool:
    """Send a best-effort plain-text message to a Discord webhook."""

    return _post_discord_content(webhook_url, message)


def send_discord_json_alert(webhook_url: str, payload: dict[str, Any]) -> bool:
    """Send a best-effort JSON code-block message to a Discord webhook."""

    return _post_discord_content(
        webhook_url,
        "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```",
    )
