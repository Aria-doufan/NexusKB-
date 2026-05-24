import hashlib
import json
import re
from typing import Any

from app.core.logger_handler import logger


SENSITIVE_KEYWORDS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
    "jwt",
    "access_key",
    "refresh_token",
    "connection_string",
}


def text_summary(value: str, max_chars: int = 200) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    if len(normalized) <= max_chars:
        return normalized
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"{normalized[:max_chars]}...<len={len(normalized)} sha256={digest}>"


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def summarize_for_audit(value: Any, max_chars: int = 200, max_items: int = 8) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value

    if isinstance(value, str):
        return text_summary(value, max_chars=max_chars)

    if isinstance(value, dict):
        summarized: dict[str, Any] = {}
        for key, item in list(value.items())[:max_items]:
            key_text = str(key)
            if any(keyword in key_text.lower() for keyword in SENSITIVE_KEYWORDS):
                summarized[key_text] = "<redacted>"
            else:
                summarized[key_text] = summarize_for_audit(item, max_chars=max_chars, max_items=max_items)
        if len(value) > max_items:
            summarized["_truncated_keys"] = len(value) - max_items
        return summarized

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        summarized_items = [
            summarize_for_audit(item, max_chars=max_chars, max_items=max_items)
            for item in items[:max_items]
        ]
        if len(items) > max_items:
            summarized_items.append(f"<truncated_items={len(items) - max_items}>")
        return summarized_items

    return text_summary(str(value), max_chars=max_chars)


def log_audit_event(name: str, **fields: Any) -> None:
    parts = [f"name={name}"]
    for key, value in fields.items():
        if value is None:
            continue
        if any(keyword in key.lower() for keyword in SENSITIVE_KEYWORDS):
            safe_value = "<redacted>"
        else:
            safe_value = summarize_for_audit(value)
        parts.append(
            f"{key}={json.dumps(safe_value, ensure_ascii=False, separators=(',', ':'))}"
        )
    logger.info("AUDIT_EVENT %s", " ".join(parts))
