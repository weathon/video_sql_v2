import re
import unicodedata
import json

from anyascii import anyascii


def normalize_text(text: str) -> str:
    chars = []
    for char in unicodedata.normalize("NFC", text):
        if "LATIN" in unicodedata.name(char, ""):
            chars.append(anyascii(char).lower())
        else:
            chars.append(char)
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def normalize_triple_atom(value: str | int | float | bool | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_text(value)
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) in (int, float):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    raise TypeError(f"unsupported semantic triple atom type: {type(value).__name__}")


def parse_hms(value: str) -> float:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"expected HH:MM:SS, received {value!r}")
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_time_range(value: str) -> tuple[float, float]:
    start, end = value.split("-", 1)
    return parse_hms(start), parse_hms(end)
