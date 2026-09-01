import re
import unicodedata

from anyascii import anyascii


def normalize_text(text: str) -> str:
    converted = []
    for character in str(text):
        if ord(character) < 128:
            converted.append(character)
            continue
        if "LATIN" in unicodedata.name(character, ""):
            converted.append(anyascii(character))
            continue
        converted.append(character)
    return re.sub(r"\s+", " ", "".join(converted).lower()).strip()
