import re
import unicodedata


ASCII_PUNCTUATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "…": "...",
    }
)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).translate(ASCII_PUNCTUATION)
    characters = []
    for character in normalized:
        if unicodedata.combining(character):
            continue
        if ord(character) < 128 or not character.isascii():
            characters.append(character)
    return re.sub(r"\s+", " ", "".join(characters).lower()).strip()
