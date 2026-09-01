from src.preprocess.normalize import normalize_text


def test_normalize_text_preserves_non_latin_and_normalizes_latin():
    assert normalize_text("  Café  中文\tHELLO  ") == "cafe 中文 hello"
