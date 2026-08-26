from src.preprocess.normalize import normalize_text, normalize_triple_atom, parse_hms, parse_time_range


def test_normalize_latin_and_preserve_non_latin():
    assert normalize_text("  Café\tStraße Łódź 東京  ") == "cafe strasse lodz 東京"


def test_parse_times():
    assert parse_hms("01:02:03.5") == 3723.5
    assert parse_time_range("00:01:00-00:02:30") == (60, 150)


def test_normalize_semantic_triple_atoms():
    assert normalize_triple_atom(None) == ""
    assert normalize_triple_atom(True) == "true"
    assert normalize_triple_atom(False) == "false"
    assert normalize_triple_atom(99) == "99"
    assert normalize_triple_atom(1.5) == "1.5"
