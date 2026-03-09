"""Unit tests for similarity and feature engineering helpers."""

import pytest

from matching_utils import (
    FEATURE_COLUMNS,
    build_idf_from_texts,
    char3_cosine,
    features_row,
    jaro_winkler,
    lev_sim,
    soundex,
    tfidf_cosine,
    token_jaccard,
)


def test_jaro_winkler_edge_cases() -> None:
    """Validate Jaro-Winkler behavior on key edge cases."""
    assert jaro_winkler("", "") == 1.0
    assert jaro_winkler("John", "John") == 1.0
    assert 0.0 <= jaro_winkler("John", "ZZZZ") < 0.5
    assert jaro_winkler(None, "John") == 0.0  # type: ignore[arg-type]


def test_lev_sim_edge_cases() -> None:
    """Validate normalized Levenshtein similarity edge cases."""
    assert lev_sim("", "") == 1.0
    assert lev_sim("SMITH", "SMITH") == 1.0
    assert lev_sim("AAAAA", "ZZZZZ") == 0.0
    assert lev_sim(None, "A") == 0.0  # type: ignore[arg-type]


def test_soundex_edge_cases() -> None:
    """Validate Soundex outputs for common cases."""
    assert soundex("Smith") == soundex("Smyth")
    assert soundex("Smith").startswith("S")
    assert soundex("") == ""
    assert soundex(None) == ""  # type: ignore[arg-type]


def test_token_jaccard_edge_cases() -> None:
    """Validate token Jaccard behavior."""
    assert token_jaccard("", "") == 1.0
    assert token_jaccard("John Smith", "John Smith") == 1.0
    assert token_jaccard("AAA BBB", "CCC DDD") == 0.0
    assert token_jaccard(None, "ABC") == 0.0  # type: ignore[arg-type]


def test_tfidf_cosine_edge_cases() -> None:
    """Validate TF-IDF cosine similarity behavior."""
    idf = build_idf_from_texts(["john smith", "mary jones", "general hospital"])
    assert tfidf_cosine("john smith", "john smith", idf) == 1.0
    assert tfidf_cosine("", "", idf) == 1.0
    # Both texts are fully OOV for this IDF map -> empty vectors on both sides.
    assert tfidf_cosine("alpha", "omega", idf) == 1.0
    # One side empty + one side in-vocab should be dissimilar.
    assert tfidf_cosine(None, "john", idf) == 0.0  # type: ignore[arg-type]


def test_char3_cosine_edge_cases() -> None:
    """Validate char 3-gram cosine similarity behavior."""
    assert char3_cosine("JOHN", "JOHN") == pytest.approx(1.0, abs=1e-12)
    assert char3_cosine("", "") == 1.0
    assert 0.0 <= char3_cosine("AAA", "ZZZ") <= 0.01
    assert char3_cosine(None, "ABC") == 0.0  # type: ignore[arg-type]


def test_features_row_schema_and_ranges() -> None:
    """Validate full feature row output schema and ranges."""
    idf = build_idf_from_texts(
        [
            "john smith",
            "jon smyth",
            "city medical group",
            "alice cooper",
        ]
    )
    row = {
        "b_first": "John",
        "b_last": "Smith",
        "b_suffix": "",
        "b_cred": "MD",
        "b_street1": "123 Main St",
        "b_city": "Austin",
        "b_state": "TX",
        "b_zip5": "78701",
        "x_first": "Jon",
        "x_last": "Smyth",
        "x_suffix": "",
        "x_cred": "MD MPH",
        "x_street1": "123 Main Street",
        "x_city": "Austin",
        "x_state": "TX",
        "x_zip5": "78701",
    }

    feat = features_row(row, idf_name=idf)
    assert set(FEATURE_COLUMNS).issubset(set(feat.keys()))
    for key, value in feat.items():
        if key.startswith("sim_") or key == "credential_overlap":
            assert 0.0 <= float(value) <= 1.0
        if key.endswith("_match") or key.startswith("miss_"):
            assert int(value) in {0, 1}
