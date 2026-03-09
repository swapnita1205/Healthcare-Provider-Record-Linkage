"""Unit tests for blocking helper functions."""

from matching_utils import (
    block_key,
    canopy_candidates,
    exact_npi_block_candidates,
    first_initial,
    last_key,
    sorted_neighborhood_candidates,
)


def test_exact_npi_blocking_zero_false_negatives() -> None:
    """Exact-NPI blocking should recover all true synthetic NPI matches."""
    records_a = [{"npi": f"{1000000000 + i}"} for i in range(20)]
    records_b = [{"profile_id": f"p{i}", "npi": f"{1000000000 + i}"} for i in range(20)]
    gold = {(f"p{i}", f"{1000000000 + i}") for i in range(20)}

    pairs = exact_npi_block_candidates(records_b, records_a)
    assert gold.issubset(pairs)


def test_sorted_neighborhood_window_2() -> None:
    """Sorted neighborhood should return at most two right candidates per key."""
    left = [
        {"profile_id": "p1", "first_name": "John", "last_name": "Smith", "state": "TX"},
        {"profile_id": "p2", "first_name": "John", "last_name": "Smith", "state": "TX"},
    ]
    right = [
        {"npi": "1", "first_name": "John", "last_name": "Smith", "state": "TX"},
        {"npi": "2", "first_name": "John", "last_name": "Smith", "state": "TX"},
        {"npi": "3", "first_name": "John", "last_name": "Smith", "state": "TX"},
    ]
    pairs = sorted_neighborhood_candidates(left, right, window=2)
    expected = {("p1", "1"), ("p1", "2"), ("p2", "1"), ("p2", "2")}
    assert set(pairs) == expected


def test_canopy_blocking_threshold() -> None:
    """Canopy blocking should include candidates over threshold."""
    left = [{"profile_id": "p1", "first_name": "Alice", "last_name": "Cooper", "state": "CA"}]
    right = [
        {"npi": "n1", "first_name": "Alice", "last_name": "Cooper", "state": "CA"},
        {"npi": "n2", "first_name": "Bob", "last_name": "Jones", "state": "CA"},
    ]
    out = canopy_candidates(left, right, threshold=0.5)
    # matching_utils normalizes IDs with uppercasing via norm()
    assert any(pid == "p1" and npi == "N1" for pid, npi, _ in out)
    assert not any(pid == "p1" and npi == "N2" for pid, npi, _ in out)


def test_block_key_generation_functions() -> None:
    """Blocking key helper functions should normalize and combine values."""
    assert first_initial("john") == "J"
    assert last_key("O'Connor") == "OCONNO"
    assert block_key("tx", "78701", "SMITH") == "TX|78701|SMITH"
