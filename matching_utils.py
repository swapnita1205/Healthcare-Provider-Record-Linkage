"""Reusable matching and blocking utilities for linkage inference.

This module mirrors feature and blocking logic used in the training pipeline,
without side effects (no file I/O at import time). It is shared by the API
and tests to keep behavior consistent.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Any

import numpy as np

_ws = re.compile(r"\s+")
_non_alnum = re.compile(r"[^A-Z0-9 ]+")

ORG_WORDS = {
    "LLC",
    "INC",
    "LTD",
    "CORP",
    "CORPORATION",
    "COMPANY",
    "CO",
    "GROUP",
    "ASSOCIATES",
    "ASSOC",
    "HOSPITAL",
    "HEALTH",
    "HEALTHCARE",
    "CLINIC",
    "CENTER",
    "CENTRE",
    "MEDICAL",
    "MED",
    "SYSTEM",
    "FOUNDATION",
    "UNIVERSITY",
    "DEPARTMENT",
    "DEPT",
    "PRACTICE",
    "INSTITUTE",
    "LAB",
    "LABS",
}
SUFFIXES = {"JR", "SR", "II", "III", "IV", "V"}
CRED_TOKS = {"MD", "DO", "DDS", "DMD", "NP", "PA", "RN", "PHD", "MBA", "MS", "MPH", "DPM", "OD"}
CHAR_DIM = 2**14

FEATURE_COLUMNS: list[str] = [
    "sim_jw_fullname",
    "sim_jw_lastname",
    "sim_lev_fullname",
    "sim_lev_lastname",
    "sim_soundex_last",
    "sim_jacc_fullname",
    "sim_jacc_lastname",
    "sim_tfidf_name",
    "sim_char3_name",
    "first_initial_match",
    "state_match",
    "zip_match",
    "zip3_match",
    "sim_jw_city",
    "sim_jw_street1",
    "org_keyword_match",
    "org_vs_person_conflict",
    "credential_overlap",
    "suffix_match",
    "miss_b_first",
    "miss_b_last",
    "miss_b_street1",
    "miss_b_city",
    "miss_b_state",
    "miss_b_zip5",
    "miss_x_first",
    "miss_x_last",
    "miss_x_street1",
    "miss_x_city",
    "miss_x_state",
    "miss_x_zip5",
    "sim_zip_num",
    "n_years_b",
]


def norm(s: str | None) -> str:
    """Normalize text to uppercase ASCII-ish alphanumeric tokens."""
    if s is None:
        return ""
    out = s.upper().strip()
    out = _ws.sub(" ", out)
    out = _non_alnum.sub("", out)
    return out


def tokens_word(s: str | None) -> list[str]:
    """Split normalized string into non-empty tokens."""
    return [t for t in norm(s).split(" ") if t]


def tokens_set(s: str | None) -> set[str]:
    """Get unique token set from normalized text."""
    return set(tokens_word(s))


def char_ngrams(s: str | None, n: int = 3) -> list[str]:
    """Return normalized character n-grams."""
    txt = norm(s).replace(" ", "")
    if len(txt) < n:
        return [txt] if txt else []
    return [txt[i : i + n] for i in range(len(txt) - n + 1)]


def street_simplify(s: str | None) -> str:
    """Simplify street text by removing unit designators."""
    txt = norm(s)
    txt = re.sub(r"\b(STE|SUITE|UNIT|APT|FL|FLOOR|BLDG|BUILDING|ROOM|RM)\b", "", txt)
    return _ws.sub(" ", txt).strip()


def first_initial(value: str | None) -> str:
    """Return first initial from normalized text."""
    txt = norm(value)
    return txt[:1] if txt else ""


def last_key(value: str | None, max_chars: int = 6) -> str:
    """Return normalized last-name prefix key for blocking."""
    return norm(value)[:max_chars]


def block_key(*parts: str | None) -> str:
    """Join blocking key parts with a pipe separator."""
    return "|".join(norm(p) for p in parts)


def zip3(value: str | None) -> str:
    """Return the first 3 zip digits from normalized zip."""
    txt = norm(value)
    return txt[:3] if len(txt) >= 3 else ""


def zip_num_sim(a: str | None, b: str | None) -> float:
    """Fuzzy numeric ZIP comparison — treats ZIP as an integer.
    Nearby ZIP codes score near 1.0; far-apart ones near 0.0.
    A gap of 10,000 (e.g. 10001 vs 20001) maps to 0.
    """
    try:
        na_s = norm(a)
        nb_s = norm(b)
        na = int(na_s[:5]) if len(na_s) >= 5 else -1
        nb = int(nb_s[:5]) if len(nb_s) >= 5 else -1
    except ValueError:
        return 0.0
    if na < 0 or nb < 0:
        return 0.0
    return float(max(0.0, 1.0 - abs(na - nb) / 10_000))


def jaro_similarity(s1: str | None, s2: str | None) -> float:
    """Compute Jaro similarity."""
    a = norm(s1)
    b = norm(s2)
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    len1 = len(a)
    len2 = len(b)
    max_dist = max(len1, len2) // 2 - 1
    match1 = [False] * len1
    match2 = [False] * len2

    matches = 0
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if match2[j]:
                continue
            if a[i] == b[j]:
                match1[i] = True
                match2[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    t = 0
    j = 0
    for i in range(len1):
        if match1[i]:
            while not match2[j]:
                j += 1
            if a[i] != b[j]:
                t += 1
            j += 1
    t /= 2.0
    return (matches / len1 + matches / len2 + (matches - t) / matches) / 3.0


def jaro_winkler(s1: str | None, s2: str | None, p: float = 0.1, max_l: int = 4) -> float:
    """Compute Jaro-Winkler similarity."""
    a = norm(s1)
    b = norm(s2)
    j = jaro_similarity(a, b)
    l = 0
    for i in range(min(len(a), len(b), max_l)):
        if a[i] == b[i]:
            l += 1
        else:
            break
    return j + l * p * (1.0 - j)


def levenshtein_dist(a: str | None, b: str | None) -> int:
    """Compute Levenshtein edit distance."""
    s1 = norm(a)
    s2 = norm(b)
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, start=1):
        cur = [i]
        for j, c2 in enumerate(s2, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if c1 == c2 else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def lev_sim(a: str | None, b: str | None) -> float:
    """Compute normalized Levenshtein similarity."""
    s1 = norm(a)
    s2 = norm(b)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    d = levenshtein_dist(s1, s2)
    m = max(len(s1), len(s2))
    return 1.0 - (d / m) if m else 1.0


_soundex_map = {
    **{c: "1" for c in "BFPV"},
    **{c: "2" for c in "CGJKQSXZ"},
    **{c: "3" for c in "DT"},
    "L": "4",
    **{c: "5" for c in "MN"},
    "R": "6",
}


def soundex(s: str | None) -> str:
    """Compute Soundex code."""
    txt = norm(s)
    if not txt:
        return ""
    first = txt[0]
    out = [first]
    last_code = _soundex_map.get(first, "")
    for ch in txt[1:]:
        code = _soundex_map.get(ch, "0")
        if code != "0" and code != last_code:
            out.append(code)
        last_code = code
    code = "".join(out).replace("0", "")
    return (code + "000")[:4]


def token_jaccard(a: str | None, b: str | None) -> float:
    """Compute token Jaccard similarity."""
    ta = tokens_set(a)
    tb = tokens_set(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def build_idf_from_texts(texts: list[str], max_vocab: int = 200_000) -> dict[str, float]:
    """Build smooth IDF dictionary from a text corpus."""
    df: dict[str, int] = defaultdict(int)
    n_docs = 0
    for v in texts:
        toks = set(tokens_word(v))
        if not toks:
            continue
        n_docs += 1
        for t in toks:
            df[t] += 1
    items = sorted(df.items(), key=lambda x: -x[1])[:max_vocab]
    idf: dict[str, float] = {}
    for t, d in items:
        idf[t] = math.log((1 + n_docs) / (1 + d)) + 1.0
    return idf


def tfidf_vec(text: str | None, idf: dict[str, float]) -> dict[str, float]:
    """Build L2-normalized sparse TF-IDF vector."""
    toks = tokens_word(text)
    if not toks:
        return {}
    tf: dict[str, int] = defaultdict(int)
    for t in toks:
        tf[t] += 1
    vec: dict[str, float] = {}
    norm2 = 0.0
    for t, c in tf.items():
        w = c * idf.get(t, 0.0)
        if w != 0.0:
            vec[t] = w
            norm2 += w * w
    if norm2 <= 0.0:
        return {}
    inv = 1.0 / math.sqrt(norm2)
    for k in list(vec.keys()):
        vec[k] *= inv
    return vec


def cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    """Compute cosine for sparse L2-normalized vectors."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    score = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None:
            score += va * vb
    return float(max(0.0, min(1.0, score)))


def tfidf_cosine(a: str | None, b: str | None, idf: dict[str, float]) -> float:
    """Compute word TF-IDF cosine similarity."""
    return cosine_sparse(tfidf_vec(a, idf), tfidf_vec(b, idf))


def hashed_char_vec(text: str | None) -> dict[int, float]:
    """Build normalized hashed char 3-gram vector."""
    grams = char_ngrams(text, 3)
    if not grams:
        return {}
    tf: dict[int, int] = defaultdict(int)
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=4).digest(), "little")
        tf[h % CHAR_DIM] += 1
    norm2 = 0.0
    vec: dict[int, float] = {}
    for i, c in tf.items():
        w = float(c)
        vec[i] = w
        norm2 += w * w
    inv = 1.0 / math.sqrt(norm2) if norm2 > 0 else 1.0
    for i in list(vec.keys()):
        vec[i] *= inv
    return vec


def cosine_sparse_int(a: dict[int, float], b: dict[int, float]) -> float:
    """Compute cosine for int-index sparse vectors."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    score = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None:
            score += va * vb
    return float(max(0.0, min(1.0, score)))


def char3_cosine(a: str | None, b: str | None) -> float:
    """Compute char 3-gram hashed cosine similarity."""
    return cosine_sparse_int(hashed_char_vec(a), hashed_char_vec(b))


def is_org_like(name: str | None) -> int:
    """Return 1 if organization-like keywords are present."""
    return int(any(t in ORG_WORDS for t in tokens_set(name)))


def suffix_token(s: str | None) -> str:
    """Extract suffix token (JR/SR/III/etc) from end of name."""
    toks = tokens_word(s)
    for t in toks[::-1]:
        if t in SUFFIXES:
            return t
    return ""


def credential_tokens(s: str | None) -> set[str]:
    """Extract known credential tokens from text."""
    return {t for t in tokens_set(s) if t in CRED_TOKS}


def overlap_ratio(a: set[str], b: set[str]) -> float:
    """Compute overlap ratio between two token sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def features_row(row: dict[str, Any], idf_name: dict[str, float]) -> dict[str, float | int]:
    """Compute full linkage feature dictionary for one pair."""
    b_first = row.get("b_first") or ""
    b_last = row.get("b_last") or ""
    x_first = row.get("x_first") or ""
    x_last = row.get("x_last") or ""

    b_full = f"{b_first} {b_last}".strip()
    x_full = f"{x_first} {x_last}".strip()

    b_st1 = row.get("b_street1") or ""
    b_city = row.get("b_city") or ""
    b_state = row.get("b_state") or ""
    b_zip = row.get("b_zip5") or ""
    b_cred = row.get("b_cred") or ""
    b_suf = row.get("b_suffix") or ""

    x_st1 = row.get("x_street1") or ""
    x_city = row.get("x_city") or ""
    x_state = row.get("x_state") or ""
    x_zip = row.get("x_zip5") or ""
    x_cred = row.get("x_cred") or ""
    x_suf = row.get("x_suffix") or ""

    miss = lambda v: int(v is None or v == "")

    jw_full = jaro_winkler(b_full, x_full)
    jw_last = jaro_winkler(b_last, x_last)
    jw_city = jaro_winkler(b_city, x_city)
    jw_st1 = jaro_winkler(street_simplify(b_st1), street_simplify(x_st1)) if (b_st1 or x_st1) else 0.0

    lev_full = lev_sim(b_full, x_full)
    lev_last = lev_sim(b_last, x_last)

    jac_full = token_jaccard(b_full, x_full)
    jac_last = token_jaccard(b_last, x_last)

    tfidf_name_sim = tfidf_cosine(b_full, x_full, idf_name)
    c3_sim = char3_cosine(b_full, x_full)

    sx_b = soundex(b_last)
    sx_x = soundex(x_last)
    sx_match = int(bool(sx_b) and bool(sx_x) and sx_b == sx_x)

    fi_match = int(first_initial(b_first) != "" and first_initial(b_first) == first_initial(x_first))
    st_match = int(bool(b_state) and bool(x_state) and norm(b_state) == norm(x_state))
    z_match = int(bool(b_zip) and bool(x_zip) and norm(b_zip) == norm(x_zip))
    z3_match = int(bool(b_zip) and bool(x_zip) and zip3(b_zip) != "" and zip3(b_zip) == zip3(x_zip))

    org_b = is_org_like(b_last) or is_org_like(b_full)
    org_x = is_org_like(x_last) or is_org_like(x_full)
    org_keyword_match = int(org_b == 1 and org_x == 1)
    org_vs_person_conflict = int((org_b == 1) ^ (org_x == 1))

    cred_overlap = overlap_ratio(credential_tokens(b_cred), credential_tokens(x_cred))

    suf_b = norm(b_suf) or suffix_token(b_full)
    suf_x = norm(x_suf) or suffix_token(x_full)
    suffix_match = int(bool(suf_b) and bool(suf_x) and suf_b == suf_x)

    # fuzzy numeric ZIP: continuous proximity beyond exact/zip3 matching
    sim_zip_num_val = zip_num_sim(b_zip, x_zip)

    # temporal: years of payment activity for the B provider, normalized to [0,1]
    # defaults to 0.0 at inference time when n_years is not in the incoming request
    raw_years = row.get("b_n_years") or 0.0
    n_years_b_val = min(float(raw_years), 20.0) / 20.0

    return {
        "sim_jw_fullname": float(jw_full),
        "sim_jw_lastname": float(jw_last),
        "sim_lev_fullname": float(lev_full),
        "sim_lev_lastname": float(lev_last),
        "sim_soundex_last": int(sx_match),
        "sim_jacc_fullname": float(jac_full),
        "sim_jacc_lastname": float(jac_last),
        "sim_tfidf_name": float(tfidf_name_sim),
        "sim_char3_name": float(c3_sim),
        "first_initial_match": int(fi_match),
        "state_match": int(st_match),
        "zip_match": int(z_match),
        "zip3_match": int(z3_match),
        "sim_jw_city": float(jw_city),
        "sim_jw_street1": float(jw_st1),
        "org_keyword_match": int(org_keyword_match),
        "org_vs_person_conflict": int(org_vs_person_conflict),
        "credential_overlap": float(cred_overlap),
        "suffix_match": int(suffix_match),
        "miss_b_first": miss(b_first),
        "miss_b_last": miss(b_last),
        "miss_b_street1": miss(b_st1),
        "miss_b_city": miss(b_city),
        "miss_b_state": miss(b_state),
        "miss_b_zip5": miss(b_zip),
        "miss_x_first": miss(x_first),
        "miss_x_last": miss(x_last),
        "miss_x_street1": miss(x_st1),
        "miss_x_city": miss(x_city),
        "miss_x_state": miss(x_state),
        "miss_x_zip5": miss(x_zip),
        "sim_zip_num": float(sim_zip_num_val),
        "n_years_b": float(n_years_b_val),
    }


def safe_predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Return positive-class probability scores for any sklearn-compatible estimator."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    return model.predict(X).astype(float)


def exact_npi_block_candidates(records_b: list[dict[str, str | None]], records_a: list[dict[str, str | None]]) -> set[tuple[str, str]]:
    """Generate exact-NPI candidate pairs (profile_id, npi_a)."""
    a_npis = {norm(r.get("npi")) for r in records_a if r.get("npi")}
    out: set[tuple[str, str]] = set()
    for rec in records_b:
        profile_id = rec.get("profile_id")
        npi = norm(rec.get("npi"))
        if profile_id and npi and npi in a_npis:
            out.add((str(profile_id), npi))
    return out


def sorted_neighborhood_candidates(
    left_records: list[dict[str, str | None]],
    right_records: list[dict[str, str | None]],
    window: int = 2,
) -> list[tuple[str, str]]:
    """Generate sorted-neighborhood pairs by state+lastkey+firstinitial key."""
    right_by_key: dict[str, list[str]] = defaultdict(list)
    for rec in right_records:
        npi = rec.get("npi")
        if not npi:
            continue
        key = block_key(rec.get("state"), last_key(rec.get("last_name")), first_initial(rec.get("first_name")))
        right_by_key[key].append(norm(npi))
    for key in right_by_key:
        right_by_key[key] = sorted(set(right_by_key[key]))[:window]

    pairs: list[tuple[str, str]] = []
    for rec in left_records:
        pid = rec.get("profile_id")
        if not pid:
            continue
        key = block_key(rec.get("state"), last_key(rec.get("last_name")), first_initial(rec.get("first_name")))
        for npi in right_by_key.get(key, []):
            pairs.append((str(pid), npi))
    return pairs


def canopy_candidates(
    left_records: list[dict[str, str | None]],
    right_records: list[dict[str, str | None]],
    threshold: float = 0.6,
) -> list[tuple[str, str, float]]:
    """Generate canopy candidates using state-constrained token Jaccard."""
    right_by_state: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
    for rec in right_records:
        npi = rec.get("npi")
        if not npi:
            continue
        st = norm(rec.get("state"))
        toks = tokens_set(f"{rec.get('first_name') or ''} {rec.get('last_name') or ''}")
        right_by_state[st].append((norm(npi), toks))

    out: list[tuple[str, str, float]] = []
    for rec in left_records:
        pid = rec.get("profile_id")
        if not pid:
            continue
        st = norm(rec.get("state"))
        btoks = tokens_set(f"{rec.get('first_name') or ''} {rec.get('last_name') or ''}")
        for npi, atoks in right_by_state.get(st, []):
            score = token_jaccard(" ".join(btoks), " ".join(atoks))
            if score >= threshold:
                out.append((str(pid), npi, float(score)))
    return out
