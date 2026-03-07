from __future__ import annotations
from pathlib import Path
import re
import math
import hashlib
import polars as pl

OUT_DIR = Path("outputs")
CAND_DIR = OUT_DIR / "candidates"
FEAT_DIR = OUT_DIR / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load provider tables + candidates
# -----------------------------
pa = pl.read_parquet(OUT_DIR / "providers_a.parquet")
pb = pl.read_parquet(OUT_DIR / "providers_b.parquet")
pc = pl.read_parquet(OUT_DIR / "providers_c.parquet")

cand_ba = pl.read_parquet(CAND_DIR / "cand_ba.parquet")
cand_bc = pl.read_parquet(CAND_DIR / "cand_bc.parquet")

# ============================================================
# Feature schema
# ============================================================
FEATURE_STRUCT = pl.Struct([
    pl.Field("sim_jw_fullname", pl.Float64),
    pl.Field("sim_jw_lastname", pl.Float64),
    pl.Field("sim_lev_fullname", pl.Float64),
    pl.Field("sim_lev_lastname", pl.Float64),
    pl.Field("sim_soundex_last", pl.Int8),

    pl.Field("sim_jacc_fullname", pl.Float64),
    pl.Field("sim_jacc_lastname", pl.Float64),

    # --- TF-IDF cosine sims---
    pl.Field("sim_tfidf_name", pl.Float64),      
    pl.Field("sim_char3_name", pl.Float64),      

    # --- structured + geo ---
    pl.Field("first_initial_match", pl.Int8),
    pl.Field("state_match", pl.Int8),
    pl.Field("zip_match", pl.Int8),
    pl.Field("zip3_match", pl.Int8),            
    pl.Field("sim_jw_city", pl.Float64),
    pl.Field("sim_jw_street1", pl.Float64),

    # --- domain-specific cues ---
    pl.Field("org_keyword_match", pl.Int8),       
    pl.Field("org_vs_person_conflict", pl.Int8),  
    pl.Field("credential_overlap", pl.Float64),   
    pl.Field("suffix_match", pl.Int8),            

    # --- missingness flags ---
    pl.Field("miss_b_first", pl.Int8),
    pl.Field("miss_b_last", pl.Int8),
    pl.Field("miss_b_street1", pl.Int8),
    pl.Field("miss_b_city", pl.Int8),
    pl.Field("miss_b_state", pl.Int8),
    pl.Field("miss_b_zip5", pl.Int8),

    pl.Field("miss_x_first", pl.Int8),
    pl.Field("miss_x_last", pl.Int8),
    pl.Field("miss_x_street1", pl.Int8),
    pl.Field("miss_x_city", pl.Int8),
    pl.Field("miss_x_state", pl.Int8),
    pl.Field("miss_x_zip5", pl.Int8),
])

# -----------------------------
# Text normalization
# -----------------------------
_ws = re.compile(r"\s+")
_non_alnum = re.compile(r"[^A-Z0-9 ]+")

ORG_WORDS = {
    "LLC","INC","LTD","CORP","CORPORATION","COMPANY","CO","GROUP","ASSOCIATES","ASSOC",
    "HOSPITAL","HEALTH","HEALTHCARE","CLINIC","CENTER","CENTRE","MEDICAL","MED","SYSTEM",
    "FOUNDATION","UNIVERSITY","DEPARTMENT","DEPT","PRACTICE","INSTITUTE","LAB","LABS"
}
SUFFIXES = {"JR","SR","II","III","IV","V"}
CRED_TOKS = {"MD","DO","DDS","DMD","NP","PA","RN","PHD","MBA","MS","MPH","DPM","OD"}

def norm(s: str | None) -> str:
    if s is None:
        return ""
    s = s.upper().strip()
    s = _ws.sub(" ", s)
    s = _non_alnum.sub("", s)
    return s

def tokens_word(s: str) -> list[str]:
    s = norm(s)
    return [t for t in s.split(" ") if t]

def tokens_set(s: str) -> set[str]:
    return set(tokens_word(s))

def char_ngrams(s: str, n: int = 3) -> list[str]:
    s = norm(s).replace(" ", "")
    if len(s) < n:
        return [s] if s else []
    return [s[i:i+n] for i in range(len(s)-n+1)]

def street_simplify(s: str) -> str:
    s = norm(s)
    s = re.sub(r"\b(STE|SUITE|UNIT|APT|FL|FLOOR|BLDG|BUILDING|ROOM|RM)\b", "", s)
    s = _ws.sub(" ", s).strip()
    return s

def first_initial(s: str) -> str:
    s = norm(s)
    return s[:1] if s else ""

def zip3(z: str) -> str:
    z = norm(z)
    return z[:3] if len(z) >= 3 else ""

# ============================================================
# Jaro-Winkler
# ============================================================
def jaro_similarity(s1: str, s2: str) -> float:
    s1 = norm(s1)
    s2 = norm(s2)
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
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
            if s1[i] == s2[j]:
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
            if s1[i] != s2[j]:
                t += 1
            j += 1
    t /= 2.0
    return (matches / len1 + matches / len2 + (matches - t) / matches) / 3.0

def jaro_winkler(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    s1n, s2n = norm(s1), norm(s2)
    j = jaro_similarity(s1n, s2n)
    l = 0
    for i in range(min(len(s1n), len(s2n), max_l)):
        if s1n[i] == s2n[i]:
            l += 1
        else:
            break
    return j + l * p * (1.0 - j)

# ============================================================
# Levenshtein similarity (normalized)
# ============================================================
def levenshtein_dist(a: str, b: str) -> int:
    a = norm(a)
    b = norm(b)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # small optimization: ensuring b is shorter in memory row
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j-1] + 1
            dele = prev[j] + 1
            sub = prev[j-1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]

def lev_sim(a: str, b: str) -> float:
    a = norm(a)
    b = norm(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    d = levenshtein_dist(a, b)
    m = max(len(a), len(b))
    return 1.0 - (d / m) if m else 1.0

# ============================================================
# Phonetic matching (Soundex)
# ============================================================
_soundex_map = {
    **{c:"1" for c in "BFPV"},
    **{c:"2" for c in "CGJKQSXZ"},
    **{c:"3" for c in "DT"},
    "L":"4",
    **{c:"5" for c in "MN"},
    "R":"6",
}
def soundex(s: str) -> str:
    s = norm(s)
    if not s:
        return ""
    first = s[0]
    out = [first]
    last_code = _soundex_map.get(first, "")
    for ch in s[1:]:
        code = _soundex_map.get(ch, "0")
        if code != "0" and code != last_code:
            out.append(code)
        last_code = code
    code = "".join(out).replace("0", "")
    code = (code + "000")[:4]
    return code

# ============================================================
# Jaccard token overlap
# ============================================================
def token_jaccard(a: str, b: str) -> float:
    A = tokens_set(a)
    B = tokens_set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)
from collections import defaultdict

# ============================================================
# TF-IDF cosine (lightweight, per-row sparse dicts)
# We build IDF dicts from provider tables once
# ============================================================
def build_idf_from_series(s: pl.Series, max_vocab: int = 200_000) -> dict[str, float]:
    # document frequency
    df = defaultdict(int)
    n_docs = 0
    for v in s.drop_nulls().to_list():
        toks = set(tokens_word(v))
        if not toks:
            continue
        n_docs += 1
        for t in toks:
            df[t] += 1

    items = sorted(df.items(), key=lambda x: -x[1])[:max_vocab]
    df = dict(items)

    idf = {}
    for t, d in df.items():
        # smoothing idf
        idf[t] = math.log((1 + n_docs) / (1 + d)) + 1.0
    return idf

def tfidf_vec(text: str, idf: dict[str, float]) -> dict[str, float]:
    toks = tokens_word(text)
    if not toks:
        return {}
    tf = defaultdict(int)
    for t in toks:
        tf[t] += 1
    # l2-normalize
    vec = {}
    norm2 = 0.0
    for t, c in tf.items():
        w = (c * idf.get(t, 0.0))
        if w != 0.0:
            vec[t] = w
            norm2 += w * w
    if norm2 <= 0.0:
        return {}
    inv = 1.0 / math.sqrt(norm2)
    for t in list(vec.keys()):
        vec[t] *= inv
    return vec

def cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    
    if len(a) > len(b):
        a, b = b, a
    s = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None:
            s += va * vb
    
    return float(max(0.0, min(1.0, s)))

# embedding-style: char-3gram tf-idf cosine using hashing trick (fixed dimensional)
CHAR_DIM = 2**14
def hashed_char_vec(text: str) -> dict[int, float]:
    grams = char_ngrams(text, 3)
    if not grams:
        return {}
    tf = defaultdict(int)
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=4).digest(), "little")
        idx = h % CHAR_DIM
        tf[idx] += 1
    norm2 = 0.0
    vec = {}
    for i, c in tf.items():
        w = float(c)
        vec[i] = w
        norm2 += w*w
    inv = 1.0 / math.sqrt(norm2) if norm2 > 0 else 1.0
    for i in list(vec.keys()):
        vec[i] *= inv
    return vec


def cosine_sparse_int(a: dict[int, float], b: dict[int, float]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    s = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None:
            s += va * vb
    return float(max(0.0, min(1.0, s)))

def build_name_series_for_idf(pb: pl.DataFrame, pa: pl.DataFrame, pc: pl.DataFrame) -> pl.Series:
    s_b = pb.select(
        pl.concat_str(
            [pl.col("first_name").cast(pl.Utf8).fill_null(""),
             pl.col("last_name").cast(pl.Utf8).fill_null("")],
            separator=" "
        ).alias("name")
    )["name"]

    s_a = pa.select(
        pl.concat_str(
            [pl.col("first_name").cast(pl.Utf8).fill_null(""),
             pl.col("last_or_org_name").cast(pl.Utf8).fill_null("")],
            separator=" "
        ).alias("name")
    )["name"]

    s_c = pc.select(
        pl.concat_str(
            [pl.col("first_name").cast(pl.Utf8).fill_null(""),
             pl.col("last_name").cast(pl.Utf8).fill_null("")],
            separator=" "
        ).alias("name")
    )["name"]

    return pl.concat([s_b, s_a, s_c], how="vertical")

# Building IDF on names once
name_series = build_name_series_for_idf(pb, pa, pc)
idf_name = build_idf_from_series(name_series, max_vocab=200_000)

# ============================================================
# Domain-specific helpers
# ============================================================
def is_org_like(name: str) -> int:
    toks = tokens_set(name)
    return int(any(t in ORG_WORDS for t in toks))

def suffix_token(s: str) -> str:
    toks = tokens_word(s)
    for t in toks[::-1]:
        if t in SUFFIXES:
            return t
    return ""

def credential_tokens(s: str) -> set[str]:
    toks = tokens_set(s)
    return set([t for t in toks if t in CRED_TOKS])

def overlap_ratio(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

# ============================================================
# Row-wise feature function
# ============================================================
def features_row(row: dict) -> dict:
    b_first = row.get("b_first") or ""
    b_last  = row.get("b_last") or ""
    x_first = row.get("x_first") or ""
    x_last  = row.get("x_last") or ""

    b_full = f"{b_first} {b_last}".strip()
    x_full = f"{x_first} {x_last}".strip()

    b_st1   = row.get("b_street1") or ""
    b_city  = row.get("b_city") or ""
    b_state = row.get("b_state") or ""
    b_zip   = row.get("b_zip5") or ""
    b_cred  = row.get("b_cred") or ""
    b_suf   = row.get("b_suffix") or ""

    x_st1   = row.get("x_street1") or ""
    x_city  = row.get("x_city") or ""
    x_state = row.get("x_state") or ""
    x_zip   = row.get("x_zip5") or ""
    x_cred  = row.get("x_cred") or ""
    x_suf   = row.get("x_suffix") or ""

    miss = lambda v: int(v is None or v == "")

    # --- JW sims ---
    jw_full = jaro_winkler(b_full, x_full)
    jw_last = jaro_winkler(b_last, x_last)
    jw_city = jaro_winkler(b_city, x_city)
    jw_st1  = jaro_winkler(street_simplify(b_st1), street_simplify(x_st1)) if (b_st1 or x_st1) else 0.0

    # --- Levenshtein sims ---
    lev_full = lev_sim(b_full, x_full)
    lev_last = lev_sim(b_last, x_last)

    # --- token overlap ---
    jac_full = token_jaccard(b_full, x_full)
    jac_last = token_jaccard(b_last, x_last)

    # --- TF-IDF cosine (word) ---
    tf_b = tfidf_vec(b_full, idf_name)
    tf_x = tfidf_vec(x_full, idf_name)
    tfidf_name_sim = cosine_sparse(tf_b, tf_x)

    # --- "embedding-style": char 3-gram cosine ---
    c3_b = hashed_char_vec(b_full)
    c3_x = hashed_char_vec(x_full)
    c3_sim = cosine_sparse_int(c3_b, c3_x)

    # --- phonetic ---
    sx_b = soundex(b_last)
    sx_x = soundex(x_last)
    sx_match = int(bool(sx_b) and bool(sx_x) and sx_b == sx_x)

    # --- structured ---
    fi_match = int(first_initial(b_first) != "" and first_initial(b_first) == first_initial(x_first))
    st_match = int(bool(b_state) and bool(x_state) and b_state == x_state)
    zip_match = int(bool(b_zip) and bool(x_zip) and b_zip == x_zip)
    zip3_match = int(bool(b_zip) and bool(x_zip) and zip3(b_zip) != "" and zip3(b_zip) == zip3(x_zip))

    # --- domain-specific ---
    org_b = is_org_like(b_last) or is_org_like(b_full)
    org_x = is_org_like(x_last) or is_org_like(x_full)
    org_keyword_match = int(org_b == 1 and org_x == 1)
    org_vs_person_conflict = int((org_b == 1) ^ (org_x == 1))

    cred_overlap = overlap_ratio(credential_tokens(b_cred), credential_tokens(x_cred))

    suf_b = norm(b_suf) or suffix_token(b_full)
    suf_x = norm(x_suf) or suffix_token(x_full)
    suffix_match = int(bool(suf_b) and bool(suf_x) and suf_b == suf_x)

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
        "zip_match": int(zip_match),
        "zip3_match": int(zip3_match),
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
    }

# ============================================================
# BA features
# ============================================================
ba = (
    cand_ba
    .join(
        pb.select([
            "profile_id", "npi",
            pl.col("first_name").alias("b_first"),
            pl.col("last_name").alias("b_last"),
            pl.col("suffix").alias("b_suffix"),
            pl.col("street1").alias("b_street1"),
            pl.col("city").alias("b_city"),
            pl.col("state").alias("b_state"),
            pl.col("zip5").alias("b_zip5"),
        ]),
        on="profile_id",
        how="left",
    )
    .join(
        pa.select([
            pl.col("npi").alias("npi_a"),
            pl.col("first_name").alias("a_first"),
            pl.col("last_or_org_name").alias("a_last"),
            pl.col("credentials").alias("a_cred"),
            pl.col("street1").alias("a_street1"),
            pl.col("city").alias("a_city"),
            pl.col("state").alias("a_state"),
            pl.col("zip5").alias("a_zip5"),
        ]),
        on="npi_a",
        how="left",
    )
    .with_columns([
        pl.col("a_first").alias("x_first"),
        pl.col("a_last").alias("x_last"),
        pl.col("a_cred").alias("x_cred"),
        pl.lit("").alias("x_suffix"),
        pl.col("a_street1").alias("x_street1"),
        pl.col("a_city").alias("x_city"),
        pl.col("a_state").alias("x_state"),
        pl.col("a_zip5").alias("x_zip5"),
        pl.lit("").alias("b_cred"),
    ])
    .with_columns([
        pl.when(pl.col("npi").is_null())
          .then(pl.lit(None, dtype=pl.Int8))
          .when(pl.col("npi") == pl.col("npi_a"))
          .then(pl.lit(1, dtype=pl.Int8))
          .otherwise(pl.lit(0, dtype=pl.Int8))
          .alias("label_weak_npi")
    ])
)

ba_feats = (
    ba.with_columns([
        pl.struct([
            "b_first","b_last","b_suffix","b_cred",
            "b_street1","b_city","b_state","b_zip5",
            "x_first","x_last","x_suffix","x_cred",
            "x_street1","x_city","x_state","x_zip5",
        ]).map_elements(features_row, return_dtype=FEATURE_STRUCT).alias("feat")
    ])
    .unnest("feat")
    .select([
        "profile_id","npi_a","pass","block_key","label_weak_npi",
        *[f.name for f in FEATURE_STRUCT.fields],
    ])
)

ba_feats.write_parquet(FEAT_DIR / "pair_features_ba.parquet")
print("Wrote:", FEAT_DIR / "pair_features_ba.parquet", ba_feats.shape)

# ============================================================
# BC features
# ============================================================
bc = (
    cand_bc
    .join(
        pb.select([
            "profile_id", "npi",
            pl.col("first_name").alias("b_first"),
            pl.col("last_name").alias("b_last"),
            pl.col("suffix").alias("b_suffix"),
            pl.col("street1").alias("b_street1"),
            pl.col("city").alias("b_city"),
            pl.col("state").alias("b_state"),
            pl.col("zip5").alias("b_zip5"),
        ]),
        on="profile_id",
        how="left",
    )
    .join(
        pc.select([
            pl.col("npi").alias("npi_c"),
            pl.col("first_name").alias("c_first"),
            pl.col("last_name").alias("c_last"),
            pl.col("state").alias("c_state"),
        ]),
        on="npi_c",
        how="left",
    )
    .with_columns([
        pl.col("c_first").alias("x_first"),
        pl.col("c_last").alias("x_last"),
        pl.lit("").alias("x_cred"),
        pl.lit("").alias("x_suffix"),
        pl.lit("").alias("x_street1"),
        pl.lit("").alias("x_city"),
        pl.col("c_state").alias("x_state"),
        pl.lit("").alias("x_zip5"),
        pl.lit("").alias("b_cred"),
    ])
    .with_columns([
        pl.when(pl.col("npi").is_null())
          .then(pl.lit(None, dtype=pl.Int8))
          .when(pl.col("npi") == pl.col("npi_c"))
          .then(pl.lit(1, dtype=pl.Int8))
          .otherwise(pl.lit(0, dtype=pl.Int8))
          .alias("label_weak_npi")
    ])
)

bc_feats = (
    bc.with_columns([
        pl.struct([
            "b_first","b_last","b_suffix","b_cred",
            "b_street1","b_city","b_state","b_zip5",
            "x_first","x_last","x_suffix","x_cred",
            "x_street1","x_city","x_state","x_zip5",
        ]).map_elements(features_row, return_dtype=FEATURE_STRUCT).alias("feat")
    ])
    .unnest("feat")
    .select([
        "profile_id","npi_c","pass","block_key","label_weak_npi",
        *[f.name for f in FEATURE_STRUCT.fields],
    ])
)

bc_feats.write_parquet(FEAT_DIR / "pair_features_bc.parquet")
print("Wrote:", FEAT_DIR / "pair_features_bc.parquet", bc_feats.shape)

print("Done. Feature files in:", FEAT_DIR)