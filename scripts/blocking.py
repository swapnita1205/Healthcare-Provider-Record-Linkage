from __future__ import annotations
from pathlib import Path
import json
import math
import hashlib
import random
from collections import defaultdict

import numpy as np
import polars as pl

OUT_DIR = Path("outputs")
CAND_DIR = OUT_DIR / "candidates"
CAND_DIR.mkdir(parents=True, exist_ok=True)

pa = pl.read_parquet(OUT_DIR / "providers_a.parquet")
pb = pl.read_parquet(OUT_DIR / "providers_b.parquet")
pc = pl.read_parquet(OUT_DIR / "providers_c.parquet")

RANDOM_STATE = 42
rng = random.Random(RANDOM_STATE)

# Helpers
def first_initial(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.to_uppercase()
        .str.slice(0, 1)
        .str.replace_all(r"[^A-Z]", "")
    )

def last_key(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.to_uppercase()
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .str.slice(0, 6)
    )

def block_key(*cols: str) -> pl.Expr:
    return pl.concat_str([pl.col(c).cast(pl.Utf8).fill_null("") for c in cols], separator="|")

def tokenize_name(first: str | None, last: str | None) -> list[str]:
    f = (first or "").strip().upper()
    l = (last or "").strip().upper()
    toks = []
    for t in (f + " " + l).split():
        t = "".join([ch for ch in t if ch.isalnum()])
        if t:
            toks.append(t)
    return toks

def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

# Building minimal blocking tables
pa_k = (
    pa.select([
        pl.col("npi").alias("npi_a"),
        pl.col("state"), pl.col("zip5"), pl.col("city"),
        pl.col("first_name").alias("a_first"),
        pl.col("last_or_org_name").alias("a_last"),
    ])
    .with_columns([
        first_initial("a_first").alias("first_init"),
        last_key("a_last").alias("last_key"),
    ])
    .with_columns([
        block_key("state", "zip5").alias("bk_state_zip"),
        block_key("state", "city").alias("bk_state_city"),
        block_key("state", "last_key", "first_init").alias("bk_state_last_fi"),
        block_key("state", "last_key").alias("bk_state_last"),
    ])
    .filter(pl.col("npi_a").is_not_null())
)

pb_k = (
    pb.select([
        pl.col("profile_id"),
        pl.col("npi").alias("npi_b"),
        pl.col("state"), pl.col("zip5"), pl.col("city"),
        pl.col("first_name").alias("b_first"),
        pl.col("last_name").alias("b_last"),
    ])
    .with_columns([
        first_initial("b_first").alias("first_init"),
        last_key("b_last").alias("last_key"),
    ])
    .with_columns([
        block_key("state", "zip5").alias("bk_state_zip"),
        block_key("state", "city").alias("bk_state_city"),
        block_key("state", "last_key", "first_init").alias("bk_state_last_fi"),
        block_key("state", "last_key").alias("bk_state_last"),
    ])
    .filter(pl.col("profile_id").is_not_null())
)

pc_k = (
    pc.select([
        pl.col("npi").alias("npi_c"),
        pl.col("state"),
        pl.col("first_name").alias("c_first"),
        pl.col("last_name").alias("c_last"),
    ])
    .with_columns([
        first_initial("c_first").alias("first_init"),
        last_key("c_last").alias("last_key"),
    ])
    .with_columns([
        block_key("state", "last_key", "first_init").alias("bk_state_last_fi"),
        block_key("state", "last_key").alias("bk_state_last"),
    ])
    .filter(pl.col("npi_c").is_not_null())
)

# Info-theory stats for blocking keys: entropy + block sizes
def key_stats(df: pl.DataFrame, key_col: str, dataset: str) -> dict:
    # counts per key
    sizes = df.group_by(key_col).len().rename({"len": "cnt"})
    tot = float(sizes["cnt"].sum()) if sizes.height else 0.0
    if tot == 0.0:
        return {
            "dataset": dataset, "key": key_col,
            "n_rows": df.height,
            "n_keys": 0,
            "entropy_bits": None,
            "avg_block": None,
            "p95_block": None,
            "max_block": None,
        }
    p = (sizes["cnt"] / tot).to_numpy()
    ent = float(-(p * np.log2(p)).sum())
    cnts = sizes["cnt"].to_numpy()
    return {
        "dataset": dataset, "key": key_col,
        "n_rows": int(df.height),
        "n_keys": int(sizes.height),
        "entropy_bits": ent,
        "avg_block": float(cnts.mean()),
        "p95_block": float(np.quantile(cnts, 0.95)),
        "max_block": float(cnts.max()),
    }

KEYS_A = ["bk_state_zip","bk_state_city","bk_state_last_fi","bk_state_last"]
KEYS_B = ["bk_state_zip","bk_state_city","bk_state_last_fi","bk_state_last"]
KEYS_C = ["bk_state_last_fi","bk_state_last"]

ks_rows = []
for k in KEYS_A:
    ks_rows.append(key_stats(pa_k, k, "A"))
for k in KEYS_B:
    ks_rows.append(key_stats(pb_k, k, "B"))
for k in KEYS_C:
    ks_rows.append(key_stats(pc_k, k, "C"))

pl.DataFrame(ks_rows).write_csv(CAND_DIR / "blocking_key_stats.csv")

# Mutual information between blocking attributes
def mutual_info_discrete(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=object)
    y = np.asarray(y, dtype=object)
    n = x.size
    if n == 0:
        return float("nan")
    
    ux, xinv = np.unique(x, return_inverse=True)
    uy, yinv = np.unique(y, return_inverse=True)
    
    joint = np.zeros((ux.size, uy.size), dtype=np.int64)
    np.add.at(joint, (xinv, yinv), 1)
    px = joint.sum(axis=1) / n
    py = joint.sum(axis=0) / n
    pxy = joint / n
    mi = 0.0
    for i in range(ux.size):
        for j in range(uy.size):
            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy[i, j] * math.log(pxy[i, j] / (px[i] * py[j]) + 1e-15, 2)
    return float(mi)

def mi_table(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    wanted = ["state", "zip5", "city", "last_key", "first_init"]
    cols = [c for c in wanted if c in df.columns]

    if len(cols) < 2:
        return pl.DataFrame(schema={"dataset": pl.Utf8, "a": pl.Utf8, "b": pl.Utf8, "mutual_info_bits": pl.Float64})

    pdf = df.select([pl.col(c).cast(pl.Utf8).fill_null("") for c in cols]).to_pandas()

    out = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            mi = mutual_info_discrete(pdf[cols[i]].values, pdf[cols[j]].values)
            out.append({"dataset": dataset, "a": cols[i], "b": cols[j], "mutual_info_bits": mi})

    return pl.DataFrame(out).sort("mutual_info_bits", descending=True)

mi_a = mi_table(pa_k, "A")
mi_b = mi_table(pb_k, "B")
mi_c = mi_table(pc_k, "C")
pl.concat([mi_a, mi_b, mi_c]).write_csv(CAND_DIR / "blocking_mutual_information.csv")

# ============================================================
# Gold-ish pairs for PR tradeoff: PB records that have NPI and exist in A/C
# ============================================================
gold_ba = (
    pb_k.filter(pl.col("npi_b").is_not_null())
        .select(["profile_id", pl.col("npi_b").alias("npi_a")])
        .join(pa_k.select(["npi_a"]), on="npi_a", how="inner")
        .unique()
)
gold_bc = (
    pb_k.filter(pl.col("npi_b").is_not_null())
        .select(["profile_id", pl.col("npi_b").alias("npi_c")])
        .join(pc_k.select(["npi_c"]), on="npi_c", how="inner")
        .unique()
)

gold_ba_set = set(map(tuple, gold_ba.to_numpy()))
gold_bc_set = set(map(tuple, gold_bc.to_numpy()))

metrics = {}
pass_reports = []

def report_pass(name: str, pairs: pl.DataFrame, left_id: str, right_id: str, gold_set: set[tuple]) -> None:
    # recall against gold-ish
    if pairs.height:
        pair_set = set(map(tuple, pairs.select([left_id, right_id]).unique().to_numpy()))
        hit = len(pair_set & gold_set)
    else:
        hit = 0
    total_gold = len(gold_set)
    recall = hit / total_gold if total_gold else None

    pass_reports.append({
        "pass": name,
        "n_pairs": int(pairs.height),
        "n_left_unique": int(pairs.select(left_id).n_unique()) if pairs.height else 0,
        "n_right_unique": int(pairs.select(right_id).n_unique()) if pairs.height else 0,
        "gold_hits": int(hit),
        "gold_total": int(total_gold),
        "gold_recall": float(recall) if recall is not None else None,
        "pairs_per_gold_hit": float(pairs.height / hit) if hit > 0 else None,
    })

# ============================================================
# A <-> C exact NPI
# ============================================================
cand_ac = (
    pa.select(pl.col("npi").alias("npi"))
      .join(pc.select(pl.col("npi").alias("npi")), on="npi", how="inner")
      .select([pl.col("npi").alias("npi_a"), pl.col("npi").alias("npi_c"), pl.lit("exact_npi").alias("pass")])
      .unique()
)
cand_ac.write_parquet(CAND_DIR / "cand_ac.parquet")
metrics["ac_exact_npi_pairs"] = cand_ac.height

# ============================================================
# Pass 1: Exact NPI BA/BC
# ============================================================
pb_has_npi = pb_k.filter(pl.col("npi_b").is_not_null()).select(["profile_id", "npi_b"])

cand_ba_npi = (
    pb_has_npi
    .join(pa_k.select(["npi_a"]), left_on="npi_b", right_on="npi_a", how="inner")
    .with_columns(pl.col("npi_b").alias("npi_a"))
    .select(["profile_id","npi_a", pl.lit("exact_npi").alias("pass"), pl.lit(None).cast(pl.Utf8).alias("block_key")])
    .unique()
)

cand_bc_npi = (
    pb_has_npi
    .join(pc_k.select(["npi_c"]), left_on="npi_b", right_on="npi_c", how="inner")
    .with_columns(pl.col("npi_b").alias("npi_c"))
    .select(["profile_id","npi_c", pl.lit("exact_npi").alias("pass"), pl.lit(None).cast(pl.Utf8).alias("block_key")])
    .unique()
)

report_pass("BA_exact_npi", cand_ba_npi, "profile_id", "npi_a", gold_ba_set)
report_pass("BC_exact_npi", cand_bc_npi, "profile_id", "npi_c", gold_bc_set)

# ============================================================
# Pass 2: Geo blocks (BA) + Name blocks (BC) with caps
# ============================================================
ZIP_BLOCK_CAP = 500
LAST_BLOCK_CAP = 2000

pb_zip_sizes = pb_k.group_by("bk_state_zip").len().rename({"len": "b_cnt"})
pa_zip_sizes = pa_k.group_by("bk_state_zip").len().rename({"len": "a_cnt"})

pb_zip_ok = (
    pb_k.join(pb_zip_sizes, on="bk_state_zip", how="left")
        .join(pa_zip_sizes, on="bk_state_zip", how="left")
        .with_columns([
            pl.coalesce([pl.col("b_cnt"), pl.lit(0)]).alias("b_cnt"),
            pl.coalesce([pl.col("a_cnt"), pl.lit(0)]).alias("a_cnt"),
        ])
        .filter((pl.col("bk_state_zip").str.len_chars() > 1) & (pl.col("b_cnt") <= ZIP_BLOCK_CAP) & (pl.col("a_cnt") <= ZIP_BLOCK_CAP))
        .drop(["b_cnt", "a_cnt"])
)

pa_zip_ok = (
    pa_k.join(pa_zip_sizes, on="bk_state_zip", how="left")
        .with_columns([pl.coalesce([pl.col("a_cnt"), pl.lit(0)]).alias("a_cnt")])
        .filter((pl.col("bk_state_zip").str.len_chars() > 1) & (pl.col("a_cnt") <= ZIP_BLOCK_CAP))
        .drop("a_cnt")
)

cand_ba_statezip = (
    pb_zip_ok.join(pa_zip_ok, on="bk_state_zip", how="inner")
    .select(["profile_id","npi_a", pl.lit("state_zip").alias("pass"), pl.col("bk_state_zip").alias("block_key")])
)

cand_ba_statecity = (
    pb_k.filter((pl.col("zip5").is_null()) & (pl.col("bk_state_city").str.len_chars() > 1))
    .join(pa_k.filter((pl.col("zip5").is_null()) & (pl.col("bk_state_city").str.len_chars() > 1)),
          on="bk_state_city", how="inner")
    .select(["profile_id","npi_a", pl.lit("state_city").alias("pass"), pl.col("bk_state_city").alias("block_key")])
)

report_pass("BA_state_zip", cand_ba_statezip, "profile_id", "npi_a", gold_ba_set)
report_pass("BA_state_city", cand_ba_statecity, "profile_id", "npi_a", gold_ba_set)

pb_last_sizes = pb_k.group_by("bk_state_last").len().rename({"len":"b_cnt"})
pc_last_sizes = pc_k.group_by("bk_state_last").len().rename({"len":"c_cnt"})

pb_last_ok = (
    pb_k.join(pb_last_sizes, on="bk_state_last", how="left")
        .join(pc_last_sizes, on="bk_state_last", how="left")
        .with_columns([
            pl.coalesce([pl.col("b_cnt"), pl.lit(0)]).alias("b_cnt"),
            pl.coalesce([pl.col("c_cnt"), pl.lit(0)]).alias("c_cnt"),
        ])
        .filter((pl.col("bk_state_last").str.len_chars() > 1) & (pl.col("b_cnt") <= LAST_BLOCK_CAP) & (pl.col("c_cnt") <= LAST_BLOCK_CAP))
        .drop(["b_cnt","c_cnt"])
)

pc_last_ok = (
    pc_k.join(pc_last_sizes, on="bk_state_last", how="left")
        .with_columns([pl.coalesce([pl.col("c_cnt"), pl.lit(0)]).alias("c_cnt")])
        .filter((pl.col("bk_state_last").str.len_chars() > 1) & (pl.col("c_cnt") <= LAST_BLOCK_CAP))
        .drop("c_cnt")
)

cand_bc_statelast = (
    pb_last_ok.join(pc_last_ok, on="bk_state_last", how="inner")
    .select(["profile_id","npi_c", pl.lit("state_lastkey").alias("pass"), pl.col("bk_state_last").alias("block_key")])
)

report_pass("BC_state_lastkey", cand_bc_statelast, "profile_id", "npi_c", gold_bc_set)

# ============================================================
# Pass 3: Sorted neighborhood
# ============================================================
WINDOW = 50

def sorted_neighborhood_pairs(left_df: pl.DataFrame, right_df: pl.DataFrame,
                              left_id: str, right_id: str,
                              block_col: str, pass_name: str, window: int) -> pl.DataFrame:
    right_list = (
        right_df.select([block_col, right_id])
        .filter(pl.col(block_col).str.len_chars() > 1)
        .group_by(block_col)
        .agg([pl.col(right_id).sort().head(window).alias("cand_list")])
    )
    return (
        left_df.select([left_id, block_col])
        .filter(pl.col(block_col).str.len_chars() > 1)
        .join(right_list, on=block_col, how="inner")
        .explode("cand_list")
        .select([pl.col(left_id), pl.col("cand_list").alias(right_id),
                 pl.lit(pass_name).alias("pass"),
                 pl.col(block_col).alias("block_key")])
    )

cand_ba_sn = sorted_neighborhood_pairs(pb_k, pa_k, "profile_id", "npi_a", "bk_state_last_fi",
                                       "sorted_neighborhood_state_last_fi", WINDOW)
cand_bc_sn = sorted_neighborhood_pairs(pb_k, pc_k, "profile_id", "npi_c", "bk_state_last_fi",
                                       "sorted_neighborhood_state_last_fi", WINDOW)

report_pass("BA_sorted_neighborhood", cand_ba_sn, "profile_id", "npi_a", gold_ba_set)
report_pass("BC_sorted_neighborhood", cand_bc_sn, "profile_id", "npi_c", gold_bc_set)

# ============================================================
# Pass 4: Canopy clustering (blocking) — within state
#   - Uses token Jaccard on {first,last} tokens as canopy similarity
#   - Efficient: works per-state and caps candidate list per canopy
#   - Thresholds: T1 forms canopy, T2 prunes
# ============================================================
CANOPY_T1 = 0.60
CANOPY_T2 = 0.80
CANOPY_MAX_CANDS = 200  # cap candidates per left record

def canopy_pairs_ba(pb_k: pl.DataFrame, pa_k: pl.DataFrame) -> pl.DataFrame:
    pb_pdf = pb_k.select(["profile_id","state","b_first","b_last"]).to_pandas()
    pa_pdf = pa_k.select(["npi_a","state","a_first","a_last"]).to_pandas()

    # pre-indexing A by state with token sets
    idx_a = defaultdict(list)
    for _, r in pa_pdf.iterrows():
        st = r["state"] or ""
        toks = set(tokenize_name(r["a_first"], r["a_last"]))
        idx_a[st].append((r["npi_a"], toks))

    out = []
    for _, r in pb_pdf.iterrows():
        st = r["state"] or ""
        btoks = set(tokenize_name(r["b_first"], r["b_last"]))
        if not st or not btoks:
            continue
        # cheap canopy: picking candidates that pass T1, then keeping those passing T2 (or top)
        cand = []
        for npi_a, atoks in idx_a.get(st, []):
            s = jaccard(btoks, atoks)
            if s >= CANOPY_T1:
                cand.append((npi_a, s))
        if not cand:
            continue
        # sorting by similarity; keep bounded
        cand.sort(key=lambda x: -x[1])
        strong = [x for x in cand if x[1] >= CANOPY_T2]
        chosen = strong[:CANOPY_MAX_CANDS] if strong else cand[:CANOPY_MAX_CANDS]
        for npi_a, _ in chosen:
            out.append((r["profile_id"], npi_a, "canopy_state_name", st))
    if not out:
        return pl.DataFrame(schema={"profile_id":pl.Utf8,"npi_a":pl.Utf8,"pass":pl.Utf8,"block_key":pl.Utf8})
    return pl.DataFrame(out, schema=["profile_id","npi_a","pass","block_key"])

cand_ba_canopy = canopy_pairs_ba(pb_k, pa_k)
report_pass("BA_canopy_state_name", cand_ba_canopy, "profile_id", "npi_a", gold_ba_set)

# ============================================================
# Pass 5: Probabilistic blocking — MinHash LSH banding on name tokens (within state)
#   - Mathematical foundation: MinHash approximates Jaccard; LSH increases collision
#   - Implementation is bounded by (state) and caps bucket sizes
# ============================================================
LSH_NUM_PERM = 64
LSH_BANDS = 8
LSH_ROWS_PER_BAND = LSH_NUM_PERM // LSH_BANDS
LSH_BUCKET_CAP = 5000

def _hash64(x: str) -> int:
    h = hashlib.blake2b(x.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)

SALTS = [rng.getrandbits(64) for _ in range(LSH_NUM_PERM)]

def minhash_signature(tokens: set[str]) -> tuple[int, ...]:
    if not tokens:
        return tuple([0] * LSH_NUM_PERM)
    vals = [_hash64(t) for t in tokens]
    sig = []
    for s in SALTS:
        m = min((v ^ s) for v in vals)
        sig.append(m)
    return tuple(sig)

def lsh_buckets(signatures: dict[str, tuple[int,...]]) -> dict[tuple[int,int], list[str]]:
    buckets = defaultdict(list)
    for _id, sig in signatures.items():
        for b in range(LSH_BANDS):
            start = b * LSH_ROWS_PER_BAND
            band = sig[start:start+LSH_ROWS_PER_BAND]
            bh = _hash64("|".join(map(str, band)))
            buckets[(b, bh)].append(_id)
    buckets = {k:v for k,v in buckets.items() if len(v) <= LSH_BUCKET_CAP}
    return buckets

def lsh_pairs_ba(pb_k: pl.DataFrame, pa_k: pl.DataFrame) -> pl.DataFrame:
    pb_pdf = pb_k.select(["profile_id","state","b_first","b_last"]).to_pandas()
    pa_pdf = pa_k.select(["npi_a","state","a_first","a_last"]).to_pandas()

    out = []
    for st in sorted(set(pa_pdf["state"].fillna("").tolist())):
        if not st:
            continue
        pa_s = pa_pdf[pa_pdf["state"].fillna("") == st]
        pb_s = pb_pdf[pb_pdf["state"].fillna("") == st]
        if pa_s.empty or pb_s.empty:
            continue

        sig_a = {}
        for _, r in pa_s.iterrows():
            toks = set(tokenize_name(r["a_first"], r["a_last"]))
            sig_a[str(r["npi_a"])] = minhash_signature(toks)

        sig_b = {}
        b_tokens = {}
        for _, r in pb_s.iterrows():
            toks = set(tokenize_name(r["b_first"], r["b_last"]))
            pid = str(r["profile_id"])
            b_tokens[pid] = toks
            sig_b[pid] = minhash_signature(toks)

        buckets_a = lsh_buckets(sig_a)
        buckets_b = lsh_buckets(sig_b)

        for k in set(buckets_a.keys()) & set(buckets_b.keys()):
            a_ids = buckets_a[k]
            b_ids = buckets_b[k]
            for pid in b_ids:
                for npi_a in a_ids[:CANOPY_MAX_CANDS]:
                    out.append((pid, npi_a, "lsh_minhash_state_name", st))

    if not out:
        return pl.DataFrame(schema={"profile_id":pl.Utf8,"npi_a":pl.Utf8,"pass":pl.Utf8,"block_key":pl.Utf8})
    return pl.DataFrame(out, schema=["profile_id","npi_a","pass","block_key"]).unique()

cand_ba_lsh = lsh_pairs_ba(pb_k, pa_k)
report_pass("BA_lsh_minhash_state_name", cand_ba_lsh, "profile_id", "npi_a", gold_ba_set)

# ============================================================
# Combining + deduping with priority
# ============================================================
PASS_PRIORITY = {
    "exact_npi": 0,
    "state_zip": 1,
    "state_city": 2,
    "state_lastkey": 2,
    "sorted_neighborhood_state_last_fi": 3,
    "canopy_state_name": 4,
    "lsh_minhash_state_name": 5,
}

def dedupe_with_priority(df: pl.DataFrame, id_left: str, id_right: str) -> pl.DataFrame:
    return (
        df.with_columns([
            pl.col("pass").map_elements(lambda x: PASS_PRIORITY.get(x, 99), return_dtype=pl.Int32).alias("pass_rank")
        ])
        .sort(["pass_rank"])
        .unique(subset=[id_left, id_right], keep="first")
        .drop("pass_rank")
    )

cand_ba_all = pl.concat([cand_ba_npi, cand_ba_statezip, cand_ba_statecity, cand_ba_sn, cand_ba_canopy, cand_ba_lsh], how="vertical_relaxed")
cand_bc_all = pl.concat([cand_bc_npi, cand_bc_statelast, cand_bc_sn], how="vertical_relaxed")

cand_ba_final = dedupe_with_priority(cand_ba_all, "profile_id", "npi_a")
cand_bc_final = dedupe_with_priority(cand_bc_all, "profile_id", "npi_c")

cand_ba_final.write_parquet(CAND_DIR / "cand_ba.parquet")
cand_bc_final.write_parquet(CAND_DIR / "cand_bc.parquet")

# ============================================================
# Write comparative pass report (precision/recall trade-off proxy)
# ============================================================
pass_df = pl.DataFrame(pass_reports).sort(["gold_recall","n_pairs"], descending=True)
pass_df.write_csv(CAND_DIR / "blocking_pass_comparison.csv")

metrics["ba_pairs_final"] = cand_ba_final.height
metrics["bc_pairs_final"] = cand_bc_final.height
metrics["WINDOW"] = WINDOW
metrics["ZIP_BLOCK_CAP"] = ZIP_BLOCK_CAP
metrics["LAST_BLOCK_CAP"] = LAST_BLOCK_CAP
metrics["CANOPY_T1"] = CANOPY_T1
metrics["CANOPY_T2"] = CANOPY_T2
metrics["LSH_NUM_PERM"] = LSH_NUM_PERM
metrics["LSH_BANDS"] = LSH_BANDS
metrics["n_providers_a"] = pa.height
metrics["n_providers_b"] = pb.height
metrics["n_providers_c"] = pc.height
metrics["gold_ba_pairs"] = len(gold_ba_set)
metrics["gold_bc_pairs"] = len(gold_bc_set)

with open(CAND_DIR / "blocking_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Wrote candidate pairs to:", CAND_DIR)
print("Wrote key stats:", CAND_DIR / "blocking_key_stats.csv")
print("Wrote mutual info:", CAND_DIR / "blocking_mutual_information.csv")
print("Wrote pass comparison:", CAND_DIR / "blocking_pass_comparison.csv")
print(json.dumps(metrics, indent=2))
